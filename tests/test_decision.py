import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from decision4rec.decision import CachedBackend, Choice, Noul, Score, decide_chunked
from decision4rec.decision.mock import OracleBackend
from decision4rec.decision.systemone import ProtocolError, SystemOneBackend


def _truth(n_users=5, n_items=30, seed=0):
    rng = np.random.default_rng(seed)
    return {"utility": rng.normal(0, 1, (n_users, n_items)), "prior": rng.normal(0, 1, n_items),
            "group": rng.integers(0, 4, n_items)}


def _choice(items):
    return Choice("pick", tuple((f"o{k + 1:02d}", f"Item [item:{i}]") for k, i in enumerate(items)))


def test_cache_returns_identical_answers_without_calling_the_model(tmp_path):
    inner = OracleBackend(_truth())
    be = CachedBackend(inner, str(tmp_path / "c.sqlite"))
    q = {"a": _choice([1, 2, 3]), "b": Noul("like? [item:4]")}
    d1 = be.decide("[user:2]", q)
    d2 = be.decide("[user:2]", q)
    assert inner.calls == 1 and d2.cached and not d1.cached
    assert d1.answers["a"].probs == d2.answers["a"].probs
    be.decide("[user:3]", q)
    assert inner.calls == 2
    be.close()


def test_chunking_splits_requests_and_preserves_answers():
    whole = OracleBackend(_truth())
    split = OracleBackend(_truth(), max_request_chars=400)
    q = {f"b{j}": _choice(list(range(j * 3, j * 3 + 3))) for j in range(8)}
    a, b = decide_chunked(whole, "[user:1]", q), decide_chunked(split, "[user:1]", q)
    assert split.calls > 1 and whole.calls == 1 and b.n_requests == split.calls
    for k in q:
        assert a.answers[k].probs == pytest.approx(b.answers[k].probs)


class _Stub(BaseHTTPRequestHandler):
    """Answers in the response shape documented by the Kev README for POST /v1/systemone."""

    fail_first = 0

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if type(self).fail_first > 0:
            type(self).fail_first -= 1
            self.send_response(503)
            self.end_headers()
            return
        answers = {}
        for name, q in body["questions"].items():
            if q["type"] == "choice":
                keys = list(q["criteria"])
                p = np.arange(len(keys), 0, -1, dtype=float)
                p /= p.sum()
                answers[name] = {"type": "choice", "choice": keys[0], "confidence": 0.2,
                                 "probabilities": dict(zip(keys, p.tolist()))}
            elif q["type"] == "noul":
                answers[name] = {"type": "noul", "noul": 0.93}
            else:
                n = len(q["criteria"])
                answers[name] = {"type": "score", "score": 1.44, "confidence": 0.3,
                                 "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                                 "probabilities": {str(i): 1.0 / n for i in range(n)}}
        out = json.dumps({"model": body["model"], "answers": answers,
                          "usage": {"input_tokens": 101, "output_tokens": 0}, "latency_ms": 12}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture()
def stub_url():
    server = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_systemone_backend_speaks_the_documented_protocol(stub_url):
    be = SystemOneBackend(stub_url, model="kev-latest", api_key="local")
    d = be.decide("state", {"c": Choice("pick", (("x", "first"), ("y", None), ("z", "third"))),
                            "n": Noul("is it?"), "s": Score("how", ("low", "mid", "high"))})
    assert list(d.answers["c"].probs) == ["x", "y", "z"]
    assert sum(d.answers["c"].probs.values()) == pytest.approx(1.0)
    assert d.answers["n"].p == pytest.approx(0.93)
    assert d.answers["s"].value == pytest.approx(1.0)        # expected level from the probabilities, not the server's own scale
    assert d.input_tokens == 101 and d.latency_ms == 12


def test_systemone_backend_retries_transient_errors(stub_url, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    _Stub.fail_first = 2
    be = SystemOneBackend(stub_url, model="kev-latest")
    assert be.decide("s", {"n": Noul("q")}).answers["n"].p == pytest.approx(0.93)


def test_systemone_backend_reports_unreachable_server(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    be = SystemOneBackend("http://127.0.0.1:9", model="x", max_retries=1, timeout=1)
    with pytest.raises(ProtocolError):
        be.decide("s", {"n": Noul("q")})


def _tiny_lm():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers

    words = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [str(i) for i in range(1, 10)] + (
        "Yes No yes no Question Options Answer Reply with the letter label of single best option and nothing "
        "else or number matching level Scale low to high User history item alpha beta gamma delta one two "
        "three which next will like how much little lot : . ( )").split()
    vocab = {"[PAD]": 0, "[UNK]": 1}
    for w in words:
        vocab.setdefault(w, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(vocab_size=len(vocab), n_positions=256, n_embd=32, n_layer=2, n_head=2)
    return fast, transformers.GPT2LMHeadModel(cfg)


def test_logit_backend_distributions_and_prefix_cache_equivalence():
    from decision4rec.decision.logit import LogitBackend

    tok, lm = _tiny_lm()
    q = {"c": Choice("which item next", (("o1", "item alpha"), ("o2", "item beta"), ("o3", "item gamma"))),
         "n": Noul("will the User like item delta"),
         "s": Score("how much", ("little", "much", "lot")),
         "c2": Choice("which item next", (("o1", "item one"), ("o2", "item two")))}
    state = "User history : item alpha . item beta . item gamma"
    cached = LogitBackend("tiny", tokenizer=tok, lm=lm, prefix_cache=True, batch_size=2)
    plain = LogitBackend("tiny", tokenizer=tok, lm=lm, prefix_cache=False, batch_size=3)
    a, b = cached.decide(state, q), plain.decide(state, q)
    assert cached.prefix_cache, "prefix cache path failed on this transformers version"
    assert sum(a.answers["c"].probs.values()) == pytest.approx(1.0)
    assert list(a.answers["c"].probs) == ["o1", "o2", "o3"]
    assert 0.0 <= a.answers["n"].p <= 1.0 and 0.0 <= a.answers["s"].value <= 2.0
    for k in ("c", "c2"):
        for key, p in a.answers[k].probs.items():
            assert p == pytest.approx(b.answers[k].probs[key], abs=1e-4)
    assert a.answers["n"].p == pytest.approx(b.answers["n"].p, abs=1e-4)
    alone = plain.decide(state, {"c": q["c"]})
    assert alone.answers["c"].probs["o1"] == pytest.approx(b.answers["c"].probs["o1"], abs=1e-5)


def test_llm_ranker_training_matches_logit_backend(tmp_path):
    """R17: the LlamaRec-style trainer learns from records, merges its adapters exactly, and scores options with the
    prompt and labels of the logit backend, so training and evaluation read the same thing."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    import json

    from decision4rec.decision.logit import LogitBackend
    from decision4rec.experiments.llmrank import add_lora, encode, label_scores, merge_lora, train

    tok, _ = _tiny_lm()
    cfg = transformers.LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=256)
    torch.manual_seed(0)
    lm = transformers.LlamaForCausalLM(cfg)

    # adapters merge exactly into the base weights
    probe = transformers.LlamaForCausalLM(cfg)
    assert add_lora(probe, r=4, alpha=8.0) == 2 * 7          # q, k, v, o, gate, up, down in each of 2 layers
    for p in probe.parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p, std=0.05)
    x = torch.tensor([[2, 3, 4, 5]])
    before = probe(input_ids=x).logits
    assert merge_lora(probe) == 14
    assert torch.allclose(before, probe(input_ids=x).logits, atol=1e-5)

    rec = {"state": "User history : item alpha . item beta",
           "questions": {"next": {"type": "choice", "instructions": "which item next",
                                  "criteria": {"o01": "item alpha", "o02": "item beta", "o03": "item gamma"},
                                  "label": "o03"}}}
    data = tmp_path / "train.jsonl"
    data.write_text("\n".join(json.dumps(rec) for _ in range(24)) + "\n")
    m = train(str(data), "tiny", str(tmp_path / "out"), lr=5e-3, lora=4, alpha=8.0, accum=2, epochs=1,
              tokenizer=tok, model=lm, checkpointing=False, log_every=2)
    assert m["records_seen"] == 24 and m["adapters"] == m["merged"] == 14
    assert m["loss_every_10_steps"][-1] < m["loss_every_10_steps"][0]
    backend = LogitBackend("tiny", tokenizer=tok, lm=lm, use_chat_template=False, prefix_cache=False, batch_size=1)
    q = rec["questions"]["next"]
    ans = backend.decide(rec["state"], {"next": Choice(q["instructions"], tuple(q["criteria"].items()))})
    ids, labs, y = encode(backend, rec)
    with torch.no_grad():
        p = torch.softmax(label_scores(lm, ids, labs), 0).tolist()
    assert y == 2 and max(ans.answers["next"].probs, key=ans.answers["next"].probs.get) == "o03"
    assert p == pytest.approx([ans.answers["next"].probs[k] for k in ("o01", "o02", "o03")], abs=1e-5)
    assert (tmp_path / "out" / "training_metrics.json").exists()


def test_hf_embedder_last_token_pooling_with_left_padding(tmp_path):
    """Qwen3-Embedding pads on the left: last-token pooling must read the last position, not the mask length."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from decision4rec.retrieval.slm import HFEmbedder

    tok, _ = _tiny_lm()
    cfg = transformers.LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=256)
    torch.manual_seed(0)
    transformers.LlamaModel(cfg).save_pretrained(tmp_path / "m")
    tok.save_pretrained(tmp_path / "m")
    for side in ("left", "right"):
        e = HFEmbedder(str(tmp_path / "m"), pooling="last", device="cpu", batch_size=4)
        e.tok.padding_side = side
        both = e._encode(["item alpha beta gamma", "item"])
        alone = e._encode(["item"])
        assert np.allclose(both[1], alone[0], atol=1e-5), side


def test_llm_ranker_shrink_options_keeps_the_target():
    """R17 v3: a question cut to k options keeps the chosen option, draws the others from the same list, and leaves
    the record unchanged when k is 0 or not smaller than the list."""
    import random
    from decision4rec.experiments.llmrank import shrink_options
    crit = {f"o{i:02d}": f"item {i}" for i in range(1, 31)}
    rec = {"state": "s", "questions": {"next": {"type": "choice", "instructions": "pick", "criteria": crit, "label": "o17"}}}
    rng = random.Random(0)
    for _ in range(50):
        q = shrink_options(rec, 5, rng)["questions"]["next"]
        assert len(q["criteria"]) == 5 and "o17" in q["criteria"] and q["label"] == "o17"
        assert all(q["criteria"][k] == crit[k] for k in q["criteria"])
    assert shrink_options(rec, 0, rng) is rec and shrink_options(rec, 30, rng) is rec
    assert rec["questions"]["next"]["criteria"] is crit and len(crit) == 30
