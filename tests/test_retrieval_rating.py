import json
import os

import numpy as np
import pytest

from decision4rec.eval import metrics as M
from decision4rec.retrieval.agents import RetrievalContext, rrf
from decision4rec.retrieval.slm import EchoWriter, HashingEmbedder

MULTI = {"kind": "multi", "tag": "multi", "base": {"kind": "itemknn"}, "embedder": {"kind": "hashing"},
         "writer": {"kind": "echo"}, "weights": "tuned", "tune_users": 80,
         "agents": ["sequential", "content", "knowledge", "collaborative", "graph", "popularity"]}


def test_ordinal_metrics():
    gold = np.array([0, 1, 2, 3, 4] * 20)
    assert M.quadratic_weighted_kappa(gold, gold, 5) == pytest.approx(1.0)
    assert M.quadratic_weighted_kappa(np.full(100, 2), gold, 5) == pytest.approx(0.0)
    full = M.scale_utilisation(np.bincount(gold, minlength=5), np.bincount(gold, minlength=5))
    squeezed = M.scale_utilisation(np.array([0, 30, 40, 30, 0]), np.bincount(gold, minlength=5))
    assert full["R"] == pytest.approx(1.0) and full["tvd"] == pytest.approx(0.0)
    assert squeezed["R"] < 0.7 and squeezed["tvd"] > 0.3


def test_small_model_stand_ins(tmp_path):
    emb = HashingEmbedder(256)
    v = emb.encode(["red racing game for console", "racing game on console in red", "cook book of pasta recipes"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert v[0] @ v[1] > v[0] @ v[2] + 0.2
    w = EchoWriter(str(tmp_path / "w.sqlite"))
    prompt = "Recent items:\n- alpha\n- beta\nDescribe the next item."
    assert w.write([prompt, prompt]) == ["beta alpha", "beta alpha"]
    again = EchoWriter(str(tmp_path / "w.sqlite"))
    again._generate = lambda prompts: pytest.fail("the cached text must be reused")
    assert again.write([prompt]) == ["beta alpha"]
    fused = rrf([[1, 2, 3], [3, 2, 9]], [1.0, 1.0])
    assert {i for i, _ in fused[:2]} == {2, 3} and len(fused) == 4      # items both lists agree on come first
    assert [i for i, _ in rrf([[1, 2, 3], [3, 2, 9]], [1.0, 0.0])] == [1, 2, 3]


def test_multi_agent_retriever_and_study(workspace):
    from decision4rec.data.schema import Dataset
    from decision4rec.experiments.retrieval_study import run_retrieval_study
    from decision4rec.experiments.runner import build_retriever, case_paths, get_cases, load_config

    path = workspace("multi", retriever=dict(MULTI, cache=os.path.join(os.path.dirname(workspace("tmp")), "retr")))
    cfg = load_config(path)
    ds = Dataset.load(cfg["dataset"])
    retriever = build_retriever(cfg["retriever"], ds)
    assert set(retriever.weights) == set(MULTI["agents"])
    assert retriever.tune_info["objective"] >= retriever.tune_info["uniform_objective"]
    hist = [[int(x[0]) for x in s[:-1]] for s in ds.seqs[:4]]
    idx, val = retriever.topk(hist, 30)
    assert idx.shape == (4, 30) and np.all(np.diff(val, axis=1) <= 1e-9)
    assert all(len(set(row)) == 30 and not set(row) & set(h) for row, h in zip(idx.tolist(), hist))
    views = {a.name: a.propose(RetrievalContext(hist[0]), 15) for a in retriever.agents}
    assert all(len(v) > 0 and not set(v) & set(hist[0]) for v in views.values())
    assert len({tuple(v) for v in views.values()}) > 3              # the points of view really differ
    cases = get_cases(cfg, ds)                                       # cases can be built from the fused agents
    assert "multi_track" in case_paths(cfg)["test"] and len(cases["test"]) == 60
    out = run_retrieval_study(path, n_users=120, cutoffs=(10, 30, 60))
    assert set(out["agents"]) == set(MULTI["agents"]) and "without_sequential" in out["leave_one_agent_out"]
    assert out["union_recall"] >= max(a["recall@60"] for a in out["agents"].values())
    assert out["fusion"]["tuned"]["recall@30"] > 0


def test_rating_study_detects_scale_compression(workspace):
    from decision4rec.experiments.rating import run_rating, to_level

    assert [to_level(r, 3) for r in (1, 2, 3, 4, 5)] == [0, 0, 1, 2, 2]
    assert [to_level(r, 2) for r in (1, 3, 4, 5)] == [0, 0, 1, 1]
    faithful = run_rating(workspace("rate_ok", backend={"score_compress": 0.0}), n_cases=60)
    squeezed = run_rating(workspace("rate_sq", backend={"score_compress": 0.7}), n_cases=60)
    f, q = faithful["variants"]["rated"], squeezed["variants"]["rated"]
    assert q["scale5"]["utilisation_R"] < f["scale5"]["utilisation_R"]
    assert q["rmse"] > f["rmse"] and q["rmse_affine"] < q["rmse"]    # an affine correction repairs compression
    assert f["rmse"] < faithful["baselines"]["global_mean"]["rmse"]
    assert f["rmse"] < faithful["variants"]["null"]["rmse"]          # the history carries the signal
    assert set(faithful["variants"]) == {"rated", "unrated", "null", "profile"}
    assert os.path.exists(os.path.join(faithful["out_dir"], "rating.md"))
    with open(os.path.join(faithful["out_dir"], "rating_rated.test.jsonl")) as fh:
        n = len(fh.readlines())
    run_rating(workspace("rate_ok", backend={"score_compress": 0.0}), n_cases=60)   # resumable
    with open(os.path.join(faithful["out_dir"], "rating_rated.test.jsonl")) as fh:
        assert len(fh.readlines()) == n


def test_retrieval_agents_run_as_behaviours_with_decision_routing(workspace):
    pytest.importorskip("activegraph")
    from decision4rec.agents.active import DecisionAgent
    from decision4rec.data.schema import Dataset
    from decision4rec.decision.base import build_backend
    from decision4rec.experiments.runner import build_retriever, get_cases, load_config, make_prompt_builder
    from decision4rec.retrieval.agents import build_multi_retriever

    cfg = load_config(workspace("ragent", K=50))
    ds = Dataset.load(cfg["dataset"])
    pb = make_prompt_builder(cfg, ds)
    case = get_cases(cfg, ds)["test"][0]
    backend = build_backend({k: v for k, v in cfg["backend"].items() if k != "cache"})
    names = ["sequential", "content", "knowledge", "collaborative", "profile", "graph", "popularity"]
    multi = build_multi_retriever({"agents": names, "embedder": {"kind": "hashing"}, "writer": {"kind": "echo"}},
                                  ds, build_retriever({"kind": "itemknn"}, ds), pb.kg)
    results = {route: DecisionAgent(backend, pb, tau_expand=float("inf"), retrieval=multi, route=route, pool=40).run(case)
               for route in ("uniform", "decision")}
    res = results["decision"]
    assert res.events.count("channel.proposed") == len(names) and res.events.count("retrieval.requested") == 1
    assert set(res.route) == set(names) and len(set(res.route.values())) > 1      # the decision set unequal weights
    assert set(results["uniform"].route.values()) == {1.0}
    assert len(res.items) == 40 and not set(res.items) & set(pb.profile_history(case))
    assert res.requests == results["uniform"].requests + 1                        # routing costs one request
    again = DecisionAgent(backend, pb, tau_expand=float("inf"), retrieval=multi, route="decision", pool=40).run(case)
    assert again.slate == res.slate and again.events == res.events


def test_hugging_face_embedder_and_writer_code_paths(tmp_path):
    """Tiny randomly initialised models saved locally: checks pooling, batching, generation and caching
    without downloading anything."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers

    from decision4rec.retrieval.slm import HFEmbedder, HFWriter

    words = "recent items of a user describe the next item racing game console pasta cook book red".split()
    vocab = {"[PAD]": 0, "[UNK]": 1, **{w: i + 2 for i, w in enumerate(dict.fromkeys(words))}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")
    torch.manual_seed(0)
    enc_dir, gen_dir = str(tmp_path / "enc"), str(tmp_path / "gen")
    transformers.BertModel(transformers.BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1,
                                                   num_attention_heads=2, intermediate_size=64)).save_pretrained(enc_dir)
    transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=len(vocab), n_positions=64, n_embd=32,
                                                         n_layer=1, n_head=2)).save_pretrained(gen_dir)
    fast.save_pretrained(enc_dir)
    fast.save_pretrained(gen_dir)

    texts = ["racing game console", "red racing game", "pasta cook book"]
    for pooling in ("cls", "mean", "last"):
        emb = HFEmbedder(enc_dir, pooling=pooling, device="cpu", batch_size=2, query_prefix="describe ")
        v = emb.encode(texts)
        assert v.shape == (3, 32) and np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
        assert np.allclose(emb.encode(texts[:1]), v[:1])                         # memoised and stable
        assert not np.allclose(emb.encode(texts[:1], is_query=True), v[:1])      # the query prefix is applied
    writer = HFWriter(gen_dir, max_new_tokens=4, batch_size=2, dtype="fp32", device="cpu",
                      cache=str(tmp_path / "w.sqlite"))
    prompts = ["recent items of a user racing game", "describe the next item", "pasta cook book"]
    first = writer.write(prompts)
    assert len(first) == 3 and all(isinstance(t, str) for t in first)
    alone = HFWriter(gen_dir, max_new_tokens=4, batch_size=1, dtype="fp32", device="cpu").write(prompts)
    assert alone == first                                                       # greedy, independent of batching
    writer._generate = lambda p: pytest.fail("cached generations must be reused")
    assert writer.write(prompts) == first
