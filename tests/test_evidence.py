import json
import re

import numpy as np
import pytest

ITEM = re.compile(r"\[item:(\d+)\]")


def _workspace_config(workspace, tmp_path, name, frac, K=12, **extra):
    arms = [{"name": "retriever", "method": "retriever"},
            {"name": "ev", "method": "luce", "m": 0, "rounds": 2, "evidence": "stats"}]
    return workspace(name, K=K, retriever={"kind": "evidence", "tag": f"ev{int(round(frac * 100)):03d}"},
                     evidence={"frac": frac}, item_text={"embedder": {"kind": "hashing"}, "cache": str(tmp_path / "emb")},
                     prompt={"item_chars": 60}, arms=arms, **extra)


def test_store_counts_regimes_statements_and_stable_candidates(workspace):
    from decision4rec.data.schema import Dataset
    from decision4rec.evidence.store import EvidenceStore
    from decision4rec.experiments.runner import load_config

    ds = Dataset.load(load_config(workspace("ev_probe"))["dataset"])
    text = np.random.default_rng(0).normal(size=(ds.n_items, 8))
    full = EvidenceStore(ds, 1.0, holdout=2, text=text)
    expect = np.zeros(ds.n_items, dtype=int)
    for s in ds.seqs:                                     # purchases of the training parts only
        for x in s[:-2]:
            expect[int(x[0])] += 1
    assert np.array_equal(full.pop, expect)
    held = EvidenceStore(ds, 1.0, holdout=3, text=text)   # for training records: every last training item left out
    assert held.pop.sum() == expect.sum() - sum(1 for s in ds.seqs if len(s) > 2)
    none = EvidenceStore(ds, 0.0, text=text)
    assert not none.has_data and none.regime_line().startswith("Store purchase data: none")
    part = EvidenceStore(ds, 0.2, text=text, seed=1)
    assert 0 < part.pop.sum() < full.pop.sum() and "partial" in part.regime_line()

    h = [int(x[0]) for x in ds.seqs[0][:-1]]
    now = full.now(0, len(h))
    c30, c60 = full.candidates(h, now, 30), full.candidates(h, now, 60)
    assert c60[:30] == c30 and len(set(c60)) == 60 and not set(c60) & set(h)   # longer lists only append
    ctx = full.context(h, now)
    tag = full.tags(ctx, c30[0])
    assert all(w in tag for w in ("sold ", "lately ", "after ", "with yours ", "text "))
    assert "sold " not in full.tags(ctx, c30[0], frozenset({"sold"}))
    nctx = none.context(h, now)
    assert none.tags(nctx, c30[0]).startswith("[text ")                  # no purchase data: text only
    assert none.candidates(h, now, 30) == [int(j) for j in np.argsort(nctx.text_rank, kind="stable")[:30]]


def test_evidence_records_keep_pool_targets_mix_regimes_and_drop_statements(workspace, tmp_path):
    from decision4rec.data.schema import Dataset
    from decision4rec.evidence.export import export_evidence_choices
    from decision4rec.evidence.store import store_for
    from decision4rec.experiments.runner import load_config

    path = _workspace_config(workspace, tmp_path, "ev_src", 1.0)
    info = export_evidence_choices([path], str(tmp_path / "ev"), n_per_dataset=25, n_heldout=4, dropout=0.3)
    assert info["train"] == 25 and info["heldout"] == 4 and info["in_pool_only"]
    cfg = load_config(path)
    ds = Dataset.load(cfg["dataset"])
    with open(tmp_path / "ev" / "train.jsonl") as f, open(tmp_path / "ev" / "train.meta.jsonl") as g:
        pairs = [(json.loads(a), json.loads(b)) for a, b in zip(f, g)]
    assert len({m["frac"] for _, m in pairs}) >= 3 and any(m["drop"] for _, m in pairs)
    for rec, meta in pairs:
        q = rec["questions"]["next"]
        seq = ds.seqs[meta["user"]]
        texts = list(q["criteria"].values())
        ids = [int(ITEM.search(t).group(1)) for t in texts]
        assert meta["target"] == int(seq[-3][0]) and len(texts) == 12
        assert ids[int(q["label"][1:]) - 1] == meta["target"]             # the label is the real next item
        store = store_for(cfg, ds, holdout=3, frac=meta["frac"])
        h = [int(x[0]) for x in seq[:-3]]
        assert set(ids) == set(store.candidates(h, store.now(meta["user"], len(h)), 12))
        if meta["frac"] == 0.0:
            assert all("[sold " not in t for t in texts) and "none" in rec["state"]
        elif "sold" in meta["drop"]:
            assert all("[sold " not in t for t in texts)
        else:
            assert all("[sold " in t for t in texts)


def test_evidence_cases_runs_and_fusion(workspace, tmp_path):
    pytest.importorskip("torch")
    from decision4rec.data.schema import Dataset
    from decision4rec.evidence.prompt import EvidencePromptBuilder
    from decision4rec.evidence.store import store_for
    from decision4rec.experiments.asc import asc_study
    from decision4rec.experiments.runner import get_cases, load_config, make_prompt_builder, run_experiment

    path = _workspace_config(workspace, tmp_path, "ev_run", 1.0, K=10)
    cfg = load_config(path)
    ds = Dataset.load(cfg["dataset"])
    pb = make_prompt_builder(cfg, ds)
    assert isinstance(pb, EvidencePromptBuilder)
    cases = get_cases(cfg, ds)
    store = store_for(cfg, ds)
    for c in cases["test"][:10]:
        assert c.cands == store.candidates(c.history, store.now(c.user, len(c.history)), 10)
    c = cases["test"][0]
    assert "Store purchase data: full." in pb.state(c) and "[sold " in pb.option(c, c.cands[0], evidence="stats")
    assert "[sold " not in pb.option(c, c.cands[0])                       # only arms that ask for evidence read it
    run_dir = run_experiment(path)
    run = asc_study([run_dir], arms=["ev"], head=5, only=["dm", "fuse"])["runs"][0]
    assert [r["variant"] for r in run["rows"]] == ["ev", "ev+fuse"]

    cold = load_config(_workspace_config(workspace, tmp_path, "ev_cold", 0.0, K=10))
    for c in get_cases(cold, ds)["test"][:5]:                              # a new store: text neighbours only
        s0 = store_for(cold, ds)
        assert c.cands == s0.candidates(c.history, s0.now(c.user, len(c.history)), 10)


def test_fused_candidates_are_stable_and_leave_history_out(workspace, tmp_path):
    from decision4rec.data.schema import Dataset
    from decision4rec.evidence.export import export_evidence_choices
    from decision4rec.evidence.store import EvidenceStore
    from decision4rec.experiments.runner import load_config

    ds = Dataset.load(load_config(workspace("ev_probe"))["dataset"])
    text = np.random.default_rng(0).normal(size=(ds.n_items, 8))
    store = EvidenceStore(ds, 1.0, holdout=2, text=text, fusion=(1, 1, 0.6, 0.3))
    for u in range(10):
        h = [int(x[0]) for x in ds.seqs[u][:-1]]
        now = store.now(u, len(h))
        c10, c40 = store.candidates(h, now, 10), store.candidates(h, now, 40)
        assert c40[:10] == c10 and len(set(c40)) == 40 and not set(c40) & set(h)
        s = store.fused_scores(h, now)
        assert all(s[c40[i]] >= s[c40[i + 1]] for i in range(39))

    path = _workspace_config(workspace, tmp_path, "ev_fused", 1.0, K=12)
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["evidence"]["fusion"] = [1, 1, 0.6, 0.3]
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    info = export_evidence_choices([path], str(tmp_path / "evf"), n_per_dataset=20, n_heldout=2, dropout=0.0)
    assert info["train"] == 20 and info["version"] == 2
