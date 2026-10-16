import json
import os
import re

import numpy as np
import pytest

FAST = {"d": 16, "lr": 1e-3, "dropout": 0.1, "n_blocks": 1, "epochs": 2}
ITEM = re.compile(r"\[item:(\d+)\]")


def test_trained_first_stage_feeds_cases_choice_records_and_fusion(workspace, tmp_path):
    pytest.importorskip("torch")
    from decision4rec.data.schema import Case, Dataset
    from decision4rec.experiments.asc import asc_study
    from decision4rec.experiments.benchmark import run_baselines
    from decision4rec.experiments.rectune import export_choices
    from decision4rec.experiments.runner import build_retriever, get_cases, load_config, make_prompt_builder, run_experiment
    from decision4rec.experiments.stage1 import chosen_dir
    from decision4rec.models import baselines as B

    bench = {"baselines": ["sasrec"], "grids": {"sasrec": [FAST]}}
    ref = workspace("rt_ref", K=15, out=str(tmp_path / "runs"), benchmark=bench)
    run_baselines(ref, device="cpu")
    stage = {"kind": "baseline", "model": "sasrec", "ref": ref, "tag": "sasrec_stage1"}
    arms = [{"name": "retriever", "method": "retriever"}, {"name": "rt", "method": "luce", "m": 0, "rounds": 2}]
    path = workspace("rt", K=12, out=str(tmp_path / "runs"), retriever=stage, arms=arms, benchmark=bench)
    cfg = load_config(path)
    ds = Dataset.load(cfg["dataset"])

    # first stage: the chosen checkpoint's own top K, history excluded, scores in decreasing order
    retr = build_retriever(cfg["retriever"], ds)
    hist = [[int(x[0]) for x in ds.seqs[u][:-1]] for u in range(20)]
    idx, val = retr.topk(hist, 12)
    assert all(not set(idx[r].tolist()) & set(hist[r]) for r in range(20))
    assert np.all(np.diff(val, axis=1) <= 1e-6)
    scorer = B.load("sasrec", chosen_dir(load_config(ref), "sasrec"), ds, "cpu")
    assert np.all(scorer.ranks(list(range(20)), hist, [int(i) for i in idx[:, 0]]) == 1)
    with pytest.raises(SystemExit):
        build_retriever({"kind": "baseline", "model": "bprmf", "ref": ref}, ds)

    # evaluation cases are the first stage's top K with its scores
    cases = get_cases(cfg, ds)
    c = cases["test"][0]
    i2, v2 = retr.topk([c.history], 12)
    assert c.cands == [int(i) for i in i2[0]] and np.allclose(c.rscores, v2[0], atol=1e-5)

    # training records: the second to last item (never a training target of the first stage) among the first
    # stage's candidates, users of evaluation cases left out, no test item read
    out = tmp_path / "rt"
    info = export_choices([path], str(out), n_per_dataset=8, n_heldout=2)
    assert info["train"] == 8 and info["heldout"] == 2 and info["datasets"][ds.name]["options"] == 12
    assert info["in_pool_only"] and info["version"] == 3 and info["datasets"][ds.name]["users_read"] >= 10
    eval_users = {c.user for split in cases.values() for c in split}
    assert info["datasets"][ds.name]["excluded_eval_users"] >= len(eval_users)
    assert os.path.exists(out / f"heldout_{ds.name}.jsonl")
    pb = make_prompt_builder(cfg, ds, need_kg=False)
    with open(out / "train.jsonl") as f, open(out / "train.meta.jsonl") as g:
        recs = [json.loads(x) for x in f]
        metas = [json.loads(x) for x in g]
    labels = set()
    for rec, meta in zip(recs, metas):
        q = rec["questions"]["next"]
        seq = ds.seqs[meta["user"]]
        assert meta["target"] == int(seq[-2][0]) and len(q["criteria"]) == 12 and q["label"] in q["criteria"]
        assert meta["user"] not in eval_users
        case = Case(cid=-1, user=meta["user"], history=[int(x[0]) for x in seq[:-2]], target=meta["target"],
                    cands=[meta["target"]], rscores=[0.0], target_pos=0)
        assert q["criteria"][q["label"]] == pb.option(case, meta["target"])
        shown = {int(m.group(1)) for t in q["criteria"].values() for m in ITEM.finditer(t)}
        assert not shown & set(case.history)                       # history items are never offered
        top, _ = retr.topk([case.history], 12)                     # the options are exactly the pool, target
        assert meta["target"] in {int(i) for i in top[0]} and shown == {int(i) for i in top[0]}  # never added
        assert rec["state"] == pb.state(case)
        labels.add(q["label"])
    assert len(labels) > 3                                              # the target's position is random

    # Kev-Rec stand-in (the oracle backend) on the cases, then the fusion with the first stage on validation users
    run_dir = run_experiment(path)
    run_baselines(path, device="cpu")
    res = asc_study([run_dir], arms=["rt"], head=5, only=["dm", "fuse", "gate", "shrink"])
    (run,) = res["runs"]
    assert [r["variant"] for r in run["rows"]] == ["rt", "rt+fuse", "rt+gate", "rt+shrink"]
    refs = run["refs"]
    assert refs["retriever (pool)"] == pytest.approx(refs["sasrec"])  # the pool is the trained model's own ranking


def test_rt_records_carry_the_arms_statements(workspace, tmp_path):
    pytest.importorskip("torch")
    from decision4rec.experiments.benchmark import run_baselines
    from decision4rec.experiments.rectune import export_choices

    bench = {"baselines": ["sasrec"], "grids": {"sasrec": [FAST]}}
    ref = workspace("rte_ref", K=15, out=str(tmp_path / "runs"), benchmark=bench)
    run_baselines(ref, device="cpu")
    stage = {"kind": "baseline", "model": "sasrec", "ref": ref, "tag": "sasrec_stage1"}
    arms = [{"name": "retriever", "method": "retriever"},
            {"name": "rte", "method": "luce", "m": 0, "rounds": 2, "evidence": "stats"}]
    path = workspace("rte", K=12, out=str(tmp_path / "runs"), retriever=stage, arms=arms, benchmark=bench, evidence={"frac": 1.0},
                     item_text={"embedder": {"kind": "hashing"}, "cache": str(tmp_path / "emb")})
    info = export_choices([path], str(tmp_path / "rte"), n_per_dataset=6, n_heldout=1)
    assert info["train"] == 6 and set(info["evidence"].values()) == {"stats"}
    with open(tmp_path / "rte" / "train.jsonl") as f:
        recs = [json.loads(x) for x in f]
    assert all("[sold " in t for r in recs for t in r["questions"]["next"]["criteria"].values())
    assert all("Store purchase data: full." in r["state"] for r in recs)


def test_agent_pool_ranker_cases_and_found_by_records(workspace, tmp_path):
    pytest.importorskip("torch")
    from decision4rec.agents.pool import pool_for
    from decision4rec.data.schema import Dataset
    from decision4rec.experiments.benchmark import run_baselines
    from decision4rec.experiments.rectune import export_choices
    from decision4rec.experiments.runner import get_cases, load_config, make_prompt_builder

    bench = {"baselines": ["sasrec"], "grids": {"sasrec": [FAST]}}
    ref = workspace("pool_ref", K=15, out=str(tmp_path / "runs"), benchmark=bench)
    run_baselines(ref, device="cpu")
    arms = [{"name": "retriever", "method": "retriever"},
            {"name": "rtm", "method": "luce", "m": 0, "rounds": 2, "evidence": "stats+found"}]
    path = workspace("rtm", K=12, out=str(tmp_path / "runs"), retriever={"kind": "pool", "tag": "pool"}, arms=arms, benchmark=bench,
                     evidence={"frac": 1.0}, pool={"ref": ref, "model": "sasrec", "users": 60, "seed": 1, "depth": 8,
                                                   "seq_depth": 15},
                     item_text={"embedder": {"kind": "hashing"}, "cache": str(tmp_path / "emb")})
    cfg = load_config(path)
    ds = Dataset.load(cfg["dataset"])
    pool = pool_for(cfg, ds)
    assert len(pool.train_users) == 60 and pool.lam >= 0.0
    cases = get_cases(cfg, ds)
    c = cases["test"][0]
    items, _ = pool.rank(c.user, c.history, 12)
    assert c.cands == items and not set(items) & set(c.history)
    pb = make_prompt_builder(cfg, ds)
    opt = pb.option(c, c.cands[0], evidence="stats+found")
    assert "found by " in opt and "[sold " in opt
    info = export_choices([path], str(tmp_path / "rtm"), n_per_dataset=5, n_heldout=1)
    with open(tmp_path / "rtm" / "train.meta.jsonl") as f:
        users = {json.loads(x)["user"] for x in f}
    assert info["train"] == 5 and not users & set(pool.train_users)
    with open(tmp_path / "rtm" / "train.jsonl") as f:
        rec = json.loads(f.readline())
    assert any("found by " in t for t in rec["questions"]["next"]["criteria"].values())


def test_kg_perspective_agents_join_the_pool(workspace, tmp_path):
    pytest.importorskip("torch")
    from decision4rec.agents.kgperspectives import NAMES, KGPerspectives
    from decision4rec.agents.pool import AGENTS, pool_for
    from decision4rec.data.schema import Dataset
    from decision4rec.experiments.benchmark import run_baselines
    from decision4rec.experiments.runner import load_config

    bench = {"baselines": ["sasrec"], "grids": {"sasrec": [FAST]}}
    ref = workspace("kgp_ref", K=15, out=str(tmp_path / "runs"), benchmark=bench)
    run_baselines(ref, device="cpu")
    path = workspace("kgp", K=12, out=str(tmp_path / "runs"), retriever={"kind": "pool", "tag": "poolp"},
                     benchmark=bench, evidence={"frac": 1.0},
                     pool={"ref": ref, "model": "sasrec", "users": 40, "seed": 1, "depth": 8, "seq_depth": 15,
                           "perspectives": "kg"},
                     item_text={"embedder": {"kind": "hashing"}, "cache": str(tmp_path / "emb")})
    cfg = load_config(path)
    ds = Dataset.load(cfg["dataset"])
    pool = pool_for(cfg, ds)
    assert pool.agents == AGENTS + tuple(NAMES) and pool.zcol == 2 * len(AGENTS) + 2 * len(NAMES)
    h = [int(x[0]) for x in ds.seqs[0][:-2]]
    L = KGPerspectives(ds, pool.store).lists(h, pool.store.now(0, len(h)), k=5)
    assert set(L) == set(NAMES) and all(len(v) <= 5 and not set(v) & set(h) for v in L.values())
    items, X, _ = pool.features(h, pool.store.now(0, len(h)), pool._sasrec_rows([0], [h])[0])
    assert X.shape[1] == pool.zcol + 6 and not set(items) & set(h)


def _script(name):
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_found_dropout_why_labels_and_crossfit_records(workspace, tmp_path):
    pytest.importorskip("torch")
    from decision4rec.agents.kgperspectives import NAMES, KGPerspectives, why_label, why_question
    from decision4rec.agents.pool import pool_for
    from decision4rec.data.schema import Dataset
    from decision4rec.experiments.benchmark import run_baselines
    from decision4rec.experiments.rectune import export_choices
    from decision4rec.experiments.runner import get_cases, load_config, make_prompt_builder

    bench = {"baselines": ["sasrec"], "grids": {"sasrec": [FAST]}}
    ref = workspace("cf_ref", K=15, out=str(tmp_path / "runs"), benchmark=bench)
    run_baselines(ref, device="cpu")
    arms = [{"name": "retriever", "method": "retriever"},
            {"name": "rtm", "method": "luce", "m": 0, "rounds": 2, "evidence": "stats+found"}]
    path = workspace("cf", K=12, out=str(tmp_path / "runs"), retriever={"kind": "pool", "tag": "pool"}, arms=arms,
                     benchmark=bench, evidence={"frac": 1.0},
                     pool={"ref": ref, "model": "sasrec", "users": 60, "seed": 1, "depth": 8, "seq_depth": 15},
                     item_text={"embedder": {"kind": "hashing"}, "cache": str(tmp_path / "emb")})
    cfg = load_config(path)
    ds = Dataset.load(cfg["dataset"])
    pool = pool_for(cfg, ds)
    cases = get_cases(cfg, ds)
    pb = make_prompt_builder(cfg, ds)

    # explanations v2: stripping "found by" gives exactly the option rendered without it
    strip = _script("drop_found").strip_found
    for c in cases["test"][:5]:
        for i in c.cands:
            assert strip(pb.option(c, i, evidence="stats+found")) == pb.option(c, i, evidence="stats")

    # cross-fitted records: only users that trained the pool's ranker, each ranked by a ranker that never saw it
    info = export_choices([path], str(tmp_path / "cf"), n_per_dataset=8, n_heldout=1, crossfit=True)
    with open(tmp_path / "cf" / "train.meta.jsonl") as f:
        metas = [json.loads(x) for x in f]
    users = {m["user"] for m in metas}
    assert info["train"] > 0 and users <= set(pool.train_users)
    assert sum(info["crossfit"][ds.name]) <= len(pool.train_users) and pb.pool is pool
    with open(tmp_path / "cf" / "train.jsonl") as f:
        assert any("found by " in t for t in json.loads(f.readline())["questions"]["next"]["criteria"].values())

    # perspectives v5: the rules label the real step; rates() reads the same rules over the past steps
    kp = KGPerspectives(ds, pool.store)
    h = [int(x[0]) for x in ds.seqs[0]]
    t = len(h) - 1
    hits = kp.explain(h[:t], h[t])
    assert len(hits) == len(NAMES) and why_label(hits) in NAMES + ("other",)
    r = kp.rates(h, back=1)
    assert np.allclose(r, (np.array(hits, dtype=float) + 0.2) / 2.0)
    q = why_question()
    assert list(q["criteria"]) == list(NAMES) + ["other"]
    _script("export_perspective_labels").main(str(tmp_path / "cf"), [path])
    _script("add_why_question").main(str(tmp_path / "cf"), str(tmp_path / "cf_v5"))
    with open(tmp_path / "cf_v5" / "train.jsonl") as f:
        recs = [json.loads(x) for x in f]
    assert len(recs) == len(metas) and all(r["questions"]["why"]["label"] in q["criteria"] for r in recs)
    assert all("next" in r["questions"] for r in recs)

    # fusion-aware records: every option carries the ranker's standardized score, aligned with the option keys
    from decision4rec.eval import metrics as M
    from decision4rec.experiments.runner import build_retriever
    info = export_choices([path], str(tmp_path / "off"), n_per_dataset=6, n_heldout=1, offsets=0.6)
    assert info["offsets"] == 0.6
    retr = build_retriever(cfg.get("retriever", {}), ds, cfg)
    K = int(cfg["eval"]["K"])
    with open(tmp_path / "off" / "train.jsonl") as f, open(tmp_path / "off" / "train.meta.jsonl") as g:
        pairs = [(json.loads(x), json.loads(y)) for x, y in zip(f, g)]
    assert pairs
    for rec, meta in pairs:
        nq = rec["questions"]["next"]
        assert list(nq["offset"]) == list(nq["criteria"])
        h = [int(x[0]) for x in ds.seqs[meta["user"]][:-2]]
        top, vals = retr.topk_users([meta["user"]], [h], K + 1)
        score = dict(zip([int(i) for i in top[0]], vals[0]))
        cands = [i for i in [int(i) for i in top[0]][:K] if i != meta["target"]][: K - 1] + [meta["target"]]
        z = M.zscore([score[i] for i in cands])
        assert nq["offset"][nq["label"]] == round(0.6 * float(z[-1]), 4)
        assert sorted(nq["offset"].values()) == sorted(round(0.6 * float(v), 4) for v in z)


def test_offset_trainer_wrapper(monkeypatch):
    torch = pytest.importorskip("torch")
    import sys
    import types
    kev, kd, kt = types.ModuleType("kev"), types.ModuleType("kev.data"), types.ModuleType("kev.train")

    def materialize(req):   # as Kev: internal option keys in the order of the request's criteria, no extra fields
        return {"questions": [{"keys": list(q["criteria"]), "label": list(q["criteria"]).index(q["label"])}
                              for q in req["questions"].values()]}

    def question_loss(z, q, dev, *args, **kw):
        return torch.nn.functional.cross_entropy(z[None], torch.tensor([q["label"]]))
    kd.materialize = kt.materialize = materialize
    kt.question_loss = question_loss
    kev.data, kev.train = kd, kt
    for name, mod in (("kev", kev), ("kev.data", kd), ("kev.train", kt)):
        monkeypatch.setitem(sys.modules, name, mod)
    wrapper = _script("kev_offset_train")
    wrapper.install()
    req = {"questions": {"next": {"criteria": {"o02": "b", "o01": "a", "none": "x"}, "label": "o01",
                                  "offset": {"o01": 1.0, "o02": -1.0}}}}
    q = kt.materialize(req)["questions"][0]
    assert q["offset"] == [-1.0, 1.0, 0.0] and q["label"] == 1
    want = -torch.log_softmax(torch.tensor([-1.0, 1.0, 0.0]), -1)[1]
    assert torch.isclose(kt.question_loss(torch.zeros(3), q, "cpu"), want)
    assert kt.materialize({"questions": {"n": {"criteria": {"a": 1}, "label": "a"}}})["questions"][0].get("offset") is None


def test_reserved_users_are_never_drawn(workspace, tmp_path):
    """R1: users reserved for the confirmation are left out of every later development draw (records, new cases,
    a new ranker), and a config with eval.users: reserved draws only them."""
    pytest.importorskip("torch")
    import shutil

    from decision4rec.agents.pool import pool_for
    from decision4rec.data.reserved import reserve, reserved_users, used_users
    from decision4rec.data.schema import Dataset
    from decision4rec.experiments.benchmark import run_baselines
    from decision4rec.experiments.rectune import export_choices
    from decision4rec.experiments.runner import get_cases, load_config

    data, out = str(tmp_path / "data"), str(tmp_path / "runs")      # own copy: the reservation stays in this test
    shutil.copytree(load_config(workspace("rs_probe"))["dataset"], data)
    bench = {"baselines": ["sasrec"], "grids": {"sasrec": [FAST]}}
    ref = workspace("rs_ref", K=15, out=out, dataset=data, benchmark=bench)
    run_baselines(ref, device="cpu")
    arms = [{"name": "retriever", "method": "retriever"},
            {"name": "rtm", "method": "luce", "m": 0, "rounds": 2, "evidence": "stats+found"}]
    path = workspace("rs", K=12, out=out, dataset=data, retriever={"kind": "pool", "tag": "pool"}, arms=arms,
                     benchmark=bench, evidence={"frac": 1.0},
                     pool={"ref": ref, "model": "sasrec", "users": 60, "seed": 1, "depth": 8, "seq_depth": 15},
                     item_text={"embedder": {"kind": "hashing"}, "cache": str(tmp_path / "emb")})
    cfg = load_config(path)
    ds = Dataset.load(data)
    get_cases(cfg, ds)
    pool_for(cfg, ds)
    info = reserve(cfg, n=30, seed=5)
    res = reserved_users(cfg)
    used = used_users(cfg, ds)
    assert info["n"] == len(res) == 30 and not res & (used["cases"] | used["ranker"] | used["records"])
    with pytest.raises(SystemExit):
        reserve(cfg, n=30, seed=5)
    export_choices([path], str(tmp_path / "rec"), n_per_dataset=400, n_heldout=5)
    with open(tmp_path / "rec" / "train.meta.jsonl") as f:
        assert not {json.loads(x)["user"] for x in f} & res
    dev2 = workspace("rs_dev2", K=12, out=out, dataset=data,
                     eval={"track": "B", "K": 12, "n_test": 150, "n_valid": 60, "seed": 9, "pool": 80})
    assert not {c.user for s in get_cases(load_config(dev2), ds).values() for c in s} & res
    conf = workspace("rs_conf", K=12, out=out, dataset=data,
                     eval={"track": "B", "K": 12, "n_test": 20, "n_valid": 5, "seed": 3, "pool": 80, "users": "reserved"})
    drawn = {c.user for s in get_cases(load_config(conf), ds).values() for c in s}
    assert drawn and drawn <= res
