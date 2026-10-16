import json
import os

import numpy as np
import pytest


def test_item_constant_study_reproduces_the_report_and_tunes_on_validation(workspace):
    pytest.importorskip("torch")
    from decision4rec.experiments.asc import ALPHAS, asc_study
    from decision4rec.experiments.benchmark import BIG, _ndcg10, run_baselines
    from decision4rec.experiments.report import report
    from decision4rec.experiments.runner import run_experiment

    arms = [{"name": "retriever", "method": "retriever"},
            {"name": "wholepool@p0", "method": "luce", "m": 0, "perm": 0},
            {"name": "luce_m10_r2", "method": "luce", "m": 10, "rounds": 2}]
    path = workspace("asc", K=20, arms=arms)
    run_baselines(path, only=["pop"], device="cpu")
    run_dir = run_experiment(path)
    report(run_dir)
    res = asc_study([run_dir], head=5)
    (run,) = res["runs"]
    assert run["arms"] == ["wholepool@p0", "luce_m10_r2"]
    names = [r["variant"] for r in run["rows"]]
    assert names[0] == "pop only" and len(names) == 1 + 8 * 2
    assert {"luce_m10_r2+pop", "luce_m10_r2+fuse", "luce_m10_r2+fuse+pop", "luce_m10_r2+gate", "luce_m10_r2+ugate",
            "luce_m10_r2+ugate+pop", "luce_m10_r2+shrink"} <= set(names)
    # the decision-model-only rows and the fusion rows equal what the report stored
    assert all(abs(v) < 1e-9 for v in run["check_against_report"].values()) and len(run["check_against_report"]) == 2
    with open(os.path.join(run_dir, "ranks.json")) as f:
        stored = json.load(f)
    rows = {r["variant"]: r for r in run["rows"]}
    for arm in ("wholepool@p0", "luce_m10_r2"):
        st = stored[f"{arm}+fuse"]["test"]
        ref = _ndcg10(np.array([BIG if st[c] is None else st[c] for c in sorted(st, key=int)], dtype=float)).mean()
        assert rows[f"{arm}+fuse"]["ndcg@10"] == pytest.approx(ref)
        a = float(rows[f"{arm}+pop"]["weights"].split("=")[1])
        assert a in ALPHAS
    n = sum(run["n_by_popularity"].values())
    assert n == run["n_test"] == 60
    for r in run["rows"]:
        weighted = sum(r["by_popularity"][g] * run["n_by_popularity"][g] for g in run["groups"]) / n
        assert weighted == pytest.approx(r["ndcg@10"])
        assert all(0.0 <= v["p"] <= 1.0 for v in r["vs"].values())
    # pop is not a reference row; trained baselines of other tests on the same users may be
    assert {"retriever (pool)", "retriever+pop"} <= set(run["refs"]) <= {"retriever (pool)", "retriever+pop", "gru4rec",
                                                                       "sasrec", "sasrecf", "sasrecf_dm", "itemknn"}
    assert run["retriever+pop"]["a"] in ALPHAS and np.isfinite(run["niche_cut"])
    assert "## " in res["markdown"]

    # the pre-registered mode computes only the named kinds and adds bootstrap intervals
    conf = asc_study([run_dir], arms=["wholepool@p0"], head=5, only=["gate", "shrink"])["runs"][0]
    assert [r["variant"] for r in conf["rows"]] == ["wholepool@p0+gate", "wholepool@p0+shrink"]
    assert conf["rows"][0]["ndcg@10"] == pytest.approx(rows["wholepool@p0+gate"]["ndcg@10"])
    for r in conf["rows"]:
        for d in r["vs"].values():
            lo, hi = d["ci95"]
            assert lo <= d["diff"] <= hi
    with pytest.raises(SystemExit):
        asc_study([run_dir], only=["nonsense"])
