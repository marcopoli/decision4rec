import os

import numpy as np
import pytest


def test_groups_partition_the_test_users_and_average_back_to_the_benchmark(workspace):
    pytest.importorskip("torch")
    from decision4rec.experiments.benchmark import run_baselines
    from decision4rec.experiments.runner import run_experiment
    from decision4rec.experiments.stratify import stratify

    arms = [{"name": "retriever", "method": "retriever"}, {"name": "wholepool@p0", "method": "luce", "m": 0, "perm": 0}]
    path = workspace("strat", K=20, arms=arms)
    summary = run_baselines(path, only=["pop", "itemknn"], device="cpu")
    run_dir = run_experiment(path)
    res = stratify([run_dir, os.path.join(run_dir, "missing")], pairs=[("strat: wholepool@p0", "pop")])
    assert res["skipped"] == [os.path.join(run_dir, "missing")]
    (d,) = res["datasets"]
    for feat in ("target popularity", "history length"):
        t = d["tables"][feat]
        table, groups = t["table"], t["groups"]
        assert sum(table["n"].values()) == d["n"] == 60
        assert {"pop", "itemknn", "retriever (pool)", "strat: wholepool@p0"} <= set(table)
        for name in ("pop", "strat: wholepool@p0"):
            weighted = sum(table[name][g] * table["n"][g] for g in groups) / d["n"]
            assert weighted == pytest.approx(table[name]["all"])
        assert all(0.0 <= table["in pool"][g] <= 1.0 for g in groups)
        assert len(t["tests"]) == len(groups) and all(0.0 <= x["p"] <= 1.0 for x in t["tests"])
    # the baseline rows average back to the numbers of the benchmark
    for name in ("pop", "itemknn"):
        assert d["tables"]["history length"]["table"][name]["all"] == pytest.approx(summary[name]["ndcg@10"])
    assert "## " in res["markdown"] and np.isfinite(d["tables"]["target popularity"]["table"]["pop"]["all"])


def test_baselines_are_shared_by_case_sets_with_the_same_users(workspace):
    pytest.importorskip("torch")
    from decision4rec.data.schema import Dataset, load_cases
    from decision4rec.experiments.benchmark import run_baselines
    from decision4rec.experiments.runner import case_paths, get_cases, load_config
    from decision4rec.experiments.stratify import baseline_ranks

    first = workspace("share_a", K=20)
    run_baselines(first, only=["pop"], device="cpu")
    other = load_config(workspace("share_b", K=20, retriever={"kind": "pop"}))
    get_cases(other, Dataset.load(other["dataset"]))
    assert case_paths(other)["test"] != case_paths(load_config(first))["test"]
    cases = load_cases(case_paths(other)["test"])
    found = baseline_ranks(other, [[c.user, c.target] for c in cases], ("pop", "no_such_model"))
    assert set(found) == {"pop"} and len(found["pop"]) == len(cases)
    assert baseline_ranks(other, [[c.user, c.target + 1] for c in cases], ("pop",)) == {}
