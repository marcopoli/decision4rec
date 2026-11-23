import os
from types import SimpleNamespace

import numpy as np

from decision4rec.data.candidates import build_cases, planned_users


class Listed:
    def topk(self, hist, k):
        return np.tile(np.arange(k), (len(hist), 1)), np.tile(np.arange(k, 0, -1.0), (len(hist), 1))


def test_planned_users_cover_the_draw_with_reserved_users_and_post_targets():
    seqs = [[(i, float(t + (5 if u % 2 == 0 and t == 5 else 0)), 1.0) for t, i in enumerate(range(10, 16))] for u in range(300)]
    ds = SimpleNamespace(n_users=300, n_items=40, seqs=seqs, extra={"cold_protocol": {"cutoff": 8.0}})
    skip = set(range(0, 300, 7))
    post = {u for u in range(300) if u % 2 == 0}
    test, _ = build_cases(ds, Listed(), "test", 20, 5, "B", 13, 50, exclude_users=skip, only_users=post)
    valid, _ = build_cases(ds, Listed(), "valid", 10, 5, "B", 13, 50, exclude_users=skip | {c.user for c in test})
    cfg = {"seed": 13, "track": "B", "n_test": 20, "n_valid": 10, "target": "post"}
    assert {c.user for c in test + valid} <= planned_users(ds, cfg, skip)
    assert not planned_users(ds, cfg, skip) & skip


def test_reserved_cases_never_share_a_file_with_development_cases():
    from decision4rec.experiments.runner import case_paths
    base = {"out": "runs/x", "retriever": {"tag": "pool"}, "eval": {"track": "B", "K": 30, "seed": 13, "n_test": 2000, "n_valid": 1000}}
    r1 = {**base, "eval": {**base["eval"], "users": "reserved"}}
    dev, res = case_paths(base), case_paths(r1)
    assert dev["valid"].endswith("pool_trackB_K30_s13_valid1000.jsonl")   # development names unchanged
    assert not set(dev.values()) & set(res.values())


def test_ranker_cases_are_cross_fitted_and_disjoint_from_development_users(tmp_path, monkeypatch):
    """C4 labels: each half of the ranker users is ranked by a ranker refit on the other half only; target second to last."""
    import pytest
    import decision4rec.agents.pool as P
    from decision4rec.data.schema import save_cases
    from decision4rec.experiments import runner as R
    seqs = [[(i, float(t), 1.0) for t, i in enumerate(range(10, 17))] for _ in range(300)]
    ds = SimpleNamespace(n_users=300, n_items=40, seqs=seqs, extra={})
    fitted = []

    class Fake:
        def __init__(self, users): self.train_users = list(users)
        def save(self, path): open(path, "w").close()

    class Ret:
        def __init__(self, pool): self.pool = pool
        def topk_users(self, users, hist, k): return Listed().topk(hist, k)

    def refit(main, users, pc):
        fitted.append(set(users))
        return Fake(users)

    monkeypatch.setattr(P, "refit", refit)
    monkeypatch.setattr(P, "PoolRetriever", Ret)
    ranker = list(range(100, 160))
    cfg = {"out": str(tmp_path), "dataset": str(tmp_path), "retriever": {"tag": "pool"},
           "eval": {"track": "B", "K": 5, "seed": 13, "pool": 50, "n_test": 30, "n_valid": 30, "users": "ranker",
                    "dev_n_test": 20, "dev_n_valid": 10}}
    dev = R.case_paths({**cfg, "eval": {k: v for k, v in cfg["eval"].items() if k != "users"} | {"n_test": 20, "n_valid": 10}})
    os.makedirs(os.path.dirname(dev["test"]))
    t, _ = build_cases(ds, Listed(), "test", 20, 5, "B", 13, 50, only_users=set(range(0, 50)))
    v, _ = build_cases(ds, Listed(), "valid", 10, 5, "B", 13, 50, only_users=set(range(50, 100)))
    save_cases(t, dev["test"]); save_cases(v, dev["valid"])
    paths = R.case_paths(cfg)
    assert paths["test"].endswith("_ranker_test30.jsonl")
    out = R._ranker_cases(cfg, ds, Ret(Fake(ranker)), paths, 5, 13, 50)
    h0, h1 = ({c.user for c in out[s]} for s in ("test", "valid"))
    assert h0 and h1 and not h0 & h1 and h0 | h1 <= set(ranker)
    assert fitted == [h1, h0]                                # "test" half ranked by a ranker fit on the other half, and back
    assert all(c.target == seqs[c.user][-2][0] for c in out["test"] + out["valid"])
    with pytest.raises(AssertionError):                       # a ranker user among the development users is refused
        R._ranker_cases(cfg, ds, Ret(Fake(ranker + [0])), paths, 5, 13, 50)
