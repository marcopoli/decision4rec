"""C1: Decision4Rec-Full on ActiveGraph runs pool -> decide -> estimate -> fuse and returns the fused top 10."""
from types import SimpleNamespace

import numpy as np

import decision4rec.agents.full_runtime as FR
from decision4rec.data.schema import Case
from decision4rec.eval.metrics import zscore
from decision4rec.operators.luce import Obs


class Ret:
    def topk_users(self, users, hist, k):
        return np.arange(100, 100 + k)[None, :], np.linspace(1.0, 0.0, k)[None, :]


def test_runtime_slate_is_the_fused_top10(monkeypatch):
    K = 12
    rng = np.random.default_rng(0)
    obs = []
    for _ in range(4):
        items = rng.permutation(K)[:6]
        p = rng.dirichlet(np.ones(6))
        obs.append(Obs(items=items, probs=p).to_json())
    monkeypatch.setattr(FR, "run_arm", lambda case, arm, backend, pb, seed=13: {"obs": obs, "cached": True})
    w = np.array([1.0, 0.5])
    agent = FR.FullAgent(Ret(), SimpleNamespace(), None, None, w, K=K)
    case = Case(cid=7, user=1, history=[1, 2, 3], target=105, cands=[], rscores=[], target_pos=-1, split="test", target_rating=0.0)
    s = agent.serve(case)
    theta = FR.fit_luce([Obs.from_json(o) for o in obs], K).theta
    want = np.stack([zscore(theta), zscore(np.linspace(1.0, 0.0, K))], 1) @ w
    assert s["slate"] == [100 + j for j in np.argsort(-want, kind="stable")[:10]]
    stages = ["request.added", "candidates.proposed", "choices.observed", "ranking.estimated", "slate.ready"]
    assert [e for e in s["events"] if e in stages] == stages
