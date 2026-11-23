"""Table pop (which targets gain) on MovieLens-1M, plain and enriched (v2): NDCG@10 by target popularity of GRU4Rec
(the best trained baseline there), of the pool with its residual ranker and of Decision4Rec-Full (4B, fused), with
paired tests against GRU4Rec within each group. Same groups as claim_tests.py pop.
PYTHONPATH=.:scripts python scripts/pop_ml1m.py"""
import numpy as np

from claim_tests import fused, listed, setup
from equivalence_ci import base
from decision4rec.eval import metrics as M

for d in ("ml1m", "ml1m_plus"):
    _, _, pool = setup(d)
    g, cases = base(d, "rtm", "gru4rec")
    f = fused(d)
    cs = [c for c in cases if c.cid in f]
    assert len(cs) >= 0.99 * len(cases), f"{d}: only {len(cs)} of {len(cases)} users"
    pop = np.array([pool.store.pop[c.target] for c in cs])
    a = np.array([g[c.cid] for c in cs]); b = np.array([listed(c) for c in cs]); e = np.array([f[c.cid] for c in cs])
    for name, m in {"rare": pop < 20, "middle": (pop >= 20) & (pop < 100), "popular": pop >= 100}.items():
        if not m.any():
            print(f"{d} {name}: no users"); continue
        print(f"{d} {name:7s} users {m.mean():.0%} | GRU4Rec {a[m].mean():.4f} | pool {b[m].mean():.4f} (p={M.paired_randomization(b[m], a[m]):.3f}) | "
              f"Full 4B {e[m].mean():.4f} (p={M.paired_randomization(e[m], a[m]):.3f})", flush=True)
