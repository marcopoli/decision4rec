"""Rung 2 of A5/A6 (docs/09, Nov 12; CPU): do the two rare-item generators, added as agents of the pool with the
residual ranker refitted, bring rare targets into the list of 30?

Step 0 (scripts/gen_probe.py) passed only at depth 200: text neighbours and shared attributes, rare items only (fewer
than 20 purchases). Here both enter the pool as agents `rare_text` and `rare_kg` (decision4rec/agents/pool.py, config key
`pool.rare`), the ranker is trained again on the same users, and the list of 30 is compared with the current pool on
the development test and validation users. Fixed before any result:
  bar   rare targets in the 30: at least +3 points in two settings, on test and on validation users;
        all targets in the 30: not down by more than 0.5 points in any setting; NDCG@10 reported, paired p by sign flip.
  else  K1/K3 close as measured negatives (accounting table).

Rung 3 (Nov 12, after rung 2 failed: +0.4 points at most): same pool, no refit, the last `slots` places of the 30 go to
the best rare candidates not already shown (rank in the rare lists). Same bar.

    python scripts/rare_pool.py out.json setting [depth] [slots]
"""
import copy
import json
import sys

import numpy as np

from decision4rec.agents.pool import PoolRetriever, pool_for
from decision4rec.data.schema import Dataset, load_cases
from decision4rec.experiments.runner import case_paths, load_config

K, RARE = 30, 20


def flip_p(d, n=10000, seed=0):
    """Two-sided paired sign-flip test of the mean of d."""
    d = np.asarray(d, dtype=float)
    if not d.any():
        return 1.0
    rng = np.random.default_rng(seed)
    m = abs(d.mean())
    s = rng.choice((-1.0, 1.0), size=(n, len(d)))
    return float(((np.abs((s * d).mean(1)) >= m - 1e-12).sum() + 1) / (n + 1))


def lists(p, cases):
    r, out = PoolRetriever(p), []
    for b in range(0, len(cases), 256):
        ch = cases[b: b + 256]
        idx, _ = r.topk_users([c.user for c in ch], [list(c.history) for c in ch], K)
        out += [list(map(int, row)) for row in idx]
    return out


def metrics(L, cases):
    hit, nd = [], []
    for row, c in zip(L, cases):
        r = row.index(c.target) if c.target in row else None
        hit.append(r is not None)
        nd.append(1.0 / np.log2(r + 2) if r is not None and r < 10 else 0.0)
    return np.array(hit, dtype=float), np.array(nd)


if __name__ == "__main__":
    out, d = sys.argv[1], sys.argv[2]
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    slots = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    cfg = load_config(f"configs/small/{d}_rtm.yaml")
    ds = Dataset.load(cfg["dataset"])
    base = pool_for(cfg, ds)
    cfg2 = copy.deepcopy(cfg)
    cfg2["pool"]["rare"] = {"depth": depth, "below": RARE}
    new = pool_for(cfg2, ds)
    new.rare_slots = slots                     # ranking time only: the checkpoint and tag stay those of rung 2
    assert new.train_users == base.train_users or not base.train_users, "rankers trained on different users"
    res = {"setting": d, "depth": depth, "slots": slots, "lam": {"base": base.lam, "rare": new.lam}}
    for split in ("test", "valid"):
        cases = sorted(load_cases(case_paths(cfg)[split]), key=lambda c: c.cid)
        rare = np.array([base.store.pop[c.target] < RARE for c in cases])
        hb, nb = metrics(lists(base, cases), cases)
        hn, nn = metrics(lists(new, cases), cases)
        row = {"users": len(cases), "rare_users": int(rare.sum()),
               "rare_in30": [100 * hb[rare].mean(), 100 * hn[rare].mean()], "p_rare": flip_p(hn[rare] - hb[rare]),
               "all_in30": [100 * hb.mean(), 100 * hn.mean()], "p_all": flip_p(hn - hb),
               "ndcg10": [nb.mean(), nn.mean()], "p_ndcg": flip_p(nn - nb)}
        res[split] = row
        print(f"{d:6s} {split:5s} rare {row['rare_users']}/{len(cases)} | rare in 30 {row['rare_in30'][0]:.1f} -> "
              f"{row['rare_in30'][1]:.1f} (p {row['p_rare']:.4f}) | all in 30 {row['all_in30'][0]:.1f} -> "
              f"{row['all_in30'][1]:.1f} (p {row['p_all']:.4f}) | NDCG@10 {row['ndcg10'][0]:.4f} -> {row['ndcg10'][1]:.4f} "
              f"(p {row['p_ndcg']:.4f}) | lam {base.lam} -> {new.lam}", flush=True)
    json.dump(res, open(out, "w"), indent=1)
