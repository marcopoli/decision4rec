"""EXPLORATORY (not in docs/11 v2): pooled R1+ F2 estimate over the four cold domains.

Each user's NDCG@10 is divided by the reference (or list) mean of the user's domain, so the pooled relative change is the
user-weighted mean of the four domain changes and no domain dominates by scale. Paired randomization (10,000) and 95%
bootstrap on the pooled scaled arrays, stratified by domain. Same Full, reference and list as r1_score.py f2.

    python scripts/r1_f2_pooled.py out.json [domain to leave out ...]   e.g. steam_cold: is the pooled result only Steam?
"""
import json
import sys

import numpy as np

import r1_score as R
from fusion_table import load
from decision4rec.eval import metrics as M
from decision4rec.experiments.asc import _ndcg


def strat_boot(F, B, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    num, den = np.zeros(n), np.zeros(n)
    for f, b in zip(F, B):
        i = rng.integers(0, len(f), (n, len(f)))
        num += f[i].sum(1)
        den += b[i].sum(1)
    return [float(100 * (q - 1)) for q in np.quantile(num / den, [0.025, 0.975])]


def main(out, drop=()):
    doms, conf, dev, seeds = R.FAMS["f2"]
    doms = tuple(d for d in doms if d not in drop)
    per, lst, cases0 = {d: [] for d in doms}, {}, {}
    for tag in seeds:
        w = R.weights(*dev, tag)
        for d in doms:
            X = load(d, conf, tag, min_valid=0)
            if X is None:
                continue
            cases, Z = X
            T = cases["test"]
            per[d].append(dict(zip(sorted(T), _ndcg({c: Z["test"][c] @ w[d] for c in T}, T))))
            lst[d] = dict(zip(sorted(T), _ndcg({c: Z["test"][c][:, 1] for c in T}, T)))
            cases0[d] = T
    res, S = {}, {"vs_ref": ([], []), "vs_list": ([], [])}
    for d in doms:
        k = sorted(set.intersection(*(set(p) for p in per[d])))
        full = np.array([np.mean([p[c] for p in per[d]]) for c in k])
        rc, rn, _ = R.ref_ndcg(d, conf, {c: cases0[d][c] for c in k})
        assert rc == k
        li = np.array([lst[d][c] for c in k])
        res[d] = {"users": len(k), "vs_ref": float(100 * (full.mean() / rn.mean() - 1)), "vs_list": float(100 * (full.mean() / li.mean() - 1))}
        for name, b in (("vs_ref", rn), ("vs_list", li)):
            S[name][0].append(full / b.mean())
            S[name][1].append(b / b.mean())
    pooled = {}
    for name, (F, B) in S.items():
        f, b = np.concatenate(F), np.concatenate(B)
        pooled[name] = {"users": len(f), "rel": float(100 * (f.mean() / b.mean() - 1)), "ci95": strat_boot(F, B),
                        "p": float(M.paired_randomization(f, b))}
        print(f"pooled {name}: {pooled[name]['rel']:+.1f}% [{pooled[name]['ci95'][0]:+.1f}, {pooled[name]['ci95'][1]:+.1f}] "
              f"p={pooled[name]['p']:.4f} over {len(f)} users", flush=True)
    for d, r in res.items():
        print(f"  {d:11s} users {r['users']} vs ref {r['vs_ref']:+.1f}% vs list {r['vs_list']:+.1f}%", flush=True)
    json.dump({"exploratory": True, "left_out": list(drop), "domains": res, "pooled": pooled}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
