"""Information accounting (V39, docs/18 A2; CPU): where the target is lost on the way to the top 10.

Per setting, development test users: share of users whose target is outside the candidates of the pool, among them
but below the 30 options, and among the 30; NDCG@10 of the list, of the fused 9B (seed 0, stable fusion) and of a
perfect reader of the 30 and of all candidates (the share of users with the target there). The same shares by the
evidence of the target (rare under 20 purchases in the training part, middle 20 to 99, popular 100 or more).
Self-check: list and fused NDCG equal scripts/fusion_table.py.

    python scripts/accounting.py out.json [setting ...]

"""
import json
import sys

import numpy as np

from fusion_table import SIX
from decision4rec.agents.pool import pool_for
from decision4rec.data.schema import Dataset, load_cases
from decision4rec.experiments.runner import case_paths, load_config
from thin_probe import GROUPS, SEEDS, per_user, runs

out, sets = sys.argv[1], sys.argv[2:] or list(SIX)
U = per_user(runs(SEEDS[0]))
res = {}
for d in sets:
    cfg = load_config(f"configs/small/{d}_rtm.yaml")
    ds = Dataset.load(cfg["dataset"])
    p = pool_for(cfg, ds)
    pop = np.zeros(ds.n_items, dtype=np.int64)
    for s in ds.seqs:
        for x in (s[:-2] if len(s) > 2 else []):
            pop[x[0]] += 1
    cases = sorted(load_cases(case_paths(cfg)["test"]), key=lambda c: c.cid)
    rank, size, mismatch = [], [], 0
    for b in range(0, len(cases), 256):
        chunk = cases[b: b + 256]
        rows = p._sasrec_rows([c.user for c in chunk], [c.history for c in chunk])
        for i, c in enumerate(chunk):
            items, _ = p.rank(c.user, c.history, 10 ** 6, None if rows is None else rows[i])
            r = items.index(c.target) + 1 if c.target in items else 0
            mismatch += (0 < r <= 30) != (c.target_pos >= 0)       # the stored case and the pool of today must agree
            rank.append(r)
            size.append(len(items))
    rank, tp = np.array(rank), pop[[c.target for c in cases]]
    assert mismatch <= 0.01 * len(cases), f"{d}: {mismatch} cases disagree with the pool"
    if d in U:
        u = U[d]
    else:                                                       # a setting outside the stored runs: it replaces its full setting in the fusion
        full = runs(SEEDS[0])
        u = per_user({**{e: full[e] for e in full if e != d.split("_")[0]}, d: runs(SEEDS[0], (d,))[d]})[d]
    assert u["cids"] == [c.cid for c in cases], f"{d}: users of the stored run differ"
    row = {"users": len(cases), "candidates_mean": float(np.mean(size)), "outside_pool": float((rank == 0).mean()),
           "below_30": float((rank > 30).mean()), "in_30": float(((rank > 0) & (rank <= 30)).mean()),
           "list": float(u["lst"].mean()), "fused": float(u["fused"].mean()),
           "perfect_reader_30": float(((rank > 0) & (rank <= 30)).mean()), "perfect_reader_pool": float((rank > 0).mean()), "groups": {}}
    row["headroom_closed_30"] = (row["fused"] - row["list"]) / max(row["perfect_reader_30"] - row["list"], 1e-9)
    line = (f"{d:9s} users {row['users']} candidates {row['candidates_mean']:.0f} | outside the pool {100 * row['outside_pool']:.1f}% "
            f"below the 30 {100 * row['below_30']:.1f}% in the 30 {100 * row['in_30']:.1f}% | NDCG@10 list {row['list']:.4f} fused {row['fused']:.4f} "
            f"perfect reader of the 30 {row['perfect_reader_30']:.4f} of the pool {row['perfect_reader_pool']:.4f} | "
            f"headroom of the 30 closed {100 * row['headroom_closed_30']:.1f}% (mismatch {mismatch})")
    print(line, flush=True)
    for name, lo, hi in GROUPS:
        g = (tp >= lo) & (tp <= (hi if hi is not None else np.inf))
        if g.sum() < 20:
            continue
        row["groups"][name] = {"users": int(g.sum()), "share": float(g.mean()), "outside_pool": float((rank[g] == 0).mean()),
                               "below_30": float((rank[g] > 30).mean()), "in_30": float(((rank[g] > 0) & (rank[g] <= 30)).mean()),
                               "list": float(u["lst"][g].mean()), "fused": float(u["fused"][g].mean())}
        t = row["groups"][name]
        print(f"   {name:7s} {100 * t['share']:4.1f}% of users | outside {100 * t['outside_pool']:.1f}% below {100 * t['below_30']:.1f}% "
              f"in the 30 {100 * t['in_30']:.1f}% | list {t['list']:.4f} fused {t['fused']:.4f}", flush=True)
    res[d] = row
    json.dump(res, open(out, "w"), indent=1)
