"""HR@10 and MRR@30 next to NDCG@10 for the main systems (Additional Materials, AM10), from the stored per-user ranks:
ranks.json of each run (rank of the target within the 30 candidates, or none) and the stored full-catalogue ranks of the
trained baselines. MRR is cut at 30 because the rerankers order 30 candidates. python logs/probe/am_metrics.py"""
import json
import os

import numpy as np

from decision4rec.data.schema import load_cases
from decision4rec.experiments.runner import case_paths, load_config
from decision4rec.experiments.stratify import baseline_ranks


def metrics(r):   # r: list of ranks (1 = first), None or > 30 for a miss
    r = np.array([np.inf if x is None else float(x) for x in r])
    return (np.mean(np.where(r <= 10, 1.0 / np.log2(r + 1.0), 0.0)), np.mean(r <= 10), np.mean(np.where(r <= 30, 1.0 / r, 0.0)))


def run_ranks(d, conf, tag, arm):
    p = f"runs/small/{d}/{d}_{conf}_{tag}/ranks.json"
    if not os.path.exists(p):
        return None
    t = json.load(open(p))[arm]["test"]
    return [t[str(i)] for i in range(len(t))]


best = {"vg": "sasrecf", "mi": "sasrecf", "steam": "sasrec", "ind": "sasrecf", "ml1m": "gru4rec", "ml1m_plus": "gru4rec"}
print("| Domain | System | NDCG@10 | HR@10 | MRR@30 |\n|---|---|---|---|---|")
for d, b in best.items():
    cfg = load_config(f"configs/small/{d}_rtm.yaml")
    cs = load_cases(case_paths(cfg)["test"])
    r = baseline_ranks(cfg, [[c.user, c.target] for c in cs], [b])[b]
    rows = [(f"{b} (best trained baseline)", [r[str(i)] for i in range(len(cs))])]
    for name, conf, tag, arm in (("pool (residual ranker)", "rtm", "kev-0.8b-rtm", "retriever"),
                                 ("Decision4Rec-Full, 0.8B, fused", "rtm", "kev-0.8b-rtm", "rtm+fuse"),
                                 ("Decision4Rec-Full, 4B, fused", "rtm", "kev-4b-rtm", "rtm+fuse"),
                                 ("Decision4Rec-List, 0.8B, fused", "rte", "kev-0.8b-rte", "rte+fuse"),
                                 ("Decision4Rec-List, 4B, fused", "rte", "kev-4b-rte", "rte+fuse"),
                                 ("Decision4Rec-KG, 0.8B, fused", "ev2", "kev-0.8b-ev2", "ev+fuse")):
        x = run_ranks(d, conf, tag, arm)
        if x is not None:
            rows.append((name, x))
    for name, x in rows:
        n, h, m = metrics(x)
        print(f"| {d} | {name} | {n:.4f} | {h:.4f} | {m:.4f} |", flush=True)
