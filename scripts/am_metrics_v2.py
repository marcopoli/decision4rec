"""AM10 rows for the runs without ranks.json (side evaluations with --no-report): HR@10 and MRR@30 from the fused
scores (validation gamma, as in the reports); the rank of the target is 1 + the candidates scored above it, a target
outside the 30 counts as a miss. Self-check: NDCG@10 from these ranks equals equivalence_ci.fused."""
import json
import numpy as np
from equivalence_ci import fused
from decision4rec.data.schema import load_cases
from decision4rec.experiments.asc import GAMMAS, SPLITS, _theta, _tune, _z
from decision4rec.experiments.runner import case_paths

def ranks(d, conf, tag, arm):
    rd = f"runs/small/{d}/{d}_{conf}_{tag}"
    cfg = json.load(open(rd + "/manifest.json"))["config"]
    cases = {s: {c.cid: c for c in load_cases(case_paths(cfg)[s])} for s in SPLITS}
    th = _theta(rd, arm, cases)
    zr = {s: {c: _z(cases[s][c].rscores) for c in cases[s]} for s in SPLITS}
    fuse = lambda g, s: {c: _z(th[s][c]) + g * zr[s][c] for c in cases[s]}
    g, _ = _tune(fuse, GAMMAS, cases)
    f = fuse(g, "test"); out = {}
    for c, case in cases["test"].items():
        t = case.target_pos
        out[c] = None if t < 0 else 1 + int(np.sum(np.asarray(f[c]) > f[c][t]))
    return out

def metrics(r):
    r = np.array([np.inf if x is None else float(x) for x in r])
    return np.mean(np.where(r <= 10, 1 / np.log2(r + 1), 0)), np.mean(r <= 10), np.mean(np.where(r <= 30, 1 / r, 0))

R = (("Decision4Rec-Full, 9B, fused", "rtm", "kev-9b-rtm", "rtm"), ("Decision4Rec-Full, 9B (8,000 records), fused", "rtm", "kev-9b-rtm8k", "rtm"),
     ("Decision4Rec-Full, 27B (8,000 records), fused", "rtm", "kev-27b-rtm8k", "rtm"), ("Decision4Rec-List, 9B, fused", "rte", "kev-9b-rte", "rte"),
     ("Decision4Rec-KG, 4B, fused", "ev2", "kev-4b-ev2", "ev"), ("Decision4Rec-KG, 9B, fused", "ev2", "kev-9b-ev2", "ev"),
     ("Decision4Rec-KG, 27B (8,000 records), fused", "ev2", "kev-27b-ev2k8", "ev"))
for d in ("vg", "mi", "steam", "ind", "ml1m", "ml1m_plus"):
    for name, conf, tag, arm in R:
        try: r = ranks(d, conf, tag, arm)
        except Exception: continue
        n, h, m = metrics([r[c] for c in sorted(r)])
        ref = np.mean(list(fused(d, tag, conf, arm).values()))
        assert abs(n - ref) < 2e-4, (d, tag, n, ref)
        print(f"| {d} | {name} | {n:.4f} | {h:.4f} | {m:.4f} |", flush=True)
