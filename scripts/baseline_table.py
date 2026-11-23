"""Stable fusion against the fair baselines, per seed and as the mean of the 9B seeds, with Holm over the four domains
(exploratory, development users). Every baseline is scored on the same test users; a + model saw the same extra targets
as the pool and the decision models (scripts/sasrecf_plus.py). The reference of each domain is the best of them on test.
python scripts/baseline_table.py out.json"""
import json
import os
import sys

import numpy as np

from fusion_table import FOUR, SIX, score
from decision4rec.agents.pool import _trained
from decision4rec.data.schema import Dataset, load_cases
from decision4rec.eval import metrics as M
from decision4rec.experiments.runner import case_paths, load_config
from decision4rec.experiments.stage1 import chosen_dir
from decision4rec.models import baselines as B

SPECS = ["rtm:kev-9b-rtm", "rtm:kev-9b-rtm_s1", "rtm:kev-9b-rtm_s2", "rtm:kev-27b-rtm", "rtm:kev-4b-rtm", "rte:kev-9b-rte", "rte:kev-27b-rte","rtm2:kev-9b-rtm", "ev2:kev-9b-ev2", "ev2:kev-27b-ev2"]
SEEDS = SPECS[:3]


def scorers(d):
    """Per-user NDCG@10 of each trained baseline, keyed by the case id of the rtm test users."""
    cfg = load_config(f"configs/small/{d}_rtm.yaml")
    ds = Dataset.load(cfg["dataset"])
    cases = load_cases(case_paths(cfg)["test"])
    args = [c.user for c in cases], [c.history for c in cases], [c.target for c in cases]
    paths = {"sasrecf": chosen_dir(load_config(cfg["pool"]["ref"]), "sasrecf"), "sasrec": _trained(cfg["out"], "sasrec"), "gru4rec": _trained(cfg["out"], "gru4rec")}
    paths.update({f"{m}+": f"runs/small/{d}/baselines/{m}_plus" for m in list(paths)})
    out = {}
    for name, path in paths.items():
        if not os.path.exists(os.path.join(path, "model.pt")):
            print(f"  {d} {name}: no checkpoint", flush=True)
            continue
        r = np.asarray(B.load(name.rstrip("+"), path, ds, "cpu").ranks(*args), dtype=float)
        out[name] = dict(zip([c.cid for c in cases], np.where(r <= 10, 1.0 / np.log2(r + 1.0), 0.0)))
    return out


def report(label, nd, base, table):
    rows = []
    for d in [x for x in SIX if x in nd]:
        best = max(base[d], key=lambda m: np.mean(list(base[d][m].values())))
        k = sorted(set(nd[d]) & set(base[d][best]))
        a, b = np.array([nd[d][i] for i in k]), np.array([base[d][best][i] for i in k])
        rows.append((d, best, a.mean(), b.mean(), M.paired_randomization(a, b), len(k)))
    hp = dict(zip([r[0] for r in rows if r[0] in FOUR], M.holm([r[4] for r in rows if r[0] in FOUR])))
    table[label] = {}
    print(f"== {label}", flush=True)
    for d, best, a, b, p, n in rows:
        table[label][d] = dict(best=best, fused=a, base=b, p=p, holm=hp.get(d))
        print(f"  {d:10s} fused {a:.4f} vs {best:9s} {b:.4f} ({100 * (a / b - 1):+.1f}%) p={p:.3f}"
              + (f" Holm {hp[d]:.3f}" if d in hp else "") + f" users {n}", flush=True)


if __name__ == "__main__":
    base = {d: scorers(d) for d in SIX}
    print({d: {m: round(float(np.mean(list(v.values()))), 4) for m, v in base[d].items()} for d in SIX}, flush=True)
    table, per = {}, {}
    for spec in SPECS:
        R = score(spec, SIX)
        per[spec] = {d: dict(zip(R[d]["cids"], R[d]["fused"])) for d in R}
        report(spec, per[spec], base, table)
        json.dump(table, open(sys.argv[1], "w"), indent=1, default=float)
    mean = {d: {c: float(np.mean([per[s][d][c] for s in SEEDS])) for c in set.intersection(*(set(per[s][d]) for s in SEEDS))} for d in SIX}
    report("9B, per-user mean of three seeds", mean, base, table)
    json.dump(table, open(sys.argv[1], "w"), indent=1, default=float)
