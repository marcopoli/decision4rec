"""Paired tests between two trained decision models evaluated under two arms (configs) on the same users, for example
the pool reader (rtm, 90 characters per item) against R19 (rtmc, 250 characters): alone, and fused with each model's
validation gamma as in the reports, plus the within-list Spearman of each model with the list order. Runs that cover
only part of the users (the 27B runs on the Spark, RUN_MAX) are compared on the users that both runs answered.
python logs/probe/arm_pairs.py rtm kev-0.8b-rtm rtmc kev-0.8b-rtmc [vg mi steam ind ml1m_plus]
An arm written config:arm reads the run of that config whose arm file has another name (rtmpr:rtm)."""
import json
import os
import sys

import numpy as np

from decision4rec.data.schema import load_cases
from decision4rec.eval import metrics as M
from decision4rec.experiments.asc import GAMMAS, SPLITS, _ndcg, _tune, _z
from decision4rec.experiments.runner import case_paths
from decision4rec.operators.luce import Obs, fit_luce

arm_a, tag_a, arm_b, tag_b = sys.argv[1:5]
domains = sys.argv[5:] or ["vg", "mi", "steam", "ind"]


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def raw(d, arm, tag):
    arm, _, farm = arm.partition(":")      # "config:arm" when the arm inside the run is named differently
    farm = farm or arm
    rd = f"runs/small/{d}/{d}_{arm}_{tag}"
    if not os.path.isdir(rd):
        return None
    cfg = json.load(open(os.path.join(rd, "manifest.json")))["config"]
    paths = case_paths(cfg)
    cases = {s: {c.cid: c for c in load_cases(paths[s])} for s in SPLITS}
    th = {}
    for s in SPLITS:
        path = os.path.join(rd, f"{farm}.{s}.jsonl")
        if not os.path.exists(path):
            return None
        th[s] = {}
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line) if line.strip() else None
            if rec and rec.get("obs") and rec["cid"] in cases[s]:
                th[s][rec["cid"]] = fit_luce([Obs.from_json(o) for o in rec["obs"]], len(cases[s][rec["cid"]].cands)).theta
    return cases, th


def scores(cases, th):
    zr = {s: {c: _z(cases[s][c].rscores) for c in cases[s]} for s in SPLITS}
    zth = {s: {c: _z(th[s][c]) for c in cases[s]} for s in SPLITS}

    def fuse(g, s):
        return {c: zth[s][c] + g * zr[s][c] for c in cases[s]}
    g, _ = _tune(fuse, GAMMAS, cases)
    T = cases["test"]
    inl = [c for c in sorted(T) if T[c].target_pos >= 0]
    return {"alone": _ndcg(th["test"], T), "fused": _ndcg(fuse(g, "test"), T), "list": _ndcg(zr["test"], T), "g": g,
            "rho": float(np.mean([spearman(th["test"][c], T[c].rscores) for c in inl]))}


for d in domains:
    ra, rb = raw(d, arm_a, tag_a), raw(d, arm_b, tag_b)
    if ra is None or rb is None:
        print(f"{d}: missing run", flush=True)
        continue
    (ca, ta), (cb, tb) = ra, rb
    common = {s: sorted(set(ta[s]) & set(tb[s])) for s in SPLITS}
    if any(ca[s][c].user != cb[s][c].user for s in SPLITS for c in common[s]):
        print(f"{d}: the two runs have different users; not paired", flush=True)
        continue
    sub = lambda cases, th: ({s: {c: cases[s][c] for c in common[s]} for s in SPLITS},
                             {s: {c: th[s][c] for c in common[s]} for s in SPLITS})
    a, b = scores(*sub(ca, ta)), scores(*sub(cb, tb))
    line = [f"{d} (test {len(common['test'])}, valid {len(common['valid'])}): rho {a['rho']:.2f} -> {b['rho']:.2f}",
            f"list {a['list'].mean():.4f} / {b['list'].mean():.4f}"]
    for k in ("alone", "fused"):
        x, y = b[k], a[k]
        line.append(f"{k} {y.mean():.4f} -> {x.mean():.4f} ({100 * (x.mean() / y.mean() - 1):+.0f}%, "
                    f"p={M.paired_randomization(x, y):.3f})")
    x, y = b["fused"], b["list"]
    line.append(f"{arm_b.split(':')[0]} fused vs list {100 * (x.mean() / y.mean() - 1):+.0f}% (p={M.paired_randomization(x, y):.3f})")
    print(" | ".join(line), flush=True)
