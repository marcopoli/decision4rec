"""Committees of trained decision models on the same users (2026-10-15): the standardized Luce worths of two or more
models are averaged before fusion with the ranker, with the committee weights and the fusion weight chosen on the
validation users, and compared with the best single model of the committee (paired randomization test on the test
users). Runs that cover only part of the users are compared on the users that every run answered.
python logs/probe/ensemble_probe.py rtm:kev-0.8b-rtm rtm:kev-4b-rtm [more arm:tag] [-- vg mi steam ind ml1m_plus]
An arm written config:arm reads the run of that config whose arm file has another name (rtmpr:rtm:tag)."""
import itertools
import json
import os
import sys

import numpy as np

from decision4rec.data.schema import load_cases
from decision4rec.eval import metrics as M
from decision4rec.experiments.asc import GAMMAS, SPLITS, _ndcg, _tune, _z
from decision4rec.experiments.runner import case_paths
from decision4rec.operators.luce import Obs, fit_luce


def raw(d, arm, tag):   # as in arm_pairs.py: the cases of a run and the Luce worths of its answered cases
    arm, _, farm = arm.partition(":")
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

args = sys.argv[1:]
if "--" in args:
    k = args.index("--")
    members, domains = args[:k], args[k + 1:]
else:
    members, domains = args, ["vg", "mi", "steam", "ind"]
WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def member(spec):   # "arm:tag" or "config:arm:tag"
    parts = spec.split(":")
    return ":".join(parts[:-1]), parts[-1]


for d in domains:
    runs = [raw(d, *member(m)) for m in members]
    if any(r is None for r in runs):
        print(f"{d}: missing run", flush=True)
        continue
    common = {s: sorted(set.intersection(*[set(th[s]) for _, th in runs])) for s in SPLITS}
    cases = {s: {c: runs[0][0][s][c] for c in common[s]} for s in SPLITS}
    if any(r[0][s][c].user != cases[s][c].user for r in runs for s in SPLITS for c in common[s]):
        print(f"{d}: the runs have different users; not compared", flush=True)
        continue
    zr = {s: {c: _z(cases[s][c].rscores) for c in common[s]} for s in SPLITS}
    zth = [{s: {c: _z(th[s][c]) for c in common[s]} for s in SPLITS} for _, th in runs]

    def committee(w, s):
        return {c: sum(wi * z[s][c] for wi, z in zip(w, zth)) for c in common[s]}

    def fused(w):
        def fuse(g, s):
            m = committee(w, s)
            return {c: m[c] + g * zr[s][c] for c in common[s]}
        g, _ = _tune(fuse, GAMMAS, cases)
        return fuse(g, "test"), g

    T = cases["test"]
    singles = []
    for i in range(len(runs)):
        w = [1.0 if j == i else 0.0 for j in range(len(runs))]
        scores, g = fused(w)
        singles.append((_ndcg(scores, T), w, g))
    best_single = max(singles, key=lambda t: t[0].mean())
    # committee weights: the first member fixed at 1, the others on a small grid, chosen on validation users
    V = cases["valid"]
    best_w, best_v = None, -1.0
    for rest in itertools.product(WEIGHTS, repeat=len(runs) - 1):
        w = [1.0, *rest]
        def fuse(g, s, w=w):
            m = committee(w, s)
            return {c: m[c] + g * zr[s][c] for c in common[s]}
        g, _ = _tune(fuse, GAMMAS, cases)
        v = _ndcg(fuse(g, "valid"), V).mean()
        if v > best_v:
            best_v, best_w = v, w
    comm, g = fused(best_w)
    nd = _ndcg(comm, T)
    eq, _ = fused([1.0] * len(runs))
    nd_eq = _ndcg(eq, T)
    ref = best_single[0]
    names = [m.split(":")[-1] for m in members]
    line = [f"{d} (test {len(common['test'])}, valid {len(common['valid'])})",
            "singles fused " + ", ".join(f"{n} {s[0].mean():.4f}" for n, s in zip(names, singles)),
            f"committee w={best_w} {nd.mean():.4f} ({100 * (nd.mean() / ref.mean() - 1):+.1f}% vs best single, p={M.paired_randomization(nd, ref):.3f})",
            f"equal weights {nd_eq.mean():.4f} ({100 * (nd_eq.mean() / ref.mean() - 1):+.1f}%, p={M.paired_randomization(nd_eq, ref):.3f})"]
    print(" | ".join(line), flush=True)
