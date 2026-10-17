"""A stronger ranker of the pool (review 2026-10-16, R26): the residual ranker with one hidden layer (configs
<d>_rtmh.yaml, pool tag poolh) against the linear residual ranker of Decision4Rec-Full (pool) and against Decision4Rec-Full
with its 4B decision model fused, on the same development test users (paired randomization tests).
python logs/probe/ranker_control.py [vg mi steam ind]
ALT=rtms compares instead the pool whose candidates are SASRecF's best items, as many as the pool (configs <d>_rtms.yaml)."""
import json
import os
import sys

import numpy as np

from decision4rec.data.schema import load_cases
from decision4rec.eval import metrics as M
from decision4rec.experiments.asc import GAMMAS, SPLITS, _ndcg, _tune, _z
from decision4rec.experiments.runner import case_paths, load_config
from decision4rec.operators.luce import Obs, fit_luce


def list_ndcg(cases):   # NDCG@10 of the order of the list itself (target position in the candidates)
    return np.array([1.0 / np.log2(c.target_pos + 2) if 0 <= c.target_pos < 10 else 0.0 for c in cases])


def fused(d, tag="kev-4b-rtm"):   # Decision4Rec-Full fused with the validation gamma, as in the reports
    rd = f"runs/small/{d}/{d}_rtm_{tag}"
    cfg = json.load(open(os.path.join(rd, "manifest.json")))["config"]
    paths = case_paths(cfg)
    cases = {s: {c.cid: c for c in load_cases(paths[s])} for s in SPLITS}
    th = {s: {} for s in SPLITS}
    for s in SPLITS:
        for line in open(os.path.join(rd, f"rtm.{s}.jsonl"), encoding="utf-8"):
            rec = json.loads(line) if line.strip() else None
            if rec and rec.get("obs") and rec["cid"] in cases[s]:
                th[s][rec["cid"]] = fit_luce([Obs.from_json(o) for o in rec["obs"]], len(cases[s][rec["cid"]].cands)).theta
    cases = {s: {c: cases[s][c] for c in th[s]} for s in SPLITS}
    zr = {s: {c: _z(cases[s][c].rscores) for c in cases[s]} for s in SPLITS}

    def fuse(g, s):
        return {c: _z(th[s][c]) + g * zr[s][c] for c in cases[s]}
    g, _ = _tune(fuse, GAMMAS, cases)
    return dict(zip(sorted(cases["test"]), _ndcg(fuse(g, "test"), cases["test"]))), g   # _ndcg follows sorted case ids


ALT = os.environ.get("ALT", "rtmh")
for d in sys.argv[1:] or ["vg", "mi", "steam", "ind"]:
    lin = load_cases(case_paths(load_config(f"configs/small/{d}_rtm.yaml"))["test"])
    hid = load_cases(case_paths(load_config(f"configs/small/{d}_{ALT}.yaml"))["test"])
    assert [c.user for c in lin] == [c.user for c in hid], "different users"
    a, b = list_ndcg(lin), list_ndcg(hid)
    f, g = fused(d)
    assert all(c.cid in f for c in lin), "test users without answers of the decision model"
    full = np.array([f[c.cid] for c in lin])
    hit = lambda cs: np.mean([c.target_pos >= 0 for c in cs])
    print(f"{d} (test {len(lin)}): linear ranker {a.mean():.4f} | {ALT} {b.mean():.4f} "
          f"({100 * (b.mean() / a.mean() - 1):+.1f}%, p={M.paired_randomization(b, a):.3f}) | target in 30: {hit(lin):.3f} / {hit(hid):.3f} | "
          f"Full 4B fused (g={g}) {full.mean():.4f}: vs {ALT} {100 * (full.mean() / b.mean() - 1):+.1f}% "
          f"(p={M.paired_randomization(full, b):.3f})", flush=True)
