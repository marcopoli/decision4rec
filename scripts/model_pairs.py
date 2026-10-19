"""Paired tests between decision models read alone on the same users (different runs, same case files)."""
import json
import os

import numpy as np

from decision4rec.data.schema import load_cases
from decision4rec.eval import metrics as M
from decision4rec.experiments.asc import SPLITS, _ndcg, _theta
from decision4rec.experiments.runner import case_paths

R = "runs/small"
DOM = {"vg": "vg", "mi": "mi", "steam": "steam", "ind": "ind", "ml1m": "ml1m"}
SETTINGS = {"rtm": ("rtm", "kev-0.8b-rtm", "kev-4b-rtm"), "ev2": ("ev", "kev-0.8b-ev2", None), "ev0": ("ev", "kev-0.8b-ev2", None)}
MODELS = {"jev": "jev-1.13.0", "rel08": "kev-0.8b", "rel4": "kev-4b"}
PAIRS = [("tr08", "jev"), ("tr08", "rel08"), ("rel4", "rel08"), ("jev", "rel4"), ("jev", "rel08"), ("tr4", "jev"), ("tr4", "tr08")]


def scores(rd, arm):
    if not os.path.exists(os.path.join(rd, "manifest.json")):
        return None
    cfg = json.load(open(os.path.join(rd, "manifest.json")))["config"]
    paths = case_paths(cfg)
    cases = {s: {c.cid: c for c in load_cases(paths[s])} for s in SPLITS}
    th = _theta(rd, arm, cases)
    if th is None:
        return None
    return _ndcg({c: th["test"][c] for c in cases["test"]}, cases["test"])


for d in DOM:
    for k, (arm, tr08, tr4) in SETTINGS.items():
        if k == "ev0" and d not in ("ind", "ml1m"):
            continue
        tags = dict(MODELS, tr08=tr08)
        if tr4:
            tags["tr4"] = tr4
        v = {m: scores(f"{R}/{d}/{d}_{k}_{t}", arm) for m, t in tags.items()}
        line = []
        for a, b in PAIRS:
            if v.get(a) is not None and v.get(b) is not None:
                x, y = v[a], v[b]
                line.append(f"{a}-{b} {100 * (x.mean() / y.mean() - 1):+.0f}% p={M.paired_randomization(x, y):.3f}")
        print(d, k, {m: round(float(x.mean()), 4) for m, x in v.items() if x is not None})
        print("   ", " | ".join(line))
