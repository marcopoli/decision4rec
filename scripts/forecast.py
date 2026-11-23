"""Forecast of the reader's gain in a scenario from its evidence alone (V34a, docs/18 A3; CPU, stored runs).

Per-user gain: log-likelihood of the target, list plus reader minus list, 9B mean of three seeds, development test users
with the target among the 30 (scripts/thin_probe.py). Candidate models, fixed before any result of this script:
  M1  gain = a + b x                    x = log10(1 + purchases of the target before the request)
  M2  M1 + c log10(mean characters of the item text of the scenario)
  M3  gain = a + b (mean log10 purchases of the 30 options)
  M4  gain = a + b min(x, k)            a floor above k purchases (added 2026-11-11 after A1: the sign turns on dense data)
  naive: the mean gain of the other settings.
Each model is fitted on five settings (equal weight per setting) and forecasts the mean gain of the sixth.

    python scripts/forecast.py out.json
"""
import json
import sys

import numpy as np
from scipy.stats import spearmanr

from decision4rec.data.schema import Dataset
from thin_probe import SEEDS, seeds_mean, with_covariates

S = with_covariates(seeds_mean())
D = {}
for d, r in S.items():
    m = np.asarray(r["pos"]) >= 0
    ds = Dataset.load(json.load(open(f"runs/small/{d}/{d}_{SEEDS[0].replace(':', '_')}/manifest.json"))["config"]["dataset"])
    idx = np.random.default_rng(0).choice(ds.n_items, min(ds.n_items, 3000), replace=False)
    chars = float(np.mean([len(ds.item_text(int(i), False)) for i in idx]))
    D[d] = {"g": (r["ll_f"] - r["ll_l"])[m], "x": np.log10(r["before"][m] + 1), "l": r["lpop"][m], "t": np.log10(chars)}
    g = D[d]["g"]
    D[d]["se"] = float(g.std(ddof=1) / np.sqrt(len(g)))
    print(f"{d:9s} users {len(g)} mean gain {g.mean():+.4f} se {D[d]['se']:.4f} sd {g.std(ddof=1):.3f} | users for se 0.02: "
          f"{int((g.std(ddof=1) / 0.02) ** 2)} with the target in the 30 | text {chars:.0f} characters", flush=True)


def design(d, model, k=None):
    x, n = D[d]["x"], len(D[d]["g"])
    cols = {"M1": [x], "M2": [x, np.full(n, D[d]["t"])], "M3": [D[d]["l"]], "M4": [np.minimum(x, k if k is not None else 9.0)]}[model]
    return np.column_stack([np.ones(n)] + cols)


def fit(train, model, k=None):
    X = np.concatenate([design(d, model, k) / np.sqrt(len(D[d]["g"])) for d in train])
    y = np.concatenate([D[d]["g"] / np.sqrt(len(D[d]["g"])) for d in train])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return b, float(((X @ b - y) ** 2).sum())


def fit_m4(train):
    best = min(((fit(train, "M4", k)[1], k) for k in np.arange(1.0, 3.6, 0.25)))
    return fit(train, "M4", best[1])[0], best[1]


names, res = list(D), {"settings": {d: {"gain": float(D[d]["g"].mean()), "se": D[d]["se"], "n": len(D[d]["g"])} for d in D}, "loso": {}}
obs = np.array([D[d]["g"].mean() for d in names])
pred = {m: [] for m in ("naive", "M1", "M2", "M3", "M4")}
for d in names:
    train = [e for e in names if e != d]
    pred["naive"].append(float(np.mean([D[e]["g"].mean() for e in train])))
    for m in ("M1", "M2", "M3"):
        pred[m].append(float((design(d, m) @ fit(train, m)[0]).mean()))
    b, k = fit_m4(train)
    pred["M4"].append(float((design(d, "M4", k) @ b).mean()))
for m, p in pred.items():
    p = np.array(p)
    rho = spearmanr(p, obs)[0] if m != "naive" else float("nan")
    res["loso"][m] = {"pred": p.tolist(), "mae": float(np.abs(p - obs).mean()), "spearman": float(rho)}
    print(f"{m:5s} error {np.abs(p - obs).mean():.4f} rank correlation {rho:+.2f} | " +
          " ".join(f"{d} {a:+.3f} (obs {o:+.3f})" for d, a, o in zip(names, p, obs)), flush=True)
res["full"] = {m: fit(names, m)[0].tolist() for m in ("M1", "M2", "M3")}
b, k = fit_m4(names)
res["full"]["M4"] = {"coef": b.tolist(), "k": float(k)}
print("fitted on all six:", json.dumps(res["full"]), flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
