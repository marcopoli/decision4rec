"""L2 after the failed blind forecast (EXPLORATORY, post hoc; docs/18 A3 fixed this next rung before the read).

M5: per-user gain = a + b min(x, k) + c s, s = share of the setting's test users with the target in the 30 (known from
the pool alone, before any reader). Added 2026-11-14 after seeing the L2 miss (post hoc, exploratory only): M6 = M4 plus the
per-user information of the list about the target (ll_l - ll_u, nats over uniform, known before any reader); M7 = that
information alone. Fitted like M4 in scripts/forecast.py (equal weight per setting, k on a grid).
Reported: leave-one-setting-out on the six development settings, the fit on all six applied to the three unseen
scenarios, and leave-one-setting-out over all nine. Also per setting: gain, share in the 30, mean x, share x < 2, list
and reader-alone information (nats over uniform), and the within-scenario slope of the gain on x in the unseen scenarios
(L1 where the reader never trained; two-sided permutation, 2,000).

    python scripts/l2_explore.py out.json
"""
import json
import sys

import numpy as np
from scipy.stats import spearmanr

import thin_probe as TP
from decision4rec.data.schema import Dataset

SEEDS = ("kev-9b-rtm", "kev-9b-rtm_s1", "kev-9b-rtm_s2")
UNSEEN = ("cds", "toys", "sw")
D = {}


def add(d, r, x_all):
    m = np.asarray(r["pos"]) >= 0
    D[d] = {"g": (r["ll_f"] - r["ll_l"])[m], "x": x_all[m], "s": float(m.mean()), "il": (r["ll_l"] - r["ll_u"])[m],
            "info_list": float((r["ll_l"] - r["ll_u"])[m].mean()), "info_reader": float((r["ll_d"] - r["ll_u"])[m].mean())}


S = TP.with_covariates(TP.seeds_mean())
for d, r in S.items():
    add(d, r, np.log10(r["before"] + 1))
for s in UNSEEN:
    U = [TP.per_user({**TP.runs(f"rtm:{t}", TP.SIX), **TP.runs(f"rtm:{t}", [s])})[s] for t in SEEDS]
    cids = sorted(set.intersection(*[set(u["cids"]) for u in U]))
    idx = [np.searchsorted(u["cids"], cids) for u in U]
    r = {k: np.mean([u[k][i] for u, i in zip(U, idx)], 0) for k in ("pos", "ll_f", "ll_l", "ll_d", "ll_u")}
    T = TP.runs(f"rtm:{SEEDS[0]}", [s])[s][0]["test"]
    cfg = json.load(open(f"runs/small/{s}/{s}_rtm_{SEEDS[0]}/manifest.json"))["config"]
    cov, _ = TP.covariates(s, [T[c] for c in cids], ds=Dataset.load(cfg["dataset"]))
    add(s, r, np.log10(np.array([cov[c][4] for c in cids], dtype=float) + 1))

DEV = [d for d in D if d not in UNSEEN]


def design(d, k, with_s):
    x, n = D[d]["x"], len(D[d]["g"])
    extra = {False: [], True: [np.full(n, D[d]["s"])], "il": [D[d]["il"]], "il_only": None}[with_s]
    if extra is None:          # M7: the list's information alone (per user, known before any reader)
        return np.column_stack([np.ones(n), D[d]["il"]])
    cols = [np.ones(n), np.minimum(x, k)] + extra
    return np.column_stack(cols)


def fit(train, k, with_s):
    X = np.concatenate([design(d, k, with_s) / np.sqrt(len(D[d]["g"])) for d in train])
    y = np.concatenate([D[d]["g"] / np.sqrt(len(D[d]["g"])) for d in train])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return b, float(((X @ b - y) ** 2).sum())


def best(train, with_s):
    k = min(np.arange(1.0, 3.6, 0.25), key=lambda k: fit(train, k, with_s)[1])
    return fit(train, k, with_s)[0], k


def pred(d, train, with_s):
    b, k = best(train, with_s)
    return float((design(d, k, with_s) @ b).mean())


res = {"settings": {}, "models": {}}
for d in D:
    g, x = D[d]["g"], D[d]["x"]
    row = {"n": len(g), "gain": float(g.mean()), "share_in30": D[d]["s"], "mean_x": float(x.mean()), "share_x_below_2": float((x < 2).mean()),
           "info_list": D[d]["info_list"], "info_reader": D[d]["info_reader"]}
    if d in UNSEEN:
        row["slope"], row["slope_p"] = TP._perm([x], [g], n=2000)
    res["settings"][d] = row
    print(f"{d:9s} n {len(g):4d} gain {g.mean():+.4f} | in 30 {D[d]['s']:.2f} | mean x {x.mean():.2f}, x<2 {(x < 2).mean():.2f} | "
          f"info list {D[d]['info_list']:+.3f}, reader alone {D[d]['info_reader']:+.3f}"
          + (f" | slope {row['slope']:+.3f} (p {row['slope_p']:.3f})" if d in UNSEEN else ""), flush=True)

obs = {d: D[d]["g"].mean() for d in D}
naive_dev = float(np.mean([obs[d] for d in DEV]))
for name, ws in (("M4", False), ("M5", True), ("M6", "il"), ("M7", "il_only")):
    lo_dev = {d: pred(d, [e for e in DEV if e != d], ws) for d in DEV}
    un = {d: pred(d, DEV, ws) for d in UNSEEN}
    lo_all = {d: pred(d, [e for e in D if e != d], ws) for d in D}
    b, k = best(DEV, ws)
    b9, k9 = best(list(D), ws)     # 2026-11-14: fit on all nine read settings, frozen for the L2 fair test (docs/r1/l2b_addendum.md)
    res["models"][name] = {"coef_dev": b.tolist(), "k": float(k), "coef_all9": b9.tolist(), "k_all9": float(k9),
                           "loso_dev": lo_dev, "unseen_from_dev": un, "loso_all9": lo_all,
                           "mae_loso_dev": float(np.mean([abs(lo_dev[d] - obs[d]) for d in DEV])),
                           "mae_unseen": float(np.mean([abs(un[d] - obs[d]) for d in UNSEEN])),
                           "mae_loso_all9": float(np.mean([abs(lo_all[d] - obs[d]) for d in D])),
                           "spearman_loso_all9": float(spearmanr([lo_all[d] for d in D], [obs[d] for d in D])[0])}
    m = res["models"][name]
    print(f"{name}: coef {np.round(b, 4).tolist()} k {k} | LOSO dev MAE {m['mae_loso_dev']:.4f} | unseen MAE {m['mae_unseen']:.4f} ("
          + ", ".join(f"{d} {un[d]:+.3f} obs {obs[d]:+.3f}" for d in UNSEEN) + f") | LOSO all nine MAE {m['mae_loso_all9']:.4f}, "
          f"rank corr {m['spearman_loso_all9']:+.2f}", flush=True)
res["observed"] = {d: float(obs[d]) for d in D}
res["naive"] = {"dev_mean": naive_dev, "all9_mean": float(np.mean(list(obs.values()))), "mae_unseen": float(np.mean([abs(naive_dev - obs[d]) for d in UNSEEN])),
                "mae_loso_all9": float(np.mean([abs(np.mean([obs[e] for e in D if e != d]) - obs[d]) for d in D]))}
print(f"naive: unseen MAE {res['naive']['mae_unseen']:.4f}, LOSO all nine MAE {res['naive']['mae_loso_all9']:.4f}", flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1, default=float)
