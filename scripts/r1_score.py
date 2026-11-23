"""R1+ scoring (docs/11 version 2, Sections 4 to 7): the primary families F1 to F3 with the references fixed in the plan.

Fusion weights never come from reserved validation users: F1 and F2 take the weights fitted on the development runs of
the same domains (stable fusion, strength 30, as fusion_table.py); F3 fits each unseen scenario on its own validation
users with the six development settings as prior (the development protocol). System: per-user mean over the 9B seeds of
the fused NDCG@10. Holm inside each family, paired randomization (10,000), 95% paired bootstrap of the relative change.

    python scripts/r1_score.py out.json dev|f1|f2|f3|f4|f4dev|l2
dev: self-check on the development runs; must reproduce logs/probe/baseline_v5.log ("9B, per-user mean of three seeds").
f4dev: self-check of F4 on the development runs; must reproduce the 9b-mean slopes of logs/probe/v40_law.json.
"""
import json
import os
import sys

import numpy as np

from fusion_table import FAMILY, FOUR, SIX, STRENGTH, load
from decision4rec.agents.pool import _trained
from decision4rec.data.schema import Dataset
from decision4rec.eval import metrics as M
from decision4rec.experiments.asc import _ndcg
from decision4rec.experiments.runner import load_config
from decision4rec.experiments.stage1 import chosen_dir
from decision4rec.models import baselines as B
from decision4rec.operators.fusion import choice_data, shrunk_weights

SEEDS = ("kev-9b-rtm", "kev-9b-rtm_s1", "kev-9b-rtm_s2")
COLD = tuple(f"{d}_cold" for d in FOUR)
UNSEEN = ("cds", "toys", "sw")
# References fixed in docs/11 v2 Section 3 (development choices, never re-chosen on fresh users).
REF = {"vg": "sasrecf+", "mi": "sasrecf+", "steam": "sasrecf+", "ind": "sasrecf+",
       "vg_cold": "sasrec", "mi_cold": "sasrec", "steam_cold": "sasrecf", "ind_cold": "sasrecf",
       "cds": "seqrules", "toys": "sasrecf+", "sw": "sasrecf+"}
# family: (domains, run conf of the scored runs, dev domains and conf for the weights, seeds)
FAMS = {"dev": (FOUR, "rtm", (SIX, "rtm"), SEEDS),
        "f1": (FOUR, "r1rtm", (SIX, "rtm"), SEEDS),
        "f2": (COLD, "r1rtm", (COLD, "rtm"), ("kev-9b-rtm_cold",)),
        "f3": (UNSEEN, "rtm", None, SEEDS),
        "s2": (FOUR, "r1rtm", (SIX, "rtm"), ("kev-27b-rtm", "kev-27b-rtm_s1"))}   # secondary rows, no claim (docs/11 v2)


def weights(domains, conf, tag, extra=None):
    D = {d: load(d, conf, tag) for d in domains}
    data = {d: choice_data(v[0]["valid"], v[1]["valid"]) for d, v in D.items() if v is not None}
    if extra:
        data.update(extra)
    return shrunk_weights(data, FAMILY, STRENGTH)


def ref_ndcg(d, conf, cases):
    """Per-user NDCG@10 of the fixed reference on the scored test cases (paths as scripts/baseline_table.py)."""
    name = REF[d]
    rcfg = load_config(f"configs/small/{d}_rtm.yaml")
    ds = Dataset.load(rcfg["dataset"])
    if name.endswith("+"):
        path = f"runs/small/{d}/baselines/{name[:-1]}_plus"
    elif not d.endswith("_cold"):
        path = _trained(rcfg["out"], name)
    else:
        path = chosen_dir(load_config(rcfg["pool"]["ref"]), name)
    cids = sorted(cases)
    C = [cases[c] for c in cids]
    r = np.asarray(B.load(name.rstrip("+"), path, ds, "cpu").ranks([c.user for c in C], [c.history for c in C], [c.target for c in C]), dtype=float)
    return cids, np.where(r <= 10, 1.0 / np.log2(r + 1.0), 0.0), path


def boot(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(a), (n, len(a)))
    return [float(100 * (q - 1)) for q in np.quantile(a[i].mean(1) / b[i].mean(1), [0.025, 0.975])]


def main(out, fam):
    doms, conf, dev, seeds = FAMS[fam]
    per, lst, cases0 = {d: [] for d in doms}, {}, {}
    for tag in seeds:
        if dev:
            w = weights(*dev, tag)
        for d in doms:
            R = load(d, conf, tag, min_valid=0 if dev else 200)   # F2 reserved runs have no validation users (all eligible go to test)
            if R is None:
                print(f"  {d} {conf} {tag}: no run", flush=True)
                continue
            cases, Z = R
            if not dev:   # unseen: own validation users, the six development settings as prior
                w = weights(SIX, "rtm", tag, {d: choice_data(cases["valid"], Z["valid"])})
            T = cases["test"]
            per[d].append(dict(zip(sorted(T), _ndcg({c: Z["test"][c] @ w[d] for c in T}, T))))
            lst[d] = dict(zip(sorted(T), _ndcg({c: Z["test"][c][:, 1] for c in T}, T)))
            cases0[d] = T
    res, tests = {}, []
    for d in doms:
        if not per[d]:
            continue
        k = sorted(set.intersection(*(set(p) for p in per[d])))
        full = np.array([np.mean([p[c] for p in per[d]]) for c in k])
        rc, rn, path = ref_ndcg(d, conf, {c: cases0[d][c] for c in k})
        assert rc == k
        li = np.array([lst[d][c] for c in k])
        res[d] = {"users": len(k), "seeds": len(per[d]), "full": full.mean(), "list": li.mean(), "ref": REF[d], "ref_path": path, "ref_ndcg": rn.mean()}
        for name, b in (("vs_ref", rn), ("vs_list", li)):
            p = M.paired_randomization(full, b)
            res[d][name] = {"rel": float(100 * (full.mean() / b.mean() - 1)), "ci95": boot(full, b), "p": float(p)}
            tests.append((d, name, p))
    for (d, name, _), h in zip(tests, M.holm([t[2] for t in tests])):
        res[d][name]["holm"] = float(h)
    for d, r in res.items():
        print(f"{d:11s} users {r['users']} seeds {r['seeds']} | Full {r['full']:.4f} vs {r['ref']} {r['ref_ndcg']:.4f} "
              f"{r['vs_ref']['rel']:+.1f}% [{r['vs_ref']['ci95'][0]:+.1f}, {r['vs_ref']['ci95'][1]:+.1f}] p={r['vs_ref']['p']:.3f} Holm {r['vs_ref']['holm']:.3f}"
              f" | vs list {r['list']:.4f} {r['vs_list']['rel']:+.1f}% p={r['vs_list']['p']:.3f} Holm {r['vs_list']['holm']:.3f}", flush=True)
    json.dump({"family": fam, "conf": conf, "seeds": list(seeds), "strength": STRENGTH, "domains": res}, open(out, "w"), indent=1, default=float)


def law_f4(out, conf):
    """F4 (L1 on fresh users): slope of the per-user gain (ll_f - ll_l, mean of the 9B seeds) on log10(1 + purchases of
    the target before the request), per domain, one-sided permutation (10,000), Holm over the four domains. Weights from
    the development runs (thin_probe.weights_of). conf rtm: self-check against logs/probe/v40_law.json (9b-mean)."""
    import thin_probe as TP
    U = []
    for tag in SEEDS:
        W = TP.weights_of(TP.runs(f"rtm:{tag}", SIX))
        U.append((tag, TP.per_user(TP.runs(f"{conf}:{tag}", FOUR), W)))
    res, ps = {}, []
    for d in FOUR:
        if any(d not in u for _, u in U):
            continue
        cids = sorted(set.intersection(*[set(u[d]["cids"]) for _, u in U]))
        idx = [np.searchsorted(u[d]["cids"], cids) for _, u in U]
        pos = U[0][1][d]["pos"][idx[0]]
        g = np.mean([(u[d]["ll_f"] - u[d]["ll_l"])[i] for (_, u), i in zip(U, idx)], 0)
        T = TP.runs(f"{conf}:{SEEDS[0]}", [d])[d][0]["test"]
        cov, _ = TP.covariates(d, [T[c] for c in cids], f"{conf}:{SEEDS[0]}")
        m = pos >= 0
        x = np.log10(np.array([cov[c][4] for c in cids], dtype=float)[m] + 1)
        g = g[m]
        sl = TP._slope(x, g)
        rng = np.random.default_rng(0)
        null = np.array([TP._slope(x, rng.permutation(g)) for _ in range(10000)])
        p = float((np.sum(null <= sl) + 1) / (len(null) + 1))
        res[d] = {"n": int(m.sum()), "gain": float(g.mean()), "slope": sl, "p_one_sided": p}
        ps.append(p)
    for d, h in zip(res, M.holm(ps)):
        res[d]["holm"] = float(h)
        print(f"{d:6s} n {res[d]['n']:5d} gain {res[d]['gain']:+.4f} slope {res[d]['slope']:+.4f} nats per tenfold, one-sided p {res[d]['p_one_sided']:.4f}, Holm {h:.4f}", flush=True)
    json.dump({"family": "f4", "conf": conf, "domains": res}, open(out, "w"), indent=1, default=float)


def law_l2(out):
    """L2: observed mean gain of the reader (mean of the 9B seeds) over the test users with the target in the 30 of each
    unseen scenario, weights of the six development settings as prior plus the scenario's validation users, against the
    frozen forecasts (logs/probe/b2_forecast_a9.json). Bars of docs/11 v2 Section 6."""
    import thin_probe as TP
    F = json.load(open("logs/probe/b2_forecast_a9.json"))
    V = json.load(open("logs/probe/v34a_forecast.json"))
    naive = float(np.mean([v["gain"] for v in V["settings"].values()]))
    res = {}
    for s in UNSEEN:
        U = []
        for tag in SEEDS:
            D = {**TP.runs(f"rtm:{tag}", SIX), **TP.runs(f"rtm:{tag}", [s])}
            if s in D:
                U.append(TP.per_user(D)[s])
        if len(U) < len(SEEDS):
            print(f"{s}: {len(U)} of {len(SEEDS)} seeds, skipped", flush=True)
            continue
        cids = sorted(set.intersection(*[set(u["cids"]) for u in U]))
        idx = [np.searchsorted(u["cids"], cids) for u in U]
        m = U[0]["pos"][idx[0]] >= 0
        g = np.mean([(u["ll_f"] - u["ll_l"])[i] for u, i in zip(U, idx)], 0)[m]
        f = F["scenarios"][f"{s}_rtm"]
        lo, hi = f["valid"]["interval80"]
        res[s] = {"n": int(m.sum()), "observed": float(g.mean()), "se": float(g.std(ddof=1) / np.sqrt(m.sum())),
                  "forecast": f["valid"]["forecast"], "interval80": [lo, hi], "inside": bool(lo <= g.mean() <= hi),
                  "forecast_test": f["test"]["forecast"], "abs_err": abs(g.mean() - f["valid"]["forecast"]), "abs_err_naive": abs(g.mean() - naive)}
        r = res[s]
        print(f"{s:5s} n {r['n']:4d} observed {r['observed']:+.4f} (se {r['se']:.4f}) | forecast {r['forecast']:+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}] inside {r['inside']} | naive {naive:+.4f}", flush=True)
    if res:
        mae, mae0 = np.mean([r["abs_err"] for r in res.values()]), np.mean([r["abs_err_naive"] for r in res.values()])
        inside = sum(r["inside"] for r in res.values())
        bar = inside >= 2 and mae < mae0 and len(res) == 3
        print(f"L2: inside {inside} of {len(res)}, MAE M4 {mae:.4f} vs naive {mae0:.4f} -> bar {'MET' if bar else 'NOT met'}", flush=True)
        json.dump({"naive": naive, "scenarios": res, "inside": inside, "mae": mae, "mae_naive": mae0, "bar_met": bar}, open(out, "w"), indent=1, default=float)


if __name__ == "__main__":
    out, fam = sys.argv[1:3]
    if fam == "f4":
        law_f4(out, "r1rtm")
    elif fam == "f4dev":
        law_f4(out, "rtm")
    elif fam == "l2":
        law_l2(out)
    else:
        main(out, fam)
