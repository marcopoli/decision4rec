"""L2 fair test scoring (docs/r1/l2b_addendum.md and l2b_addendum_note_mind.md; docs/11 v2 Section 6).

Per scenario (seed mean of the 9B readers, weights: the six development settings as prior plus the scenario's own
validation users, as F3 and law_l2): observed gain over the test users with the target in the 30; forecasts M4 (frozen:
b2_forecast_a9.json for cds, toys, sw; b2_forecast_l2b.json for the new ones) and M5 (l2b_m5_forecast.json, new ones).
- primary: nine scenarios, at least 7 inside the 80% interval, rank correlation >= 0.6, MAE below naive 0.0659; the eight
  without mind with 6 of 8 reported next to it; the verdict needs both (the stricter reading);
- S-L2a: fresh scenarios, all but one inside (5 of 6), MAE below naive; S-L2b: M5 on the fresh ones, MAE below the naive
  of the nine (0.0661) and rank correlation >= 0.6; S-L1u: slope of the gain on x per fresh scenario, one-sided
  permutation (10,000), Holm over the fresh ones, at least 4 of 6; S-F3x: Full vs reference (best of the seven baselines
  on the test users, the A9 rule) and Full vs list, NDCG@10, paired randomization (10,000), Holm over all these tests;
- descriptive: rank correlation of the gain with reader-alone minus list information over every scenario and setting.

    python scripts/l2b_score.py out.json
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

import r1_score as R
import thin_probe as TP
from fusion_table import SIX
from decision4rec.data.schema import Dataset
from decision4rec.eval import metrics as M
from decision4rec.experiments.asc import _ndcg
from decision4rec.experiments.runner import load_config
from decision4rec.models import baselines as B
from decision4rec.operators.fusion import choice_data

OLD = ("cds", "toys", "sw")
FRESH = ("baby", "pet", "groc", "arts", "ml32m", "mind")
NAIVE, NAIVE9 = 0.0659, 0.0661
BASES = ("gru4rec", "sasrec", "sasrecf", "seqrules", "contentknn", "transtext", "sasrecf+")


def gains(s):
    U = []
    for tag in R.SEEDS:
        D = {**TP.runs(f"rtm:{tag}", SIX), **TP.runs(f"rtm:{tag}", [s])}
        if s in D:
            U.append(TP.per_user(D)[s])
    if len(U) < len(R.SEEDS):
        return None
    cids = sorted(set.intersection(*[set(u["cids"]) for u in U]))
    idx = [np.searchsorted(u["cids"], cids) for u in U]
    m = U[0]["pos"][idx[0]] >= 0
    mean = {k: np.mean([u[k][i] for u, i in zip(U, idx)], 0)[m] for k in ("ll_f", "ll_l", "ll_d", "ll_u")}
    cfg = load_config(f"configs/small/{s}_rtm.yaml")
    T = TP.runs(f"rtm:{R.SEEDS[0]}", [s])[s][0]["test"]
    keep = [c for c, k in zip(cids, m) if k]
    cov, _ = TP.covariates(s, [T[c] for c in keep], ds=Dataset.load(cfg["dataset"]))
    x = np.log10(np.array([cov[c][4] for c in keep], dtype=float) + 1)
    return {"g": mean["ll_f"] - mean["ll_l"], "x": x, "info_list": float((mean["ll_l"] - mean["ll_u"]).mean()),
            "info_reader": float((mean["ll_d"] - mean["ll_u"]).mean())}


def f3x(s):
    """Full (seed mean, fused) vs the best of the seven baselines on the test users, and vs the pool list."""
    per, lst, T = [], None, None
    for tag in R.SEEDS:
        X = R.load(s, "rtm", tag)
        if X is None:
            return None
        cases, Z = X
        w = R.weights(SIX, "rtm", tag, {s: choice_data(cases["valid"], Z["valid"])})
        T = cases["test"]
        per.append(dict(zip(sorted(T), _ndcg({c: Z["test"][c] @ w[s] for c in T}, T))))
        lst = dict(zip(sorted(T), _ndcg({c: Z["test"][c][:, 1] for c in T}, T)))
    k = sorted(set.intersection(*(set(p) for p in per)))
    full = np.array([np.mean([p[c] for p in per]) for c in k])
    li = np.array([lst[c] for c in k])
    cfg = load_config(f"configs/small/{s}_rtm.yaml")
    ds = Dataset.load(cfg["dataset"])
    C = [T[c] for c in k]
    args = [c.user for c in C], [c.history for c in C], [c.target for c in C]
    base = {}
    for b in BASES:        # the six benchmark baselines from their per-case test ranks, SASRecF+ from its checkpoint
        if b == "sasrecf+":
            path = f"{cfg['out']}/baselines/sasrecf_plus"
            if not os.path.exists(os.path.join(path, "info.json")):
                continue
            r = np.asarray(B.load("sasrecf", path, ds, "cpu").ranks(*args), dtype=float)
        else:
            f = glob.glob(f"{cfg['out']}/benchmark/pool_*/{b}.json")
            if len(f) != 1:
                print(f"  {s}: {b} benchmark record {f}", flush=True)
                continue
            rk = json.load(open(f[0]))["ranks"]["test"]
            r = np.array([rk[str(c)] for c in k], dtype=float)
        base[b] = np.where(r <= 10, 1.0 / np.log2(r + 1.0), 0.0)
    ref = max(base, key=lambda b: base[b].mean())
    out = {"users": len(k), "full": full.mean(), "list": li.mean(), "ref": ref, "ref_ndcg": base[ref].mean(),
           "baselines": {b: v.mean() for b, v in base.items()}}
    for name, b in (("vs_ref", base[ref]), ("vs_list", li)):
        out[name] = {"rel": float(100 * (full.mean() / b.mean() - 1)), "ci95": R.boot(full, b), "p": float(M.paired_randomization(full, b))}
    return out


def main(out):
    F4 = json.load(open("logs/probe/b2_forecast_a9.json"))["scenarios"]
    if os.path.exists("logs/probe/b2_forecast_l2b.json"):
        F4.update(json.load(open("logs/probe/b2_forecast_l2b.json"))["scenarios"])
    F5 = json.load(open("logs/probe/l2b_m5_forecast.json"))["scenarios"] if os.path.exists("logs/probe/l2b_m5_forecast.json") else {}
    res = {}
    for s in OLD + FRESH:
        G = gains(s)
        if G is None or f"{s}_rtm" not in F4:
            print(f"{s}: not ready (reader seeds or forecast missing)", flush=True)
            continue
        f = F4[f"{s}_rtm"]["valid"]
        obs = float(G["g"].mean())
        r = {"n": len(G["g"]), "observed": obs, "se": float(G["g"].std(ddof=1) / np.sqrt(len(G["g"]))),
             "m4": f["forecast"], "m4_interval80": f["interval80"], "inside": bool(f["interval80"][0] <= obs <= f["interval80"][1]),
             "info_list": G["info_list"], "info_reader": G["info_reader"]}
        if f"{s}_rtm" in F5:
            r["m5"] = F5[f"{s}_rtm"]["valid"]["forecast"]
        sl = TP._slope(G["x"], G["g"])
        rng = np.random.default_rng(0)
        null = np.array([TP._slope(G["x"], rng.permutation(G["g"])) for _ in range(10000)])
        r.update({"slope": sl, "slope_p_one_sided": float((np.sum(null <= sl) + 1) / (len(null) + 1))})
        if s in FRESH:
            r["f3x"] = f3x(s)
        res[s] = r
        print(f"{s:6s} n {r['n']:4d} observed {obs:+.4f} | M4 {r['m4']:+.4f} [{f['interval80'][0]:+.4f}, {f['interval80'][1]:+.4f}] "
              f"inside {r['inside']}" + (f" | M5 {r['m5']:+.4f}" if "m5" in r else "") + f" | slope {sl:+.4f} p {r['slope_p_one_sided']:.4f}"
              + f" | info list {r['info_list']:+.3f} reader {r['info_reader']:+.3f}", flush=True)

    def bars(names, need_inside, naive, key="m4"):
        S = [s for s in names if s in res and key in res[s]]
        if len(S) < len(names):
            return {"complete": False, "have": S}
        o, p = np.array([res[s]["observed"] for s in S]), np.array([res[s][key] for s in S])
        mae, mae0 = float(np.mean(np.abs(o - p))), float(np.mean(np.abs(o - naive)))
        rho = float(spearmanr(p, o)[0])
        inside = sum(res[s]["inside"] for s in S) if key == "m4" else None
        return {"complete": True, "n": len(S), "inside": inside, "need_inside": need_inside, "mae": mae, "mae_naive": mae0, "rho": rho}

    V = {"primary_nine": bars(OLD + FRESH, 7, NAIVE), "primary_eight": bars(OLD + FRESH[:-1], 6, NAIVE),
         "s_l2a": bars(FRESH, 5, NAIVE), "s_l2b": bars(FRESH, None, NAIVE9, "m5")}
    for k in ("primary_nine", "primary_eight"):
        v = V[k]
        if v["complete"]:
            v["met"] = v["inside"] >= v["need_inside"] and v["rho"] >= 0.6 and v["mae"] < v["mae_naive"]
    if V["s_l2a"]["complete"]:
        V["s_l2a"]["met"] = V["s_l2a"]["inside"] >= 5 and V["s_l2a"]["mae"] < V["s_l2a"]["mae_naive"]
    if V["s_l2b"]["complete"]:
        V["s_l2b"]["met"] = V["s_l2b"]["mae"] < V["s_l2b"]["mae_naive"] and V["s_l2b"]["rho"] >= 0.6
    if V["primary_nine"]["complete"] and V["primary_eight"]["complete"]:
        V["primary_verdict"] = V["primary_nine"]["met"] and V["primary_eight"]["met"]
    fr = [s for s in FRESH if s in res]
    if len(fr) == len(FRESH):
        hs = M.holm([res[s]["slope_p_one_sided"] for s in fr])
        for s, h in zip(fr, hs):
            res[s]["slope_holm"] = float(h)
        sig = sum(res[s]["slope"] < 0 and res[s]["slope_holm"] < 0.05 for s in fr)
        V["s_l1u"] = {"significant_negative": sig, "met": sig >= 4}
        tests = [(s, n) for s in fr if res[s].get("f3x") for n in ("vs_ref", "vs_list")]
        for (s, n), h in zip(tests, M.holm([res[s]["f3x"][n]["p"] for s, n in tests])):
            res[s]["f3x"][n]["holm"] = float(h)
        V["s_f3x"] = {f"{s}_{n}": {k: res[s]["f3x"][n][k] for k in ("rel", "p", "holm")} for s, n in tests}
    allg = {**{s: (r["observed"], r["info_reader"] - r["info_list"]) for s, r in res.items()}}
    for d, r in json.load(open("logs/probe/l2_explore.json"))["settings"].items():
        allg.setdefault(d, (r["gain"], r["info_reader"] - r["info_list"]))
    V["descriptive_rho_gain_vs_reader_minus_list"] = float(spearmanr([v[1] for v in allg.values()], [v[0] for v in allg.values()])[0])
    V["descriptive_n"] = len(allg)
    print(json.dumps(V, indent=1, default=float), flush=True)
    json.dump({"scenarios": res, "verdicts": V}, open(out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1])
