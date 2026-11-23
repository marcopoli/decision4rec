"""C2 scorer (prereg docs/r1/c2_prereg.md, SHA d68b8ee8...): reads logs/c2/<d>.jsonl from scripts/c2_stream.py.

L4a: fresh fused NDCG@10 above frozen fused, paired sign-flip randomization over events pooled over the four domains
(10,000, one-sided). L4b: in fresh mode, slope of the nats gain (ll_f - ll_l, events with the target in the 30) on
log10(1 + purchases of the target before the event), gain and x centred per domain (domain fixed effects), one-sided
permutation within domains (10,000); per-domain slopes with Holm. Also by week, targets born after the cut, seconds.
Secondary (no bar): on the events with a retrained row (day 30 on), share of the retrained-over-frozen fused gain that
fresh recovers, (fresh - frozen) / (retrained - frozen), with a 95% bootstrap interval over events (10,000), per domain
and pooled.

    python scripts/c2_score.py out.json [domain ...]
"""
import json
import sys

import numpy as np

from decision4rec.eval import metrics as M

FOUR = ("vg", "mi", "steam", "ind")
N = 10000


def slope(x, y):
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum())


def load(d):
    R = {}
    for r in map(json.loads, open(f"logs/c2/{d}.jsonl")):
        R.setdefault(r["mode"], {})[r["eid"]] = r
    return R


def share(z, f, r, rng):
    """(fresh - frozen) / (retrained - frozen) of the means, and its percentile bootstrap interval over events."""
    est = float((f - z).mean() / (r - z).mean()) if (r - z).mean() else float("nan")
    bs = []
    for _ in range(N):
        i = rng.integers(0, len(z), len(z))
        dn = (r[i] - z[i]).mean()
        bs.append((f[i] - z[i]).mean() / dn if dn else np.nan)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {"events": int(len(z)), "frozen": z.mean(), "fresh": f.mean(), "retrained": r.mean(), "share": est,
            "ci95": [float(lo), float(hi)], "p_retrained_vs_frozen_two_sided": float(M.paired_randomization(r, z))}


def secondary(doms, rng):
    res, Z, F, R_ = {}, [], [], []
    for d in doms:
        R = load(d)
        if "retrained" not in R:
            continue
        e = sorted(set(R["retrained"]) & set(R["frozen"]) & set(R["fresh"]))
        if not e:
            continue
        z, f, r = (np.array([R[m][i]["ndcg_f"] for i in e]) for m in ("frozen", "fresh", "retrained"))
        res[d] = share(z, f, r, rng)
        Z.append(z); F.append(f); R_.append(r)
    if Z:
        res["pooled"] = share(np.concatenate(Z), np.concatenate(F), np.concatenate(R_), rng)
        for d, v in res.items():
            print(f"secondary {d:6s} {v['events']} events | fused frozen {v['frozen']:.4f} fresh {v['fresh']:.4f} retrained {v['retrained']:.4f} | "
                  f"share {v['share']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]", flush=True)
    return res


def main(out, doms):
    rng = np.random.default_rng(0)
    res, diffs, X, Y, G = {}, [], [], [], []
    for d in doms:
        R = load(d)
        e = sorted(set(R["frozen"]) & set(R["fresh"]))
        fz = np.array([R["frozen"][i]["ndcg_f"] for i in e]); fr = np.array([R["fresh"][i]["ndcg_f"] for i in e])
        lz = np.array([R["frozen"][i]["ndcg_l"] for i in e]); lr = np.array([R["fresh"][i]["ndcg_l"] for i in e])
        ins = [R["fresh"][i] for i in e if R["fresh"][i]["pos"] >= 0]
        x = np.log10(1 + np.array([r["sold_before"] for r in ins], dtype=float))
        g = np.array([r["ll_f"] - r["ll_l"] for r in ins])
        sl = slope(x, g)
        null = np.array([slope(x, rng.permutation(g)) for _ in range(N)])
        weeks = sorted({R["fresh"][i]["week"] for i in e})
        born = [i for i in e if R["fresh"][i]["born_after_cut"]]
        res[d] = {"events": len(e), "ndcg": {"frozen_list": lz.mean(), "frozen_fused": fz.mean(), "fresh_list": lr.mean(), "fresh_fused": fr.mean()},
                  "fresh_minus_frozen": float(fr.mean() - fz.mean()), "p_fresh_gt_frozen_two_sided": float(M.paired_randomization(fr, fz)),
                  "target_in_30": {"frozen": float(np.mean([R["frozen"][i]["pos"] >= 0 for i in e])), "fresh": float(np.mean([R["fresh"][i]["pos"] >= 0 for i in e]))},
                  "slope_fresh": sl, "p_slope_one_sided": float((np.sum(null <= sl) + 1) / (N + 1)), "n_slope": len(ins),
                  "by_week": {int(w): {m: float(np.mean([R[m][i]["ndcg_f"] for i in e if R[m][i]["week"] == w])) for m in ("frozen", "fresh")} for w in weeks},
                  "born_after_cut": {"share": len(born) / len(e), "frozen_fused": float(np.mean([R["frozen"][i]["ndcg_f"] for i in born])) if born else None,
                                     "fresh_fused": float(np.mean([R["fresh"][i]["ndcg_f"] for i in born])) if born else None},
                  "seconds_per_event": {m: float(np.median([R[m][i]["seconds"] for i in e])) for m in ("frozen", "fresh")}}
        diffs.append(fr - fz); X.append(x - x.mean()); Y.append(g - g.mean()); G.append(np.full(len(x), len(G)))
    for (d, r), h in zip(res.items(), M.holm([r["p_slope_one_sided"] for r in res.values()])):
        r["holm_slope"] = float(h)
    D = np.concatenate(diffs)
    obs = D.mean()
    flips = np.array([(D * rng.choice([-1, 1], len(D))).mean() for _ in range(N)])
    pa = float((np.sum(flips >= obs) + 1) / (N + 1))
    x, y, gid = np.concatenate(X), np.concatenate(Y), np.concatenate(G)
    sl = slope(x, y)
    null = []
    for _ in range(N):
        yp = y.copy()
        for k in np.unique(gid):
            m = gid == k
            yp[m] = rng.permutation(y[m])
        null.append(slope(x, yp))
    pb = float((np.sum(np.array(null) <= sl) + 1) / (N + 1))
    pooled = {"events": int(len(D)), "fresh_minus_frozen": float(obs), "p_L4a_one_sided": pa, "L4a": pa < 0.05 and obs > 0,
              "slope_fresh": sl, "p_L4b_one_sided": pb, "L4b": pb < 0.05 and sl < 0}
    pooled["L4_confirmed"] = pooled["L4a"] and pooled["L4b"]
    for d, r in res.items():
        n = r["ndcg"]
        print(f"{d:6s} {r['events']} events | fused frozen {n['frozen_fused']:.4f} fresh {n['fresh_fused']:.4f} ({r['fresh_minus_frozen']:+.4f}) | "
              f"list frozen {n['frozen_list']:.4f} fresh {n['fresh_list']:.4f} | slope {r['slope_fresh']:+.4f} nats/tenfold p {r['p_slope_one_sided']:.4f} Holm {r['holm_slope']:.4f}", flush=True)
    print(f"pooled: fresh - frozen {obs:+.4f} p {pa:.4f} (L4a {pooled['L4a']}); slope {sl:+.4f} p {pb:.4f} (L4b {pooled['L4b']}); L4 confirmed {pooled['L4_confirmed']}", flush=True)
    json.dump({"domains": res, "pooled": pooled, "secondary_retrained": secondary(doms, rng)}, open(out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or FOUR)
