"""A4, specification curve (V38; docs/18; CPU): how much does the fused-over-list gain of the 9B reader depend on the
choices of the analysis?

Specifications (5 x 5 x 3 x 2 = 150 per setting), on the stored runs of kev-9b-rtm (scripts/fusion_table.py load):
  metric  NDCG@5, NDCG@10, NDCG@20, HR@10, MRR
  fusion  stable weights with strength 0 (weights of the other settings alone), 10, 30 (the paper), 100; grid
          (list weight tuned on NDCG@10 of the fitting users, asc.GAMMAS)
  seed    reader seeds 0, 1, 2
  users   fit on validation users and score test users (the paper), or the two swapped
For each: fused against list, relative change and paired sign-flip p. No bar: report per setting the share of
specifications with a positive gain and with p < .05. Output: JSON and a figure (sorted effects, filled markers p < .05).

    python scripts/spec_curve.py out.json [fig.pdf]
"""
import json
import sys

import numpy as np

from fusion_table import FAMILY, SIX
from rare_pool import flip_p
from thin_probe import SEEDS, runs
from decision4rec.eval import metrics as M
from decision4rec.experiments.asc import GAMMAS
from decision4rec.operators.fusion import choice_data, leave_one_domain_out, shrunk_weights

METRICS = (("ndcg@5", 5), ("ndcg@10", 10), ("ndcg@20", 20), ("hr@10", 10), ("mrr", 10))
FUSIONS = ("s0", "s10", "s30", "s100", "grid")


def metric(S, cases, cids, name, k):
    return np.array([M.per_case_metrics(M.target_rank(S[c], cases[c].target_pos), k)[name] for c in cids])


def weights(D, fit, fusion):
    data = {d: choice_data(D[d][0][fit], D[d][1][fit]) for d in D}
    if fusion == "s0":
        return leave_one_domain_out(data, FAMILY)
    return shrunk_weights(data, FAMILY, float(fusion[1:]))


def grid(cases, Z, fit):
    C, cids = cases[fit], sorted(cases[fit])
    best, bv = None, -1.0
    for g in GAMMAS:
        v = metric({c: Z[fit][c][:, 0] + g * Z[fit][c][:, 1] for c in cids}, C, cids, "ndcg@10", 10).mean()
        if v > bv + 1e-12:
            best, bv = g, v
    return best


if __name__ == "__main__":
    out = sys.argv[1]
    fig = sys.argv[2] if len(sys.argv) > 2 else None
    rows = []
    for si, spec in enumerate(SEEDS):
        D = runs(spec)
        for fit, ev in (("valid", "test"), ("test", "valid")):
            W = {f: weights(D, fit, f) for f in FUSIONS if f != "grid"}
            for d, (cases, Z) in D.items():
                C, cids = cases[ev], sorted(cases[ev])
                lst = {c: Z[ev][c][:, 1] for c in cids}
                g = grid(cases, Z, fit)
                for f in FUSIONS:
                    fused = {c: (Z[ev][c][:, 0] + g * Z[ev][c][:, 1]) if f == "grid" else Z[ev][c] @ W[f][d] for c in cids}
                    for name, k in METRICS:
                        a, b = metric(lst, C, cids, name, k), metric(fused, C, cids, name, k)
                        rows.append({"setting": d, "seed": si, "users": f"{fit}->{ev}", "fusion": f, "metric": name,
                                     "list": a.mean(), "fused": b.mean(), "rel": (b.mean() - a.mean()) / a.mean() if a.mean() else float("nan"),
                                     "p": flip_p(b - a), "n": len(cids)})
        print(f"seed {si} done: {len(rows)} specifications", flush=True)
    summ = {}
    for d in sorted({r["setting"] for r in rows}, key=lambda x: SIX.index(x) if x in SIX else 99):
        R = [r for r in rows if r["setting"] == d]
        rel = np.array([r["rel"] for r in R])
        summ[d] = {"n": len(R), "positive": float(np.mean(rel > 0)), "pos_sig": float(np.mean([(r["rel"] > 0) and r["p"] < .05 for r in R])),
                   "neg_sig": float(np.mean([(r["rel"] < 0) and r["p"] < .05 for r in R])),
                   "median_rel": float(np.median(rel)), "q10_rel": float(np.quantile(rel, .1)), "q90_rel": float(np.quantile(rel, .9))}
        s = summ[d]
        print(f"{d:10s} specs {s['n']} | positive {100 * s['positive']:.0f}% | positive p<.05 {100 * s['pos_sig']:.0f}% | "
              f"negative p<.05 {100 * s['neg_sig']:.0f}% | median {100 * s['median_rel']:+.1f}% (10-90%: "
              f"{100 * s['q10_rel']:+.1f} to {100 * s['q90_rel']:+.1f})", flush=True)
    json.dump({"summary": summ, "rows": rows}, open(out, "w"), indent=1)
    if fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ds = list(summ)
        f, axes = plt.subplots(1, len(ds), figsize=(2.4 * len(ds), 2.6), sharey=True)
        for ax, d in zip(np.atleast_1d(axes), ds):
            R = sorted((r for r in rows if r["setting"] == d), key=lambda r: r["rel"])
            x = np.arange(len(R))
            y = np.array([100 * r["rel"] for r in R])
            sig = np.array([r["p"] < .05 for r in R])
            ax.scatter(x[~sig], y[~sig], s=6, facecolors="none", edgecolors="0.5", linewidths=0.6)
            ax.scatter(x[sig], y[sig], s=6, color="0.1")
            ax.axhline(0, color="0.6", lw=0.6)
            ax.set_title(d, fontsize=8)
            ax.set_xticks([])
            ax.tick_params(labelsize=7)
        np.atleast_1d(axes)[0].set_ylabel("fused over list (%)", fontsize=8)
        f.tight_layout()
        f.savefig(fig)
