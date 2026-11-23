"""A8, beyond-accuracy tables (docs/18; CPU, stored runs): what does fusing the 9B reader change in the top 10 besides
accuracy?

For the development test users of each setting, the top 10 of the list and of the fused system (list + kev-9b-rtm,
stable weights of thin_probe.per_user), over the 30 options:
  coverage   distinct items in all top 10s, as a share of the catalogue
  pop        mean log10(1 + purchases in the training part) of the top 10 (count as thin_probe.covariates)
  rare       share of the top 10 with fewer than 20 purchases
  new        share of the top 10 first seen in the 30 days before the request (settings with real time stamps)
  NDCG@10 by history length (1-5, 6-10, 11-20, over 20 items)
Paired sign-flip p (scripts/rare_pool.flip_p) for pop, rare, new and per history bucket. Output JSON and a LaTeX table.

    python scripts/beyond.py out.json out.tex
"""
import json
import sys

import numpy as np

from fusion_table import SIX
from rare_pool import flip_p
from thin_probe import SEEDS, runs
from decision4rec.data.schema import Dataset
from decision4rec.eval import metrics as M
from decision4rec.operators.fusion import choice_data, shrunk_weights
from fusion_table import FAMILY, STRENGTH

BUCKETS = ((1, 5), (6, 10), (11, 20), (21, 10 ** 9))


def store_stats(d):
    ds = Dataset.load(json.load(open(f"runs/small/{d}/{d}_rtm_kev-9b-rtm/manifest.json"))["config"]["dataset"])
    pop, first, times = np.zeros(ds.n_items), np.full(ds.n_items, np.inf), []
    for s in ds.seqs:
        for x in s:
            first[x[0]] = min(first[x[0]], float(x[1]))
            times.append(float(x[1]))
        for x in (s[:-2] if len(s) > 2 else []):
            pop[x[0]] += 1
    span = max(times) - min(times)
    day = 86400e3 if span > 1e11 else 86400.0 if span > 1e7 else None
    return ds, pop, first, day


if __name__ == "__main__":
    out, tex = sys.argv[1], sys.argv[2]
    D = runs(SEEDS[0])
    w = shrunk_weights({d: choice_data(D[d][0]["valid"], D[d][1]["valid"]) for d in D}, FAMILY, STRENGTH)
    res = {}
    for d in [e for e in SIX if e in D]:
        cases, Z = D[d]
        T, cids = cases["test"], sorted(cases["test"])
        ds, pop, first, day = store_stats(d)
        per = {k: {"list": [], "fused": []} for k in ("pop", "rare", "new", "ndcg")}
        top = {"list": set(), "fused": set()}
        hl = []
        for c in cids:
            case = T[c]
            cand = np.asarray(case.cands)
            seq = ds.seqs[case.user]
            k = len(case.history)
            assert seq[k][0] == case.target, f"{d} {c}: target differs from the data"
            now = float(seq[k][1])
            hl.append(k)
            for name, s in (("list", Z["test"][c][:, 1]), ("fused", Z["test"][c] @ w[d])):
                t10 = cand[np.argsort(-s, kind="stable")[:10]]
                top[name].update(t10.tolist())
                per["pop"][name].append(np.log10(1 + pop[t10]).mean())
                per["rare"][name].append((pop[t10] < 20).mean())
                per["new"][name].append(((now - first[t10]) <= 30 * day).mean() if day else np.nan)
                per["ndcg"][name].append(M.per_case_metrics(M.target_rank(s, case.target_pos))["ndcg@10"])
        hl = np.array(hl)
        row = {"users": len(cids), "items": int(ds.n_items),
               "coverage": {n: len(top[n]) / ds.n_items for n in top}}
        for k in ("pop", "rare", "new"):
            a, b = np.array(per[k]["list"]), np.array(per[k]["fused"])
            row[k] = {"list": float(np.nanmean(a)), "fused": float(np.nanmean(b)),
                      "p": flip_p(np.nan_to_num(b - a)) if day or k != "new" else None}
        a, b = np.array(per["ndcg"]["list"]), np.array(per["ndcg"]["fused"])
        row["ndcg_by_history"] = {}
        for lo, hi in BUCKETS:
            m = (hl >= lo) & (hl <= hi)
            if m.sum() >= 30:
                row["ndcg_by_history"][f"{lo}-{hi if hi < 10 ** 9 else 'max'}"] = {
                    "users": int(m.sum()), "list": float(a[m].mean()), "fused": float(b[m].mean()), "p": flip_p(b[m] - a[m])}
        res[d] = row
        print(f"{d:10s} coverage {100 * row['coverage']['list']:.2f}% -> {100 * row['coverage']['fused']:.2f}% | "
              f"pop {row['pop']['list']:.3f} -> {row['pop']['fused']:.3f} (p {row['pop']['p']:.4f}) | rare "
              f"{100 * row['rare']['list']:.1f}% -> {100 * row['rare']['fused']:.1f}% (p {row['rare']['p']:.4f}) | new "
              + (f"{100 * row['new']['list']:.1f}% -> {100 * row['new']['fused']:.1f}% (p {row['new']['p']:.4f})" if day else "n/a")
              + " | NDCG@10 by history " + " ".join(f"{b_}: {v['list']:.4f}->{v['fused']:.4f} (n {v['users']}, p {v['p']:.3f})"
                                                     for b_, v in row["ndcg_by_history"].items()), flush=True)
    json.dump(res, open(out, "w"), indent=1)
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r"Setting & \multicolumn{2}{c}{Coverage (\%)} & \multicolumn{2}{c}{Popularity} & \multicolumn{2}{c}{Rare (\%)} & \multicolumn{2}{c}{New (\%)} \\",
             r" & List & Fused & List & Fused & List & Fused & List & Fused \\", r"\midrule"]
    star = lambda p: "$^{*}$" if p is not None and p < .05 else ""
    for d, r in res.items():
        new = (f"{100 * r['new']['list']:.1f} & {100 * r['new']['fused']:.1f}{star(r['new']['p'])}" if r["new"]["p"] is not None else "-- & --")
        lines.append(f"{d} & {100 * r['coverage']['list']:.2f} & {100 * r['coverage']['fused']:.2f} & {r['pop']['list']:.2f} & "
                     f"{r['pop']['fused']:.2f}{star(r['pop']['p'])} & {100 * r['rare']['list']:.1f} & {100 * r['rare']['fused']:.1f}"
                     f"{star(r['rare']['p'])} & {new} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(tex, "w").write("\n".join(lines) + "\n")
