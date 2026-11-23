"""B5 scoring: UniSRec against SASRecF, the fixed reference and the system (Full), on the same test cases.

Full as scripts/r1_score.py: per-user mean over the three 9B seeds of the fused NDCG@10; development settings take the
weights fitted on the six development settings, unseen scenarios their own validation users with the six as prior (F3).
Paired randomization (10,000), 95% paired bootstrap of the relative change, Holm over settings for Full vs UniSRec.

    python scripts/b5_score.py out.json <variant> d1 [d2 ...]     variant: dir suffix, e.g. transductive_ft
    python scripts/b5_score.py out.json selftest d1 [...]         SASRecF in place of UniSRec: must give 0% and p = 1
"""
import json
import sys

import numpy as np

import r1_score as R
from fusion_table import SIX
from decision4rec.agents.pool import _trained
from decision4rec.data.schema import Dataset
from decision4rec.eval import metrics as M
from decision4rec.experiments.asc import _ndcg
from decision4rec.experiments.runner import load_config
from decision4rec.models import baselines as B
from decision4rec.operators.fusion import choice_data


def nd(r):
    r = np.asarray(r, dtype=float)
    return np.where(r <= 10, 1.0 / np.log2(r + 1.0), 0.0)


def full(d):
    per, T = [], None
    for tag in R.SEEDS:
        X = R.load(d, "rtm", tag)
        if X is None:
            continue
        cases, Z = X
        w = R.weights(SIX, "rtm", tag) if d in SIX else R.weights(SIX, "rtm", tag, {d: choice_data(cases["valid"], Z["valid"])})
        T = cases["test"]
        per.append(dict(zip(sorted(T), _ndcg({c: Z["test"][c] @ w[d] for c in T}, T))))
    return per, T


def cmp(a, b):
    return {"rel": float(100 * (a.mean() / b.mean() - 1)), "ci95": R.boot(a, b), "p": float(M.paired_randomization(a, b))}


def main(out, variant, doms):
    res, tests = {}, []
    for d in doms:
        test = variant == "selftest"
        if not test:
            uni = json.load(open(f"runs/small/{d}/baselines/unisrec_{variant}/test_ranks.json"))
            info = json.load(open(f"runs/small/{d}/baselines/unisrec_{variant}/info.json"))
        per, T = full(d)
        if not per:
            print(f"{d}: no reader run" + ("" if test else f", UniSRec only: NDCG@10 {nd(list(uni.values())).mean():.4f}"), flush=True)
            continue
        k = sorted(set.intersection(*(set(p) for p in per)) & (set(T) if test else {int(c) for c in uni}))
        if not test and len(k) != len(uni):
            print(f"{d}: {len(uni)} UniSRec cases, {len(k)} shared with Full", flush=True)
        F = np.array([np.mean([p[c] for p in per]) for c in k])
        cfg = load_config(f"configs/small/{d}_rtm.yaml")
        ds = Dataset.load(cfg["dataset"])
        C = [T[c] for c in k]
        S = nd(B.load("sasrecf", _trained(cfg["out"], "sasrecf"), ds, "cpu").ranks([c.user for c in C], [c.history for c in C], [c.target for c in C]))
        if test:
            U, info = S.copy(), {}
        else:
            U = nd([uni[str(c)] for c in k])
        r = {"users": len(k), "seeds": len(per), "unisrec": U.mean(), "sasrecf": S.mean(), "full": F.mean(),
             "best_epoch": info.get("best_epoch"), "smoke": info.get("smoke"),
             "unisrec_vs_sasrecf": cmp(U, S), "full_vs_unisrec": cmp(F, U)}
        if d in R.REF:
            rc, rn, path = R.ref_ndcg(d, "rtm", {c: T[c] for c in k})
            assert rc == k
            r.update({"ref": R.REF[d], "ref_ndcg": rn.mean(), "unisrec_vs_ref": cmp(U, rn)})
        if test:
            assert r["unisrec_vs_sasrecf"]["rel"] == 0 and r["unisrec_vs_sasrecf"]["p"] == 1, r["unisrec_vs_sasrecf"]
        res[d] = r
        tests.append((d, r["full_vs_unisrec"]["p"]))
    for (d, _), h in zip(tests, M.holm([t[1] for t in tests])):
        res[d]["full_vs_unisrec"]["holm"] = float(h)
    for d, r in res.items():
        f = r["full_vs_unisrec"]
        print(f"{d:9s} users {r['users']} | UniSRec {r['unisrec']:.4f} vs SASRecF {r['sasrecf']:.4f} "
              f"{r['unisrec_vs_sasrecf']['rel']:+.1f}% p={r['unisrec_vs_sasrecf']['p']:.3f}"
              + (f" | vs {r['ref']} {r['ref_ndcg']:.4f} {r['unisrec_vs_ref']['rel']:+.1f}%" if "ref" in r else "")
              + f" | Full {r['full']:.4f} vs UniSRec {f['rel']:+.1f}% [{f['ci95'][0]:+.1f}, {f['ci95'][1]:+.1f}] p={f['p']:.3f} Holm {f['holm']:.3f}",
              flush=True)
    json.dump({"variant": variant, "settings": res}, open(out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
