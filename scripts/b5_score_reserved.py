"""UniSRec against Decision4Rec-Full and the fixed reference on the 10,000 reserved users of the four domains (added
after the pre-registration of docs/r1/b5_prereg.md, which fixed the development users for these settings; the same test
and the same Holm correction, over the four domains).

Full as scripts/r1_score.py f1: per-user mean over the three 9B seeds of the fused NDCG@10 with the weights fitted on the
development runs. UniSRec: the transductive fine-tuned models of the plan, trained once, ranking the reserved users
(scripts/b5_unisrec.py fit --score-npz). Reference: SASRecF+ (docs/11 v2 Section 3).

    python scripts/b5_score_reserved.py out.json
"""
import json
import sys

import numpy as np

import r1_score as R
from fusion_table import FOUR, SIX
from decision4rec.agents.pool import _trained
from decision4rec.data.schema import Dataset
from decision4rec.eval import metrics as M
from decision4rec.experiments.asc import _ndcg
from decision4rec.experiments.runner import load_config
from decision4rec.models import baselines as B


def nd(r):
    r = np.asarray(r, dtype=float)
    return np.where(r <= 10, 1.0 / np.log2(r + 1.0), 0.0)


def cmp(a, b):
    return {"rel": float(100 * (a.mean() / b.mean() - 1)), "ci95": R.boot(a, b), "p": float(M.paired_randomization(a, b))}


res, tests = {}, []
for d in FOUR:
    per, T = [], None
    for tag in R.SEEDS:
        X = R.load(d, "r1rtm", tag, min_valid=200)
        if X is None:
            continue
        cases, Z = X
        w = R.weights(SIX, "rtm", tag)
        T = cases["test"]
        per.append(dict(zip(sorted(T), _ndcg({c: Z["test"][c] @ w[d] for c in T}, T))))
    uni = json.load(open(f"runs/small/{d}/baselines/unisrec_transductive_ft_reserved/test_ranks.json"))
    k = sorted(set.intersection(*(set(p) for p in per)) & {int(c) for c in uni})
    assert len(k) == len(uni) == 10000, (d, len(k), len(uni))
    F = np.array([np.mean([p[c] for p in per]) for c in k])
    U = nd([uni[str(c)] for c in k])
    rc, rn, path = R.ref_ndcg(d, "r1rtm", {c: T[c] for c in k})
    assert rc == k
    cfg = load_config(f"configs/small/{d}_rtm.yaml")
    C = [T[c] for c in k]
    S = nd(B.load("sasrecf", _trained(cfg["out"], "sasrecf"), Dataset.load(cfg["dataset"]), "cpu").ranks([c.user for c in C], [c.history for c in C], [c.target for c in C]))
    res[d] = {"users": len(k), "seeds": len(per), "full": float(F.mean()), "unisrec": float(U.mean()), "sasrecf": float(S.mean()), "unisrec_vs_sasrecf": cmp(U, S), "ref": R.REF[d],
              "ref_ndcg": float(rn.mean()), "full_vs_unisrec": cmp(F, U), "unisrec_vs_ref": cmp(U, rn)}
    tests.append((d, res[d]["full_vs_unisrec"]["p"]))
for (d, _), h in zip(tests, M.holm([t[1] for t in tests])):
    res[d]["full_vs_unisrec"]["holm"] = float(h)
for d, r in res.items():
    f = r["full_vs_unisrec"]
    print(f"{d:6s} users {r['users']} seeds {r['seeds']} | UniSRec {r['unisrec']:.4f} ref {r['ref']} {r['ref_ndcg']:.4f} "
          f"({r['unisrec_vs_ref']['rel']:+.1f}%, p={r['unisrec_vs_ref']['p']:.3f}) | Full {r['full']:.4f} vs UniSRec "
          f"{f['rel']:+.1f}% [{f['ci95'][0]:+.1f}, {f['ci95'][1]:+.1f}] p={f['p']:.4f} Holm {f['holm']:.4f}", flush=True)
json.dump({"users": "reserved", "domains": res}, open(sys.argv[1], "w"), indent=1, default=float)
