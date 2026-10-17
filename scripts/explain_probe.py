"""Explanations by removing one family of statements (Roadmap R9).

    python scripts/explain_probe.py configs/small/vg_rtm.yaml configs/backends/kev-0.8b-rtm.yaml [n_users]

For every test case the decision model answers the request of the full architecture, then the same request with one
family of statements left out (sold, lately, after, with, text) or without the "found by" lines. For each family:
mean change of the probability of the item ranked first, share of cases whose first choice changes, and NDCG@10 of
the decision without the family (an ablation at inference time), and the within-list Spearman correlation of the
answer with the order of the first stage (for the pool: its residual ranker), which shows what makes the two agree. The family whose removal changes the first item's
probability most is the explanation of that recommendation; a deletion test checks that it matters more than a family
drawn at random.
"""

import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

from decision4rec.data.schema import Dataset
from decision4rec.decision.base import build_backend
from decision4rec.eval import metrics as M
from decision4rec.evidence.store import FAMILIES
from decision4rec.experiments.runner import get_cases, load_config, make_prompt_builder

VARIANTS = ("full",) + FAMILIES + ("found",)


def ndcg(p, y):
    if y < 0:
        return 0.0
    r = int((p > p[y]).sum()) + 1
    return 1.0 / np.log2(r + 1) if r <= 10 else 0.0


def main(cfg_path: str, backend_path: str, n: int) -> None:
    cfg = load_config(cfg_path, backend_path)
    ds = Dataset.load(cfg["dataset"])
    cases = get_cases(cfg, ds)["test"][:n]
    pb = make_prompt_builder(cfg, ds)
    backend = build_backend(cfg["backend"])
    P = {v: [] for v in VARIANTS}
    Y = []
    for c in cases:
        order = list(range(len(c.cands)))
        for v in VARIANTS:
            pb.drop = frozenset([v]) if v in FAMILIES else frozenset()
            q = pb.choice(c, order, evidence="stats" if v == "found" else "stats+found")
            d = backend.decide(pb.state(c), {"next": q})
            pr = d.answers["next"].probs
            P[v].append(np.array([pr.get(k, 0.0) for k in q.keys()]))
        pb.drop = frozenset()
        Y.append(c.cands.index(c.target) if c.target in c.cands else -1)
    full = np.array(P["full"])
    top = full.argmax(1)
    rows, delta = {}, {}
    for v in VARIANTS:
        A = np.array(P[v])
        nd = np.mean([ndcg(A[i], Y[i]) for i in range(len(Y))])
        rhos = np.array([spearmanr(A[i], cases[i].rscores).statistic for i in range(len(Y))])
        rho = round(float(np.nanmean(rhos)), 3)
        if v == "full":
            rho_full = rhos
            rows[v] = {"ndcg10": round(float(nd), 4), "spearman_first_stage": rho}
            continue
        ok = ~np.isnan(rhos) & ~np.isnan(rho_full)                  # constant answers have no rank correlation
        dp = full[np.arange(len(top)), top] - A[np.arange(len(top)), top]
        delta[v] = np.abs(dp)
        rows[v] = {"mean_abs_dp_top": round(float(np.abs(dp).mean()), 4), "top_changes": round(float((A.argmax(1) != top).mean()), 3),
                   "ndcg10_without": round(float(nd), 4), "ndcg10_change": round(float(nd - rows["full"]["ndcg10"]), 4),
                   "spearman_first_stage": rho, "spearman_p_vs_full": round(float(M.paired_randomization(rhos[ok], rho_full[ok])), 4)}
    D = np.stack([delta[v] for v in VARIANTS[1:]], 1)
    expl = D.argmax(1)
    rng = np.random.default_rng(0)
    rand = rng.integers(0, D.shape[1], len(expl))
    share = {v: round(float((expl == k).mean()), 3) for k, v in enumerate(VARIANTS[1:])}
    out = {"dataset": ds.name, "backend": cfg["name"], "users": len(cases), "families": rows, "explanation_share": share,
           "deletion_test": {"chosen_family_mean_dp": round(float(D[np.arange(len(expl)), expl].mean()), 4),
                             "random_family_mean_dp": round(float(D[np.arange(len(rand)), rand].mean()), 4)}}
    path = os.path.join(cfg["out"], f"explain_{cfg['name']}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 500)
