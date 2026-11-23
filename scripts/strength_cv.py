"""Strength of the prior of the stable fusion, chosen on validation users only (2026-10-30): five-fold cross-validated
log-likelihood of the validation choices of every variant and setting, for each strength. No test user is read.
PYTHONPATH=scripts:. python logs/probe/strength_cv.py"""
import numpy as np

from fusion_table import FAMILY, GROUPS, SIX, load
from decision4rec.operators.fusion import choice_data, fit_conditional_logit, leave_one_domain_out

STRENGTHS = (0.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 1e9)


def loglik(X, y, b):
    u = X @ b
    return (u[np.arange(len(y)), y] - np.log(np.exp(u - u.max(1, keepdims=True)).sum(1)) - u.max(1)).sum()


specs = [s for name in ("sizes", "sizes8k", "list", "kg", "kg8k") for s in GROUPS[name][2]]
total = {k: 0.0 for k in STRENGTHS}
by = {}
for sp in specs:
    conf, tag = sp.split(":")
    D = {d: load(d, conf, tag) for d in SIX}
    D = {d: v for d, v in D.items() if v is not None}
    data = {d: choice_data(D[d][0]["valid"], D[d][1]["valid"]) for d in D}
    others = leave_one_domain_out(data, FAMILY)
    for d, (X, y) in data.items():
        fold = np.random.default_rng(0).integers(0, 5, len(y))
        for k in STRENGTHS:
            ll = 0.0
            for f in range(5):
                tr, te = fold != f, fold == f
                b = fit_conditional_logit(X[tr], y[tr], prior=others[d], strength=k) if k < 1e8 else others[d]
                ll += loglik(X[te], y[te], b)
            total[k] += ll / len(y)
            by.setdefault(d, {}).setdefault(k, 0.0)
            by[d][k] += ll / len(y)
    print(sp, "done", flush=True)
print("strength: mean validation log-likelihood per user, summed over variants and settings", flush=True)
for k in STRENGTHS:
    print(f"{k:g}: {total[k]:.3f} | " + " ".join(f"{d} {by[d][k]:.2f}" for d in by), flush=True)
