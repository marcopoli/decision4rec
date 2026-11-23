"""Fusion of a decision model with a list as a conditional logit (McFadden).

The utility of option i is b1 z(theta_i) + b2 z(r_i), the same form as the fused score, and the weights are fitted by
maximum likelihood on validation choices: the users whose target is among the options. A grid on validation NDCG@10
reads the same users through a noisy metric (about 200 positive users per domain), and its choice can move from 0.5 to
8 between resamples of the users. The likelihood uses the whole distribution over the options, so one fit is stable,
and the domains can share strength: a domain starts from the weights fitted on the others, so a new domain needs no
validation users, and its own validation choices move it as far as they can bear.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np


def fit_conditional_logit(X: np.ndarray, y: np.ndarray, l2: float = 1e-3, max_iter: int = 50, tol: float = 1e-6,
                          weights: Optional[np.ndarray] = None, prior: Optional[np.ndarray] = None,
                          strength: float = 0.0) -> np.ndarray:
    """Weights of the utilities X b (X: users, options, terms; y: the chosen option) by Newton steps. `weights`: of each
    user in the likelihood. `prior`, `strength`: a Gaussian prior on b around `prior`, worth about `strength` users."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
    n, p = len(y), X.shape[2]
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float) * n / np.sum(weights)
    b = np.full(p, 0.1) if prior is None else np.asarray(prior, dtype=float).copy()
    centre = np.zeros(p) if prior is None else np.asarray(prior, dtype=float)
    pen = l2 * n + strength
    for _ in range(max_iter):
        u = X @ b
        q = np.exp(u - u.max(1, keepdims=True))
        q /= q.sum(1, keepdims=True)
        mean = (q[..., None] * X).sum(1)
        grad = (w[:, None] * (X[np.arange(n), y] - mean)).sum(0) - pen * (b - centre)
        d = X - mean[:, None, :]
        hess = -np.einsum("n,nk,nkp,nkq->pq", w, q, d, d) - pen * np.eye(p)
        step = np.linalg.solve(hess, grad)
        b = b - step
        if np.abs(step).max() < tol:
            break
    return b


def choice_data(cases: dict, Z: dict) -> tuple[np.ndarray, np.ndarray]:
    """The validation choices of one domain: `Z[cid]` holds the terms of each option of `cases[cid]`. Users without the
    target among the options choose nothing here; cases with another number of options than most are left out."""
    K = Counter(len(c.cands) for c in cases.values()).most_common(1)[0][0]
    keys = [c for c in sorted(cases) if cases[c].target_pos >= 0 and len(cases[c].cands) == K]
    return np.stack([Z[c] for c in keys]), np.array([cases[c].target_pos for c in keys])


def leave_one_domain_out(data: dict, family: Optional[dict] = None) -> dict:
    """Weights for each domain of `data` (domain: (X, y)), fitted on the other domains only, each domain counting the
    same whatever its number of positive users. `family` joins settings of one domain (the two forms of MovieLens-1M):
    they share one share of the likelihood, and neither is fitted on the other."""
    family = family or {}
    of = lambda d: family.get(d, d)
    out = {}
    for d in data:
        rest = [x for x in data if of(x) != of(d)]
        size = {of(x): sum(len(data[z][1]) for z in rest if of(z) == of(x)) for x in rest}
        out[d] = fit_conditional_logit(np.concatenate([data[x][0] for x in rest]), np.concatenate([data[x][1] for x in rest]),
                                       weights=np.concatenate([np.full(len(data[x][1]), 1.0 / size[of(x)]) for x in rest]))
    return out


def shrunk_weights(data: dict, family: Optional[dict] = None, strength: float = 30.0) -> dict:
    """Weights for each domain, fitted on its own validation choices with the weights of the other domains as a prior:
    a domain with few positive users stays near them, one with many moves to its own (empirical Bayes). A strength of 30
    users has the best cross-validated likelihood on the validation users (scripts/strength_cv.py)."""
    others = leave_one_domain_out(data, family)
    return {d: fit_conditional_logit(*data[d], prior=others[d], strength=strength) for d in data}
