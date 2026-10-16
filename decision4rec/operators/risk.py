"""Risk control from pool probabilities: selective recommendation and conformal slates."""

from __future__ import annotations

import numpy as np


def risk_coverage(conf: np.ndarray, loss: np.ndarray) -> dict:
    """Selective risk as coverage grows when cases are served in order of decreasing confidence.

    AURC is the area under that curve; a lower value means confidence ranks errors better.
    """
    conf, loss = np.asarray(conf, dtype=float), np.asarray(loss, dtype=float)
    if len(conf) == 0:
        return {"aurc": float("nan"), "coverage": [], "risk": []}
    order = np.argsort(-conf, kind="stable")
    risk = np.cumsum(loss[order]) / np.arange(1, len(loss) + 1)
    cov = np.arange(1, len(loss) + 1) / len(loss)
    grid = np.linspace(0.1, 1.0, 10)
    at = [float(risk[min(int(np.ceil(g * len(loss))) - 1, len(loss) - 1)]) for g in grid]
    return {"aurc": float(risk.mean()), "coverage": grid.tolist(), "risk": at}


def conformal_quantile(nonconformity: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile with the finite-sample correction."""
    s = np.sort(np.asarray(nonconformity, dtype=float))
    n = len(s)
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float("inf") if k > n else float(s[k - 1])


def lac_set(probs: np.ndarray, qhat: float) -> np.ndarray:
    """Least-ambiguous set: every label whose nonconformity 1 - p is within the quantile."""
    return np.flatnonzero(1.0 - np.asarray(probs, dtype=float) <= qhat + 1e-12)


def conformal_report(cal: list[tuple[np.ndarray, int]], test: list[tuple[np.ndarray, int]],
                     alphas=(0.1, 0.2, 0.3)) -> list[dict]:
    """Coverage and size of conformal label sets.

    Each entry is (probabilities over labels, index of the true label). When an
    outside option is used it is one of the labels, so a set containing it
    states that the right item may be missing from the candidates.
    """
    out = []
    if not cal or not test:
        return out
    s_cal = np.array([1.0 - p[y] for p, y in cal])
    for a in alphas:
        q = conformal_quantile(s_cal, a)
        sets = [lac_set(p, q) for p, _ in test]
        out.append({"alpha": a, "target_coverage": 1 - a,
                    "coverage": float(np.mean([y in s for s, (_, y) in zip(sets, test)])),
                    "mean_set_size": float(np.mean([len(s) for s in sets])),
                    "empty_sets": float(np.mean([len(s) == 0 for s in sets]))})
    return out
