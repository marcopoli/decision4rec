"""Ranking metrics, calibration measures and paired significance tests (numpy only)."""

from __future__ import annotations

import numpy as np


def target_rank(scores: np.ndarray, target_pos: int) -> float:
    """1-based rank of the target under `scores` (higher is better), inf if absent. Ties count against it."""
    if target_pos < 0:
        return float("inf")
    s = np.asarray(scores, dtype=float)
    return float(1 + np.sum(s > s[target_pos]) + np.sum(s == s[target_pos]) - 1)


def zscore(x) -> np.ndarray:
    """Standardise within one candidate set. Non-finite entries take the smallest finite value."""
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, np.nanmin(x[np.isfinite(x)]) if np.isfinite(x).any() else 0.0)
    return (x - x.mean()) / (x.std() + 1e-9)


def per_case_metrics(rank: float, k: int = 10) -> dict:
    hit = rank <= k
    return {f"ndcg@{k}": float(1.0 / np.log2(rank + 1.0)) if hit else 0.0,
            f"hr@{k}": float(hit), "mrr": 0.0 if np.isinf(rank) else float(1.0 / rank)}


# ---- calibration -----------------------------------------------------------

def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    if len(conf) == 0:
        return float("nan")
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        sel = bins == b
        if sel.any():
            total += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(total)


def reliability(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> list[dict]:
    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    return [{"bin": b, "n": int((bins == b).sum()), "confidence": float(conf[bins == b].mean()),
             "accuracy": float(correct[bins == b].mean())} for b in range(n_bins) if (bins == b).any()]


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Area under the ROC curve by the rank statistic, nan if one class is empty."""
    score, label = np.asarray(score, dtype=float), np.asarray(label, dtype=bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sorted_s = score[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i: j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def fit_temperature(logits: list[np.ndarray], labels: list[int]) -> float:
    """Temperature minimising the negative log-likelihood of the true labels (golden-section search)."""
    def nll(log_t: float) -> float:
        t = np.exp(log_t)
        total = 0.0
        for z, y in zip(logits, labels):
            z = np.asarray(z, dtype=float) / t
            z = z - z.max()
            total -= z[y] - np.log(np.exp(z).sum())
        return total / max(len(labels), 1)

    lo, hi = np.log(0.05), np.log(20.0)
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = hi - phi * (hi - lo), lo + phi * (hi - lo)
    fc, fd = nll(c), nll(d)
    for _ in range(60):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - phi * (hi - lo)
            fc = nll(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + phi * (hi - lo)
            fd = nll(d)
    return float(np.exp((lo + hi) / 2.0))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def fit_platt(p: np.ndarray, y: np.ndarray, iters: int = 200) -> tuple[float, float]:
    """Logistic recalibration q = sigmoid(a * logit(p) + b) by damped Newton steps."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p))
    y = np.asarray(y, dtype=float)
    a, b = 1.0, 0.0
    for _ in range(iters):
        q = _sigmoid(a * x + b)
        g = np.array([np.sum((q - y) * x), np.sum(q - y)])
        w = q * (1 - q)
        h = np.array([[np.sum(w * x * x) + 1e-3, np.sum(w * x)], [np.sum(w * x), np.sum(w) + 1e-3]])
        step = np.linalg.solve(h, g)
        norm = np.abs(step).max()
        if norm > 2.0:
            step = step * (2.0 / norm)
        a, b = a - step[0], b - step[1]
        if norm < 1e-8:
            break
    return float(a), float(b)


def apply_platt(p: np.ndarray, a: float, b: float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return _sigmoid(a * np.log(p / (1 - p)) + b)


# ---- ordinal scales --------------------------------------------------------

def quadratic_weighted_kappa(pred: np.ndarray, gold: np.ndarray, n_levels: int) -> float:
    """Agreement between two ordinal labelings, 1 = perfect, 0 = chance."""
    pred, gold = np.asarray(pred, dtype=int), np.asarray(gold, dtype=int)
    if len(pred) == 0 or n_levels < 2:
        return float("nan")
    obs = np.zeros((n_levels, n_levels))
    np.add.at(obs, (gold, pred), 1.0)
    idx = np.arange(n_levels)
    w = (idx[:, None] - idx[None, :]) ** 2 / (n_levels - 1) ** 2
    exp = np.outer(obs.sum(1), obs.sum(0)) / obs.sum()
    den = (w * exp).sum()
    return float(1.0 - (w * obs).sum() / den) if den > 0 else float("nan")


def scale_utilisation(pred_dist: np.ndarray, gold_dist: np.ndarray) -> dict:
    """Effective-support ratio R = exp(H(pred)) / exp(H(gold)) and total variation distance between
    the distribution of predicted levels and of gold levels (arXiv:2609.38827). R below 1 means the
    model uses fewer effective levels than the data."""
    def entropy(p):
        p = np.asarray(p, dtype=float)
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())
    p = np.asarray(pred_dist, dtype=float) / max(np.sum(pred_dist), 1e-12)
    q = np.asarray(gold_dist, dtype=float) / max(np.sum(gold_dist), 1e-12)
    return {"R": float(np.exp(entropy(p)) / np.exp(entropy(q))), "tvd": float(0.5 * np.abs(p - q).sum())}


# ---- significance ----------------------------------------------------------

def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_randomization(a: np.ndarray, b: np.ndarray, n_perm: int = 10000, seed: int = 0) -> float:
    """Two-sided p-value for mean(a) != mean(b) over paired cases (sign-flip test)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(d) == 0 or np.allclose(d, 0):
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    return float((np.sum(np.abs((signs * d).mean(1)) >= obs - 1e-15) + 1) / (n_perm + 1))


def paired_randomization_large(a: np.ndarray, b: np.ndarray, n_perm: int = 2000, seed: int = 0,
                               chunk: int = 100) -> float:
    """The sign-flip test of `paired_randomization` for very many cases. Pairs with no difference do not change
    the flipped sums, so only the others are drawn, and permutations are drawn in chunks to bound memory."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    d = d[d != 0]
    if n == 0 or len(d) == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(d.sum()) / n
    hits = 0
    for start in range(0, n_perm, chunk):
        m = min(chunk, n_perm - start)
        signs = rng.choice([-1.0, 1.0], size=(m, len(d)))
        hits += int(np.sum(np.abs(signs @ d) / n >= obs - 1e-15))
    return float((hits + 1) / (n_perm + 1))


def bootstrap_ci_large(x: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0,
                       chunk: int = 50) -> tuple[float, float]:
    """`bootstrap_ci` for very many cases, resampling in chunks to bound memory."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.concatenate([x[rng.integers(0, len(x), (min(chunk, n_boot - s), len(x)))].mean(1)
                            for s in range(0, n_boot, chunk)])
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()
