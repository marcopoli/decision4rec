"""Position-augmented Luce model fitted to soft choice distributions.

A decision model asked to choose within a block B returns a full distribution
p_B. The model is

    pi_B(k) = w[i_k] * b[k] / (w_out + sum_j w[i_j] * b[j])

with item worths w, position effects b and an optional outside worth. The fit
minimises sum_B KL(p_B || pi_B) by MM updates under a weak Gamma prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

_EPS = 1e-12


@dataclass
class Obs:
    """One answered block: items in presented order, their probabilities, the outside probability."""

    items: np.ndarray
    probs: np.ndarray
    p_out: float = float("nan")

    @property
    def has_outside(self) -> bool:
        return not np.isnan(self.p_out)

    def to_json(self) -> dict:
        return {"items": [int(i) for i in self.items], "probs": [float(p) for p in self.probs],
                "p_out": None if not self.has_outside else float(self.p_out)}

    @classmethod
    def from_json(cls, d: dict) -> "Obs":
        p_out = d.get("p_out")
        return cls(items=np.asarray(d["items"], dtype=int), probs=np.asarray(d["probs"], dtype=float),
                   p_out=float("nan") if p_out is None else float(p_out))


@dataclass
class LuceFit:
    theta: np.ndarray        # log worths, centred on the items
    theta_out: float         # log outside worth on the same scale, nan without outside option
    residual: float          # mean KL(p_B || pi_B): zero iff the answers are Luce consistent
    n_iter: int
    item_dispersion: np.ndarray  # per-item std of log p minus log pi across blocks

    def pool_probs(self, temperature: float = 1.0) -> tuple[np.ndarray, float]:
        """Choice shares over the whole pool, and the outside share (0 without outside option)."""
        z = self.theta / temperature
        if np.isnan(self.theta_out):
            z = z - z.max()
            p = np.exp(z)
            return p / p.sum(), 0.0
        zo = self.theta_out / temperature
        top = max(z.max(), zo)
        p, po = np.exp(z - top), np.exp(zo - top)
        total = p.sum() + po
        return p / total, float(po / total)


def _flatten(obs: list[Obs], beta: Optional[np.ndarray]):
    items = np.concatenate([o.items for o in obs]).astype(int)
    probs = np.concatenate([o.probs for o in obs]).astype(float)
    pos = np.concatenate([np.arange(len(o.items)) for o in obs])
    oid = np.concatenate([np.full(len(o.items), c) for c, o in enumerate(obs)])
    if beta is None:
        b = np.ones(len(items))
    else:
        beta = np.asarray(beta, dtype=float)
        b = np.exp(beta[np.minimum(pos, len(beta) - 1)])
    out_mask = np.array([o.has_outside for o in obs])
    p_out = np.array([o.p_out if o.has_outside else 0.0 for o in obs])
    return items, probs, pos, oid, b, out_mask, p_out


def fit_luce(obs: list[Obs], n_items: int, beta: Optional[np.ndarray] = None, prior: float = 1e-5,
             max_iter: int = 2000, tol: float = 1e-7, w0: Optional[np.ndarray] = None) -> LuceFit:
    if not obs:
        return LuceFit(np.zeros(n_items), float("nan"), 0.0, 0, np.zeros(n_items))
    items, probs, pos, oid, b, out_mask, p_out = _flatten(obs, beta)
    n_obs = len(obs)
    has_out = bool(out_mask.any())
    num = prior + np.bincount(items, weights=probs, minlength=n_items)
    num_out = prior + p_out.sum()
    w = np.ones(n_items) if w0 is None else np.asarray(w0, dtype=float).copy()
    w_out = 1.0
    it = 0
    for it in range(1, max_iter + 1):
        z = np.bincount(oid, weights=w[items] * b, minlength=n_obs)
        if has_out:
            z = z + w_out * out_mask
        den = prior + np.bincount(items, weights=b / z[oid], minlength=n_items)
        w_new = num / den
        if has_out:
            w_out_new = num_out / (prior + (out_mask / z).sum())
        else:
            w_out_new = w_out
        # Convergence is judged on centred log worths: the likelihood is scale free, so the
        # common scale is set by the weak prior alone and drifts slowly without affecting anything.
        step = np.log(w_new) - np.log(w)
        centre = step.mean()
        delta = np.max(np.abs(step - centre))
        if has_out:
            delta = max(delta, abs(np.log(w_out_new) - np.log(w_out) - centre))
        w, w_out = w_new, w_out_new
        if delta < tol:
            break
    z = np.bincount(oid, weights=w[items] * b, minlength=n_obs)
    if has_out:
        z = z + w_out * out_mask
    pi = w[items] * b / z[oid]
    kl = np.bincount(oid, weights=probs * (np.log(probs + _EPS) - np.log(pi + _EPS)), minlength=n_obs)
    if has_out:
        pi_out = w_out / z
        kl = kl + out_mask * p_out * (np.log(p_out + _EPS) - np.log(pi_out + _EPS))
    res = np.log(probs + 1e-6) - np.log(pi + 1e-6)
    cnt = np.maximum(np.bincount(items, minlength=n_items), 1)
    mean = np.bincount(items, weights=res, minlength=n_items) / cnt
    var = np.bincount(items, weights=(res - mean[items]) ** 2, minlength=n_items) / cnt
    theta = np.log(w)
    shift = theta.mean()
    return LuceFit(theta=theta - shift,
                   theta_out=float(np.log(w_out) - shift) if has_out else float("nan"),
                   residual=float(kl.mean()), n_iter=it, item_dispersion=np.sqrt(var))


def fit_position_bias(groups: list[tuple[list[Obs], int]], n_pos: int, prior: float = 0.1,
                      item_prior: float = 1e-5, outer: int = 200, tol: float = 1e-6) -> np.ndarray:
    """Shared position effects beta (beta[0] = 0) from many users' observations.

    Identified only when items appear at different positions within a user,
    that is with at least two rounds or two permutations per user.
    """
    beta = np.zeros(n_pos)
    warm: list[Optional[np.ndarray]] = [None] * len(groups)
    for _ in range(outer):
        num = np.zeros(n_pos)
        den = np.zeros(n_pos)
        for g, (obs, n_items) in enumerate(groups):
            fit = fit_luce(obs, n_items, beta=beta, prior=item_prior, w0=warm[g], max_iter=300, tol=1e-7)
            w = np.exp(fit.theta)
            warm[g] = w
            w_out = np.exp(fit.theta_out) if not np.isnan(fit.theta_out) else 0.0
            for o in obs:
                k = np.minimum(np.arange(len(o.items)), n_pos - 1)
                bk = np.exp(beta[k])
                zc = (w[o.items] * bk).sum() + (w_out if o.has_outside else 0.0)
                np.add.at(num, k, o.probs)
                np.add.at(den, k, w[o.items] / zc)
        b_new = (prior + num) / (prior + den)
        beta_new = np.log(b_new) - np.log(b_new[0])
        done = np.max(np.abs(beta_new - beta)) < tol
        beta = beta_new
        if done:
            break
    return beta


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall tau-a between two score vectors over the same items."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = len(a)
    if n < 2:
        return 1.0
    da = np.sign(a[:, None] - a[None, :])
    db = np.sign(b[:, None] - b[None, :])
    return float((da * db).sum() / (n * (n - 1)))
