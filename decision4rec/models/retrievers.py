"""Training-free retrievers (popularity, windowed item kNN) and the retriever interface."""

from __future__ import annotations

from typing import Optional

import numpy as np


def train_prefix(seq: list, split: str) -> list[int]:
    """Items visible when predicting the split target: all but the last two (valid) or the last one (test)."""
    cut = {"train": 2, "valid": 2, "test": 1}[split]
    return [int(x[0]) for x in seq[:-cut]] if len(seq) > cut else []


def windowed_cooc(seqs: list, window: int = 5, holdout: int = 2, top: Optional[int] = None) -> dict[int, dict[int, int]]:
    """Symmetric co-occurrence counts of items within `window` steps, on the training prefix only.
    Vectorised so that datasets with tens of millions of interactions stay tractable. With `top`,
    only the strongest neighbours of each item are kept."""
    lengths = [max(len(s) - holdout, 0) for s in seqs]
    total = int(sum(lengths))
    if total == 0:
        return {}
    items = np.empty(total, dtype=np.int64)
    owner = np.empty(total, dtype=np.int64)
    pos = 0
    for u, (s, n) in enumerate(zip(seqs, lengths)):
        if n:
            items[pos: pos + n] = [x[0] for x in s[:n]]
            owner[pos: pos + n] = u
            pos += n
    n_items = int(items.max()) + 1
    codes = []
    for off in range(1, window + 1):
        if off >= total:
            break
        a, b = items[:-off], items[off:]
        ok = (owner[:-off] == owner[off:]) & (a != b)
        codes.append(a[ok] * n_items + b[ok])
        codes.append(b[ok] * n_items + a[ok])
    if not codes or sum(len(c) for c in codes) == 0:
        return {}
    code, count = np.unique(np.concatenate(codes), return_counts=True)
    del codes
    src, dst = code // n_items, code % n_items
    order = np.lexsort((dst, -count, src))
    src, dst, count = src[order], dst[order], count[order]
    if top is not None:
        first = np.flatnonzero(np.r_[True, src[1:] != src[:-1]])
        rank = np.arange(len(src)) - np.repeat(first, np.diff(np.r_[first, len(src)]))
        keep = rank < top
        src, dst, count = src[keep], dst[keep], count[keep]
    out: dict[int, dict[int, int]] = {}
    for a, b, c in zip(src.tolist(), dst.tolist(), count.tolist()):
        out.setdefault(a, {})[b] = c
    return out


class Retriever:
    name = "retriever"

    def scores(self, history: list[int]) -> np.ndarray:
        raise NotImplementedError

    def topk(self, histories: list[list[int]], k: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-k item indices and scores per history, history items excluded."""
        idx, val = [], []
        for h in histories:
            s = self.scores(h).astype(float).copy()
            if h:
                s[np.asarray(h, dtype=int)] = -np.inf
            kk = min(k, len(s))
            top = np.argpartition(-s, kk - 1)[:kk]
            top = top[np.argsort(-s[top], kind="stable")]
            idx.append(top)
            val.append(s[top])
        return np.stack(idx), np.stack(val)


class PopRetriever(Retriever):
    name = "pop"

    def __init__(self, seqs: list, n_items: int, holdout: int = 2):
        c = np.zeros(n_items)
        for seq in seqs:
            for x in (seq[:-holdout] if holdout else seq):
                c[int(x[0])] += 1
        self.pop = np.log1p(c)

    def scores(self, history: list[int]) -> np.ndarray:
        return self.pop


class ItemKNNRetriever(Retriever):
    name = "itemknn"

    def __init__(self, seqs: list, n_items: int, window: int = 5, recent: int = 10, holdout: int = 2):
        self.n_items = n_items
        self.recent = recent
        self.cooc = windowed_cooc(seqs, window=window, holdout=holdout, top=100)
        self.deg = np.zeros(n_items)
        for i, c in self.cooc.items():
            self.deg[i] = sum(c.values())
        self.pop = PopRetriever(seqs, n_items, holdout).pop

    def scores(self, history: list[int]) -> np.ndarray:
        s = 1e-3 * self.pop.copy()
        for rank, h in enumerate(reversed(history[-self.recent:])):
            decay = 0.8 ** rank
            for j, c in self.cooc.get(h, {}).items():
                s[j] += decay * c / np.sqrt(self.deg[h] * self.deg[j] + 1e-9)
        return s


def sequential_rules(seqs: list, window: int = 10, holdout: int = 2, top: Optional[int] = 200) -> dict[int, dict[int, float]]:
    """Directed rules a -> b, weighted 1 / gap for every b that follows a within `window` steps of the same
    training prefix (the sequential rules of Ludewig and Jannach, 2018). With `top`, only the strongest
    successors of each item are kept."""
    lengths = [max(len(s) - holdout, 0) for s in seqs]
    total = int(sum(lengths))
    if total == 0:
        return {}
    items = np.empty(total, dtype=np.int64)
    owner = np.empty(total, dtype=np.int64)
    pos = 0
    for u, (s, n) in enumerate(zip(seqs, lengths)):
        if n:
            items[pos: pos + n] = [x[0] for x in s[:n]]
            owner[pos: pos + n] = u
            pos += n
    n_items = int(items.max()) + 1
    codes, weights = [], []
    for off in range(1, window + 1):
        if off >= total:
            break
        a, b = items[:-off], items[off:]
        ok = (owner[:-off] == owner[off:]) & (a != b)
        codes.append(a[ok] * n_items + b[ok])
        weights.append(np.full(int(ok.sum()), 1.0 / off))
    if not codes or sum(len(c) for c in codes) == 0:
        return {}
    code, inv = np.unique(np.concatenate(codes), return_inverse=True)
    w = np.bincount(inv, weights=np.concatenate(weights))
    src, dst = code // n_items, code % n_items
    order = np.lexsort((dst, -w, src))
    src, dst, w = src[order], dst[order], w[order]
    if top is not None:
        first = np.flatnonzero(np.r_[True, src[1:] != src[:-1]])
        rank = np.arange(len(src)) - np.repeat(first, np.diff(np.r_[first, len(src)]))
        keep = rank < top
        src, dst, w = src[keep], dst[keep], w[keep]
    out: dict[int, dict[int, float]] = {}
    for a, b, c in zip(src.tolist(), dst.tolist(), w.tolist()):
        out.setdefault(a, {})[b] = c
    return out


def _unit(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float32)
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


class SeqRulesRetriever(Retriever):
    """Untrained, recency-weighted pairwise probe: successors of the last items under sequential rules."""
    name = "seqrules"

    def __init__(self, seqs: list, n_items: int, recent: int = 3, decay: float = 0.5, window: int = 10, holdout: int = 2):
        self.rules = sequential_rules(seqs, window=window, holdout=holdout)
        self.recent, self.decay = recent, decay
        pop = PopRetriever(seqs, n_items, holdout).pop
        self.tie = 1e-6 * pop / max(pop.max(), 1e-9)          # breaks ties only

    def scores(self, history: list[int]) -> np.ndarray:
        s = self.tie.copy()
        for rank, h in enumerate(reversed(history[-self.recent:])):
            for j, w in self.rules.get(h, {}).items():
                s[j] += self.decay ** rank * w
        return s


class ContentKNNRetriever(Retriever):
    """Items whose text is closest to the last items of the history; reaches items nobody bought."""
    name = "contentknn"

    def __init__(self, matrix: np.ndarray, pop: np.ndarray, recent: int = 3, decay: float = 0.5):
        self.unit = _unit(matrix)
        self.recent, self.decay = recent, decay
        self.tie = 1e-6 * pop / max(pop.max(), 1e-9)

    def scores(self, history: list[int]) -> np.ndarray:
        q = np.zeros(self.unit.shape[1], dtype=np.float32)
        for rank, h in enumerate(reversed(history[-self.recent:])):
            q += self.decay ** rank * self.unit[h]
        return self.unit @ q + self.tie


class TransTextRetriever(Retriever):
    """Candidates within two transition hops of the last two items, ranked by text similarity to the last item:
    our implementation of the training-free shortcut heuristic described by Han et al. (2026, arXiv 2605.07125),
    whose code is not public. Items outside the two hops rank below every candidate."""
    name = "transtext"

    def __init__(self, seqs: list, n_items: int, matrix: np.ndarray, first: int = 50, second: int = 10,
                 window: int = 10, holdout: int = 2):
        self.rules = sequential_rules(seqs, window=window, holdout=holdout)
        self.unit = _unit(matrix)
        self.first, self.second = first, second
        pop = PopRetriever(seqs, n_items, holdout).pop
        self.tie = 1e-6 * pop / max(pop.max(), 1e-9)

    def _succ(self, i: int, k: int) -> list[int]:
        return list(self.rules.get(i, {}))[:k]                 # already sorted by weight

    def scores(self, history: list[int]) -> np.ndarray:
        s = self.tie - 10.0
        if not history:
            return s
        hop1 = {j for a in history[-2:] for j in self._succ(a, self.first)}
        hop2 = {j for a in hop1 for j in self._succ(a, self.second)}
        cand = np.fromiter(hop1 | hop2, dtype=np.int64)
        if len(cand):
            s[cand] = self.unit[cand] @ self.unit[history[-1]] + self.tie[cand]
        return s
