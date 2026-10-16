"""Training-free behavioural evidence of one store, read by the evidence-reading decision model (docs/08).

A store sees the training part of the sequences of a share of its users: 1.0 is a mature platform, 0.0 a new
platform with a catalogue only. From those logs it keeps purchases per item with their times, directed
transitions between items (sequential rules) and co-occurrence within a window. Each number becomes a short
statement on a scale relative to the store (percentiles and ranks), so that the same words mean the same thing in
any domain, which is what lets one decision model, trained on some stores, read the evidence of another.

The store also generates candidates from the same evidence (successors of the recent items, items co-bought with
the history, items popular at that time, items whose text is close to the recent ones), so recall and ranking use
one source and a new store needs no trained model.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..data.schema import Dataset
from ..models.retrievers import _unit, sequential_rules, windowed_cooc

FAMILIES = ("sold", "lately", "after", "with", "text")
LEGEND = ("Candidate evidence, relative to this store: sold = all purchases; lately = last 30 days; after last = "
          "taken right after your last item; with yours = your items it is bought with; text = closeness to your "
          "last item.")
# Control of gap G2 (raw counts instead of positions on scales of the store): the same statements as numbers.
LEGEND_RAW = ("Candidate evidence, counts in this store: sold = all purchases; lately = purchases in the last 30 days; "
              "after last = weighted count of times taken right after your last item; with yours = your items it is "
              "bought with; text = closeness to your last item.")
QUOTAS = (("after", 10), ("with", 8), ("text", 8), ("lately", 4))      # shares of a candidate list of 30


def _levels(values: np.ndarray) -> Optional[np.ndarray]:
    v = np.asarray(values, dtype=float)
    v = v[v > 0]
    return np.quantile(v, [0.5, 0.9, 0.99]) if len(v) else None


def _bucket(x: float, q: Optional[np.ndarray], words: tuple[str, str, str, str]) -> str:
    """words for: at least the 99th, 90th, 50th percentile of positive values, and below."""
    if q is None:
        return words[3]
    return words[0] if x >= q[2] else words[1] if x >= q[1] else words[2] if x >= q[0] else words[3]


class Context:
    """What the statements of one case need, computed once per case."""

    def __init__(self, history: list[int], now: Optional[float], after: dict, after_last: dict, with_n: dict,
                 text_rank: Optional[np.ndarray]):
        self.history, self.now, self.after, self.after_last = history, now, after, after_last
        self.with_n, self.text_rank = with_n, text_rank


class EvidenceStore:
    def __init__(self, ds: Dataset, frac: float = 1.0, holdout: int = 2, seed: int = 0,
                 text: Optional[np.ndarray] = None, days: float = 30.0, recent: int = 3, decay: float = 0.5,
                 fusion: Optional[tuple] = None, raw: bool = False):
        self.ds, self.frac, self.holdout, self.recent, self.decay = ds, float(frac), holdout, recent, decay
        self.raw = raw                                # statements as raw counts, not levels of the store (G2 control)
        self.legend = LEGEND_RAW if raw else LEGEND
        self.fusion = tuple(float(w) for w in fusion) if fusion else None    # weights: after, with, text, lately
        n_items = ds.n_items
        if frac >= 1.0:
            keep = np.ones(ds.n_users, dtype=bool)
        elif frac <= 0.0:
            keep = np.zeros(ds.n_users, dtype=bool)
        else:
            keep = np.random.default_rng(seed).random(ds.n_users) < frac
        seqs = [s if keep[u] else [] for u, s in enumerate(ds.seqs)]
        parts = [s[:-holdout] if len(s) > holdout else [] for s in seqs]
        items = np.array([int(x[0]) for p in parts for x in p], dtype=np.int64)
        times = np.array([float(x[1]) for p in parts for x in p], dtype=np.float64)
        self.has_data = len(items) > 0
        order = np.lexsort((times, items))
        self.t_item = times[order]                                     # times grouped by item, ascending
        grid = np.arange(n_items)
        self.start = np.searchsorted(items[order], grid, side="left")
        self.end = np.searchsorted(items[order], grid, side="right")
        self.pop = (self.end - self.start).astype(np.int64)
        all_times = np.array([float(x[1]) for s in ds.seqs for x in s], dtype=np.float64)
        day = 86_400_000.0 if len(all_times) and float(np.median(all_times)) > 1e11 else 86_400.0   # ms or s
        self.window = days * day
        self.rules = sequential_rules(seqs, window=10, holdout=holdout) if self.has_data else {}
        self.cooc = windowed_cooc(seqs, window=5, holdout=holdout, top=100) if self.has_data else {}
        self.deg = np.zeros(n_items)
        for i, c in self.cooc.items():
            self.deg[i] = sum(c.values())
        self.q_sold = _levels(self.pop)
        self.q_after = _levels(np.array([w for d in self.rules.values() for w in d.values()]))
        # what was popular when: top items of the trailing window at regular snapshots
        g = np.argsort(times, kind="stable")
        self.g_items, self.g_times = items[g], times[g]
        self.snap, self.snap_top, lately = np.empty(0), [], []
        if self.has_data:
            step = self.window / 2
            self.snap = np.arange(self.g_times[0] + step, self.g_times[-1] + step, step)
            for b in self.snap:
                lo, hi = np.searchsorted(self.g_times, [b - self.window, b])
                cnt = np.bincount(self.g_items[lo:hi], minlength=n_items)
                top = np.argsort(-cnt, kind="stable")[:60]
                self.snap_top.append(top[cnt[top] > 0])
                lately.append(cnt[cnt > 0])
        self.q_lately = _levels(np.concatenate(lately)) if lately else None
        self.unit = _unit(text) if text is not None else None

    # ---- evidence of one case ------------------------------------------------
    def now(self, user: int, n_hist: int) -> Optional[float]:
        """Time of the last history item: the moment the recommendation is made."""
        s = self.ds.seqs[user]
        return float(s[n_hist - 1][1]) if 0 < n_hist <= len(s) else None

    def recent_count(self, item: int, now: Optional[float]) -> int:
        if now is None:
            return 0
        t = self.t_item[self.start[item]: self.end[item]]
        return int(np.searchsorted(t, now, side="left") - np.searchsorted(t, now - self.window, side="left"))

    def context(self, history: list[int], now: Optional[float]) -> Context:
        after: dict[int, float] = {}
        after_last: dict[int, float] = {}
        for r, h in enumerate(reversed(history[-self.recent:])):
            for j, w in self.rules.get(h, {}).items():
                after[j] = after.get(j, 0.0) + self.decay ** r * w
                if r == 0:
                    after_last[j] = w
        with_n: dict[int, int] = {}
        for h in set(history[-10:]):
            for j in self.cooc.get(h, {}):
                with_n[j] = with_n.get(j, 0) + 1
        rank = None
        if self.unit is not None and history:
            q = np.zeros(self.unit.shape[1], dtype=np.float32)
            for r, h in enumerate(reversed(history[-self.recent:])):
                q += self.decay ** r * self.unit[h]
            sims = self.unit @ q
            sims[np.asarray(history, dtype=int)] = -np.inf
            rank = np.empty(len(sims), dtype=np.int64)
            rank[np.argsort(-sims, kind="stable")] = np.arange(1, len(sims) + 1)
        return Context(history, now, after, after_last, with_n, rank)

    def tags(self, ctx: Context, item: int, drop: frozenset = frozenset()) -> str:
        """Compact statements, about 20 tokens: 30 options must fit one Kev training row."""
        parts = []
        if self.has_data:
            if "sold" not in drop:
                parts.append("sold " + (str(int(self.pop[item])) if self.raw else "none yet" if self.pop[item] == 0 else
                                        _bucket(self.pop[item], self.q_sold, ("top 1%", "top 10%", "top half", "rare"))))
            if "lately" not in drop and ctx.now is not None:
                n = self.recent_count(item, ctx.now)
                parts.append("lately " + (str(int(n)) if self.raw else "none" if n == 0 else
                                          _bucket(n, self.q_lately, ("very high", "high", "some", "low"))))
            if "after" not in drop:
                w = ctx.after.get(item, 0.0)
                where = "after last" if ctx.after_last.get(item, 0.0) > 0 else "after recent"
                parts.append(f"{where} " + (f"{w:.1f}" if self.raw else "never" if w == 0 else
                                           _bucket(w, self.q_after, ("very often", "often", "sometimes", "rarely"))))
            if "with" not in drop:
                n = ctx.with_n.get(item, 0)
                parts.append(f"with yours {n}" if self.raw else f"with yours {min(n, 5)}" + ("+" if n > 5 else ""))
        if "text" not in drop and ctx.text_rank is not None:
            r = int(ctx.text_rank[item])
            parts.append("text " + ("top 10" if r <= 10 else "top 50" if r <= 50 else "top 500" if r <= 500 else "far"))
        return "[" + "; ".join(parts) + "]" if parts else ""

    def regime_line(self) -> str:
        if not self.has_data:
            return "Store purchase data: none, this store has a catalogue only."
        if self.frac >= 1.0:
            return "Store purchase data: full."
        return f"Store purchase data: partial, from about {self.frac:.0%} of its customers."

    # ---- candidates ------------------------------------------------------------
    def generators(self, history: list[int], now: Optional[float], ctx: Optional[Context] = None) -> dict[str, list[int]]:
        """Ranked lists of each evidence generator (history items may appear; `candidates` removes them)."""
        ctx = ctx or self.context(history, now)
        lists: dict[str, list[int]] = {"after": [j for j, _ in sorted(ctx.after.items(), key=lambda t: (-t[1], t[0]))]}
        knn: dict[int, float] = {}
        for r, h in enumerate(reversed(history[-10:])):
            for j, c in self.cooc.get(h, {}).items():
                knn[j] = knn.get(j, 0.0) + 0.8 ** r * c / np.sqrt(self.deg[h] * self.deg[j] + 1e-9)
        lists["with"] = [j for j, _ in sorted(knn.items(), key=lambda t: (-t[1], t[0]))]
        lists["lately"] = []
        if len(self.snap) and now is not None:
            s = int(np.searchsorted(self.snap, now, side="right")) - 1
            lists["lately"] = [int(j) for j in self.snap_top[s]] if s >= 0 else []
        lists["text"] = [int(j) for j in np.argsort(ctx.text_rank, kind="stable")[: 500 + len(history)]] \
            if ctx.text_rank is not None else []     # a fixed length, so the order never depends on k
        return lists

    def fused_scores(self, history: list[int], now: Optional[float]) -> np.ndarray:
        """One score per item: each generator's score scaled to at most 1 in this case, weighted by `fusion`
        (after, with, text, lately). On development users of the source domains it finds the target more often
        than the quota mix (scripts/recall_probe.py, 2026-10-10)."""
        wa, ww, wt, wl = self.fusion
        ctx = self.context(history, now)
        s = 1e-9 * self.pop / max(int(self.pop.max()), 1)                    # ties: most sold first
        if ctx.after:
            idx, v = np.fromiter(ctx.after.keys(), int), np.fromiter(ctx.after.values(), float)
            s[idx] += wa * v / v.max()
        knn: dict[int, float] = {}
        for r, h in enumerate(reversed(history[-10:])):
            for j, c in self.cooc.get(h, {}).items():
                knn[j] = knn.get(j, 0.0) + 0.8 ** r * c / np.sqrt(self.deg[h] * self.deg[j] + 1e-9)
        if knn:
            idx, v = np.fromiter(knn.keys(), int), np.fromiter(knn.values(), float)
            s[idx] += ww * v / v.max()
        if ctx.text_rank is not None:                                        # 1 at rank 1, 0 from rank 500
            s += wt * np.maximum(0.0, 1.0 - np.log(ctx.text_rank) / np.log(500.0))
        if wl and now is not None and len(self.snap):
            i = int(np.searchsorted(self.snap, now, side="right")) - 1
            if i >= 0 and len(self.snap_top[i]):
                top = self.snap_top[i]
                s[top] += wl * (1.0 - np.arange(len(top)) / len(top))
        s[np.asarray(history, dtype=int)] = -np.inf
        return s

    def candidates(self, history: list[int], now: Optional[float], k: int,
                   quotas: Optional[tuple] = None) -> list[int]:
        """The evidence generators, history excluded: one fused score when the store has `fusion` weights, else
        interleaved by quota. The order does not depend on k, so the first 30 of a longer list are the list of 30
        that training records use."""
        if self.fusion and self.has_data and quotas is None:
            return [int(j) for j in np.argsort(-self.fused_scores(history, now), kind="stable")[:k]]
        lists = self.generators(history, now)
        quotas = (quotas or QUOTAS) if self.has_data else (("text", 1),)
        seen, out = set(history), []
        pos = {name: 0 for name in lists}
        credit = {name: 0.0 for name, _ in quotas}
        total = float(sum(q for _, q in quotas))
        fallback = [name for name, _ in quotas] + [n for n in ("after", "with", "text", "lately") if n not in credit]

        def take(name: str) -> bool:
            lst = lists[name]
            while pos[name] < len(lst):
                j = lst[pos[name]]
                pos[name] += 1
                if j not in seen:
                    seen.add(j)
                    out.append(j)
                    return True
            return False

        while len(out) < k:
            for name, q in quotas:                        # smooth weighted round robin over the generators
                credit[name] += q
            pick = max(credit, key=lambda n: credit[n])
            credit[pick] -= total
            if not take(pick) and not any(take(n) for n in fallback):
                break
        for j in np.argsort(-self.pop, kind="stable"):    # a store too small for k: most sold, then any
            if len(out) >= k:
                break
            if int(j) not in seen:
                seen.add(int(j))
                out.append(int(j))
        return out[:k]


class EvidenceRetriever:
    """Retriever over a store, for evaluation cases (the candidates depend on when the user is served)."""

    def __init__(self, store: EvidenceStore):
        self.store = store

    def topk_users(self, users: list[int], histories: list[list[int]], k: int):
        idx = np.array([self.store.candidates(h, self.store.now(u, len(h)), k) for u, h in zip(users, histories)])
        val = np.tile(np.arange(k, 0, -1, dtype=float), (len(histories), 1))   # the generators' order
        return idx, val

    def topk(self, histories: list[list[int]], k: int):
        raise SystemExit("the evidence retriever needs the users, so build its cases with `build_cases`")


_STORES: dict[tuple, EvidenceStore] = {}


def store_for(cfg: dict, ds: Dataset, holdout: int = 2, frac: Optional[float] = None) -> EvidenceStore:
    """One store per dataset, share of users and holdout, shared by the retriever and the prompt builder."""
    from ..experiments.benchmark import prepare_dataset
    ecfg = cfg.get("evidence") or {}
    frac = float(ecfg.get("frac", 1.0) if frac is None else frac)
    seed, days = int(ecfg.get("seed", 0)), float(ecfg.get("days", 30.0))
    fusion = tuple(float(w) for w in ecfg["fusion"]) if ecfg.get("fusion") else None
    raw = bool(ecfg.get("raw", False))
    key = (cfg["dataset"], frac, holdout, seed, days, fusion, raw)
    if key not in _STORES:
        prepare_dataset({**cfg, "item_text": cfg.get("item_text") or ecfg}, ds, ["contentknn"])
        _STORES[key] = EvidenceStore(ds, frac, holdout, seed, ds.extra.get("item_text_matrix"), days, fusion=fusion, raw=raw)
    return _STORES[key]
