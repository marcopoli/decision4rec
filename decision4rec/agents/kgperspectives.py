"""Perspective agents grounded in the knowledge graph (version 4): no text is generated.

Each perspective is a typed rule over the behavior of the scenario (co-purchase, transitions, what sells lately) and
the attributes of the items (kind and maker), so it carries behavior that the generative versions (1 and 2) lacked:

    complement     bought with the recent items, but of another kind (an accessory for a console, strings for a guitar)
    continuation   taken after the recent items and of the same maker or series (sequels, next models, add-ons);
                   on MovieLens also the films that follow a recent one in Wikidata
    loyalty        the same maker as the recent items (brand, developer, director), what sells lately first
    replenishment  the same kind and maker as an older item of the user (a new one of what was used before)
    exploration    what sells lately in kinds next to the user's kinds (by co-purchase between kinds), not yet tried

The kind of an item is its deepest catalogue category (or the category at `kind_depth`), or its first genre; the maker is any of brand, manufacturer,
store, developer, publisher, director or studio. Every list is ranked and excludes the history.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Optional

import numpy as np

KIND = ("category", "genre", "type of item")
MAKER = ("brand", "manufacturer", "store", "developer", "publisher", "director", "studio")
NAMES = ("complement", "continuation", "loyalty", "replenishment", "exploration")


def _norm(t: str) -> str:
    return re.sub(r"\s*\(\d{4}\)\s*$", "", t or "").strip().lower()


class KGPerspectives:
    def __init__(self, ds, store, recent: int = 3, makers_max: int = 400, kind_depth: Optional[int] = None):
        self.ds, self.store, self.recent = ds, store, recent
        n = ds.n_items
        self.kind: list[Optional[str]] = [None] * n
        self.makers: list[frozenset] = [frozenset()] * n
        self.series: list[frozenset] = [frozenset()] * n
        title = {}
        for i, it in enumerate(ds.items):
            cats = [v for r, v in it.get("attrs", []) if r == "category"]
            if kind_depth and cats:                  # a coarser kind: the category at this depth of the path
                cats = cats[: kind_depth][-1:]
            kinds = cats[-1:] or [v for r, v in it.get("attrs", []) if r in KIND][:1]
            self.kind[i] = kinds[0] if kinds else None
            self.makers[i] = frozenset(f"{r}:{v}".lower() for r, v in it.get("attrs", []) if r in MAKER and v)
            self.series[i] = frozenset(v.lower() for r, v in it.get("attrs", []) if r == "series")
            title.setdefault(_norm(it.get("title", "")), i)
        # sequels (MovieLens enriched): "followed by" points to the next film by title
        self.next_of: dict[int, list[int]] = defaultdict(list)
        for i, it in enumerate(ds.items):
            for r, v in it.get("attrs", []):
                j = title.get(_norm(v))
                if j is None or j == i:
                    continue
                if r == "followed by":
                    self.next_of[i].append(j)
                elif r == "follows":
                    self.next_of[j].append(i)
        pop = store.pop
        self.by_maker: dict[str, list[int]] = defaultdict(list)
        self.by_kind_maker: dict[tuple, list[int]] = defaultdict(list)
        for i in np.argsort(-pop, kind="stable"):
            i = int(i)
            for m in self.makers[i]:
                if len(self.by_maker[m]) < makers_max:
                    self.by_maker[m].append(i)
                if self.kind[i] is not None and len(self.by_kind_maker[(self.kind[i], m)]) < makers_max:
                    self.by_kind_maker[(self.kind[i], m)].append(i)
        # kinds that are bought together (co-purchase between kinds)
        kk: dict[str, Counter] = defaultdict(Counter)
        for a, nb in store.cooc.items():
            ka = self.kind[a]
            if ka is None:
                continue
            for b, c in nb.items():
                kb = self.kind[b]
                if kb is not None and kb != ka:
                    kk[ka][kb] += c
        self.near_kinds = {k: {x for x, _ in c.most_common(5)} for k, c in kk.items()}

    def lists(self, history: list[int], now: Optional[float], k: int = 20, g: Optional[dict] = None) -> dict[str, list[int]]:
        seen = set(history)
        recent = history[-self.recent:]
        g = g or self.store.generators(history, now)
        rkinds = {self.kind[h] for h in recent if self.kind[h] is not None}
        rmakers = Counter(m for h in recent for m in self.makers[h])
        rseries = set().union(*(self.series[h] for h in recent)) if recent else set()
        ukinds = {self.kind[h] for h in history if self.kind[h] is not None}

        def take(seq, cond=lambda j: True):
            out = []
            for j in seq:
                j = int(j)
                if j not in seen and j not in out and cond(j):
                    out.append(j)
                    if len(out) >= k:
                        break
            return out

        lately = g.get("lately") or []
        lately_rank = {j: r for r, j in enumerate(lately)}
        out = {}
        out["complement"] = take(g.get("with", []), lambda j: self.kind[j] is not None and self.kind[j] not in rkinds)
        seq = [j for h in reversed(recent) for j in self.next_of.get(h, [])]
        out["continuation"] = take(seq + list(g.get("after", [])),
                                   lambda j: bool(self.makers[j] & set(rmakers)) or bool(self.series[j] & rseries)
                                   or j in seq)
        cand = []
        for m, _ in rmakers.most_common(4):
            cand += self.by_maker.get(m, [])[:100]
        cand = sorted(set(cand), key=lambda j: (lately_rank.get(j, 10 ** 6), -int(self.store.pop[j])))
        out["loyalty"] = take(cand)
        older = history[:-self.recent]
        pairs = Counter((self.kind[h], m) for h in older if self.kind[h] is not None for m in self.makers[h])
        cand = []
        for p, _ in pairs.most_common(5):
            cand += self.by_kind_maker.get(p, [])[:50]
        out["replenishment"] = take(sorted(set(cand), key=lambda j: -int(self.store.pop[j])))
        near = set().union(*(self.near_kinds.get(x, set()) for x in ukinds)) - ukinds if ukinds else set()
        out["exploration"] = take(lately, lambda j: self.kind[j] in near)
        return out

    def explain(self, prefix: list[int], x: int) -> tuple[bool, ...]:
        """Which perspectives (in the order of NAMES) explain the step from `prefix` to item `x`, by the same rules
        as the lists, read from the statistics of the store."""
        store, t = self.store, len(prefix)
        prev, k = prefix[-3:], self.kind[x]
        pk = {self.kind[p] for p in prev}
        pm = set().union(*(self.makers[p] for p in prev))
        ps = set().union(*(self.series[p] for p in prev))
        older = prefix[max(0, t - 50):max(0, t - 3)]
        last = prefix[-1] if prefix else None
        return (bool(any(x in store.cooc.get(p, {}) for p in prev) and k is not None and k not in pk),
                bool((x in store.rules.get(last, {}) or x in self.next_of.get(last, []))
                     and (self.makers[x] & pm or self.series[x] & ps)),
                bool(self.makers[x] & pm),
                any(k is not None and self.kind[o] == k and bool(self.makers[x] & self.makers[o]) for o in older),
                bool(k is not None and k not in {self.kind[o] for o in prefix}))

    def rates(self, history: list[int], back: int = 50) -> np.ndarray:
        """Version 5a: share of the user's own past steps that each perspective explains (Laplace-smoothed)."""
        h = history
        cnt, n = np.zeros(len(NAMES)), 0
        for t in range(max(3, len(h) - back), len(h)):
            cnt += np.array(self.explain(h[:t], h[t]), dtype=float)
            n += 1
        return (cnt + 0.2) / (n + 1.0)


# Version 5: the decision model is asked, in the same request as the next choice, which point of view explains the
# user's next step. Labels come from the rules above applied to the real next item (no model output is a label).
WHY_ORDER = ("continuation", "complement", "replenishment", "loyalty", "exploration")   # most specific rule first
WHY_TEXT = {"complement": "something that goes with a recent item but is of another kind",
            "continuation": "the next one after a recent item, of the same maker or series",
            "loyalty": "another item of the same brand, maker or creator as the recent items",
            "replenishment": "a new one of the same kind and maker as an older item",
            "exploration": "a popular item of a nearby kind the user has not tried",
            "other": "none of these"}
WHY_INSTRUCTIONS = "Based on the user's interaction history, which kind of choice will the user make next?"


def why_label(hits) -> str:
    """One label for a step: the most specific perspective that explains it, or 'other'."""
    on = {n for n, h in zip(NAMES, hits) if h}
    return next((n for n in WHY_ORDER if n in on), "other")


def why_question() -> dict:
    """The wire form of the version 5 question (Kev choice), options in a fixed order."""
    return {"type": "choice", "instructions": WHY_INSTRUCTIONS,
            "criteria": {n: WHY_TEXT[n] for n in NAMES + ("other",)}}

