"""Item knowledge graph: typed attribute edges plus collaborative co-consumption edges.

Both kinds of edge are treated uniformly as evidence that links a candidate to
the user's history, so the decision model arbitrates between content,
knowledge and collaborative signals expressed in one vocabulary.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Optional

from ..data.schema import Dataset
from ..models.retrievers import windowed_cooc


class ItemKG:
    def __init__(self, ds: Dataset, cooc_window: int = 5, cooc_top: int = 20, max_df: float = 0.5):
        self.ds = ds
        self.attrs: list[frozenset] = [frozenset((r, e) for r, e in it.get("attrs", [])) for it in ds.items]
        self.ent_items: dict[tuple, list[int]] = defaultdict(list)
        for i, a in enumerate(self.attrs):
            for key in a:
                self.ent_items[key].append(i)
        n = max(ds.n_items, 1)
        # Attributes shared by a large share of the catalogue carry no preference signal.
        self.idf = {k: math.log(n / len(v)) for k, v in self.ent_items.items() if len(v) / n <= max_df}
        self.cooc: dict[int, dict[int, int]] = windowed_cooc(ds.seqs, window=cooc_window, holdout=2, top=cooc_top)

    def similarity(self, i: int, j: int) -> float:
        """IDF-weighted Jaccard similarity of two items' attribute sets."""
        a, b = self.attrs[i], self.attrs[j]
        inter = sum(self.idf.get(k, 0.0) for k in a & b)
        union = sum(self.idf.get(k, 0.0) for k in a | b)
        return inter / union if union > 0 else 0.0

    def facts(self, item: int, history: list[int], kinds: str = "kg+cf", max_facts: int = 3,
              mask_titles: bool = False) -> list[tuple[str, float]]:
        """Evidence linking `item` to the history, strongest first, as (text, weight)."""
        out: list[tuple[str, float]] = []
        if "kg" in kinds:
            share: Counter = Counter()
            for h in history:
                for key in self.attrs[item] & self.attrs[h]:
                    if key in self.idf:
                        share[key] += 1
            for (rel, ent), cnt in share.items():
                noun = "item" if cnt == 1 else "items"
                out.append((f"same {rel} '{ent}' as {cnt} history {noun}", self.idf[(rel, ent)] * math.sqrt(cnt)))
        if "cf" in kinds:
            for h in history:
                c = self.cooc.get(h, {}).get(item)
                if c:
                    who = "a history item" if mask_titles else f"'{self.ds.items[h].get('title', '')[:60]}'"
                    out.append((f"often chosen by users who chose {who}", 1.0 + math.log1p(c)))
        out.sort(key=lambda t: -t[1])
        return out[:max_facts]

    def profile(self, history: list[int], top: int = 4, min_count: int = 2) -> Optional[str]:
        """Recurring attributes of the history, the graph's summary of the user's stable preferences."""
        cnt: Counter = Counter()
        for h in history:
            for key in self.attrs[h]:
                if key in self.idf:
                    cnt[key] += 1
        best = sorted(((c * self.idf[k], k, c) for k, c in cnt.items() if c >= min_count), reverse=True)[:top]
        if not best:
            return None
        return "; ".join(f"{rel} '{ent}' ({c})" for _, (rel, ent), c in best)

    def neighbours(self, seeds: list[int], exclude: set, top: int) -> list[int]:
        """Expansion candidates: items linked to the seeds by co-consumption or rare shared attributes."""
        score: Counter = Counter()
        for s in seeds:
            for j, c in self.cooc.get(s, {}).items():
                if j not in exclude:
                    score[j] += 1.0 + math.log1p(c)
            for key in self.attrs[s]:
                w = self.idf.get(key)
                if w and len(self.ent_items[key]) <= 200:
                    for j in self.ent_items[key]:
                        if j not in exclude and j != s:
                            score[j] += 0.2 * w
        return [j for j, _ in score.most_common(top)]

    # ---- user modelling support ------------------------------------------

    def hypotheses(self, history: list[int], max_h: int = 24) -> list[tuple[str, str, int]]:
        """Preference hypotheses for a history: informative attributes of consumed items,
        ordered by support times rarity. Returns (relation, entity, support)."""
        cnt: Counter = Counter()
        for h in history:
            for key in self.attrs[h]:
                if key in self.idf:
                    cnt[key] += 1
        return rank_hypotheses([(rel, ent, c, self.idf[(rel, ent)]) for (rel, ent), c in cnt.items()], max_h)

    def nests(self, items: list[int], relation: str, max_n: int = 10) -> list[str]:
        """Most frequent informative values of `relation` among `items`: the candidate intents."""
        cnt: Counter = Counter()
        for i in items:
            for rel, ent in self.attrs[i]:
                if rel == relation and (rel, ent) in self.idf:
                    cnt[ent] += 1
        return [e for e, _ in sorted(cnt.items(), key=lambda t: (-t[1], t[0]))[:max_n]]

    def retrieve_by_preferences(self, strength: dict[tuple, float], exclude: set, top: int,
                                min_p: float = 0.6, max_fanout: int = 2000) -> list[int]:
        """Catalogue items reached by user -> preferred entity -> item paths, best graph score first."""
        score: Counter = Counter()
        for key, p in strength.items():
            if p >= min_p and key in self.idf and len(self.ent_items[key]) <= max_fanout:
                for j in self.ent_items[key]:
                    if j not in exclude:
                        score[j] += self.idf[key] * (2.0 * p - 1.0)
        return [j for j, _ in sorted(score.items(), key=lambda t: (-t[1], t[0]))[:top]]


def rank_hypotheses(rows: list[tuple[str, str, int, float]], max_h: int) -> list[tuple[str, str, int]]:
    """Deterministic ordering shared by the batch path and the agent: support * idf, then name."""
    rows = sorted(rows, key=lambda t: (-t[2] * t[3], t[0], t[1]))
    return [(rel, ent, int(c)) for rel, ent, c, _ in rows[:max_h]]
