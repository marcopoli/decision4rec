"""Perspective agents: a small instruction-tuned language model imagines the user's next item from one point of view,
and the item closest to that text is retrieved (generate, then retrieve).

Each perspective is one way in which a next purchase follows the past ones. The prompt shows a few real examples of
(recent items, next item) taken from the logs of the same store, from users who are not evaluated, then the history
of the user, and asks for one short description of the next item under that perspective. No model is trained.

    complement     an accessory, add-on or part for a recent item
    continuation   the next one in a series, a sequel or an upgrade
    replenishment  a consumable or a replacement of something used up
    loyalty        an item of the same brand, franchise, maker or creator
    exploration    a popular item of a nearby kind that the user has not tried

The perspective that found a candidate becomes evidence that the decision model reads ("found by: complement").
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..data.schema import Dataset
from .slm import Embedder, Writer

PERSPECTIVES = {
    "complement": "an accessory, add-on or part that goes with one of the recent items",
    "continuation": "the next one in a series, a sequel or an upgrade of a recent item",
    "replenishment": "a consumable or a replacement for something the user has used up",
    "loyalty": "another item of the same brand, franchise, maker or creator as the recent items",
    "exploration": "a popular item of a nearby kind that the user has not tried yet",
}


def _title(ds: Dataset, i: int, chars: int = 90) -> str:
    return (ds.items[i].get("title") or ds.item_text(i))[:chars]


def _line(ds: Dataset, i: int, chars: int = 140) -> str:
    """Title and catalogue details (category, brand), for the history lines of version 2."""
    return ds.item_text(i)[:chars]


def shots(ds: Dataset, users: list[int], n: int = 3, recent: int = 3) -> list[tuple[list[str], str]]:
    """Real examples from the training part of the logs: the last `recent` items before an item, and that item."""
    out = []
    for u in users:
        s = ds.seqs[u][:-2]                         # training part only
        if len(s) > recent:
            out.append(([_title(ds, int(x[0])) for x in s[-recent - 1:-1]], _title(ds, int(s[-1][0]))))
        if len(out) >= n:
            break
    return out


def prompt(ds: Dataset, perspective: str, history: list[int], examples: list[tuple[list[str], str]],
           recent: int = 5, n_ideas: int = 1) -> str:
    lines = [f"You recommend products in a store. Think about {PERSPECTIVES[perspective]}.",
             "Examples of what customers of this store bought next:"]
    for past, nxt in examples:
        lines.append("Recent: " + "; ".join(past))
        lines.append(f"Next: {nxt}")
    lines.append("Now this customer. Recent items, oldest to newest:")
    if n_ideas == 1:
        lines += [f"- {_title(ds, i)}" for i in history[-recent:]]
        lines.append(f"Write only the name of one product that is {PERSPECTIVES[perspective]}.")
    else:
        lines += [f"- {_line(ds, i)}" for i in history[-recent:]]
        lines.append(f"Write {n_ideas} different short product descriptions, one per line, each {PERSPECTIVES[perspective]}. "
                     "Name the kind of product and its main features. No numbering.")
    return "\n".join(lines)


class PerspectiveAgents:
    """One agent per perspective; all share the writer, the embedder and the item text matrix."""

    def __init__(self, ds: Dataset, writer: Writer, embedder: Embedder, item_matrix: np.ndarray,
                 example_users: list[int], names: Optional[list[str]] = None, n_ideas: int = 1):
        self.ds, self.writer, self.embedder = ds, writer, embedder
        m = np.asarray(item_matrix, dtype=np.float32)
        self.unit = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
        self.names = list(names or PERSPECTIVES)
        self.examples = shots(ds, example_users)
        self.n_ideas = n_ideas

    def propose(self, histories: list[list[int]], k: int) -> dict[str, list[list[int]]]:
        """For every perspective, the k items closest to the imagined next item of each history."""
        out = {}
        for name in self.names:
            texts = self.writer.write([prompt(self.ds, name, h, self.examples, n_ideas=self.n_ideas) for h in histories])
            lists = []
            for h, text in zip(histories, texts):
                ideas = [t.strip(" -*0123456789.") for t in text.split("\n") if t.strip()][: self.n_ideas] or [text]
                sims = self.embedder.encode(ideas, is_query=True) @ self.unit.T
                sims[:, np.asarray(h, dtype=int)] = -np.inf
                per = max(1, k // len(ideas))
                got = list(dict.fromkeys(int(j) for row in sims for j in np.argsort(-row)[:per]))
                lists.append(got[:k])
            out[name] = lists
        return out
