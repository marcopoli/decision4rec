"""Retrieval agents: several points of view on what the user may choose next.

Each agent returns its own ranked candidates. Small language models do the
retrieval: an encoder embeds three textual views of every item (content,
knowledge-graph attributes, and a verbalised collaborative neighbourhood), and
an optional generator writes a hypothetical next item from each point of view.
The classical sequential model is one more agent, not a special case.

    sequential     SASRec or item kNN on the interaction sequence (collaborative, non-linguistic)
    content        items whose description resembles the recent history
    knowledge      items whose graph attributes resemble those of the recent history
    collaborative  items whose co-consumption neighbourhood, verbalised and embedded, resembles the history's
    profile        items matching the decision-estimated preference profile and intent (needs the user model)
    graph          items reached by preference paths, or by graph neighbours of the history
    popularity     fallback

Proposals are merged by weighted reciprocal rank fusion. Weights are uniform,
tuned on validation users, or set per user by a decision (see the agent runtime).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..data.schema import Dataset
from ..graph.kg import ItemKG
from ..graph.profile import OTHER_KEY, Profile
from ..models.retrievers import PopRetriever, Retriever
from .slm import Embedder, Writer, build_embedder, build_writer

VIEW_POINTS = {
    "content": "the content and style of the items",
    "knowledge": "attributes such as category, brand, genre and creator",
    "collaborative": "what users with similar histories choose next",
}


@dataclass
class RetrievalContext:
    history: list[int]
    prof: Optional[Profile] = None
    intent: Optional[dict] = None


def rrf(lists: list[list[int]], weights: Optional[list[float]] = None, c: float = 60.0) -> list[tuple[int, float]]:
    """Weighted reciprocal rank fusion, best first. Ties are broken by item id for determinism."""
    score: dict[int, float] = {}
    for n, items in enumerate(lists):
        w = 1.0 if weights is None else weights[n]
        if w <= 0:
            continue
        for rank, item in enumerate(items):
            score[item] = score.get(item, 0.0) + w / (c + rank + 1.0)
    return sorted(score.items(), key=lambda t: (-t[1], t[0]))


class ItemViews:
    """Three textual views of every item and their embedding matrices."""

    VIEWS = ("content", "knowledge", "collaborative")

    def __init__(self, ds: Dataset, kg: ItemKG, embedder: Embedder, cache_dir: Optional[str] = None,
                 n_neighbours: int = 8, doc_chars: int = 300):
        self.ds, self.kg, self.embedder = ds, kg, embedder
        self.cache_dir, self.n_neighbours, self.doc_chars = cache_dir, n_neighbours, doc_chars
        self._mats: dict[str, np.ndarray] = {}

    def _title(self, i: int) -> str:
        return (self.ds.items[i].get("title") or f"item {i}")[:60]

    def doc(self, view: str, i: int) -> str:
        if view == "content":
            return self.ds.item_text(i)[: self.doc_chars]
        if view == "knowledge":
            attrs = sorted(k for k in self.kg.attrs[i] if k in self.kg.idf) or sorted(self.kg.attrs[i])
            return "; ".join(f"{rel}: {ent}" for rel, ent in attrs)[: self.doc_chars]
        if view == "collaborative":
            near = sorted(self.kg.cooc.get(i, {}).items(), key=lambda t: (-t[1], t[0]))[: self.n_neighbours]
            return ("Chosen together with: " + "; ".join(self._title(j) for j, _ in near)) if near else ""
        raise ValueError(view)

    def matrix(self, view: str) -> np.ndarray:
        if view in self._mats:
            return self._mats[view]
        path = None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            path = os.path.join(self.cache_dir, f"emb_{self.embedder.tag}_{view}_{self.ds.n_items}.npy")
        if path and os.path.exists(path):
            mat = np.load(path)
        else:
            docs = [self.doc(view, i) for i in range(self.ds.n_items)]
            filled = [i for i, d in enumerate(docs) if d]
            vec = self.embedder.encode([docs[i] for i in filled])
            mat = np.zeros((self.ds.n_items, vec.shape[1]), dtype=np.float32)
            mat[filled] = vec
            self.embedder._memo.clear()          # item documents are not needed as memoised queries
            if path:
                np.save(path, mat)
        self._mats[view] = mat
        return mat

    def search(self, view: str, query: np.ndarray, k: int, exclude: list[int]) -> list[int]:
        scores = self.matrix(view) @ query
        if exclude:
            scores[np.asarray(exclude, dtype=int)] = -np.inf
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        return [int(i) for i in top[np.argsort(-scores[top], kind="stable")] if np.isfinite(scores[i])]


class RetrievalAgent:
    name = "agent"
    needs_profile = False

    def prepare(self, ctxs: list[RetrievalContext], k: int) -> None:
        """Optional batching hook: warm caches for many users at once."""

    def propose(self, ctx: RetrievalContext, k: int) -> list[int]:
        raise NotImplementedError


class SequentialAgent(RetrievalAgent):
    name = "sequential"

    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self._memo: dict[tuple, list[int]] = {}

    def prepare(self, ctxs: list[RetrievalContext], k: int) -> None:
        todo = [c.history for c in ctxs if (tuple(c.history), k) not in self._memo]
        if todo:
            idx, _ = self.retriever.topk(todo, k)
            for h, row in zip(todo, idx):
                self._memo[(tuple(h), k)] = [int(i) for i in row]

    def propose(self, ctx: RetrievalContext, k: int) -> list[int]:
        self.prepare([ctx], k)
        return self._memo[(tuple(ctx.history), k)]


class DenseAgent(RetrievalAgent):
    """One point of view. Queries are the recent items seen through that view, plus, with a writer,
    a hypothetical next item described by a small language model from the same point of view."""

    def __init__(self, view: str, views: ItemViews, writer: Optional[Writer] = None, recent: int = 5,
                 decay: float = 0.7):
        self.name, self.view, self.views, self.writer = view, view, views, writer
        self.recent, self.decay = recent, decay

    def prompt(self, ctx: RetrievalContext) -> str:
        lines = ["Recent items of a user, oldest to newest:"]
        for h in ctx.history[-self.recent:]:
            line = f"- {self.views.doc('content', h)[:120]}"
            if self.view == "knowledge":
                line = f"- {self.views.doc('knowledge', h)[:160]}"
            elif self.view == "collaborative":
                near = self.views.doc("collaborative", h)
                line += f" ({near[:160]})" if near else ""
            lines.append(line)
        lines.append(f"Considering only {VIEW_POINTS[self.view]}, describe in one short sentence the item "
                     "this user is most likely to choose next. Reply with the description only.")
        return "\n".join(lines)

    def prepare(self, ctxs: list[RetrievalContext], k: int) -> None:
        if self.writer is not None:
            texts = self.writer.write([self.prompt(c) for c in ctxs if c.history])
            self.views.embedder.encode(texts, is_query=True)

    def propose(self, ctx: RetrievalContext, k: int) -> list[int]:
        mat = self.views.matrix(self.view)
        lists, weights = [], []
        for rank, h in enumerate(reversed(ctx.history[-self.recent:])):
            if np.any(mat[h]):
                lists.append(self.views.search(self.view, mat[h], k, ctx.history))
                weights.append(self.decay ** rank)
        if self.writer is not None and ctx.history:
            text = self.writer.write([self.prompt(ctx)])[0]
            if text:
                query = self.views.embedder.encode([text], is_query=True)[0]
                lists.append(self.views.search("content", query, k, ctx.history))
                weights.append(1.0)
        return [i for i, _ in rrf(lists, weights)][:k]


class ProfileAgent(RetrievalAgent):
    """Retrieves with the decision-estimated user model as the query."""

    name = "profile"
    needs_profile = True

    def __init__(self, views: ItemViews):
        self.views = views

    def propose(self, ctx: RetrievalContext, k: int) -> list[int]:
        if ctx.prof is None:
            return []
        top = ctx.prof.top(8)
        if not top:
            return []
        text = "; ".join(f"{q.rel}: {q.ent}" for q in top)
        if ctx.intent:
            best = sorted(((p, n) for n, p in ctx.intent.items() if n != OTHER_KEY), reverse=True)[:2]
            text += "; likely next: " + ", ".join(n for _, n in best)
        query = self.views.embedder.encode([text], is_query=True)[0]
        lists = [self.views.search("knowledge", query, k, ctx.history),
                 self.views.search("content", query, k, ctx.history)]
        return [i for i, _ in rrf(lists)][:k]


class GraphAgent(RetrievalAgent):
    """Preference paths when the user model is available, graph neighbours of the history otherwise."""

    name = "graph"

    def __init__(self, kg: ItemKG):
        self.kg = kg

    def propose(self, ctx: RetrievalContext, k: int) -> list[int]:
        exclude = set(ctx.history)
        if ctx.prof is not None and ctx.prof.prefs:
            return self.kg.retrieve_by_preferences(ctx.prof.strength(), exclude, k)
        return self.kg.neighbours(ctx.history[-5:], exclude, k)


class PopularityAgent(RetrievalAgent):
    name = "popularity"

    def __init__(self, ds: Dataset):
        pop = PopRetriever(ds.seqs, ds.n_items).pop
        self.order = [int(i) for i in np.argsort(-pop, kind="stable")]

    def propose(self, ctx: RetrievalContext, k: int) -> list[int]:
        seen = set(ctx.history)
        out = []
        for i in self.order:
            if i not in seen:
                out.append(i)
                if len(out) == k:
                    break
        return out


class MultiAgentRetriever(Retriever):
    """Fusion of retrieval agents. Usable anywhere a retriever is, so cases can be built from it."""

    name = "multi"

    def __init__(self, agents: list[RetrievalAgent], weights: Optional[dict[str, float]] = None,
                 depth: int = 200, n_items: int = 0):
        self.agents = agents
        self.weights = weights or {a.name: 1.0 for a in agents}
        self.depth, self.n_items = depth, n_items

    def propose_all(self, ctx: RetrievalContext, k: int, with_profile_agents: bool = True) -> dict[str, list[int]]:
        return {a.name: a.propose(ctx, k) for a in self.agents
                if with_profile_agents or not a.needs_profile}

    def fuse(self, lists: dict[str, list[int]], weights: Optional[dict[str, float]] = None,
             k: Optional[int] = None) -> tuple[list[int], list[float]]:
        weights = weights or self.weights
        names = [n for n in lists if lists[n]]
        ranked = rrf([lists[n] for n in names], [float(weights.get(n, 1.0)) for n in names])
        ranked = ranked[:k] if k else ranked
        return [i for i, _ in ranked], [s for _, s in ranked]

    def topk(self, histories: list[list[int]], k: int) -> tuple[np.ndarray, np.ndarray]:
        depth = max(k, self.depth)
        ctxs = [RetrievalContext(history=h) for h in histories]
        for a in self.agents:
            if not a.needs_profile:
                a.prepare(ctxs, depth)
        idx = np.zeros((len(histories), k), dtype=int)
        val = np.full((len(histories), k), -1.0)
        for r, ctx in enumerate(ctxs):
            items, scores = self.fuse(self.propose_all(ctx, depth, with_profile_agents=False), k=k)
            if len(items) < k:                                   # pad so the interface always returns k items
                seen = set(items) | set(ctx.history)
                items += [i for i in range(self.n_items) if i not in seen][: k - len(items)]
                scores += [0.0] * (len(items) - len(scores))
            idx[r, : len(items)] = items
            val[r, : len(scores)] = scores
        return idx, val

    def tune(self, ds: Dataset, n_users: int = 500, seed: int = 13, grid=(0.0, 0.25, 0.5, 1.0, 2.0),
             cutoffs=(20, 50, 200)) -> dict:
        """Coordinate ascent on agent weights. The objective is the mean hit rate at the cutoffs on
        validation targets (the second-to-last item of sampled users), so test targets are never seen."""
        rng = np.random.default_rng(seed)
        users = [u for u in rng.permutation(ds.n_users).tolist() if len(ds.seqs[u]) >= 5][:n_users]
        ctxs = [RetrievalContext(history=[int(x[0]) for x in ds.seqs[u][:-2]]) for u in users]
        targets = [int(ds.seqs[u][-2][0]) for u in users]
        static = [a for a in self.agents if not a.needs_profile]
        depth = max(cutoffs)
        for a in static:
            a.prepare(ctxs, depth)
        lists = [{a.name: a.propose(c, depth) for a in static} for c in ctxs]

        def objective(w: dict) -> float:
            total = 0.0
            for per_user, target in zip(lists, targets):
                items, _ = self.fuse(per_user, w, k=depth)
                rank = items.index(target) if target in items else depth
                total += sum(rank < c for c in cutoffs) / len(cutoffs)
            return total / max(len(targets), 1)

        w = {a.name: 1.0 for a in static}
        best = objective(w)
        for _ in range(2):
            for name in list(w):
                for g in grid:
                    trial = {**w, name: g}
                    if sum(trial.values()) <= 0:
                        continue
                    val = objective(trial)
                    if val > best + 1e-9:
                        best, w = val, trial
        self.weights = {**{a.name: 1.0 for a in self.agents}, **w}
        return {"weights": self.weights, "objective": best, "uniform_objective": objective({a.name: 1.0 for a in static}),
                "n_users": len(users)}


def build_multi_retriever(cfg: dict, ds: Dataset, base: Optional[Retriever] = None,
                          kg: Optional[ItemKG] = None) -> MultiAgentRetriever:
    """Build the agents named in `cfg["agents"]`. `base` is the sequential model for the `sequential` agent."""
    kg = kg or ItemKG(ds)
    names = cfg.get("agents") or ["sequential", "content", "knowledge", "collaborative", "graph", "popularity"]
    dense = [n for n in names if n in ItemViews.VIEWS or n == "profile"]
    views = None
    writer = None
    if dense:
        views = ItemViews(ds, kg, build_embedder(cfg.get("embedder")), cfg.get("cache"))
        wcfg = dict(cfg.get("writer") or {})
        if wcfg.get("kind", "none") != "none" and cfg.get("cache") and "cache" not in wcfg:
            os.makedirs(cfg["cache"], exist_ok=True)
            wcfg["cache"] = os.path.join(cfg["cache"], "writer.sqlite")
        writer = build_writer(wcfg)
    agents: list[RetrievalAgent] = []
    for n in names:
        if n == "sequential":
            if base is None:
                raise ValueError("the sequential agent needs a base retriever (retriever.base in the config)")
            agents.append(SequentialAgent(base))
        elif n in ItemViews.VIEWS:
            agents.append(DenseAgent(n, views, writer))
        elif n == "profile":
            agents.append(ProfileAgent(views))
        elif n == "graph":
            agents.append(GraphAgent(kg))
        elif n == "popularity":
            agents.append(PopularityAgent(ds))
        else:
            raise ValueError(f"unknown retrieval agent {n!r}")
    weights = cfg.get("weights", "uniform")
    retriever = MultiAgentRetriever(agents, weights if isinstance(weights, dict) else None,
                                    depth=int(cfg.get("depth", 200)), n_items=ds.n_items)
    if weights == "tuned":
        retriever.tune_info = retriever.tune(ds, n_users=int(cfg.get("tune_users", 500)), seed=int(cfg.get("seed", 13)))
    return retriever
