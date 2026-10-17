"""Multi-agent candidate pool with a residual fast ranker (docs/08, section 7; paper Section 4.6).

Agents propose candidates from different points of view. SASRecF, when the scenario has one, is one agent: its best
`seq_depth` items enter the pool whole, so the pool never loses a target SASRecF finds. The other agents add their best
`depth` items each: transitions, co-purchase, text, popular now and the fused evidence score (all read from the
knowledge graph of the scenario, `decision4rec.evidence.store`), items sharing rare catalogue attributes with the recent
items (knowledge-graph agent) and popularity.

A residual ranker scores every candidate as z(SASRecF score) + lam * f(features), with f started at zero, so lam = 0 is
SASRecF's own order. It is trained on users of no evaluation case (second-to-last item as target, statistics leaving
out the last two items of every user) and the users it was trained on are recorded, because a decision model must not
be trained on them: the ranker's top K would already know their answers (rule 3 of the paper).

For each candidate the pool also keeps which agents found it, written as one more statement ("found by ...").
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Optional

import numpy as np

from ..data.schema import Dataset

AGENTS = ("sasrecf", "evidence", "after", "with", "text", "lately", "kg", "pop")
NAMES = {"sasrecf": "sequence model", "evidence": "evidence", "after": "transitions", "with": "co-purchase",
         "text": "text", "lately": "popular now", "kg": "shared attributes", "pop": "best sellers",
         # perspective agents of the knowledge graph (version 4, decision4rec.agents.kgperspectives), when enabled
         "complement": "complement", "continuation": "continuation", "loyalty": "same maker",
         "replenishment": "replenishment", "exploration": "exploration"}
Z_COL = 2 * len(AGENTS)                     # without perspective agents; AgentPool.zcol in general


def persp_names() -> list[str]:
    from .kgperspectives import NAMES as P
    return list(P)


def kg_agent(kg, history: list[int], k: int) -> list[int]:
    score, seen = Counter(), set(history)
    for s in history[-3:]:
        for key in sorted(kg.attrs[s]):          # fixed order: sums and ties must not depend on the hash seed
            w = kg.idf.get(key)
            if w and len(kg.ent_items[key]) <= 200:
                for j in kg.ent_items[key]:
                    if j not in seen:
                        score[j] += w
    return [j for j, _ in score.most_common(k)]


class AgentPool:
    def __init__(self, ds: Dataset, store, scorer=None, kg=None, depth: int = 20, seq_depth: int = 50, persp=None):
        self.ds, self.store, self.scorer, self.kg = ds, store, scorer, kg
        self.persp = persp                           # KGPerspectives or None
        self.agents = AGENTS + (tuple(persp_names()) if persp is not None else ())
        self.zcol = 2 * len(self.agents)
        self.hidden = 0                              # 0: linear residual; > 0: one hidden layer
        self.persp_slots = 0                         # slots of the top k reserved for perspective candidates
        self.persp_profile = False                   # version 5a features
        self.sasrecf_cands = False                   # ablation: candidates = SASRecF's best items, as many as the pool
        self.depth, self.seq_depth = depth, seq_depth
        self.popular = [int(j) for j in np.argsort(-store.pop, kind="stable")[:300]]
        self.net = None
        self.lam = 0.0
        self.mu = self.sd = None
        self.train_users: list[int] = []
        self._found: dict = {}

    # ---- agents and features ----------------------------------------------------
    def _sasrec_rows(self, users: list[int], histories: list[list[int]]) -> Optional[np.ndarray]:
        if self.scorer is None:
            return None
        import torch
        with torch.no_grad():
            m = self.scorer.scores(users, histories).cpu().numpy().astype(np.float64)
        for r, h in enumerate(histories):
            m[r, np.asarray(h, dtype=int)] = -np.inf
        return m

    def lists(self, history: list[int], now, row: Optional[np.ndarray]) -> dict[str, list[int]]:
        seen = set(history)
        g = self.store.generators(history, now)
        out = {"sasrecf": [int(j) for j in np.argsort(-row)[: self.seq_depth]] if row is not None else [],
               "evidence": self.store.candidates(history, now, self.depth)}
        for a in ("after", "with", "text", "lately"):
            out[a] = [j for j in g[a] if j not in seen][: self.depth]
        out["kg"] = kg_agent(self.kg, history, self.depth) if self.kg is not None else []
        out["pop"] = [j for j in self.popular if j not in seen][: self.depth]
        if self.persp is not None:
            out.update(self.persp.lists(history, now, self.depth, g=g))
        return out

    def features(self, history: list[int], now, row: Optional[np.ndarray]):
        L = self.lists(history, now, row)
        pool = list(dict.fromkeys(j for a in self.agents for j in L[a]))
        if self.sasrecf_cands and row is not None:   # same size, same features: tests candidates against signals
            pool = [int(j) for j in np.argsort(-row, kind="stable")[: len(pool)]]
        pos = {a: {j: r for r, j in enumerate(L[a])} for a in self.agents}
        ctx = self.store.context(history, now)
        prof = self.persp.rates(history) if (self.persp is not None and self.persp_profile) else None
        PN = persp_names() if prof is not None else ()
        if row is not None:
            sc = row[pool]
            z = (sc - sc.mean()) / (sc.std() + 1e-9)
        else:
            z = np.zeros(len(pool))
        X = []
        for k, j in enumerate(pool):
            f = []
            for a in self.agents:
                r = pos[a].get(j)
                f += [0.0 if r is None else 1.0, 0.0 if r is None else 1.0 / (1.0 + r)]
            f += [z[k], np.log1p(self.store.pop[j]), np.log1p(self.store.recent_count(j, now)), ctx.after.get(j, 0.0),
                  float(ctx.with_n.get(j, 0)), np.log(ctx.text_rank[j]) if ctx.text_rank is not None else 0.0]
            if prof is not None:                         # version 5a: membership weighted by the user's own mode
                f += [(0.0 if pos[a].get(j) is None else 1.0 / (1.0 + pos[a][j])) * prof[q] for q, a in enumerate(PN)]
            X.append(f)
        return pool, np.asarray(X, dtype=np.float32), pos

    # ---- residual ranker -------------------------------------------------------
    def _net(self, d: int, hidden: int = 0):
        import torch
        net = torch.nn.Linear(d, 1) if not hidden else torch.nn.Sequential(
            torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
        last = net if not hidden else net[-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)
        return net

    def _score(self, X: np.ndarray) -> np.ndarray:
        if self.net is None or self.lam == 0.0:
            return X[:, self.zcol].astype(np.float64)
        import torch
        with torch.no_grad():
            f = self.net(torch.tensor((X - self.mu) / self.sd)).squeeze(-1).numpy()
        return X[:, self.zcol] + self.lam * f

    def fit(self, users: list[int], lams=(0.1, 0.3, 1.0, 3.0), steps: int = 400, seed: int = 0, batch: int = 256) -> dict:
        """Train on `users` (target: second-to-last item, history before it); lam and steps on a fifth of them."""
        import torch
        torch.manual_seed(seed)
        hist = [[int(x[0]) for x in self.ds.seqs[u][:-2]] for u in users]
        tgt = [int(self.ds.seqs[u][-2][0]) for u in users]
        Xs, Ys = [], []
        for s in range(0, len(users), batch):
            us, hs = users[s: s + batch], hist[s: s + batch]
            rows = self._sasrec_rows(us, hs)
            for r, (u, h) in enumerate(zip(us, hs)):
                pool, X, _ = self.features(h, self.store.now(u, len(h)), None if rows is None else rows[r])
                Xs.append(X)
                Ys.append(pool.index(tgt[s + r]) if tgt[s + r] in pool else -1)
        keep = [n for n, y in enumerate(Ys) if y >= 0]
        rng = np.random.default_rng(seed)
        perm = rng.permutation(keep).tolist()
        sel, fit = perm[: len(perm) // 5], perm[len(perm) // 5:]
        self.mu = np.concatenate([Xs[n] for n in fit]).mean(0)
        self.sd = np.concatenate([Xs[n] for n in fit]).std(0) + 1e-6
        self.mu[self.zcol], self.sd[self.zcol] = 0.0, 1.0

        def pad(idx):
            m = max(len(Xs[n]) for n in idx)
            F = np.zeros((len(idx), m, Xs[idx[0]].shape[1]), dtype=np.float32)
            M = np.zeros((len(idx), m), dtype=bool)
            for r, n in enumerate(idx):
                F[r, :len(Xs[n])], M[r, :len(Xs[n])] = Xs[n], True
            Z = torch.tensor(F[:, :, self.zcol])
            return torch.tensor((F - self.mu) / self.sd), torch.tensor(M), Z, torch.tensor([Ys[n] for n in idx])

        Ff, Mf, Zf, yf = pad(fit)
        Fs, Ms, Zs, ys = pad(sel)

        def ndcg_sel(net, lam):
            with torch.no_grad():
                s = (Zs + lam * net(Fs).squeeze(-1)).masked_fill(~Ms, float("-inf"))
                t = s[torch.arange(len(ys)), ys]
                r = (s > t[:, None]).sum(1) + 1
                return torch.where(r <= 10, 1.0 / torch.log2(r.float() + 1.0), torch.zeros(len(r))).mean().item()

        best = (ndcg_sel(lambda F: torch.zeros(F.shape[:2] + (1,)), 0.0), 0.0, None, 0)
        d = Ff.shape[2]
        for lam in lams:
            net = self._net(d, self.hidden)
            opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
            for step in range(1, steps + 1):
                s = (Zf + lam * net(Ff).squeeze(-1)).masked_fill(~Mf, float("-inf"))
                loss = torch.nn.functional.cross_entropy(s, yf)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if step % 50 == 0:
                    v = ndcg_sel(net, lam)
                    if v > best[0]:
                        best = (v, lam, {k: t.clone() for k, t in net.state_dict().items()}, step)
        self.lam = best[1]
        if best[2] is not None:
            self.net = self._net(d, self.hidden)
            self.net.load_state_dict(best[2])
        self.train_users = [int(u) for u in users]
        return {"lam": self.lam, "step": best[3], "selection_ndcg10": best[0], "users": len(users),
                "target_in_pool": len(keep) / max(len(users), 1)}

    def save(self, path: str) -> None:
        import torch
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"lam": self.lam, "mu": self.mu, "sd": self.sd, "train_users": self.train_users,
                    "state": None if self.net is None else self.net.state_dict(),
                    "depth": self.depth, "seq_depth": self.seq_depth, "hidden": self.hidden}, path)

    def load(self, path: str) -> "AgentPool":
        import torch
        d = torch.load(path, weights_only=False)
        self.lam, self.mu, self.sd, self.train_users = d["lam"], d["mu"], d["sd"], d["train_users"]
        if d["state"] is not None:
            self.hidden = int(d.get("hidden", 0))
            self.net = self._net(len(self.mu), self.hidden)
            self.net.load_state_dict(d["state"])
        return self

    # ---- use -------------------------------------------------------------------
    def rank(self, user: int, history: list[int], k: int, row: Optional[np.ndarray] = None):
        now = self.store.now(user, len(history))
        if row is None:
            rows = self._sasrec_rows([user], [history])
            row = None if rows is None else rows[0]
        pool, X, pos = self.features(history, now, row)
        s = self._score(X)
        order = list(np.argsort(-s, kind="stable")[:k])
        if self.persp_slots and self.persp is not None:
            # version 4.3: the last slots go to the best candidates of the perspective agents not already shown,
            # so the decision model can read them with their "found by" line
            from .kgperspectives import NAMES as P
            idx = {j: i for i, j in enumerate(pool)}
            keep = order[: k - self.persp_slots]
            shown = set(keep)
            best = sorted({idx[j] for a in P for j in pos[a] if idx[j] not in shown},
                          key=lambda i: (min(pos[a].get(pool[i], 10 ** 6) for a in P), -s[i]))
            order = keep + best[: self.persp_slots]
            order += [i for i in np.argsort(-s, kind="stable") if i not in set(order)][: k - len(order)]
        items = [pool[i] for i in order]
        self._found[(user, len(history))] = {j: {a: pos[a][j] for a in self.agents if j in pos[a]} for j in items}
        return items, s[order]

    def found_by(self, user: int, n_hist: int, item: int) -> str:
        hits = self._found.get((user, n_hist), {}).get(item)
        if not hits:
            return ""
        parts = []
        for a in self.agents:
            if a in hits:
                parts.append(f"{NAMES[a]} #{hits[a] + 1}" if a == "sasrecf" else NAMES[a])
        return "found by " + ", ".join(parts)


def refit(pool: AgentPool, users: list[int], pc: dict) -> AgentPool:
    """A copy of `pool` (same agents, store, scorer and settings) whose residual ranker is trained on `users` only.
    Used to cross-fit training records: a user's candidates come from a ranker that never saw that user."""
    p2 = AgentPool(pool.ds, pool.store, pool.scorer, pool.kg, pool.depth, pool.seq_depth, pool.persp)
    p2.hidden, p2.persp_slots, p2.persp_profile = pool.hidden, pool.persp_slots, pool.persp_profile
    p2.sasrecf_cands = pool.sasrecf_cands
    p2.fit(users, lams=tuple(float(x) for x in pc.get("lams", (0.1, 0.3, 1.0, 3.0))), steps=int(pc.get("steps", 400)))
    return p2


class PoolRetriever:
    """Retriever interface over an AgentPool, for evaluation cases and training records."""

    def __init__(self, pool: AgentPool):
        self.pool = pool

    def topk_users(self, users: list[int], histories: list[list[int]], k: int):
        rows = self.pool._sasrec_rows(users, histories)
        idx, val = [], []
        for r, (u, h) in enumerate(zip(users, histories)):
            items, s = self.pool.rank(u, h, k, None if rows is None else rows[r])
            seen = set(items) | set(h)
            items = items + [j for j in self.pool.popular if j not in seen][: k - len(items)]
            if len(items) < k:      # long histories (MovieLens) can cover the popular list: go on through the catalogue
                seen = set(items) | set(h)
                rest = (int(j) for j in np.argsort(-self.pool.store.pop, kind="stable") if int(j) not in seen)
                items += [j for _, j in zip(range(k - len(items)), rest)]
            idx.append(items[:k])
            val.append(list(s) + [float(min(s)) - 1.0 - n for n in range(k - len(s))])
        return np.array(idx), np.array(val, dtype=float)[:, :k]

    def topk(self, histories, k):
        raise SystemExit("the agent pool needs the users: use topk_users")


_POOLS: dict = {}


def pool_for(cfg: dict, ds: Dataset) -> AgentPool:
    """The pool of a config (section `pool:`), trained once and cached next to the config's runs.

        pool: {ref: configs/small/vg_semantic.yaml, model: sasrecf, users: 15000, seed: 11}
    """
    from ..evidence.store import store_for
    from ..experiments.benchmark import prepare_dataset
    from ..experiments.rectune import _case_users
    from ..experiments.runner import load_config
    from ..experiments.stage1 import chosen_dir
    from ..graph.kg import ItemKG
    from ..models import baselines as B
    pc = cfg.get("pool") or {}
    key = (cfg["dataset"], json.dumps(pc, sort_keys=True))
    if key in _POOLS:
        return _POOLS[key]
    store = store_for(cfg, ds)
    scorer = None
    if pc.get("ref"):
        ref = load_config(pc["ref"])
        model = pc.get("model", "sasrecf")
        prepare_dataset(ref, ds, [model])
        scorer = B.load(model, chosen_dir(ref, model), ds, pc.get("device", "cpu"))
    persp = None
    if pc.get("perspectives") == "kg":
        from .kgperspectives import KGPerspectives
        persp = KGPerspectives(ds, store, kind_depth=pc.get("kind_depth"))
    pool = AgentPool(ds, store, scorer, ItemKG(ds), int(pc.get("depth", 20)), int(pc.get("seq_depth", 50)), persp)
    tag = f"pool_{pc.get('model', 'none') if pc.get('ref') else 'none'}_u{int(pc.get('users', 15000))}_s{int(pc.get('seed', 11))}"
    if persp is not None:
        tag += "_kgp" + (f"_k{int(pc['kind_depth'])}" if pc.get("kind_depth") else "")
    lams = tuple(float(x) for x in pc.get("lams", (0.1, 0.3, 1.0, 3.0)))
    steps = int(pc.get("steps", 400))
    if "lams" in pc or "steps" in pc:
        tag += f"_l{'-'.join(f'{x:g}' for x in lams)}_t{steps}"
    pool.hidden = int(pc.get("hidden", 0))
    pool.persp_slots = int(pc.get("persp_slots", 0))      # used at ranking time only: the ranker is the same
    pool.persp_profile = bool(pc.get("persp_profile", False))
    if pool.persp_profile:
        tag += "_prof"
    if pool.hidden:
        tag += f"_h{pool.hidden}"
    pool.sasrecf_cands = pc.get("cands") == "sasrecf"
    if pool.sasrecf_cands:
        tag += "_cs"
    path = os.path.join(cfg["out"], "pool", f"{tag}.pt")
    if os.path.exists(path):
        pool.load(path)
    else:
        from ..data.candidates import planned_users
        excluded = _case_users(cfg) | planned_users(ds, cfg["eval"])   # also the cases this config will draw
        rng = np.random.default_rng(int(pc.get("seed", 11)))
        users = [u for u in rng.permutation(ds.n_users).tolist() if u not in excluded and len(ds.seqs[u]) >= 4]
        info = pool.fit(users[: int(pc.get("users", 15000))], lams=lams, steps=steps)
        pool.save(path)
        with open(path.replace(".pt", ".json"), "w") as f:
            json.dump(info, f, indent=1)
        print(f"[pool] {ds.name}: trained the residual ranker {info}", flush=True)
    _POOLS[key] = pool
    return pool
