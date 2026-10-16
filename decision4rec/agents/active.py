"""Decision-native recommendation agent on the ActiveGraph runtime.

The user model lives on the graph and is maintained by events. Control flow is
code, every judgment is a typed decision.

    interaction.added   -> observe    : new entities and beliefs for unseen attributes of the item
                        -> reinforce  : relation behaviour on belief -about-> entity edges, fires only
                                        for entities the item carries, adds support and marks stale
    slate.requested     -> profile    : one request, a `noul` per stale belief (preference strength)
    profile.updated     -> context    : one `choice` over the candidate nests (current intent)
    context.ready       -> retrieve   : first tranche of candidates, or, with retrieval agents,
    retrieval.requested -> retriever:* : one behaviour per point of view (sequential, content, knowledge,
                                        collaborative, profile, graph), each proposes its own candidates
    channel.proposed    -> route      : one `choice` over the agents' slates sets the fusion weights
    candidates.proposed -> decide     : block-design choices with outside option, Luce fit
    ranking.estimated   -> control    : if the outside option is likely, widen the set along
                                        user -> preferred entity -> item paths, else stop
    slate.ready         -> risk       : emit the slate or abstain

Beliefs are refreshed incrementally: an interaction only invalidates the
beliefs about entities it touches. Runs are replayable and forkable; a
counterfactual replays the log without one interaction.

The user model is built from the last `profile_hist` interactions. Older ones are kept as a
prefix of the session: retrieval agents read the whole history and never propose a seen item.
With `gamma` > 0 the final ranking fuses the Luce utilities with the retrieval score, as the
batch system does. Every round is recorded in `trace`, so the outcome of any expansion
threshold can be read from one run that always expands.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..data.schema import Case
from ..decision.base import DecisionBackend
from ..graph.kg import rank_hypotheses
from ..graph.profile import Pref, Profile, ask_intent, ask_preferences, graph_score
from ..operators.luce import Obs, fit_luce
from ..operators.rank import Arm, Cost, extend_observations
from ..decision.base import decide_chunked
from ..decision.types import Choice
from ..eval.metrics import zscore
from ..prompts import PromptBuilder

INSTR_ROUTE = ("Based on the user's interaction history, which set of items best matches what this user "
               "is most likely to choose next?")


def _key(rel: str, ent: str) -> str:
    return f"{rel}\t{ent}"


def fuse_scores(theta, rscores, gamma: float) -> np.ndarray:
    """Final ranking score of a set of examined items: Luce utilities, fused with the retrieval score."""
    theta = np.asarray(theta, dtype=float)
    if gamma <= 0 or len(theta) < 2:
        return theta
    return zscore(theta) + gamma * zscore(rscores)


@dataclass
class AgentResult:
    items: list[int]              # item ids examined or available, pool first, graph expansions after
    scores: np.ndarray            # aligned with `items`, -inf for items never examined
    slate: list[int]              # item ids, best first
    p_out: float
    confidence: float
    abstain: bool
    rounds: int
    n_examined: int
    n_graph_added: int            # candidates that came from preference paths, outside the retriever pool
    profile: dict                 # "relation\tentity" -> {"p", "support", "stale"}
    intent: Optional[dict]
    route: Optional[dict]         # retrieval agent -> fusion weight, None without retrieval agents
    tokens: int
    requests: int
    latency_ms: float
    events: list[str]
    anchor_event: str
    runtime: object = field(repr=False, default=None)
    theta: Optional[np.ndarray] = None     # Luce utilities aligned with `items`, -inf if never examined
    rscores: Optional[np.ndarray] = None   # retrieval scores aligned with `items`, 0 for items it did not rank
    trace: list = field(default_factory=list)   # one record per round: active items, utilities, cost so far
    n_pool: int = 0                         # items[:n_pool] came from retrieval, the rest from preference paths

    def target_pos(self, target: int) -> int:
        return self.items.index(target) if target in self.items else -1


class DecisionAgent:
    def __init__(self, backend: DecisionBackend, pb: PromptBuilder, k0: int = 20, step: int = 15,
                 max_rounds: int = 2, m: int = 5, rounds: int = 2, tau_expand: float = 0.5,
                 tau_abstain: float = 0.0, top_k: int = 10, user_model: bool = True, intent: bool = True,
                 evidence: str = "pref", kg_expand: bool = True, seed: int = 13, retrieval=None,
                 route: str = "decision", pool: int = 50, depth: int = 100, gamma: float = 0.0):
        """`retrieval` is a MultiAgentRetriever: its agents then run as behaviours and replace the fixed
        candidate pool. `route` sets the fusion weights: uniform, tuned (validation) or decision.
        `pool` is the number of fused candidates the session keeps, `k0` of them are compared first.
        `gamma` weights the retrieval score in the final ranking (0 ranks by the Luce utilities alone)."""
        self.backend, self.pb = backend, pb
        self.k0, self.step, self.max_rounds, self.m = k0, step, max_rounds, m
        self.tau_expand, self.tau_abstain, self.top_k = tau_expand, tau_abstain, top_k
        self.user_model, self.use_intent, self.kg_expand, self.seed = user_model, intent, kg_expand, seed
        self.arm = Arm(name="agent", method="luce", m=m, rounds=rounds, outside=True,
                       evidence=evidence if user_model else "none")
        self.retrieval, self.route, self.pool, self.depth = retrieval, route, pool, depth
        self.gamma = float(gamma)
        self._cases: dict[int, Case] = {}
        self._cost: dict[str, Cost] = {}

    # ---- reading the user model off the graph ------------------------------

    def _beliefs(self, graph, data: dict) -> list:
        return [graph.get_object(b) for b in data["beliefs"]]

    def _profile(self, graph, data: dict) -> Profile:
        return Profile(prefs=[Pref(b.data["rel"], b.data["ent"], float(b.data["p"]), int(b.data["support"]),
                                   float(b.data["idf"]))
                              for b in self._beliefs(graph, data) if b.data["p"] is not None])

    @staticmethod
    def _seen(data: dict) -> list[int]:
        """Everything the user interacted with: the older prefix plus the observed interactions."""
        return list(data.get("prefix", [])) + list(data["history"])

    def _work_case(self, data: dict) -> Case:
        """The case as the graph currently sees it: history and candidates come from the session."""
        base = self._cases[data["cid"]]
        return dataclasses.replace(base, history=list(data["history"]), cands=list(data["items"]),
                                   rscores=[0.0] * len(data["items"]))

    # ---- behaviours: user model -------------------------------------------

    def _observe(self, event, graph, ctx) -> None:
        sid, item = event.payload["sid"], int(event.payload["item"])
        data = graph.get_object(sid).data
        index, beliefs = dict(data["index"]), list(data["beliefs"])
        kg = self.pb.kg
        for rel, ent in sorted(k for k in kg.attrs[item] if k in kg.idf and _key(*k) not in index):
            entity = graph.add_object("entity", {"rel": rel, "ent": ent})
            belief = graph.add_object("belief", {"rel": rel, "ent": ent, "idf": float(kg.idf[(rel, ent)]),
                                                 "p": None, "support": 1, "stale": True})
            graph.add_relation(belief.id, entity.id, "about")
            index[_key(rel, ent)] = entity.id
            beliefs.append(belief.id)
        graph.patch_object(sid, {"history": data["history"] + [item], "index": index, "beliefs": beliefs})

    @staticmethod
    def _reinforce(relation, event, graph, ctx) -> None:
        """Runs once per belief whose entity the new item carries (the event references the entity id)."""
        belief = graph.get_object(relation.source)
        graph.patch_object(belief.id, {"support": belief.data["support"] + 1, "stale": True})

    def _refresh_profile(self, event, graph, ctx) -> None:
        sid = event.payload["sid"]
        data = graph.get_object(sid).data
        asked = 0
        if self.user_model:
            stale = {(b.data["rel"], b.data["ent"]): b for b in self._beliefs(graph, data) if b.data["stale"]}
            hyps = rank_hypotheses([(r, e, b.data["support"], b.data["idf"]) for (r, e), b in stale.items()],
                                   self.pb.max_h)
            if hyps:
                case = self._cases[data["cid"]]
                state = self.pb.profile_state(case, history=data["history"])
                prof = ask_preferences(self.backend, state, hyps, self.pb.kg.idf, tag=self.pb.ds.user_tag)
                self._cost[sid].add_raw(prof.tokens, prof.requests, prof.latency_ms)
                for q in prof.prefs:
                    graph.patch_object(stale[(q.rel, q.ent)].id, {"p": q.p, "stale": False})
                asked = len(hyps)
        then = "slate" if event.type == "slate.requested" else None
        graph.emit("profile.updated", {"sid": sid, "asked": asked, "then": then})

    def _context(self, event, graph, ctx) -> None:
        if event.payload.get("then") != "slate":
            return
        sid = event.payload["sid"]
        data = graph.get_object(sid).data
        if self.use_intent and self.user_model and self.pb.intent_relation:
            case = self._cases[data["cid"]]
            nests = self.pb.kg.nests(list(data["items"]) + data["history"][-5:], self.pb.intent_relation,
                                     self.pb.max_intents)
            intent, tokens, requests, latency = ask_intent(
                self.backend, self.pb.profile_state(case, history=data["history"]), nests,
                self.pb.intent_relation, tag=self.pb.ds.user_tag)
            self._cost[sid].add_raw(tokens, requests, latency)
            graph.patch_object(sid, {"intent": intent})
        graph.emit("context.ready", {"sid": sid})

    # ---- behaviours: ranking loop -----------------------------------------

    def _retrieve(self, event, graph, ctx) -> None:
        sid = event.payload["sid"]
        if self.retrieval is not None:
            graph.emit("retrieval.requested", {"sid": sid})
            return
        n = len(graph.get_object(sid).data["items"])
        graph.emit("candidates.proposed", {"sid": sid, "round": 0, "new": list(range(min(self.k0, n)))})

    def _retrieval_behavior(self, agent):
        """One retrieval agent as a behaviour: it reads the history and the user model from the graph
        and proposes candidates from its own point of view."""
        from ..retrieval.agents import RetrievalContext

        def propose(event, graph, ctx) -> None:
            sid = event.payload["sid"]
            data = graph.get_object(sid).data
            rctx = RetrievalContext(history=self._seen(data),
                                    prof=self._profile(graph, data) if self.user_model else None,
                                    intent=data.get("intent"))
            graph.emit("channel.proposed", {"sid": sid, "channel": agent.name,
                                            "items": [int(i) for i in agent.propose(rctx, self.depth)]})
        return propose

    def _route(self, event, graph, ctx) -> None:
        """Collect the agents' proposals; once all are in, weight them and fuse into the candidate pool."""
        sid = event.payload["sid"]
        data = graph.get_object(sid).data
        proposals = {**data.get("proposals", {}), event.payload["channel"]: event.payload["items"]}
        if len(proposals) < len(self.retrieval.agents):
            graph.patch_object(sid, {"proposals": proposals})
            return
        names = [n for n, items in proposals.items() if items]
        weights = {n: 1.0 for n in names}
        if self.route == "tuned":
            weights = {n: float(self.retrieval.weights.get(n, 1.0)) for n in names}
        elif self.route == "decision" and len(names) > 1:
            case = self._work_case(data)
            prof = self._profile(graph, data) if self.user_model else None
            state = self.pb.state(case, profile="decision" if self.user_model else "none", prof=prof,
                                  intent=data.get("intent"))
            options = tuple((f"c{j + 1:02d}", "; ".join(self.pb.ds.item_text(i)[:80] for i in proposals[n][:3]))
                            for j, n in enumerate(names))
            d = decide_chunked(self.backend, state, {"route": Choice(INSTR_ROUTE, options)})
            self._cost[sid].add(d)
            probs = d.answers["route"].probs
            floor = 0.5 / len(names)
            weights = {n: floor + float(probs[f"c{j + 1:02d}"]) for j, n in enumerate(names)}
        ranked, fused = self.retrieval.fuse({n: proposals[n] for n in names}, weights)
        items = ranked[: self.pool]
        graph.patch_object(sid, {"proposals": proposals, "items": items, "n_pool": len(items), "route": weights,
                                 "rscore": {str(i): float(s) for i, s in zip(ranked, fused)}})
        graph.emit("candidates.proposed", {"sid": sid, "round": 0, "new": list(range(min(self.k0, len(items))))})

    def _decide(self, event, graph, ctx) -> None:
        sid, rnd = event.payload["sid"], event.payload["round"]
        data = graph.get_object(sid).data
        case = self._work_case(data)
        K = len(case.cands)
        cost = self._cost[sid]
        prof = self._profile(graph, data) if self.user_model else None
        state = self.pb.state(case, profile="decision" if self.user_model else "none", prof=prof,
                              intent=data.get("intent"))
        rng = np.random.default_rng([self.seed, case.cid, rnd])
        prior = [Obs.from_json(o) for o in data["obs"]]
        obs = extend_observations(self.backend, self.pb, case, self.arm, state, rng, cost, prior,
                                  list(event.payload["new"]), self.m, prof)
        active = sorted(set(data["active"]) | set(event.payload["new"]))
        fit = fit_luce(obs, K)
        w = np.exp(fit.theta[active])
        w_out = float(np.exp(fit.theta_out))
        total = w.sum() + w_out
        step = {"round": rnd, "active": active, "theta": [float(fit.theta[j]) for j in active],
                "p_out": w_out / total, "n_items": K, "tokens": cost.tokens, "requests": cost.requests,
                "latency_ms": cost.latency_ms}
        graph.patch_object(sid, {"obs": [o.to_json() for o in obs], "active": active, "round": rnd,
                                 "theta": [float(t) for t in fit.theta], "p_out": w_out / total,
                                 "confidence": float(w.max() / total), "trace": data["trace"] + [step]})
        graph.emit("ranking.estimated", {"sid": sid, "round": rnd, "p_out": w_out / total})

    def _control(self, event, graph, ctx) -> None:
        sid = event.payload["sid"]
        data = graph.get_object(sid).data
        if data["p_out"] > self.tau_expand and data["round"] < self.max_rounds:
            items, new = self._next_tranche(graph, data)
            if new:
                graph.patch_object(sid, {"items": items})
                graph.emit("candidates.proposed", {"sid": sid, "round": data["round"] + 1, "new": new})
                return
        graph.emit("slate.ready", {"sid": sid})

    def _next_tranche(self, graph, data: dict) -> tuple[list[int], list[int]]:
        """Next candidates. Pool items come first, ordered by the strength of the preference paths that
        reach them. With `kg_expand`, half of the tranche is drawn from the whole catalogue along
        user -> preferred entity -> item paths, which is how a retrieval miss can be repaired."""
        items = list(data["items"])
        active = set(data["active"])
        remaining = [j for j in range(len(items)) if j not in active]
        kg = self.pb.kg
        strength = self._profile(graph, data).strength() if self.user_model else {}
        if strength:
            remaining.sort(key=lambda j: (-graph_score(kg.attrs[items[j]], strength, kg.idf), j))
        n_graph = self.step // 2 if (self.kg_expand and strength) else 0
        new = remaining[: self.step - n_graph]
        if n_graph:
            exclude = set(items) | set(self._seen(data))
            for item in kg.retrieve_by_preferences(strength, exclude, n_graph):
                items.append(int(item))
                new.append(len(items) - 1)
        if len(new) < self.step:
            taken = set(new)
            new += [j for j in remaining if j not in taken][: self.step - len(new)]
        return items, new

    def _risk(self, event, graph, ctx) -> None:
        sid = event.payload["sid"]
        data = graph.get_object(sid).data
        active = list(data["active"])
        rmap = data["rscore"]
        final = fuse_scores([data["theta"][j] for j in active],
                            [rmap.get(str(data["items"][j]), 0.0) for j in active], self.gamma)
        score = {j: float(s) for j, s in zip(active, final)}
        slate = sorted(active, key=lambda j: -score[j])[: self.top_k]
        abstain = bool(data["confidence"] < self.tau_abstain)
        graph.patch_object(sid, {"slate": slate, "abstain": abstain, "scores": {str(j): s for j, s in score.items()}})
        graph.emit("slate.emitted", {"sid": sid, "slate": [data["items"][j] for j in slate], "abstain": abstain})

    def behaviors(self) -> list:
        from activegraph import Behavior, RelationBehavior
        extra = []
        if self.retrieval is not None:
            extra = [Behavior(name=f"retriever:{a.name}", fn=self._retrieval_behavior(a), on=["retrieval.requested"])
                     for a in self.retrieval.agents]
            extra.append(Behavior(name="route", fn=self._route, on=["channel.proposed"]))
        return extra + [Behavior(name="observe", fn=self._observe, on=["interaction.added"]),
                RelationBehavior(name="reinforce", fn=self._reinforce, relation_type="about",
                                 on=["interaction.added"]),
                Behavior(name="profile", fn=self._refresh_profile, on=["slate.requested", "profile.requested"]),
                Behavior(name="context", fn=self._context, on=["profile.updated"]),
                Behavior(name="retrieve", fn=self._retrieve, on=["context.ready"]),
                Behavior(name="decide", fn=self._decide, on=["candidates.proposed"]),
                Behavior(name="control", fn=self._control, on=["ranking.estimated"]),
                Behavior(name="risk", fn=self._risk, on=["slate.ready"])]

    # ---- harness ----------------------------------------------------------

    @staticmethod
    def _emit(graph, etype: str, payload: dict):
        from activegraph import Event
        ev = Event(id=graph.ids.event(), type=etype, payload=payload, actor="harness", timestamp=graph.clock.now())
        graph.emit(ev)
        return ev

    def start(self, case: Case, persist_to: Optional[str] = None):
        """Open a session: an empty user model and the retriever pool. Returns (runtime, session id, anchor)."""
        from activegraph import Frame, Graph, Runtime
        graph = Graph()
        runtime = Runtime(graph, behaviors=self.behaviors(), frame=Frame(goal="recommend the next item"),
                          budget={"max_events": 200000}, persist_to=persist_to)
        self._cases[case.cid] = case
        session = graph.add_object("session", {
            "cid": case.cid, "history": [], "index": {}, "beliefs": [], "items": list(case.cands), "intent": None,
            "prefix": [int(i) for i in case.history[: max(len(case.history) - self.pb.profile_hist, 0)]],
            "rscore": {str(i): float(s) for i, s in zip(case.cands, case.rscores)},
            "n_pool": len(case.cands), "proposals": {}, "route": None, "trace": [], "scores": {},
            "obs": [], "active": [], "round": 0, "theta": [], "p_out": 1.0, "confidence": 0.0, "slate": None})
        self._cost[session.id] = Cost()
        return runtime, session.id, graph.events[-1].id

    def observe(self, runtime, sid: str, item: int) -> None:
        """Feed one interaction. The payload names the entities of the item that already exist on the
        graph, so the relation behaviour fires for exactly those beliefs."""
        graph = runtime.graph
        index = graph.get_object(sid).data["index"]
        kg = self.pb.kg
        known = [index[_key(r, e)] for r, e in sorted(kg.attrs[item]) if _key(r, e) in index]
        self._emit(graph, "interaction.added", {"sid": sid, "item": int(item), "entities": known})
        runtime.run_until_idle()

    def refresh(self, runtime, sid: str) -> int:
        """Refresh stale beliefs without asking for a slate. Returns the tokens spent so far."""
        self._emit(runtime.graph, "profile.requested", {"sid": sid})
        runtime.run_until_idle()
        return self._cost[sid].tokens

    def beliefs(self, runtime, sid: str) -> dict:
        graph = runtime.graph
        return {_key(b.data["rel"], b.data["ent"]): {"p": b.data["p"], "support": b.data["support"],
                                                     "stale": b.data["stale"]}
                for b in self._beliefs(graph, graph.get_object(sid).data)}

    def _result(self, runtime, sid: str, anchor: str) -> AgentResult:
        graph = runtime.graph
        if runtime.errors:
            raise RuntimeError(f"agent behaviour failed: {runtime.errors[0]}")
        data = graph.get_object(sid).data
        if data.get("slate") is None:
            raise RuntimeError("agent stopped before emitting a slate (budget exhausted?)")
        items = list(data["items"])
        scores = np.full(len(items), -np.inf)
        theta = np.full(len(items), -np.inf)
        theta[data["active"]] = np.asarray(data["theta"])[data["active"]]
        for j, s in data["scores"].items():
            scores[int(j)] = s
        rscores = np.array([data["rscore"].get(str(i), 0.0) for i in items], dtype=float)
        cost = self._cost[sid]
        n_pool = int(data["n_pool"])
        return AgentResult(items=items, scores=scores, slate=[items[j] for j in data["slate"]],
                           p_out=data["p_out"], confidence=data["confidence"], abstain=data["abstain"],
                           rounds=data["round"], n_examined=len(data["active"]),
                           n_graph_added=sum(1 for j in data["active"] if j >= n_pool),
                           profile=self.beliefs(runtime, sid), intent=data.get("intent"), route=data.get("route"),
                           tokens=cost.tokens,
                           requests=cost.requests, latency_ms=cost.latency_ms,
                           events=[e.type for e in graph.events], anchor_event=anchor, runtime=runtime,
                           theta=theta, rscores=rscores, trace=list(data["trace"]), n_pool=n_pool)

    def _request(self, runtime, sid: str, anchor: str) -> AgentResult:
        self._emit(runtime.graph, "slate.requested", {"sid": sid})
        runtime.run_until_idle()
        return self._result(runtime, sid, anchor)

    def run(self, case: Case, persist_to: Optional[str] = None) -> AgentResult:
        runtime, sid, anchor = self.start(case, persist_to)
        for item in self.pb.profile_history(case):
            self.observe(runtime, sid, item)
        return self._request(runtime, sid, anchor)

    def counterfactual(self, base: AgentResult, case: Case, drop: int) -> AgentResult:
        """Fork the recorded run before any interaction and replay the history without item `drop`:
        the user model, the intent and the slate are all recomputed from the altered log.
        Requires a run created with `persist_to` (ActiveGraph forks SQLite-backed runs)."""
        fork = base.runtime.fork(at_event=base.anchor_event, label=f"drop-{drop}", behaviors=self.behaviors())
        sid = next(o.id for o in fork.graph.all_objects() if o.type == "session")
        self._cost[sid] = Cost()
        for item in self.pb.profile_history(case):
            if item != drop:
                self.observe(fork, sid, item)
        return self._request(fork, sid, base.anchor_event)
