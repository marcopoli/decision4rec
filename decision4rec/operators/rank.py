"""Ranking arms: how a decision backend is queried for one evaluation case.

An arm only produces raw observations (answered blocks or pointwise
probabilities). Scores, debiasing, fusion and calibration are derived later
from the stored observations without new model calls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from ..data.schema import Case
from ..decision.base import DecisionBackend, decide_chunked
from ..decision.types import OUTSIDE_KEY, Decision
from ..graph.profile import Profile, graph_score, intent_logprob
from ..prompts import PromptBuilder
from .design import components, random_partition
from .luce import Obs, fit_luce


@dataclass
class Arm:
    name: str
    method: str = "luce"        # retriever | pointwise | luce | graph
    m: int = 0                  # block size, 0 means the whole pool in one block
    rounds: int = 1             # independent random partitions of the pool
    adaptive: int = 0           # extra rounds that re-compare the current head of the ranking
    outside: bool = False       # add the outside option to every block
    evidence: str = "none"      # kinds joined by '+': pref (decision-estimated preference paths), kg, cf
    profile: str = "none"       # none | count (attribute frequencies) | decision (model-estimated edges)
    intent: bool = False        # ask the model for the user's current intent over the candidate nests
    null_state: bool = False    # history-free prior (leakage and popularity control)
    mask_titles: bool = False   # hide item titles (memorisation control)
    perm: int = 0               # permutation id, changes the presentation order only
    top_k: int = 10
    max_cases: int = 0          # run on the first n cases only (0 = all), for control arms

    def __post_init__(self) -> None:
        if isinstance(self.profile, bool):
            self.profile = "count" if self.profile else "none"

    @property
    def needs_user_model(self) -> bool:
        return not self.null_state and (self.method == "graph" or self.profile == "decision"
                                        or "pref" in self.evidence.split("+"))

    @classmethod
    def from_dict(cls, d: dict) -> "Arm":
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)


class Cost:
    def __init__(self) -> None:
        self.tokens, self.latency_ms, self.requests, self.stages, self.cached = 0, 0.0, 0, 0, True

    def add(self, d: Decision) -> None:
        self.tokens += d.input_tokens
        self.latency_ms += d.latency_ms
        self.requests += d.n_requests
        self.stages += 1
        self.cached = self.cached and d.cached

    def add_raw(self, tokens: int, requests: int, latency_ms: float) -> None:
        self.tokens += tokens
        self.requests += requests
        self.latency_ms += latency_ms


def _ask(backend: DecisionBackend, pb: PromptBuilder, case: Case, arm: Arm, state: str,
         blocks: list[list[int]], cost: Cost, prof: Optional[Profile] = None) -> list[Obs]:
    opt = {"evidence": arm.evidence, "mask_titles": arm.mask_titles, "prof": prof}
    questions = {f"b{j}": pb.choice(case, b, outside=arm.outside, **opt) for j, b in enumerate(blocks)}
    d = decide_chunked(backend, state, questions)
    cost.add(d)
    out = []
    for j, b in enumerate(blocks):
        probs = d.answers[f"b{j}"].probs
        p = np.array([probs[f"o{k + 1:02d}"] for k in range(len(b))], dtype=float)
        p_out = float(probs[OUTSIDE_KEY]) if arm.outside else float("nan")
        out.append(Obs(items=np.asarray(b, dtype=int), probs=p, p_out=p_out))
    return out


def extend_observations(backend: DecisionBackend, pb: PromptBuilder, case: Case, arm: Arm, state: str,
                        rng: np.random.Generator, cost: Cost, obs: list[Obs], new_items: list[int],
                        m: int, prof: Optional[Profile] = None) -> list[Obs]:
    """Compare `new_items` (positions in `case.cands`) in `arm.rounds` random partitions sent as one
    request, then add bridge blocks of component leaders until every item seen so far is connected."""
    K = len(case.cands)
    blocks = [b for _ in range(max(arm.rounds, 1)) for b in random_partition(new_items, m, rng)]
    obs = obs + _ask(backend, pb, case, arm, state, blocks, cost, prof)
    seen = {int(i) for o in obs for i in o.items}
    while True:
        comps = [c for c in components([o.items.tolist() for o in obs], K) if c[0] in seen]
        if len(comps) <= 1:
            return obs
        theta = fit_luce(obs, K).theta
        leaders = [max(c, key=lambda i: theta[i]) for c in comps]
        obs = obs + _ask(backend, pb, case, arm, state, random_partition(leaders, m, rng), cost, prof)


def luce_observations(backend: DecisionBackend, pb: PromptBuilder, case: Case, arm: Arm, state: str,
                      rng: np.random.Generator, cost: Cost, prof: Optional[Profile] = None) -> list[Obs]:
    """Stage 1: `rounds` random partitions in one request. Stage 2: bridge blocks until the comparison
    hypergraph is connected. Stage 3: adaptive rounds that re-compare the head of the ranking."""
    K = len(case.cands)
    m = K if arm.m in (0, None) or arm.m >= K else arm.m
    obs = extend_observations(backend, pb, case, arm, state, rng, cost, [], list(range(K)), m, prof)
    for _ in range(arm.adaptive):
        theta = fit_luce(obs, K).theta
        head = np.argsort(-theta)[: max(2 * arm.top_k, m)]
        if len(head) < 2:
            break
        obs = obs + _ask(backend, pb, case, arm, state, random_partition(head, m, rng), cost, prof)
    return obs


def user_context(backend: DecisionBackend, pb: PromptBuilder, case: Case, arm: Arm, cost: Cost, rec: dict):
    """Build the decision-native user model an arm asks for and log what the analysis needs:
    graph scores, intent log-probabilities and the judged hypotheses with their outcome labels."""
    prof = intent = None
    if arm.needs_user_model:
        prof = pb.user_model(backend, case, arm.mask_titles)
        cost.add_raw(prof.tokens, prof.requests, prof.latency_ms)
        strength, kg = prof.strength(), pb.kg
        rec["graph"] = [graph_score(kg.attrs[i], strength, kg.idf) for i in case.cands]
        target_attrs = kg.attrs[case.target]
        rec["prefs"] = [[q.rel, q.ent, q.p, q.support, q.idf, int((q.rel, q.ent) in target_attrs)]
                        for q in prof.prefs]
    if arm.intent and not arm.null_state:
        intent, tokens, requests, latency = pb.intent(backend, case, arm.mask_titles)
        cost.add_raw(tokens, requests, latency)
        rec["intent"] = [intent_logprob(pb.kg.attrs[i], intent, pb.intent_relation) for i in case.cands]
    return prof, intent


def run_arm(case: Case, arm: Arm, backend: DecisionBackend, pb: PromptBuilder, seed: int = 13) -> dict:
    """Execute one arm on one case and return a JSON-serialisable record of raw observations."""
    cost = Cost()
    rec: dict = {"cid": case.cid, "arm": arm.name, "obs": None, "point": None}
    if arm.method == "retriever":
        cost.cached = False
    else:
        prof, intent = user_context(backend, pb, case, arm, cost, rec)
        state = pb.state(case, null_state=arm.null_state, profile=arm.profile, mask_titles=arm.mask_titles,
                         prof=prof, intent=intent)
        if arm.method == "graph":
            pass                                   # ranks by the graph score alone, no ranking question
        elif arm.method == "pointwise":
            opt = {"evidence": arm.evidence, "mask_titles": arm.mask_titles, "prof": prof}
            questions = {f"q{ci}": pb.noul(case, item, **opt) for ci, item in enumerate(case.cands)}
            d = decide_chunked(backend, state, questions)
            cost.add(d)
            rec["point"] = [float(d.answers[f"q{ci}"].p) for ci in range(len(case.cands))]
        elif arm.method == "luce":
            rng = np.random.default_rng([seed, case.cid, arm.perm])
            obs = luce_observations(backend, pb, case, arm, state, rng, cost, prof)
            rec["obs"] = [o.to_json() for o in obs]
        else:
            raise ValueError(f"unknown method {arm.method!r}")
    rec.update(tokens=cost.tokens, latency_ms=cost.latency_ms, requests=cost.requests,
               stages=cost.stages, cached=cost.cached)
    return rec
