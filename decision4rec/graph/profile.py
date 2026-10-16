"""Decision-native user model.

The knowledge graph proposes hypotheses (attributes of the items a user
consumed), the decision model judges each one with a `noul` question, and the
answers become weighted user -> entity preference edges. The same model
answers one `choice` question about the user's current intent over the nests
(categories, genres) present among the candidates.

These edges are then used three ways: as a compact profile in the state, as
path evidence (user -> preferred entity -> candidate) attached to candidates,
and as a model-free graph score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ..decision.base import DecisionBackend, decide_chunked
from ..decision.types import Choice, Noul

INSTR_PREF = ("Based on the interaction history, does this user have a clear preference for items "
              "whose {rel} is '{ent}'?")
INSTR_INTENT = "Based on the interaction history, which kind of item is the user most likely to choose next?"
OTHER_KEY = "other"


@dataclass
class Pref:
    rel: str
    ent: str
    p: float
    support: int
    idf: float


@dataclass
class Profile:
    prefs: list[Pref] = field(default_factory=list)
    tokens: int = 0
    requests: int = 0
    latency_ms: float = 0.0

    def strength(self) -> dict[tuple, float]:
        return {(q.rel, q.ent): q.p for q in self.prefs}

    def top(self, n: int = 6, min_p: float = 0.5) -> list[Pref]:
        return sorted((q for q in self.prefs if q.p >= min_p), key=lambda q: (-q.p, q.rel, q.ent))[:n]


def pref_question(rel: str, ent: str, tag: bool = False) -> Noul:
    text = INSTR_PREF.format(rel=rel, ent=ent)
    return Noul(text + (f" [attr:{rel}={ent}]" if tag else ""))


def ask_preferences(backend: DecisionBackend, state: str, hyps: list[tuple[str, str, int]], idf: dict,
                    tag: bool = False) -> Profile:
    """One request: a `noul` per hypothesis over the same history state."""
    if not hyps:
        return Profile()
    questions = {f"h{j}": pref_question(rel, ent, tag) for j, (rel, ent, _) in enumerate(hyps)}
    d = decide_chunked(backend, state, questions)
    prefs = [Pref(rel, ent, float(d.answers[f"h{j}"].p), sup, float(idf[(rel, ent)]))
             for j, (rel, ent, sup) in enumerate(hyps)]
    return Profile(prefs=prefs, tokens=d.input_tokens, requests=d.n_requests, latency_ms=d.latency_ms)


def intent_question(nests: list[str], relation: str, tag: bool = False) -> Choice:
    options = [(f"n{j + 1:02d}", f"{relation}: {n}" + (f" [attr:{relation}={n}]" if tag else ""))
               for j, n in enumerate(nests)]
    options.append((OTHER_KEY, "Something else"))
    return Choice(INSTR_INTENT, tuple(options))


def ask_intent(backend: DecisionBackend, state: str, nests: list[str], relation: str,
               tag: bool = False) -> tuple[dict[str, float], int, int, float]:
    """Distribution over candidate nests plus `other`. Returns (probabilities, tokens, requests, latency)."""
    if not nests:
        return {OTHER_KEY: 1.0}, 0, 0, 0.0
    d = decide_chunked(backend, state, {"intent": intent_question(nests, relation, tag)})
    probs = d.answers["intent"].probs
    out = {n: float(probs[f"n{j + 1:02d}"]) for j, n in enumerate(nests)}
    out[OTHER_KEY] = float(probs[OTHER_KEY])
    return out, d.input_tokens, d.n_requests, d.latency_ms


def graph_score(attrs: frozenset, strength: dict[tuple, float], idf: dict) -> float:
    """Evidence carried by user -> preferred entity -> item paths. Zero for unprofiled attributes."""
    return float(sum(idf[k] * (2.0 * strength[k] - 1.0) for k in attrs if k in strength and k in idf))


def intent_logprob(attrs: frozenset, intent: dict[str, float], relation: str) -> float:
    """Log-probability of the most likely nest an item belongs to, `other` if it belongs to none."""
    best = max((intent[e] for r, e in attrs if r == relation and e in intent), default=None)
    return math.log(max(best if best is not None else intent.get(OTHER_KEY, 0.0), 1e-6))


def pref_facts(attrs: frozenset, profile: Optional[Profile], max_facts: int, min_p: float = 0.5) -> list[tuple[str, float]]:
    """Path evidence for a candidate: its attributes that the user is judged to prefer."""
    if profile is None:
        return []
    hits = [q for q in profile.prefs if q.p >= min_p and (q.rel, q.ent) in attrs]
    hits.sort(key=lambda q: (-q.p * q.idf, q.rel, q.ent))
    return [(f"matches the user's preference for {q.rel} '{q.ent}' (strength {q.p:.2f})", q.p * q.idf)
            for q in hits[:max_facts]]
