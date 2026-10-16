"""Typed questions and answers of the decision layer.

The wire format follows the System One protocol (`POST /v1/systemone`): a
`state` plus a map of named questions of type `choice`, `noul` or `score`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

OUTSIDE_KEY = "none"


@dataclass(frozen=True)
class Choice:
    """Pick one of several named options. `options` keeps presentation order."""

    instructions: str
    options: tuple[tuple[str, Optional[str]], ...]

    def keys(self) -> list[str]:
        return [k for k, _ in self.options]

    def to_wire(self) -> dict:
        return {"type": "choice", "instructions": self.instructions,
                "criteria": {k: v for k, v in self.options}}


@dataclass(frozen=True)
class Noul:
    """Probability that a yes/no statement holds."""

    instructions: str

    def to_wire(self) -> dict:
        return {"type": "noul", "instructions": self.instructions}


@dataclass(frozen=True)
class Score:
    """Position on an ordered, low-to-high scale."""

    instructions: str
    levels: tuple[str, ...]

    def to_wire(self) -> dict:
        return {"type": "score", "instructions": self.instructions,
                "criteria": list(self.levels)}


Question = Union[Choice, Noul, Score]


@dataclass
class Answer:
    kind: str
    probs: Optional[dict[str, float]] = None  # choice: option key -> p, score: level index -> p
    p: Optional[float] = None                 # noul: probability of yes
    value: Optional[float] = None             # score: expected level (0-based)

    def to_json(self) -> dict:
        return {"kind": self.kind, "probs": self.probs, "p": self.p, "value": self.value}

    @classmethod
    def from_json(cls, d: dict) -> "Answer":
        return cls(kind=d["kind"], probs=d.get("probs"), p=d.get("p"), value=d.get("value"))


@dataclass
class Decision:
    answers: dict[str, Answer]
    input_tokens: int = 0
    latency_ms: float = 0.0
    n_requests: int = 1
    cached: bool = False
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"answers": {k: a.to_json() for k, a in self.answers.items()},
                "input_tokens": self.input_tokens, "latency_ms": self.latency_ms,
                "n_requests": self.n_requests}

    @classmethod
    def from_json(cls, d: dict) -> "Decision":
        return cls(answers={k: Answer.from_json(a) for k, a in d["answers"].items()},
                   input_tokens=int(d.get("input_tokens", 0)),
                   latency_ms=float(d.get("latency_ms", 0.0)),
                   n_requests=int(d.get("n_requests", 1)))


def merge(decisions: list[Decision]) -> Decision:
    """Combine decisions that answered disjoint question sets over the same state."""
    out = Decision(answers={}, input_tokens=0, latency_ms=0.0, n_requests=0, cached=True)
    for d in decisions:
        out.answers.update(d.answers)
        out.input_tokens += d.input_tokens
        out.latency_ms += d.latency_ms
        out.n_requests += d.n_requests
        out.cached = out.cached and d.cached
    return out
