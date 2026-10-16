"""A synthetic decision backend with known ground truth.

It answers from a utility matrix and can inject the pathologies the framework
is meant to measure: position bias, similarity-driven IIA violations and a
history-free prior. Used by tests and by the synthetic end-to-end run.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Mapping, Optional

import numpy as np

from .base import DecisionBackend
from .types import OUTSIDE_KEY, Answer, Choice, Decision, Noul, Question, Score

_ITEM = re.compile(r"\[item:(\d+)\]")
_USER = re.compile(r"\[user:(\d+)\]")
_ATTR = re.compile(r"\[attr:([^=\]]+)=([^\]]+)\]")
_NEST = re.compile(r"\[nest:(\d+)\]")
RELATION_KEYS = ["complement", "substitute", "series", "unrelated"]


def _noise(seed: str, scale: float) -> float:
    if scale <= 0:
        return 0.0
    h = int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return (h - 0.5) * 2.0 * scale


class OracleBackend(DecisionBackend):
    def __init__(self, truth: str | dict, name: str = "oracle", temperature: float = 1.0,
                 pos_bias: float = 0.0, similarity: float = 0.0, noise: float = 0.0,
                 outside_utility: float = 0.0, score_compress: float = 0.0,
                 max_request_chars: Optional[int] = None):
        if isinstance(truth, str):
            z = np.load(truth)
            truth = {k: z[k] for k in z.files}
        self.util = np.asarray(truth["utility"], dtype=float)        # users x items
        self.prior = np.asarray(truth["prior"], dtype=float)          # items
        self.group = np.asarray(truth["group"], dtype=int)            # items -> nest id
        # Optional attribute tastes, users x entities, keyed by relation name ("taste_category" -> "category").
        self.taste = {k[len("taste_"):]: np.asarray(v, dtype=float) for k, v in truth.items() if k.startswith("taste_")}
        self.name = name
        self.temperature = temperature
        self.pos_bias = pos_bias
        self.similarity = similarity
        self.noise = noise
        self.outside_utility = outside_utility
        self.score_compress = score_compress      # 0 = faithful scale, towards 1 = answers collapse to the middle
        self.max_request_chars = max_request_chars
        self.calls = 0

    def fingerprint(self) -> str:
        return (f"oracle:{self.name}:{self.temperature}:{self.pos_bias}:{self.similarity}:"
                f"{self.noise}:{self.outside_utility}:{self.score_compress}")

    def _utility(self, user: Optional[int], item: int) -> float:
        return float(self.prior[item] if user is None else self.util[user, item])

    def _taste(self, user: Optional[int], text: str) -> Optional[float]:
        """Ground-truth taste of the user for an attribute named by an `[attr:relation=entity]` tag."""
        hit = _ATTR.search(text or "")
        if not hit or hit.group(1) not in self.taste:
            return None
        if user is None:
            return 0.0
        idx = int(re.sub(r"\D", "", hit.group(2)) or 0)
        return float(self.taste[hit.group(1)][user, idx])

    def decide(self, state: str, questions: Mapping[str, Question]) -> Decision:
        self.calls += 1
        m = _USER.search(state)
        user = int(m.group(1)) if m else None
        answers: dict[str, Answer] = {}
        first = _ITEM.search(state)
        state_item = int(first.group(1)) if first and user is None else None
        for name, q in questions.items():
            if isinstance(q, Choice) and state_item is not None:
                # Item-level questions of the semantic layer. Facets: an option tagged [nest:k] is the true
                # category of the item in the state. Relations: items of one category are of the same series.
                other = _ITEM.search(q.instructions)
                if q.keys() == RELATION_KEYS and other:
                    same = self.group[state_item] == self.group[int(other.group(1))]
                    z = np.array([1.0, 0.5, 2.0, 0.0] if same else [0.0, 0.0, 0.0, 2.5])
                else:
                    z = np.array([3.0 if (m := _NEST.search(desc or "")) and int(m.group(1)) == self.group[state_item]
                                  else 0.0 for _, desc in q.options])
                p = np.exp(z - z.max())
                p /= p.sum()
                answers[name] = Answer(kind="choice", probs={k: float(v) for k, v in zip(q.keys(), p)})
                continue
            if isinstance(q, Noul) and state_item is not None and _NEST.search(q.instructions):
                g = int(_NEST.search(q.instructions).group(1))
                z = 3.0 if self.group[state_item] == g else -3.0
                answers[name] = Answer(kind="noul", p=float(1.0 / (1.0 + math.exp(-z))))
                continue
            if isinstance(q, Choice):
                items = []
                for key, desc in q.options:
                    hit = _ITEM.search(desc or "")
                    items.append(int(hit.group(1)) if hit else None)
                real = [i for i in items if i is not None]
                counts = {g: sum(1 for i in real if self.group[i] == g) for g in {self.group[i] for i in real}}
                logits = []
                pos = 0
                for (key, desc), item in zip(q.options, items):
                    if item is None:
                        taste = self._taste(user, desc)
                        if taste is not None:
                            logits.append(taste / self.temperature)
                        else:
                            logits.append(self.outside_utility if key == OUTSIDE_KEY else -30.0)
                        continue
                    z = self._utility(user, item) / self.temperature
                    z -= self.pos_bias * math.log1p(pos)
                    z -= self.similarity * math.log(counts[self.group[item]])
                    z += _noise(f"{user}:{item}:{sorted(real)}", self.noise)
                    logits.append(z)
                    pos += 1
                z = np.array(logits)
                p = np.exp(z - z.max())
                p /= p.sum()
                answers[name] = Answer(kind="choice", probs={k: float(v) for k, v in zip(q.keys(), p)})
            elif isinstance(q, Noul):
                hit = _ITEM.search(q.instructions)
                taste = self._taste(user, q.instructions)
                if hit:
                    z = self._utility(user, int(hit.group(1))) / self.temperature
                else:
                    z = (taste or 0.0) / self.temperature
                answers[name] = Answer(kind="noul", p=float(1.0 / (1.0 + math.exp(-z))))
            elif isinstance(q, Score):
                # The synthetic rating is 3 + utility / 2 on a 1 to 5 scale; levels are spread over that range.
                n = len(q.levels)
                hit = _ITEM.search(q.instructions)
                mu = 3.0 + (self._utility(user, int(hit.group(1))) / 2.0 if hit and user is not None else 0.0)
                mu = 3.0 + (mu - 3.0) * (1.0 - self.score_compress)
                centres = np.linspace(1.0, 5.0, n)
                z = -((centres - mu) ** 2) / (2 * 0.7 ** 2)
                pr = np.exp(z - z.max())
                pr /= pr.sum()
                answers[name] = Answer(kind="score", probs={str(i): float(v) for i, v in enumerate(pr)},
                                       value=float((np.arange(n) * pr).sum()))
        tokens = (len(state) + sum(len(str(q.to_wire())) for q in questions.values())) // 4
        return Decision(answers=answers, input_tokens=tokens, latency_ms=0.0, n_requests=1)
