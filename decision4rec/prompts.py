"""State and option serialisation. Every string the decision model reads is built here."""

from __future__ import annotations

from collections import Counter
from typing import Optional

from .data.schema import Case, Dataset
from .decision.base import DecisionBackend
from .decision.types import OUTSIDE_KEY, Choice, Noul
from .graph.kg import ItemKG
from .graph.profile import OTHER_KEY, Profile, ask_intent, ask_preferences, pref_facts

# Instruction of arXiv:2609.40241, kept verbatim for comparability.
INSTR_CHOICE = ("Based on the user's interaction history, which candidate item is the user most likely "
                "to interact with next?")
INSTR_NOUL = "Based on the user's interaction history, will the user interact with this item next? Item: {item}"
OUTSIDE_DESC = "None of the listed candidates"
NULL_STATE = "No interaction history is available for this user."


class PromptBuilder:
    def __init__(self, ds: Dataset, kg: Optional[ItemKG] = None, hist_len: int = 10, max_facts: int = 3,
                 item_chars: int = 400, profile_hist: int = 30, profile_item_chars: int = 120, max_h: int = 24,
                 intent_relation: Optional[str] = None, max_intents: int = 10):
        self.ds, self.kg = ds, kg
        self.hist_len, self.max_facts, self.item_chars = hist_len, max_facts, item_chars
        self.profile_hist, self.profile_item_chars, self.max_h = profile_hist, profile_item_chars, max_h
        self.max_intents = max_intents
        self.intent_relation = intent_relation or self._default_relation()
        self._memo: dict = {}

    def _default_relation(self) -> Optional[str]:
        if self.kg is None:
            return None
        rels = Counter(rel for rel, _ in self.kg.idf)
        for name in ("category", "genre"):
            if rels.get(name):
                return name
        return rels.most_common(1)[0][0] if rels else None

    # ---- history views ---------------------------------------------------

    def history(self, case: Case) -> list[int]:
        return case.history[-self.hist_len:]

    def profile_history(self, case: Case, drop: Optional[int] = None) -> list[int]:
        return [h for h in case.history[-self.profile_hist:] if h != drop]

    def _compact(self, item: int, mask_titles: bool) -> str:
        it = self.ds.items[item]
        parts = ([] if mask_titles else [it.get("title") or ""]) + [it.get("meta") or ""]
        return (" | ".join(p for p in parts if p) or f"item {item}")[: self.profile_item_chars]

    def profile_state(self, case: Case, mask_titles: bool = False, drop: Optional[int] = None,
                      history: Optional[list[int]] = None) -> str:
        """Long, compact history: the state on which the user model is estimated. `history` overrides
        the case history, used by the agent whose history lives on the graph."""
        hist = self.profile_history(case, drop) if history is None else history
        lines = ["User interaction history (oldest to newest):"]
        lines += [f"{n}. {self._compact(h, mask_titles)}" for n, h in enumerate(hist, 1)]
        if self.ds.user_tag:
            lines.append(f"[user:{case.user}]")
        return "\n".join(lines)

    # ---- decision-native user model ---------------------------------------

    def user_model(self, backend: DecisionBackend, case: Case, mask_titles: bool = False,
                   drop: Optional[int] = None) -> Profile:
        """Preference edges of the user: hypotheses from the graph, strengths from the decision model."""
        key = ("prof", backend.fingerprint(), case.user, len(case.history), mask_titles, drop)
        if key not in self._memo:
            hyps = self.kg.hypotheses(self.profile_history(case, drop), self.max_h)
            self._memo[key] = ask_preferences(backend, self.profile_state(case, mask_titles, drop), hyps,
                                              self.kg.idf, tag=self.ds.user_tag)
        return self._memo[key]

    def intent(self, backend: DecisionBackend, case: Case, mask_titles: bool = False,
               drop: Optional[int] = None) -> tuple[dict[str, float], int, int, float]:
        """Distribution over the nests (categories, genres) present among the candidates."""
        key = ("intent", backend.fingerprint(), case.user, len(case.history), tuple(case.cands), mask_titles, drop)
        if key not in self._memo:
            nests = self.kg.nests(list(case.cands) + self.history(case)[-5:], self.intent_relation, self.max_intents)
            self._memo[key] = ask_intent(backend, self.profile_state(case, mask_titles, drop), nests,
                                         self.intent_relation, tag=self.ds.user_tag)
        return self._memo[key]

    # ---- ranking state and options ----------------------------------------

    def state(self, case: Case, null_state: bool = False, profile: str = "none", mask_titles: bool = False,
              drop: Optional[int] = None, prof: Optional[Profile] = None,
              intent: Optional[dict[str, float]] = None, ratings: Optional[dict[int, float]] = None) -> str:
        """`profile` is none, count (attribute frequencies) or decision (model-estimated preference edges).
        `drop` removes one history item, used for counterfactuals."""
        if null_state:
            return NULL_STATE
        hist = [h for h in self.history(case) if h != drop]
        lines = []
        if profile == "decision" and prof is not None:
            top = prof.top()
            if top:
                lines.append("User preference profile (estimated strength from 0 to 1):")
                lines += [f"- {q.rel} '{q.ent}': {q.p:.2f}" for q in top]
        elif profile == "count" and self.kg is not None:
            summary = self.kg.profile(hist)
            if summary:
                lines.append(f"Recurring in the history: {summary}")
        if intent:
            best = sorted(((p, n) for n, p in intent.items() if n != OTHER_KEY), reverse=True)[:3]
            if best:
                lines.append("Likely next kind of item: " + ", ".join(f"{n} ({p:.2f})" for p, n in best))
        lines.append("User interaction history (oldest to newest):")
        for n, h in enumerate(hist, 1):
            line = f"{n}. {self.ds.item_text(h, mask_titles)[: self.item_chars]}"
            if ratings is not None and h in ratings:
                line += f" (rated {int(round(ratings[h]))}/5)"
            lines.append(line)
        if self.ds.user_tag:
            lines.append(f"[user:{case.user}]")
        return "\n".join(lines)

    def facts(self, case: Case, item: int, evidence: str = "none", mask_titles: bool = False,
              prof: Optional[Profile] = None) -> list[str]:
        """Evidence for a candidate. Kinds are joined with '+': pref (preference paths estimated by the
        decision model), kg (attributes shared with the history), cf (co-consumption)."""
        if evidence == "none" or self.kg is None:
            return []
        kinds = set(evidence.split("+"))
        scored: list[tuple[str, float]] = []
        if "pref" in kinds:
            scored += pref_facts(self.kg.attrs[item], prof, self.max_facts)
        rest = "+".join(k for k in ("kg", "cf") if k in kinds)
        if rest:
            scored += self.kg.facts(item, self.history(case), rest, self.max_facts, mask_titles)
        if "pref" in kinds:           # preference paths first, then the remaining evidence
            return [t for t, _ in scored][: self.max_facts]
        return [t for t, _ in sorted(scored, key=lambda t: -t[1])][: self.max_facts]

    def option(self, case: Case, item: int, evidence: str = "none", mask_titles: bool = False,
               facts: Optional[list[str]] = None, prof: Optional[Profile] = None) -> str:
        text = self.ds.item_text(item, mask_titles)[: self.item_chars]
        if facts is None:
            facts = self.facts(case, item, evidence, mask_titles, prof)
        if facts:
            text += " || Evidence: " + "; ".join(facts)
        if self.ds.user_tag and "[item:" not in text:
            text += f" [item:{item}]"
        return text

    def choice(self, case: Case, block: list[int], outside: bool = False, **opt) -> Choice:
        """A choice over `block`, positions of `case.cands` in presentation order. Keys are neutral."""
        options = [(f"o{k + 1:02d}", self.option(case, case.cands[ci], **opt)) for k, ci in enumerate(block)]
        if outside:
            options.append((OUTSIDE_KEY, OUTSIDE_DESC))
        return Choice(instructions=INSTR_CHOICE, options=tuple(options))

    def noul(self, case: Case, item: int, **opt) -> Noul:
        return Noul(instructions=INSTR_NOUL.format(item=self.option(case, item, **opt)))
