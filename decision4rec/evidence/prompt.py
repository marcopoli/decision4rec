"""Prompts of the evidence-reading decision model: the usual history and item text, plus the store's evidence.

The state gains one line on how much purchase data the store has and a legend of the statements; every option
gains a short block of statements when the arm asks for `evidence: stats`. Training records (`export-evidence`)
and evaluation runs use this same builder, so the model reads at test time exactly what it learned to read.
"""

from __future__ import annotations

import threading
from typing import Optional

from ..data.schema import Case
from ..prompts import PromptBuilder
from .store import LEGEND, Context, EvidenceStore


class EvidencePromptBuilder(PromptBuilder):
    def __init__(self, ds, store: EvidenceStore, pool=None, **kw):
        super().__init__(ds, None, **kw)
        self.store, self.pool = store, pool          # pool: the multi-agent pool, for the "found by" statement
        self.drop: frozenset = frozenset()          # statement families left out (training-time dropout)
        self._last = threading.local()              # the context of the last case, per thread (runs use workers)

    def context(self, case: Case) -> Context:
        key = (case.user, len(case.history), case.target)
        if getattr(self._last, "key", None) != key:
            self._last.ctx = self.store.context(case.history, self.store.now(case.user, len(case.history)))
            self._last.key = key
        return self._last.ctx

    def state(self, case: Case, null_state: bool = False, **kw) -> str:
        s = super().state(case, null_state=null_state, **kw)
        return s if null_state else f"{s}\n{self.store.regime_line()}\n{LEGEND}"

    def option(self, case: Case, item: int, evidence: str = "none", mask_titles: bool = False,
               facts: Optional[list[str]] = None, prof=None) -> str:
        text = super().option(case, item, "none", mask_titles, facts, prof)
        kinds = (evidence or "none").split("+")
        if "stats" in kinds:
            tags = self.store.tags(self.context(case), item, self.drop)
            if "found" in kinds and self.pool is not None:
                found = self.found(case, item)
                if found:
                    tags = (tags[:-1] + "; " + found + "]") if tags else f"[{found}]"
            if tags:
                text += " " + tags
        return text

    def found(self, case: Case, item: int) -> str:
        """Which agents of the pool proposed the item, ranked once per case."""
        key = (case.user, len(case.history))
        if key not in self.pool._found:
            self.pool.rank(case.user, case.history, max(len(case.cands), 30))
        return self.pool.found_by(case.user, len(case.history), item)
