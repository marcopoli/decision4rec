"""Backend protocol, request chunking, content-addressed cache and the backend factory."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from typing import Mapping, Optional

from .types import Decision, Question, merge


class DecisionBackend:
    """A decision model: one state and typed questions in, one distribution per question out."""

    name: str = "backend"
    # Upper bound on characters (state plus questions) per request, None for no limit.
    max_request_chars: Optional[int] = None

    def fingerprint(self) -> str:
        """Identifies everything that can change an answer (model, version, prompt format)."""
        return self.name

    def decide(self, state: str, questions: Mapping[str, Question]) -> Decision:
        raise NotImplementedError

    def close(self) -> None:
        pass


def _wire_size(q: Question) -> int:
    return len(json.dumps(q.to_wire(), ensure_ascii=False))


def decide_chunked(backend: DecisionBackend, state: str,
                   questions: Mapping[str, Question]) -> Decision:
    """Answer all questions, splitting them over several requests if the backend has a size limit.

    Splitting is exact for question-isolated backends: an answer depends on the
    state and its own question only.
    """
    limit = backend.max_request_chars
    if limit is None or not questions:
        return backend.decide(state, questions)
    budget = max(limit - len(state), 1)
    chunks: list[dict[str, Question]] = [{}]
    used = 0
    for name, q in questions.items():
        size = _wire_size(q)
        if chunks[-1] and used + size > budget:
            chunks.append({})
            used = 0
        chunks[-1][name] = q
        used += size
    if len(chunks) == 1:
        return backend.decide(state, questions)
    return merge([backend.decide(state, c) for c in chunks])


def request_key(fingerprint: str, state: str, questions: Mapping[str, Question]) -> str:
    payload = {"b": fingerprint, "s": state,
               "q": [[n, q.to_wire()] for n, q in questions.items()]}
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class CachedBackend(DecisionBackend):
    """SQLite cache keyed by (backend fingerprint, state, questions).

    Decision models do not sample, so a cached answer equals a fresh one. The
    cache makes runs resumable and replays exact.
    """

    def __init__(self, inner: DecisionBackend, path: str):
        self.inner = inner
        self.name = inner.name
        self.max_request_chars = inner.max_request_chars
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS decisions (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        self._db.commit()
        self.hits = 0
        self.misses = 0

    def fingerprint(self) -> str:
        return self.inner.fingerprint()

    def decide(self, state: str, questions: Mapping[str, Question]) -> Decision:
        key = request_key(self.fingerprint(), state, questions)
        with self._lock:
            row = self._db.execute("SELECT v FROM decisions WHERE k = ?", (key,)).fetchone()
        if row is not None:
            self.hits += 1
            d = Decision.from_json(json.loads(row[0]))
            d.cached = True
            return d
        d = self.inner.decide(state, questions)
        self.misses += 1
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO decisions (k, v) VALUES (?, ?)",
                             (key, json.dumps(d.to_json())))
            self._db.commit()
        return d

    def close(self) -> None:
        with self._lock:
            self._db.close()
        self.inner.close()


def build_backend(cfg: dict) -> DecisionBackend:
    """Build a backend from a config dict. `kind` selects the implementation."""
    cfg = dict(cfg)
    kind = cfg.pop("kind")
    cache = cfg.pop("cache", None)
    if kind == "systemone":
        from .systemone import SystemOneBackend
        backend: DecisionBackend = SystemOneBackend(**cfg)
    elif kind == "logit":
        from .logit import LogitBackend
        backend = LogitBackend(**cfg)
    elif kind == "oracle":
        from .mock import OracleBackend
        backend = OracleBackend(**cfg)
    else:
        raise ValueError(f"unknown backend kind: {kind!r}")
    if cache:
        backend = CachedBackend(backend, cache)
    return backend
