"""HTTP backend for any server that speaks the System One protocol.

Works with the hosted TypeSafe endpoint, OpenRouter, and local Kev servers
(`python -m kev.serve`), which expose the same `POST /v1/systemone` contract.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Mapping, Optional

from .base import DecisionBackend
from .types import Answer, Choice, Decision, Noul, Question, Score


class ProtocolError(RuntimeError):
    pass


class SystemOneBackend(DecisionBackend):
    # Spending guard for paid endpoints: JEV_MAX_MTOK caps the input tokens one process may send (millions).
    _sent = 0
    _lock = threading.Lock()

    def __init__(self, base_url: str, model: str, name: Optional[str] = None,
                 api_key: Optional[str] = None, api_key_env: Optional[str] = None,
                 path: str = "/v1/systemone", timeout: float = 300.0, max_retries: int = 6,
                 max_request_chars: Optional[int] = None, version_tag: str = ""):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = name or model
        self.path = path
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_request_chars = max_request_chars
        self.version_tag = version_tag
        self.api_key = api_key or (os.environ.get(api_key_env) if api_key_env else None)

    def fingerprint(self) -> str:
        # The URL is deliberately excluded: the same weights behind another port are the same model.
        return f"systemone:{self.name}:{self.model}:{self.version_tag}"

    def _post(self, body: dict) -> dict:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = self.base_url + self.path
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:500]
                retryable = e.code == 429 or e.code >= 500
                if not retryable or attempt == self.max_retries:
                    raise ProtocolError(f"HTTP {e.code} from {url}: {detail}") from e
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == self.max_retries:
                    raise ProtocolError(f"cannot reach {url}: {e}") from e
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
        raise ProtocolError(f"no response from {url}")

    def decide(self, state: str, questions: Mapping[str, Question]) -> Decision:
        body = {"state": state, "model": self.model,
                "questions": {n: q.to_wire() for n, q in questions.items()}}
        cap = float(os.environ.get("JEV_MAX_MTOK") or 0)
        if cap and SystemOneBackend._sent > cap * 1e6:
            raise ProtocolError(f"token budget of {cap}M input tokens reached; raise JEV_MAX_MTOK to continue")
        t0 = time.perf_counter()
        raw = self._post(body)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        answers = parse_answers(raw, questions)
        usage = raw.get("usage") or {}
        tokens = int(usage.get("input_tokens") or 0)
        if tokens == 0:
            tokens = (len(state) + sum(len(json.dumps(q.to_wire())) for q in questions.values())) // 4
        with SystemOneBackend._lock:
            SystemOneBackend._sent += tokens
        return Decision(answers=answers, input_tokens=tokens,
                        latency_ms=float(raw.get("latency_ms") or wall_ms), n_requests=1,
                        extra={"wall_ms": wall_ms})


def parse_answers(raw: dict, questions: Mapping[str, Question]) -> dict[str, Answer]:
    """Validate a System One response against the questions that were asked."""
    got = raw.get("answers")
    if not isinstance(got, dict):
        raise ProtocolError(f"response has no 'answers' object: {str(raw)[:300]}")
    out: dict[str, Answer] = {}
    for name, q in questions.items():
        a = got.get(name)
        if not isinstance(a, dict):
            raise ProtocolError(f"missing answer for question {name!r}")
        if isinstance(q, Choice):
            probs = a.get("probabilities")
            if not isinstance(probs, dict):
                raise ProtocolError(f"choice answer {name!r} has no 'probabilities'")
            keys = q.keys()
            unknown = set(probs) - set(keys)
            if unknown:
                raise ProtocolError(f"choice answer {name!r} returned unknown options {sorted(unknown)[:5]}")
            out[name] = Answer(kind="choice", probs={k: float(probs.get(k, 0.0)) for k in keys})
        elif isinstance(q, Noul):
            if "noul" not in a:
                raise ProtocolError(f"noul answer {name!r} has no 'noul' field")
            out[name] = Answer(kind="noul", p=float(a["noul"]))
        elif isinstance(q, Score):
            probs = a.get("probabilities") or {}
            p = {str(i): float(probs.get(str(i), 0.0)) for i in range(len(q.levels))}
            # The expected level (0-based) comes from the probabilities. Servers differ in how they scale
            # their own "score" field (Kev rounds it like a probability), so it is used only as a fallback.
            value = sum(int(k) * v for k, v in p.items()) if probs else a.get("score")
            if value is None:
                raise ProtocolError(f"score answer {name!r} has neither 'probabilities' nor 'score'")
            out[name] = Answer(kind="score", probs=p, value=float(value))
        else:
            raise TypeError(f"unsupported question type: {type(q)}")
    return out
