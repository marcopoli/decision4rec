"""Any causal language model as a decision backend.

Each question becomes a prompt whose answer is a single label token. The
distribution is the exact next-token probability restricted to the labels, so
no text is generated. The state prefix is encoded once and its key/value cache
is reused by every question, which mirrors the question isolation of native
decision models.
"""

from __future__ import annotations

import copy
import itertools
import string
import time
import warnings
from typing import Mapping, Optional

from .base import DecisionBackend
from .types import Answer, Choice, Decision, Noul, Question, Score

_SENTINEL = "<<DECISION4REC-SPLIT>>"


class LogitBackend(DecisionBackend):
    def __init__(self, model: str, name: Optional[str] = None, device: Optional[str] = None,
                 dtype: str = "bf16", batch_size: int = 4, use_chat_template: bool = True,
                 prefix_cache: bool = True, load_in_4bit: bool = False,
                 max_request_chars: Optional[int] = None, tokenizer=None, lm=None):
        import torch

        self.torch = torch
        self.model_id = model
        self.name = name or model.split("/")[-1]
        self.batch_size = batch_size
        self.use_chat_template = use_chat_template
        self.prefix_cache = prefix_cache
        self.max_request_chars = max_request_chars
        if tokenizer is None or lm is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model)
            kwargs: dict = {}
            if load_in_4bit:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
                kwargs["device_map"] = "auto"
            else:
                kwargs["torch_dtype"] = {"bf16": torch.bfloat16, "fp16": torch.float16,
                                         "fp32": torch.float32}[dtype]
            if not load_in_4bit:
                if device is None:
                    device = "cuda" if torch.cuda.is_available() else (
                        "mps" if torch.backends.mps.is_available() else "cpu")
                if device == "mps":
                    kwargs["torch_dtype"] = torch.float32      # half precision is unreliable on Apple MPS
            lm = AutoModelForCausalLM.from_pretrained(model, **kwargs)
            if not load_in_4bit:
                lm = lm.to(device)
        self.tok = tokenizer
        self.lm = lm.eval()
        self.device = next(self.lm.parameters()).device
        self.pad_id = self.tok.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tok.eos_token_id if self.tok.eos_token_id is not None else 0
        self._label_cache: dict[int, list[str]] = {}
        self._warned = False

    def fingerprint(self) -> str:
        return f"logit:{self.model_id}:chat={int(self.use_chat_template)}:v1"

    # ---- label vocabulary -------------------------------------------------

    def _single(self, text: str) -> Optional[int]:
        ids = self.tok.encode(text, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    def _variants(self, label: str) -> list[int]:
        ids = {self._single(v) for v in (label, " " + label)}
        return sorted(i for i in ids if i is not None)

    def _labels(self, n: int) -> list[str]:
        """n label strings, each a single token and none a prefix of another."""
        if n in self._label_cache:
            return self._label_cache[n]
        if n <= 26:
            pool = list(string.ascii_uppercase)
        else:
            pool = ["".join(p) for p in itertools.product(string.ascii_uppercase, repeat=2)]
        seen: set[int] = set()
        labels: list[str] = []
        for cand in pool:
            ids = self._variants(cand)
            if ids and not (set(ids) & seen):
                labels.append(cand)
                seen.update(ids)
            if len(labels) == n:
                break
        if len(labels) < n:
            raise ValueError(f"tokenizer offers only {len(labels)} single-token labels, {n} requested; "
                             "use smaller blocks")
        self._label_cache[n] = labels
        return labels

    # ---- prompt construction ---------------------------------------------

    def _frame(self, state: str) -> tuple[str, str]:
        """Text before and after the question part, with the state inside the prefix."""
        body = state.rstrip() + "\n\n" + _SENTINEL
        if self.use_chat_template and getattr(self.tok, "chat_template", None):
            text = self.tok.apply_chat_template([{"role": "user", "content": body}],
                                                tokenize=False, add_generation_prompt=True)
        else:
            text = body + "\nAnswer:"
        pre, post = text.split(_SENTINEL)
        return pre, post

    def _question_text(self, q: Question) -> tuple[str, list[list[int]]]:
        if isinstance(q, Choice):
            labels = self._labels(len(q.options))
            lines = [f"{lab}: {desc if desc else key}" for lab, (key, desc) in zip(labels, q.options)]
            text = (f"Question: {q.instructions}\nOptions:\n" + "\n".join(lines)
                    + "\nReply with the letter label of the single best option and nothing else.")
            return text, [self._variants(lab) for lab in labels]
        if isinstance(q, Noul):
            text = f"Question: {q.instructions}\nReply with Yes or No and nothing else."
            yes = sorted(set(self._variants("Yes") + self._variants("yes")))
            no = sorted(set(self._variants("No") + self._variants("no")))
            return text, [yes, no]
        if isinstance(q, Score):
            if len(q.levels) > 9:
                raise ValueError("score questions support at most 9 levels in the logit backend")
            lines = [f"{i + 1}: {lev}" for i, lev in enumerate(q.levels)]
            text = (f"Question: {q.instructions}\nScale (low to high):\n" + "\n".join(lines)
                    + "\nReply with the number of the best matching level and nothing else.")
            return text, [self._variants(str(i + 1)) for i in range(len(q.levels))]
        raise TypeError(f"unsupported question type: {type(q)}")

    # ---- forward passes ---------------------------------------------------

    def _pad(self, rows: list[list[int]]):
        torch = self.torch
        width = max(len(r) for r in rows)
        ids = torch.full((len(rows), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
            mask[i, : len(r)] = 1
        return ids.to(self.device), mask.to(self.device)

    def _last_logits(self, logits, rows: list[list[int]]):
        idx = self.torch.tensor([len(r) - 1 for r in rows], device=logits.device)
        return logits[self.torch.arange(len(rows), device=logits.device), idx].float()

    def _forward_plain(self, prefix_ids: list[int], suffixes: list[list[int]]):
        outs = []
        for s in range(0, len(suffixes), self.batch_size):
            rows = [prefix_ids + suf for suf in suffixes[s: s + self.batch_size]]
            ids, mask = self._pad(rows)
            logits = self.lm(input_ids=ids, attention_mask=mask, use_cache=False).logits
            outs.append(self._last_logits(logits, rows))
        return self.torch.cat(outs, 0)

    def _expand_cache(self, past, n: int):
        past = copy.deepcopy(past)
        if hasattr(past, "batch_repeat_interleave"):
            past.batch_repeat_interleave(n)
            return past
        return tuple(tuple(t.repeat_interleave(n, dim=0) for t in layer) for layer in past)

    def _forward_cached(self, prefix_ids: list[int], suffixes: list[list[int]]):
        torch = self.torch
        pre = torch.tensor([prefix_ids], dtype=torch.long, device=self.device)
        past = self.lm(input_ids=pre, use_cache=True).past_key_values
        outs = []
        for s in range(0, len(suffixes), self.batch_size):
            rows = suffixes[s: s + self.batch_size]
            ids, mask = self._pad(rows)
            full_mask = torch.cat([torch.ones((len(rows), len(prefix_ids)), dtype=torch.long,
                                              device=self.device), mask], dim=1)
            logits = self.lm(input_ids=ids, attention_mask=full_mask,
                             past_key_values=self._expand_cache(past, len(rows)),
                             use_cache=True).logits
            outs.append(self._last_logits(logits, rows))
        return torch.cat(outs, 0)

    def decide(self, state: str, questions: Mapping[str, Question]) -> Decision:
        torch = self.torch
        t0 = time.perf_counter()
        pre, post = self._frame(state)
        prefix_ids = self.tok.encode(pre, add_special_tokens=True)
        names, suffixes, label_ids = [], [], []
        for name, q in questions.items():
            text, labs = self._question_text(q)
            if any(len(ids) == 0 for ids in labs):
                raise ValueError(f"tokenizer has no single-token answer label for question {name!r}")
            names.append(name)
            suffixes.append(self.tok.encode(text + post, add_special_tokens=False))
            label_ids.append(labs)
        with torch.no_grad():
            logits = None
            if self.prefix_cache and len(suffixes) > 1:
                try:
                    logits = self._forward_cached(prefix_ids, suffixes)
                except Exception as e:  # cache layouts differ across transformers versions
                    if not self._warned:
                        warnings.warn(f"prefix cache disabled, falling back to plain batching: {e!r}")
                        self._warned = True
                    self.prefix_cache = False
            if logits is None:
                logits = self._forward_plain(prefix_ids, suffixes)
        answers: dict[str, Answer] = {}
        for row, name, labs in zip(logits, names, label_ids):
            q = questions[name]
            scores = torch.stack([torch.logsumexp(row[torch.tensor(ids, device=row.device)], 0)
                                  for ids in labs])
            probs = torch.softmax(scores, 0).tolist()
            if isinstance(q, Choice):
                answers[name] = Answer(kind="choice", probs=dict(zip(q.keys(), probs)))
            elif isinstance(q, Noul):
                answers[name] = Answer(kind="noul", p=probs[0])
            else:
                answers[name] = Answer(kind="score", probs={str(i): p for i, p in enumerate(probs)},
                                       value=sum(i * p for i, p in enumerate(probs)))
        tokens = len(prefix_ids) + sum(len(s) for s in suffixes)
        return Decision(answers=answers, input_tokens=tokens,
                        latency_ms=(time.perf_counter() - t0) * 1000.0, n_requests=1)
