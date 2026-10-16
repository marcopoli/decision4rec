"""Small language models used by the retrieval agents.

Two roles. An *embedder* maps text to a vector for dense retrieval. A *writer*
generates one short hypothetical description of the user's next item from a
given point of view (generate-then-retrieve). Both are swappable by config and
both have dependency-free stand-ins so the pipeline can be tested without
downloading a model.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
import zlib
from typing import Optional

import numpy as np

_WORD = re.compile(r"[a-z0-9]+")


class Embedder:
    tag = "embedder"
    query_prefix = ""

    def __init__(self) -> None:
        self._memo: dict[str, np.ndarray] = {}

    def _encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """L2-normalised vectors, memoised per text."""
        keys = [(self.query_prefix + t) if is_query else t for t in texts]
        missing = [k for k in dict.fromkeys(keys) if k not in self._memo]
        if missing:
            for k, v in zip(missing, self._encode(missing)):
                self._memo[k] = v
        return np.stack([self._memo[k] for k in keys]) if keys else np.zeros((0, 1), dtype=np.float32)


class HashingEmbedder(Embedder):
    """Signed feature hashing of word unigrams and bigrams. No model, deterministic."""

    def __init__(self, dim: int = 512):
        super().__init__()
        self.dim = dim
        self.tag = f"hash{dim}"

    def _encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, text in enumerate(texts):
            words = _WORD.findall(text.lower())
            for tok in words + [f"{a}_{b}" for a, b in zip(words, words[1:])]:
                h = zlib.crc32(tok.encode())
                out[r, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norm, 1e-9)


class HFEmbedder(Embedder):
    """A small encoder from Hugging Face, for example BAAI/bge-small-en-v1.5 (33M parameters)."""

    def __init__(self, model: str, pooling: str = "cls", max_length: int = 256, batch_size: int = 128,
                 query_prefix: str = "", device: Optional[str] = None, dtype: Optional[str] = None):
        super().__init__()
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tag = model.split("/")[-1]
        self.pooling, self.max_length, self.batch_size, self.query_prefix = pooling, max_length, batch_size, query_prefix
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model)
        load = {"dtype": {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]} if dtype else {}
        self.lm = AutoModel.from_pretrained(model, **load).to(device).eval()   # large encoders: dtype bf16

    def _encode(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.no_grad():
            for s in range(0, len(texts), self.batch_size):
                enc = self.tok(texts[s: s + self.batch_size], padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt").to(self.device)
                hidden = self.lm(**enc).last_hidden_state
                if self.pooling == "cls":
                    vec = hidden[:, 0]
                elif self.pooling == "last":     # the last real token; Qwen3-Embedding pads on the left
                    if self.tok.padding_side == "left":
                        vec = hidden[:, -1]
                    else:
                        last = enc["attention_mask"].sum(1) - 1
                        vec = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
                else:
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
                out.append(torch.nn.functional.normalize(vec.float(), dim=-1).cpu().numpy())
        return np.concatenate(out, 0).astype(np.float32)


class Writer:
    """Generates one short text per prompt. Results are cached, so a rerun reproduces the same queries."""

    tag = "writer"

    def __init__(self, cache: Optional[str] = None):
        self._memo: dict[str, str] = {}
        self._lock = threading.Lock()
        self._db = None
        if cache:
            self._db = sqlite3.connect(cache, check_same_thread=False)
            self._db.execute("CREATE TABLE IF NOT EXISTS texts (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
            self._db.commit()

    def _key(self, prompt: str) -> str:
        return hashlib.sha256(f"{self.tag}\n{prompt}".encode()).hexdigest()

    def _generate(self, prompts: list[str]) -> list[str]:
        raise NotImplementedError

    def write(self, prompts: list[str]) -> list[str]:
        missing = []
        for p in dict.fromkeys(prompts):
            if p in self._memo:
                continue
            row = None
            if self._db is not None:
                with self._lock:
                    row = self._db.execute("SELECT v FROM texts WHERE k = ?", (self._key(p),)).fetchone()
            if row is not None:
                self._memo[p] = row[0]
            else:
                missing.append(p)
        if missing:
            for p, text in zip(missing, self._generate(missing)):
                self._memo[p] = text
                if self._db is not None:
                    with self._lock:
                        self._db.execute("INSERT OR REPLACE INTO texts (k, v) VALUES (?, ?)", (self._key(p), text))
            if self._db is not None:
                with self._lock:
                    self._db.commit()
        return [self._memo[p] for p in prompts]


class EchoWriter(Writer):
    """Stand-in for tests: returns the item lines of the prompt, most recent first."""

    tag = "echo"

    def _generate(self, prompts: list[str]) -> list[str]:
        return [" ".join(reversed([ln[2:] for ln in p.splitlines() if ln.startswith("- ")][-3:])) for p in prompts]


class HFWriter(Writer):
    """A small instruction-tuned generator, for example Qwen/Qwen2.5-1.5B-Instruct. Greedy decoding."""

    def __init__(self, model: str, max_new_tokens: int = 48, batch_size: int = 16, dtype: str = "bf16",
                 device: Optional[str] = None, cache: Optional[str] = None):
        super().__init__(cache)
        import torch

        self.torch = torch
        self.tag = f"{model.split('/')[-1]}:{max_new_tokens}"
        self.max_new_tokens, self.batch_size = max_new_tokens, batch_size
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self._model, self._dtype = model, dtype
        self.tok = self.lm = None            # loaded on the first prompt that is not in the cache

    def _load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch = self.torch
        self.tok = AutoTokenizer.from_pretrained(self._model)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        kind = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[self._dtype]
        if self.device == "mps":
            kind = torch.float32            # half precision is unreliable on Apple MPS
        self.lm = AutoModelForCausalLM.from_pretrained(self._model, torch_dtype=kind).to(self.device).eval()

    def _generate(self, prompts: list[str]) -> list[str]:
        if self.lm is None:
            self._load()
        torch = self.torch
        out = []
        with torch.no_grad():
            for s in range(0, len(prompts), self.batch_size):
                chunk = prompts[s: s + self.batch_size]
                if getattr(self.tok, "chat_template", None):
                    chunk = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                          add_generation_prompt=True) for p in chunk]
                enc = self.tok(chunk, return_tensors="pt", padding=True).to(self.device)
                gen = self.lm.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                       pad_token_id=self.tok.pad_token_id)
                for row in gen[:, enc["input_ids"].shape[1]:]:
                    out.append(self.tok.decode(row, skip_special_tokens=True).strip().split("\n")[0])
        return out


def build_embedder(cfg: Optional[dict]) -> Embedder:
    cfg = dict(cfg or {"kind": "hashing"})
    kind = cfg.pop("kind")
    if kind == "hashing":
        return HashingEmbedder(**cfg)
    if kind == "hf":
        return HFEmbedder(**cfg)
    raise ValueError(f"unknown embedder kind {kind!r}")


def build_writer(cfg: Optional[dict]) -> Optional[Writer]:
    if not cfg or cfg.get("kind", "none") == "none":
        return None
    cfg = dict(cfg)
    kind = cfg.pop("kind")
    if kind == "echo":
        return EchoWriter(cfg.get("cache"))
    if kind == "hf":
        return HFWriter(**cfg)
    raise ValueError(f"unknown writer kind {kind!r}")
