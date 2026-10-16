"""Item facets answered once per item by a decision model (E1).

A schema (configs/facets/*.yaml) lists typed questions about a product: `choice` questions with a fixed set
of options and `noul` statements. Each item is one request whose state is the item text. The answers are
stored as one JSON line per item, so an interrupted run resumes where it stopped.

The answers become attributes of SASRecF (`sasrecf_dm`): the most likely option of a choice facet when its
probability reaches the schema threshold, and the facet itself for a noul statement judged true. A control
assigns the same choice facets by text-embedding similarity (`sasrecf_emb`), so that the gain of the decision
model can be told apart from the gain of any text model.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import yaml

from ..data.schema import Dataset
from ..decision.base import DecisionBackend, decide_chunked
from ..decision.types import Choice, Noul, Question


@dataclass
class Facet:
    name: str
    kind: str                                   # "choice" or "noul"
    instructions: str
    options: list[tuple[str, str]] = field(default_factory=list)
    match: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class Schema:
    domain: str
    facets: list[Facet]
    item_chars: int = 400
    threshold: float = 0.4

    def questions(self) -> dict[str, Question]:
        return {f.name: (Choice(f.instructions, tuple(f.options)) if f.kind == "choice" else Noul(f.instructions))
                for f in self.facets}


def load_schema(path: str) -> Schema:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    facets = []
    for q in raw["questions"]:
        kind = q["type"]
        if kind not in ("choice", "noul"):
            raise ValueError(f"{path}: facet {q['name']} has type {kind!r}; only choice and noul are supported")
        opts = [(str(k), str(v)) for k, v in (q.get("options") or {}).items()]
        if kind == "choice" and len(opts) < 2:
            raise ValueError(f"{path}: choice facet {q['name']} needs at least two options")
        facets.append(Facet(q["name"], kind, q["instructions"], opts,
                            {str(k): [str(w) for w in v] for k, v in (q.get("match") or {}).items()}))
    return Schema(raw.get("domain", ""), facets, int(raw.get("item_chars", 400)), float(raw.get("threshold", 0.4)))


def item_state(ds: Dataset, i: int, chars: int) -> str:
    return "Product: " + ds.item_text(i)[:chars]


def read_answers(path: str) -> dict[int, dict]:
    """item -> {facet: {option: p}} for choice facets, {facet: p} for noul facets."""
    out: dict[int, dict] = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[int(r["item"])] = r["answers"]
    return out


def compile_facets(ds: Dataset, backend: DecisionBackend, schema: Schema, out_path: str, workers: int = 1,
                   limit: Optional[int] = None) -> dict:
    """Ask the schema about every item not yet in `out_path`. Returns counts and cost."""
    questions = schema.questions()
    done = set(read_answers(out_path))
    todo = [i for i in range(ds.n_items) if i not in done][: limit]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tokens, t0, last = 0, time.time(), time.time()
    print(f"[facets] {ds.name}: {len(todo)} of {ds.n_items} items to ask, {len(questions)} questions each", flush=True)

    def ask(i: int):
        return decide_chunked(backend, item_state(ds, i, schema.item_chars), questions)

    with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(max(1, workers)) as ex:
        for n, (i, dec) in enumerate(zip(todo, ex.map(ask, todo)), 1):
            ans = {}
            for fc in schema.facets:
                a = dec.answers[fc.name]
                ans[fc.name] = {k: float(v) for k, v in a.probs.items()} if fc.kind == "choice" else float(a.p)
            f.write(json.dumps({"item": i, "answers": ans, "tokens": dec.input_tokens}) + "\n")
            tokens += dec.input_tokens
            now = time.time()
            if n == len(todo) or n % 500 == 0 or now - last > 600:
                f.flush()
                last = now
                per = (now - t0) / n
                print(f"[facets] {n}/{len(todo)} items, {per:.2f} s per item, about "
                      f"{per * (len(todo) - n) / 60:.0f} min left", flush=True)
    return {"items_asked": len(todo), "items_total": len(read_answers(out_path)), "input_tokens": tokens}


def embedding_answers(ds: Dataset, schema: Schema, item_matrix: np.ndarray, embedder) -> dict[int, dict]:
    """Control: choice facets assigned by cosine similarity between the item text and the option texts.
    Noul facets have no embedding counterpart and are left out. Probabilities are a softmax of the similarity
    with temperature 0.05; only the most likely option is used downstream."""
    out: dict[int, dict] = {i: {} for i in range(ds.n_items)}
    norms = np.linalg.norm(item_matrix, axis=1, keepdims=True)
    items = item_matrix / np.maximum(norms, 1e-9)
    for fc in schema.facets:
        if fc.kind != "choice":
            continue
        opt = embedder.encode([f"{fc.instructions} {desc}" for _, desc in fc.options], is_query=True)
        opt = opt / np.maximum(np.linalg.norm(opt, axis=1, keepdims=True), 1e-9)
        sim = items @ opt.T / 0.05
        p = np.exp(sim - sim.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        for i in range(ds.n_items):
            if norms[i, 0] > 0:
                out[i][fc.name] = {k: float(v) for (k, _), v in zip(fc.options, p[i])}
    return out


def write_answers(answers: dict[int, dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i in sorted(answers):
            f.write(json.dumps({"item": i, "answers": answers[i], "tokens": 0}) + "\n")


def facet_attrs(answers: dict[int, dict], schema: Schema, n_items: int, threshold: Optional[float] = None,
                prefix: str = "dm") -> list[list[list[str]]]:
    """Attributes per item: [f"{prefix}:{facet}", option] for a confident choice, [.., "yes"] for a true noul."""
    thr = schema.threshold if threshold is None else threshold
    out: list[list[list[str]]] = [[] for _ in range(n_items)]
    for i, ans in answers.items():
        for fc in schema.facets:
            a = ans.get(fc.name)
            if a is None:
                continue
            if fc.kind == "choice":
                key = max(a, key=a.get)
                if a[key] >= thr:
                    out[i].append([f"{prefix}:{fc.name}", key])
            elif a >= 0.5:
                out[i].append([f"{prefix}:{fc.name}", "yes"])
    return out


def attrs_from_config(fcfg: dict, ds: Dataset) -> dict[str, list]:
    """The facet attributes named by a config's `facets:` section, for the baselines that read them."""
    schema = load_schema(fcfg["schema"])
    out = {}
    for source in ("dm", "emb"):
        path = fcfg.get(source)
        if path and os.path.exists(path):
            answers = read_answers(path)
            # the embedding control assigns its most likely option whatever its probability
            out[source] = facet_attrs(answers, schema, ds.n_items, threshold=0.0 if source == "emb" else None,
                                      prefix=source)
    return out


def fingerprint(attrs: list) -> str:
    return hashlib.sha1(json.dumps(attrs, sort_keys=True).encode()).hexdigest()[:10]


def _silver(ds: Dataset, i: int, fc: Facet) -> Optional[str]:
    """The option named by the raw metadata of item i, if exactly one option is named."""
    text = " ; ".join(str(v) for _, v in ds.items[i].get("attrs", [])).lower()
    hits = [k for k, words in fc.match.items()
            if any(re.search(rf"(?<![a-z0-9]){re.escape(w.lower())}(?![a-z0-9])", text) for w in words)]
    return hits[0] if len(hits) == 1 else None


def facet_summary(ds: Dataset, schema: Schema, answers: dict[int, dict], label: str) -> dict:
    """Coverage, abstention, option shares and, where the schema has `match` words, agreement with metadata."""
    out = {"label": label, "items": len(answers), "facets": {}}
    for fc in schema.facets:
        vals = [a[fc.name] for a in answers.values() if fc.name in a]
        if not vals:
            continue
        if fc.kind == "noul":
            p = np.array(vals)
            out["facets"][fc.name] = {"kind": "noul", "share_true": float((p >= 0.5).mean()), "mean_p": float(p.mean())}
            continue
        top = [max(a, key=a.get) for a in vals]
        conf = np.array([a[k] for a, k in zip(vals, top)])
        keys, counts = np.unique(top, return_counts=True)
        info = {"kind": "choice", "mean_top_p": float(conf.mean()),
                "abstain_rate": float((conf < schema.threshold).mean()),
                "shares": {str(k): float(c / len(top)) for k, c in sorted(zip(keys, counts), key=lambda t: -t[1])}}
        if fc.match:
            pairs = [(max(answers[i][fc.name], key=answers[i][fc.name].get), s) for i in answers
                     if fc.name in answers[i] and (s := _silver(ds, i, fc)) is not None]
            info["metadata_check"] = {"items_with_label": len(pairs),
                                      "agreement": float(np.mean([a == s for a, s in pairs])) if pairs else None}
        out["facets"][fc.name] = info
    return out
