"""Typed relations between items, judged by a decision model (E2).

For an anchor item A, the candidates B are its strongest co-occurrence neighbours in the training sequences
and its nearest neighbours by text. One request per anchor: the state is A, and one choice question per B asks
how B relates to A: complement (used together, accessory, add-on, consumable), substitute (one or the other),
same series or product line, or unrelated. These questions are decided by the text of both items.

Evaluation (`relation_eval`): for each user, the related items of the last few history items join the top 50
of SASRec, and the score of a candidate is its standardised SASRec score plus a weighted sum of its relation
probabilities. With weight 0 the ranking is SASRec's. Two controls use the same pairs without the decision
model: the co-occurrence strength of the pair, and the mere presence of the pair. Weights are fitted on
validation users and applied once to test users.
"""

from __future__ import annotations

import itertools
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from ..data.schema import Dataset, load_cases
from ..decision.base import DecisionBackend, decide_chunked
from ..decision.types import Choice
from ..eval import metrics as M

RELATIONS = (
    ("complement", "Used together with the product in the state, or adds to it: an accessory, add-on, part or consumable"),
    ("substitute", "An alternative to the product in the state: a buyer would usually choose one or the other"),
    ("series", "From the same series, franchise or product line as the product in the state"),
    ("unrelated", "None of the above"),
)
TYPES = ("complement", "substitute", "series")


def anchors_of(cases: list, last: int) -> list[int]:
    seen: dict[int, None] = {}
    for c in cases:
        for a in c.history[-last:]:
            seen.setdefault(int(a), None)
    return list(seen)


def candidate_pairs(ds: Dataset, anchors: list[int], cooc: dict, content: Optional[np.ndarray],
                    n_cooc: int = 10, n_content: int = 10) -> dict[int, list[int]]:
    """Up to n_cooc co-occurrence neighbours and n_content text neighbours of each anchor, without repeats."""
    out = {}
    unit = None
    if content is not None:
        unit = content / np.maximum(np.linalg.norm(content, axis=1, keepdims=True), 1e-9)
    for a in anchors:
        near = [j for j, _ in sorted(cooc.get(a, {}).items(), key=lambda t: (-t[1], t[0]))[:n_cooc]]
        if unit is not None and np.any(unit[a]):
            sim = unit @ unit[a]
            sim[a] = -np.inf
            top = np.argpartition(-sim, min(n_content, len(sim) - 1))[: n_content + 1]
            near += [int(j) for j in top[np.argsort(-sim[top], kind="stable")] if np.isfinite(sim[j])][:n_content]
        out[a] = list(dict.fromkeys(j for j in near if j != a))
    return out


def read_relations(path: str) -> dict[int, dict[int, list[float]]]:
    """anchor -> candidate -> probabilities in the order of RELATIONS."""
    out: dict[int, dict[int, list[float]]] = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[int(r["anchor"])] = {int(b): p for b, p in zip(r["cands"], r["probs"])}
    return out


def judge_relations(ds: Dataset, backend: DecisionBackend, pairs: dict[int, list[int]], out_path: str,
                    item_chars: int = 300, other_chars: int = 200, workers: int = 1) -> dict:
    done = set(read_relations(out_path))
    todo = [a for a in pairs if a not in done and pairs[a]]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"[relations] {ds.name}: {len(todo)} anchors to judge, {sum(len(pairs[a]) for a in todo)} pairs", flush=True)
    tokens, t0, last = 0, time.time(), time.time()

    def ask(a: int):
        qs = {f"r{j}": Choice(f"Other product: {ds.item_text(b)[:other_chars]}. How is the other product related "
                              "to the product described in the state?", RELATIONS)
              for j, b in enumerate(pairs[a])}
        return decide_chunked(backend, "Product: " + ds.item_text(a)[:item_chars], qs)

    keys = [k for k, _ in RELATIONS]
    with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(max(1, workers)) as ex:
        for n, (a, dec) in enumerate(zip(todo, ex.map(ask, todo)), 1):
            probs = [[float(dec.answers[f"r{j}"].probs[k]) for k in keys] for j in range(len(pairs[a]))]
            f.write(json.dumps({"anchor": a, "cands": pairs[a], "probs": probs, "tokens": dec.input_tokens}) + "\n")
            tokens += dec.input_tokens
            now = time.time()
            if n == len(todo) or n % 200 == 0 or now - last > 600:
                f.flush()
                last = now
                per = (now - t0) / n
                print(f"[relations] {n}/{len(todo)} anchors, {per:.2f} s per anchor, about "
                      f"{per * (len(todo) - n) / 60:.0f} min left", flush=True)
    return {"anchors_judged": len(todo), "input_tokens": tokens}


def _features(case, rel: dict, cooc: dict, last: int, decay: float):
    """Related candidates of a case and, per candidate, the decayed relation probabilities (3), the
    co-occurrence strength and the pair presence, summed over the last history items."""
    hist = set(case.history)
    acc: dict[int, np.ndarray] = {}
    for rank, a in enumerate(reversed(case.history[-last:])):
        w = decay ** rank
        top = max(cooc.get(a, {}).values(), default=0) or 1
        for b, p in rel.get(int(a), {}).items():
            if b in hist:
                continue
            v = acc.setdefault(b, np.zeros(5))
            v[:3] += w * np.asarray(p[:3])
            v[3] += w * cooc.get(a, {}).get(b, 0) / top
            v[4] += w
    return acc


def relation_eval(cases: dict, retriever, rel: dict, cooc: dict, last: int = 3, decay: float = 0.7,
                  refs: Optional[dict[str, np.ndarray]] = None, tail: Optional[np.ndarray] = None) -> dict:
    """`cases`: {"valid": [...], "test": [...]} with SASRec top-K candidates. `retriever.score_items` gives
    SASRec scores of any item. `refs`: per-test-user NDCG@10 of other methods (same order as cases["test"])."""
    data = {}
    for split, cs in cases.items():
        feats = [_features(c, rel, cooc, last, decay) for c in cs]
        union = [list(c.cands) + [b for b in f if b not in set(c.cands)] for c, f in zip(cs, feats)]
        logits = retriever.score_items([c.history for c in cs], union)
        rows = []
        for c, f, u, lg in zip(cs, feats, union, logits):
            x = np.array([f.get(b, np.zeros(5)) for b in u])
            pos = u.index(c.target) if c.target in u else -1
            rows.append((M.zscore(lg), x, pos, len(c.cands)))
        data[split] = rows

    def ndcg(split: str, w_rel: np.ndarray, w_cooc: float, w_pres: float) -> np.ndarray:
        out = []
        for zl, x, pos, k in data[split]:
            s = zl + x[:, :3] @ w_rel + w_cooc * x[:, 3] + w_pres * x[:, 4]
            out.append(M.per_case_metrics(M.target_rank(s, pos))["ndcg@10"])
        return np.array(out)

    betas = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    mixes = [np.array(m, dtype=float) for m in itertools.product((0.0, 0.5, 1.0), repeat=3) if any(m)]

    def tune(grid, make):
        best, best_v = None, -1.0
        for g in grid:
            v = float(ndcg("valid", *make(g)).mean())
            if v > best_v + 1e-12:
                best, best_v = g, v
        return best

    variants = {"sasrec (top 50)": ((np.zeros(3), 0.0, 0.0), "")}
    b, m = tune([(b, m) for b in betas for m in mixes], lambda g: (g[0] * g[1], 0.0, 0.0))
    variants["relations, decision model"] = ((b * m, 0.0, 0.0), f"beta={b}, complement/substitute/series={m.tolist()}")
    b2 = tune(betas, lambda g: (np.zeros(3), g, 0.0))
    variants["co-occurrence, same pairs"] = ((np.zeros(3), b2, 0.0), f"beta={b2}")
    b3 = tune(betas, lambda g: (np.zeros(3), 0.0, g))
    variants["pair presence, same pairs"] = ((np.zeros(3), 0.0, b3), f"beta={b3}")
    bb = tune([(b, m, c) for b in betas for m in mixes for c in betas], lambda g: (g[0] * g[1], g[2], 0.0))
    variants["relations + co-occurrence"] = ((bb[0] * bb[1], bb[2], 0.0),
                                             f"beta={bb[0]}, mix={bb[1].tolist()}, co-occurrence={bb[2]}")
    test = {name: ndcg("test", *w) for name, (w, _) in variants.items()}
    base = test["sasrec (top 50)"]
    compare = {"sasrec (top 50)": base, "co-occurrence, same pairs": test["co-occurrence, same pairs"], **(refs or {})}
    rows = []
    for name, (w, note) in variants.items():
        v = test[name]
        row = {"variant": name, "weights": note, "ndcg@10": float(v.mean()), "vs": {}}
        for ref, rv in compare.items():
            if ref != name:
                row["vs"][ref] = {"diff": float(v.mean() - rv.mean()), "p": M.paired_randomization(v, rv)}
        if tail is not None and tail.any():
            row["tail_ndcg@10"] = float(v[tail].mean())
            row["tail_vs_sasrec"] = {"diff": float(v[tail].mean() - base[tail].mean()),
                                     "p": M.paired_randomization(v[tail], base[tail])}
        rows.append(row)
    recall = {"top 50": float(np.mean([pos >= 0 and pos < k for _, _, pos, k in data["test"]])),
              "top 50 + related": float(np.mean([pos >= 0 for _, _, pos, _ in data["test"]])),
              "mean related per user": float(np.mean([len(x) - k for _, x, _, k in data["test"]]))}
    return {"rows": rows, "recall": recall, "refs": {k: float(v.mean()) for k, v in (refs or {}).items()}}


def relation_stats(rel: dict, cooc: dict, ds: Dataset) -> dict:
    """How the judged pairs split over the relation types, for co-occurrence and text neighbours."""
    keys = [k for k, _ in RELATIONS]
    top = Counter()
    by_source = {"co-occurrence": Counter(), "text only": Counter()}
    n = 0
    for a, cands in rel.items():
        for b, p in cands.items():
            k = keys[int(np.argmax(p))]
            top[k] += 1
            by_source["co-occurrence" if b in cooc.get(a, {}) else "text only"][k] += 1
            n += 1
    share = lambda c: {k: round(c[k] / max(sum(c.values()), 1), 3) for k in keys}
    return {"pairs": n, "anchors": len(rel), "shares": share(top), "by_source": {s: share(c) for s, c in by_source.items()}}
