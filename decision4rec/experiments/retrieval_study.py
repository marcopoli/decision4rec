"""Retrieval study: what each retrieval agent finds, alone and fused.

Answers whether language-model retrieval can replace or should complement the
sequential model. Per agent it reports recall and NDCG on held-out next items,
the targets only that agent finds, and the loss when it is removed from the
fusion. Fusion weights are tuned on validation targets only.
"""

from __future__ import annotations

import json
import os

import numpy as np

from ..data.schema import Dataset
from ..eval.metrics import bootstrap_ci
from ..retrieval.agents import RetrievalContext
from .runner import build_retriever, load_config


def _rank(items: list[int], target: int) -> int:
    return items.index(target) if target in items else 10 ** 9


def _summary(ranks: np.ndarray, cutoffs) -> dict:
    out = {f"recall@{c}": float(np.mean(ranks < c)) for c in cutoffs}
    ndcg = np.where(ranks < 10, 1.0 / np.log2(ranks + 2.0), 0.0)
    out["ndcg@10"] = float(ndcg.mean())
    out["ndcg@10_ci95"] = list(bootstrap_ci(ndcg))
    return out


def run_retrieval_study(cfg_path: str, n_users: int = 1000, cutoffs=(20, 50, 200)) -> dict:
    cfg = load_config(cfg_path)
    rcfg = dict(cfg["retriever"])
    if rcfg.get("kind") != "multi":
        raise SystemExit("the retrieval study needs a config whose retriever has kind: multi")
    ds = Dataset.load(cfg["dataset"])
    rcfg["weights"] = "uniform"
    retriever = build_retriever(rcfg, ds)
    seed = int(cfg["eval"].get("seed", 13))
    rng = np.random.default_rng([seed, 0])
    users = [u for u in rng.permutation(ds.n_users).tolist() if len(ds.seqs[u]) >= 5][:n_users]
    ctxs = [RetrievalContext(history=[int(x[0]) for x in ds.seqs[u][:-1]]) for u in users]
    targets = [int(ds.seqs[u][-1][0]) for u in users]
    static = [a for a in retriever.agents if not a.needs_profile]
    depth = max(cutoffs)
    for a in static:
        a.prepare(ctxs, depth)
    lists = [{a.name: a.propose(c, depth) for a in static} for c in ctxs]
    names = [a.name for a in static]

    agents = {}
    found = {n: np.array([t in per[n] for per, t in zip(lists, targets)]) for n in names}
    for n in names:
        ranks = np.array([_rank(per[n], t) for per, t in zip(lists, targets)])
        others = np.any([found[m] for m in names if m != n], axis=0) if len(names) > 1 else np.zeros(len(users), bool)
        agents[n] = {**_summary(ranks, cutoffs), "unique_hits_share": float(np.mean(found[n] & ~others))}

    def fused(weights: dict) -> np.ndarray:
        return np.array([_rank(retriever.fuse(per, weights, k=depth)[0], t) for per, t in zip(lists, targets)])

    uniform = {n: 1.0 for n in names}
    tune = retriever.tune(ds, n_users=int(rcfg.get("tune_users", 500)), seed=seed, cutoffs=cutoffs)
    tuned = {n: float(tune["weights"].get(n, 1.0)) for n in names}
    fusion = {"uniform": _summary(fused(uniform), cutoffs), "tuned": _summary(fused(tuned), cutoffs)}
    ablation = {f"without_{n}": _summary(fused({**tuned, n: 0.0}), cutoffs) for n in names if len(names) > 1}
    out = {"dataset": ds.name, "n_users": len(users), "agents": agents, "fusion": fusion,
           "leave_one_agent_out": ablation, "tuned_weights": tuned,
           "union_recall": float(np.mean(np.any(list(found.values()), axis=0))),
           "note": "union_recall is the share of targets found by at least one agent within the depth, "
                   "the ceiling any fusion of these agents can reach"}
    out_dir = os.path.join(cfg["out"], cfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "retrieval_study.json"), "w") as f:
        json.dump(out, f, indent=1)
    cols = [f"recall@{c}" for c in cutoffs] + ["ndcg@10"]
    md = [f"# Retrieval study: {ds.name}", "", f"{len(users)} test users, depth {depth}. Union recall: {out['union_recall']:.3f}.",
          "", "| source | " + " | ".join(cols) + " | unique hits |", "|---|" + "---|" * (len(cols) + 1)]
    for n in names:
        md.append(f"| agent: {n} | " + " | ".join(f"{agents[n][c]:.4f}" for c in cols) + f" | {agents[n]['unique_hits_share']:.3f} |")
    for n, r in fusion.items():
        md.append(f"| fusion: {n} | " + " | ".join(f"{r[c]:.4f}" for c in cols) + " | |")
    for n, r in ablation.items():
        md.append(f"| fusion tuned, {n.replace('_', ' ')} | " + " | ".join(f"{r[c]:.4f}" for c in cols) + " | |")
    md += ["", "Tuned weights: " + ", ".join(f"{n}={w:g}" for n, w in tuned.items())]
    with open(os.path.join(out_dir, "retrieval_study.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    return out
