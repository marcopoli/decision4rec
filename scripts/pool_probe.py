"""Does a multi-agent candidate pool find more targets than SASRecF alone? Development validation users.

    python scripts/pool_probe.py configs/small/vg_ev2.yaml configs/small/vg_semantic.yaml [n_users]

Agents: SASRecF (trained), the store's fused evidence and its single generators (transitions, co-purchase, text,
popular now), a knowledge-graph agent (items that share rare catalogue attributes with the last three items) and
popularity. Validation cases of the first evidence pilot (history s[:-2], target s[-2]); all statistics leave out
the last two items of every user. Weights of the weighted reciprocal rank fusion are tuned on the first half of the
users by coordinate search and reported on the second half, so the reported recall is not tuned on itself.
"""

import math
import sys
from collections import Counter

import numpy as np
import torch

from decision4rec.data.schema import Dataset, load_cases
from decision4rec.evidence.store import store_for
from decision4rec.experiments.benchmark import prepare_dataset
from decision4rec.experiments.runner import case_paths, load_config
from decision4rec.experiments.stage1 import chosen_dir
from decision4rec.graph.kg import ItemKG
from decision4rec.models import baselines as B
from decision4rec.retrieval.agents import rrf

DEPTH = 100


def kg_agent(kg: ItemKG, h: list[int], k: int) -> list[int]:
    score: Counter = Counter()
    seen = set(h)
    for s in h[-3:]:
        for key in kg.attrs[s]:
            w = kg.idf.get(key)
            if w and len(kg.ent_items[key]) <= 200:
                for j in kg.ent_items[key]:
                    if j not in seen:
                        score[j] += w
    return [j for j, _ in score.most_common(k)]


def main(ev_path: str, ref_path: str, n: int) -> None:
    cfg, ref = load_config(ev_path), load_config(ref_path)
    ds = Dataset.load(cfg["dataset"])
    store = store_for(cfg, ds)
    kg = ItemKG(ds)
    prepare_dataset(ref, ds, ["sasrecf"])
    scorer = B.load("sasrecf", chosen_dir(ref, "sasrecf"), ds, "cpu")
    cases = load_cases(case_paths({**cfg, "retriever": {"tag": "ev100"}})["valid"])[:n]
    pop = [int(j) for j in np.argsort(-store.pop, kind="stable")[: DEPTH + 200]]
    lists = []
    for s in range(0, len(cases), 128):
        part = cases[s: s + 128]
        with torch.no_grad():
            m = scorer.scores([c.user for c in part], [c.history for c in part]).cpu().numpy()
        for c, row in zip(part, m):
            h, seen = c.history, set(c.history)
            row[np.asarray(h, dtype=int)] = -np.inf
            now = store.now(c.user, len(h))
            g = store.generators(h, now)
            L = {"sasrecf": [int(j) for j in np.argsort(-row)[:DEPTH]],
                 "evidence": store.candidates(h, now, DEPTH)}
            for name in ("after", "with", "text", "lately"):
                L[name] = [j for j in g[name] if j not in seen][:DEPTH]
            L["kg"] = kg_agent(kg, h, DEPTH)
            L["pop"] = [j for j in pop if j not in seen][:DEPTH]
            lists.append(L)
    names = list(lists[0])
    tgt = [c.target for c in cases]
    half = len(cases) // 2

    def recall(w: dict, idx: range, k: int) -> float:
        hit = 0
        for u in idx:
            fused = rrf([lists[u][a] for a in names], [w.get(a, 0.0) for a in names])
            hit += tgt[u] in {i for i, _ in fused[:k]}
        return hit / len(idx)

    tr, te = range(half), range(half, len(cases))
    for a in names:
        print(f"{ds.name:22s} agent {a:9s} recall@30 {recall({a: 1.0}, te, 30):.3f}  recall@50 {recall({a: 1.0}, te, 50):.3f}", flush=True)
    for k in (30, 50):
        w = {"sasrecf": 1.0}
        best = recall(w, tr, k)
        for _ in range(2):                                     # coordinate search on the first half
            for a in names:
                for v in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
                    t = {**w, a: v}
                    r = recall(t, tr, k)
                    if r > best + 1e-9:
                        best, w = r, t
        base = recall({"sasrecf": 1.0}, te, k)
        got = recall(w, te, k)
        print(f"{ds.name:22s} K={k}: SASRecF alone {base:.3f}  multi-agent pool {got:.3f} ({(got / base - 1) * 100:+.0f}%)  "
              f"weights {{{', '.join(f'{a}: {v}' for a, v in w.items() if v)}}}", flush=True)
        print(f"{ds.name:22s} K={k}: uniform fusion of all agents {recall({a: 1.0 for a in names}, te, k):.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1000)
