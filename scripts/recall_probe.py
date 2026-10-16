"""Recall at K of candidate generators on development validation users (target = second to last item).

    python scripts/recall_probe.py configs/small/vg_ev.yaml [n_users]

Reads the validation cases of an evidence config (history s[:-2], target s[-2]); statistics leave out the last
two items of every user, so no target is seen. Reports each generator alone, quota mixes, a score fusion, the
training-free baselines and the union of generators (an upper bound for any mix of them).
"""

import sys

import numpy as np

from decision4rec.data.schema import Dataset, load_cases
from decision4rec.evidence.store import store_for
from decision4rec.experiments.runner import case_paths, load_config
from decision4rec.experiments.stage1 import BaselineRetriever
from decision4rec.models import baselines as B

K = 30
MIXES = {
    "current a10 w8 t8 l4": (("after", 10), ("with", 8), ("text", 8), ("lately", 4)),
    "a8 w14 t4 l4": (("after", 8), ("with", 14), ("text", 4), ("lately", 4)),
    "a14 w10 t3 l3": (("after", 14), ("with", 10), ("text", 3), ("lately", 3)),
    "a12 w12 t6": (("after", 12), ("with", 12), ("text", 6)),
    "a10 w10 t5 l5": (("after", 10), ("with", 10), ("text", 5), ("lately", 5)),
}


def fused(store, h, now, w):
    n = store.ds.n_items
    s = np.zeros(n)
    ctx = store.context(h, now)
    for key, val in (("after", ctx.after),):
        if val:
            idx = np.fromiter(val.keys(), int); v = np.fromiter(val.values(), float)
            s[idx] += w[0] * v / v.max()
    knn = {}
    for r, x in enumerate(reversed(h[-10:])):
        for j, c in store.cooc.get(x, {}).items():
            knn[j] = knn.get(j, 0.0) + 0.8 ** r * c / np.sqrt(store.deg[x] * store.deg[j] + 1e-9)
    if knn:
        idx = np.fromiter(knn.keys(), int); v = np.fromiter(knn.values(), float)
        s[idx] += w[1] * v / v.max()
    if ctx.text_rank is not None:
        s += w[2] * np.maximum(0.0, 1.0 - np.log(ctx.text_rank) / np.log(500.0))
    if w[3] and now is not None and len(store.snap):
        i = int(np.searchsorted(store.snap, now, side="right")) - 1
        if i >= 0:
            top = store.snap_top[i]
            s[top] += w[3] * (1.0 - np.arange(len(top)) / len(top))
    s += 1e-9 * store.pop / max(store.pop.max(), 1)
    s[np.asarray(h, dtype=int)] = -np.inf
    return np.argsort(-s, kind="stable")[:K].tolist()


def main(path: str, n: int) -> None:
    cfg = load_config(path)
    ds = Dataset.load(cfg["dataset"])
    store = store_for(cfg, ds)
    cases = load_cases(case_paths(cfg)["valid"])[:n]
    res = {}

    def hit(name, lists):
        res[name] = float(np.mean([c.target in set(l[:K]) for c, l in zip(cases, lists)]))
        print(f"{ds.name:22s} {name:28s} recall@{K} {res[name]:.3f}", flush=True)

    gens = [store.generators(c.history, store.now(c.user, len(c.history))) for c in cases]
    for g in ("after", "with", "text", "lately"):
        hit(f"only {g}", [[j for j in G[g] if j not in set(c.history)] for c, G in zip(cases, gens)])
    u = np.mean([c.target in {j for g in ("after", "with", "text", "lately")
                              for j in [x for x in G[g] if x not in set(c.history)][:K]} for c, G in zip(cases, gens)])
    print(f"{ds.name:22s} {'union of 4 x 30 (bound)':28s} recall@120 {u:.3f}", flush=True)
    for name, q in MIXES.items():
        hit(name, [store.candidates(c.history, store.now(c.user, len(c.history)), K, q) for c in cases])
    for w in ((1, 1, 0.3, 0.2), (1, 1.5, 0.3, 0.2), (1.5, 1, 0.3, 0.2), (1, 1, 0.6, 0.3), (1, 1, 0.15, 0.1)):
        hit(f"fused {w}", [fused(store, c.history, store.now(c.user, len(c.history)), w) for c in cases])
    hist = [c.history for c in cases]
    for m in ("pop", "itemknn", "seqrules", "transtext"):
        idx, _ = BaselineRetriever(B.Scorer(m, ds)).topk(hist, K)
        hit(f"baseline {m}", [r.tolist() for r in idx])


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
