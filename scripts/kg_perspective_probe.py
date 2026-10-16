"""Perspective agents of the knowledge graph (version 4): how many targets do they add to the multi-agent pool, and
do they help its order? Development validation users, the same test as versions 1 and 2.

    python scripts/kg_perspective_probe.py configs/small/vg_rtm.yaml

Coverage: target in the pool, in the union of the five lists (20 each), and only in the lists (share of the pool's
hits). Order: 5-fold cross-validation over users of s + w . f, where s is the residual ranker and f marks, for every
pool candidate, the reciprocal rank in each perspective list (0 if absent); targets that only the lists find are out
of the pool and count as misses here.
"""

import os
import sys

import numpy as np
import torch

from decision4rec.agents.kgperspectives import NAMES, KGPerspectives
from decision4rec.agents.pool import pool_for
from decision4rec.data.schema import Dataset, load_cases
from decision4rec.experiments.runner import case_paths, load_config


def ndcg(s, y):
    if y < 0:
        return 0.0
    r = int((s > s[y]).sum()) + 1
    return 1.0 / np.log2(r + 1) if r <= 10 else 0.0


def fit(rows, steps=300):
    w = torch.zeros(rows[0][1].shape[1], requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.05, weight_decay=1e-3)
    data = [(torch.tensor(s), torch.tensor(F), y) for s, F, y in rows if y >= 0]
    for _ in range(steps):
        loss = sum(-torch.log_softmax(s + F @ w, 0)[y] for s, F, y in data) / len(data)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach().numpy()


def main(path: str, n: int) -> None:
    cfg = load_config(path)
    ds = Dataset.load(cfg["dataset"])
    pool = pool_for(cfg, ds)
    kgp = KGPerspectives(ds, pool.store, kind_depth=int(os.environ['KIND_DEPTH']) if os.environ.get('KIND_DEPTH') else None)
    cases = load_cases(case_paths({**cfg, "retriever": {"tag": "ev100"}})["valid"])[:n]
    in_pool = in_p = only = 0
    per = dict.fromkeys(NAMES, 0)
    per_only = dict.fromkeys(NAMES, 0)
    rows = []
    for s0 in range(0, len(cases), 128):
        part = cases[s0: s0 + 128]
        sas = pool._sasrec_rows([c.user for c in part], [c.history for c in part])
        for r, c in enumerate(part):
            now = pool.store.now(c.user, len(c.history))
            items, X, _ = pool.features(c.history, now, sas[r])
            L = kgp.lists(c.history, now)
            union = {j for v in L.values() for j in v}
            a, b = c.target in set(items), c.target in union
            in_pool += a
            in_p += b
            only += b and not a
            for k in NAMES:
                per[k] += c.target in set(L[k])
                per_only[k] += (c.target in set(L[k])) and not a
            pos = {k: {j: q for q, j in enumerate(L[k])} for k in NAMES}
            F = np.array([[1.0 / (1 + pos[k][j]) if j in pos[k] else 0.0 for k in NAMES] for j in items], dtype=np.float32)
            F = (F - F.mean(0)) / (F.std(0) + 1e-6)
            rows.append((pool._score(X).astype(np.float32), F, items.index(c.target) if a else -1))
    m = len(cases)
    print(f"{ds.name}: {m} validation users; target in pool {in_pool / m:.3f}, in the 5 lists {in_p / m:.3f}, "
          f"only in the lists {only / m:.3f} (+{only / max(in_pool, 1) * 100:.1f}% of the pool's hits)")
    print("  per perspective, recall@20 / only there: " + "  ".join(f"{k} {per[k] / m:.3f}/{per_only[k] / m:.3f}" for k in NAMES))
    folds = np.random.default_rng(0).permutation(len(rows)) % 5
    base, new = np.zeros(len(rows)), np.zeros(len(rows))
    for f in range(5):
        w = fit([rows[i] for i in range(len(rows)) if folds[i] != f])
        for i in np.where(folds == f)[0]:
            s, F, y = rows[i]
            base[i], new[i] = ndcg(s, y), ndcg(s + F @ w, y)
    d = new - base
    p = float((np.abs((np.random.default_rng(1).choice([-1.0, 1.0], size=(10000, len(d))) * d).mean(1)) >= abs(d.mean())).mean())
    print(f"  pool order NDCG@10 {base.mean():.4f}; with perspective features {new.mean():.4f} ({d.mean():+.4f}, p={p:.3f}); "
          f"weights (fold 5) {dict(zip(NAMES, np.round(w, 3)))}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
