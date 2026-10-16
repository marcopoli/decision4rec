"""Perspective agents version 5a: personal perspective weights. Does the user's own past mode (how often each
perspective explained what the user took next) tell which perspective list to trust for this user?

    python scripts/persp_profile_probe.py configs/small/vg_rtmp.yaml

Pool: the multi-agent pool with the five perspective agents of the knowledge graph (configs *_rtmp.yaml). For every
user, rate_p = share of the user's past steps that perspective p explains (rules of decision4rec.agents.kgperspectives,
applied to the user's own history, Laplace-smoothed). For every pool candidate: in_p = 1 / (1 + rank in list p), or 0.
Development validation users, 5-fold cross-validation over users, softmax cross-entropy on the pool score s:
    A: s + w . in            (perspective membership, the same for every user)
    B: s + w . in + v . (in * rate)   (membership weighted by the user's own mode)
"""

import sys

import numpy as np
import torch

from decision4rec.agents.kgperspectives import NAMES
from decision4rec.agents.pool import pool_for
from decision4rec.data.schema import Dataset
from decision4rec.experiments.runner import get_cases, load_config


def ndcg(s, y):
    if y < 0:
        return 0.0
    r = int((s > s[y]).sum()) + 1
    return 1.0 / np.log2(r + 1) if r <= 10 else 0.0


def fit(rows, cols, steps=300):
    w = torch.zeros(len(cols), requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.05, weight_decay=1e-3)
    data = [(torch.tensor(s), torch.tensor(F[:, cols]), y) for s, F, y in rows if y >= 0]
    for _ in range(steps):
        loss = sum(-torch.log_softmax(s + F @ w, 0)[y] for s, F, y in data) / len(data)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach().numpy()


def rates(kgp, store, h, back=50):
    """Share of the user's past steps explained by each perspective (Laplace smoothing toward 1/5)."""
    cnt, n = np.zeros(len(NAMES)), 0
    lo = max(3, len(h) - back)
    for t in range(lo, len(h)):
        x, prev, k = h[t], h[t - 3:t], kgp.kind[h[t]]
        pk = {kgp.kind[p] for p in prev}
        pm = set().union(*(kgp.makers[p] for p in prev))
        ps = set().union(*(kgp.series[p] for p in prev))
        older = h[max(0, t - 50):t - 3]
        hit = [
            any(x in store.cooc.get(p, {}) for p in prev) and k is not None and k not in pk,
            (x in store.rules.get(h[t - 1], {}) or x in kgp.next_of.get(h[t - 1], [])) and bool(kgp.makers[x] & pm or kgp.series[x] & ps),
            bool(kgp.makers[x] & pm),
            any(k is not None and kgp.kind[o] == k and kgp.makers[x] & kgp.makers[o] for o in older),
            k is not None and k not in {kgp.kind[o] for o in h[:t]},
        ]
        cnt += np.array(hit, dtype=float)
        n += 1
    return (cnt + 0.2) / (n + 1.0)


def main(path: str, n: int) -> None:
    cfg = load_config(path)
    ds = Dataset.load(cfg["dataset"])
    pool = pool_for(cfg, ds)
    kgp, store = pool.persp, pool.store
    cases = get_cases(cfg, ds)["valid"][:n]
    rows, R = [], []
    for s0 in range(0, len(cases), 128):
        part = cases[s0: s0 + 128]
        sas = pool._sasrec_rows([c.user for c in part], [c.history for c in part])
        for r, c in enumerate(part):
            now = store.now(c.user, len(c.history))
            items, X, pos = pool.features(c.history, now, sas[r])
            rt = rates(kgp, store, c.history)
            R.append(rt)
            inl = np.array([[1.0 / (1 + pos[p][j]) if j in pos[p] else 0.0 for p in NAMES] for j in items], dtype=np.float32)
            F = np.concatenate([inl, inl * rt[None, :].astype(np.float32)], 1)
            # no standardization within the user: it would cancel the user's rate, constant over the candidates
            rows.append((pool._score(X).astype(np.float32), F, items.index(c.target) if c.target in items else -1))
    R = np.array(R)
    allF = np.concatenate([F for _, F, _ in rows])
    mu, sd = allF.mean(0), allF.std(0) + 1e-6           # one scaling for all users
    rows = [(s, (F - mu) / sd, y) for s, F, y in rows]
    print(f"{ds.name}: {len(rows)} validation users, target in pool {np.mean([y >= 0 for *_, y in rows]):.3f}; "
          f"mean past-mode rates " + ", ".join(f"{p} {v:.2f}" for p, v in zip(NAMES, R.mean(0))))
    folds = np.random.default_rng(0).permutation(len(rows)) % 5
    k = len(NAMES)
    sets = {"A membership": list(range(k)), "B membership + personal": list(range(2 * k))}
    res = {"pool": np.zeros(len(rows)), **{s: np.zeros(len(rows)) for s in sets}}
    for f in range(5):
        tr = [rows[i] for i in range(len(rows)) if folds[i] != f]
        ws = {s: fit(tr, cols) for s, cols in sets.items()}
        for i in np.where(folds == f)[0]:
            s, F, y = rows[i]
            res["pool"][i] = ndcg(s, y)
            for name, cols in sets.items():
                res[name][i] = ndcg(s + F[:, cols] @ ws[name], y)
    rng = np.random.default_rng(1)
    for name, v in res.items():
        for ref in ("pool", "A membership"):
            if name == ref or (ref == "A membership" and name != "B membership + personal"):
                continue
            d = v - res[ref]
            p = float((np.abs((rng.choice([-1.0, 1.0], size=(10000, len(d))) * d).mean(1)) >= abs(d.mean())).mean())
            print(f"  {name:26s} NDCG@10 {v.mean():.4f}  vs {ref}: {d.mean():+.4f} (p={p:.3f})")
    print(f"  pool NDCG@10 {res['pool'].mean():.4f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
