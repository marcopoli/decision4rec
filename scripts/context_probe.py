"""Context statements as features of the residual ranker (V28a, docs/17 item B; CPU, exploratory).

What neither the ranker nor the readers see today: whether an option existed before the request and how often it had
sold by then (born), its age, the pace of the user and the basket of the last day (time), and the ratings the user gave
(rating). Each family joins the ranker of the paper fitted by maximum likelihood (as logs/probe/ranker_ml_probe.py), on
the ranker users of the paper. Reports the held-out likelihood, and list NDCG@10 and the share of targets in the 30 on
development test and validation users, against the same ranker without the family.

    python scripts/context_probe.py vg out.json          (CTX_MAX=600 for a smoke run)
"""
import json
import os
import sys

import numpy as np
import torch

from decision4rec.agents.pool import pool_for
from decision4rec.data.schema import Dataset, load_cases
from decision4rec.eval import metrics as M
from decision4rec.experiments.runner import case_paths, load_config

torch.set_num_threads(2)
d, out = sys.argv[1], sys.argv[2]
MAX = int(os.environ.get("CTX_MAX", "0"))
cfg = load_config(f"configs/small/{d}_rtm.yaml")
ds = Dataset.load(cfg["dataset"])
p = pool_for(cfg, ds)
st, z0 = p.store, p.zcol
DAY = st.window / 30.0                                  # the store's unit of one day (milliseconds or seconds)
RATED = len({float(x[2]) for s in ds.seqs[:2000] for x in s}) > 1
NAMES = ["born", "sold before", "age", "age x list", "basket x with", "basket x after", "basket x list", "pace x list",
         "pace x after", "pace x sold", "after liked", "with liked", "text liked", "last rating x after", "last rating x with",
         "last rating x list", "last rating x text", "recent ratings x list"]
FAMILIES = {"born": [0, 1], "time": list(range(2, 10)), "rating": list(range(10, 18)), "all": list(range(18))}


def extras(u, h, now, pool, X):
    """The context columns of each option of `pool` (NAMES), from what is known when the recommendation is made."""
    s, k = ds.seqs[u], len(h)
    tim = np.array([float(x[1]) for x in s[:k]])
    rc = np.array([(float(x[2]) - 3.0) / 2.0 for x in s[:k]]) if RATED else np.zeros(k)     # ratings 1..5 to -1..1
    r_last, r3 = (float(rc[-1]), float(rc[-3:].mean())) if k else (0.0, 0.0)
    basket = float(np.log1p(max((now - tim[-10:] < DAY).sum() - 1, 0))) if k else 0.0        # other items of the last day
    gaps = np.diff(tim[-4:]) / DAY
    pace = float(np.log1p(max(np.median(gaps), 0.0))) if len(gaps) else 0.0
    liked = dict(zip(h[-10:], rc[-10:]))                # the rating of the latest occurrence of each recent item
    n_rec = min(st.recent, k)
    q = sum(st.decay ** r * rc[-1 - r] * st.unit[h[-1 - r]] for r in range(n_rec)) if st.unit is not None and k else None
    z, lpop, after, with_n, trank = X[:, z0], X[:, z0 + 1], X[:, z0 + 3], X[:, z0 + 4], X[:, z0 + 5]
    E = np.zeros((len(pool), len(NAMES)), dtype=np.float32)
    for n, j in enumerate(pool):
        t = st.t_item[st.start[j]: st.end[j]]
        before = int(np.searchsorted(t, now, side="left")) if now is not None else len(t)
        age = float(np.log1p(max(now - t[0], 0.0) / DAY)) if before > 0 else 0.0
        al = sum(st.decay ** r * st.rules.get(h[-1 - r], {}).get(j, 0.0) * rc[-1 - r] for r in range(n_rec))
        wl = sum(r for i, r in liked.items() if j in st.cooc.get(i, {}))
        tl = float(st.unit[j] @ q) if q is not None else 0.0
        E[n] = [float(before > 0), np.log1p(before), age, age * z[n], basket * with_n[n], basket * after[n], basket * z[n],
                pace * z[n], pace * after[n], pace * lpop[n], al, wl, tl, r_last * after[n], r_last * with_n[n],
                r_last * z[n], r_last * trank[n], r3 * z[n]]
    return E


def collect(us, hs):
    Xs, Es, pools = [], [], []
    for s in range(0, len(us), 256):
        rows = p._sasrec_rows(us[s: s + 256], hs[s: s + 256])
        for r, (u, h) in enumerate(zip(us[s: s + 256], hs[s: s + 256])):
            now = st.now(u, len(h))
            pool, X, _ = p.features(h, now, rows[r])
            Xs.append(X)
            Es.append(extras(u, h, now, pool, X))
            pools.append(pool)
    return Xs, Es, pools


users = list(p.train_users)[: MAX or None]
Xs, Es, pools = collect(users, [[int(x[0]) for x in ds.seqs[u][:-2]] for u in users])
Ys = [pl.index(t) if t in pl else -1 for pl, t in zip(pools, [int(ds.seqs[u][-2][0]) for u in users])]
keep = [n for n, y in enumerate(Ys) if y >= 0]
perm = np.random.default_rng(0).permutation(keep).tolist()
sel, fit = perm[: len(perm) // 5], perm[len(perm) // 5:]
A = [np.column_stack([x, e]) for x, e in zip(Xs, Es)]
nb = Xs[0].shape[1]
mu = np.concatenate([A[n] for n in fit]).mean(0)
sd = np.concatenate([A[n] for n in fit]).std(0) + 1e-6
mu[z0], sd[z0] = 0.0, 1.0
print(f"{d}: ranker users {len(users)}, target in the pool {len(keep)}, fit {len(fit)}, held out {len(sel)}, ratings {RATED}, "
      f"options born before the request {np.concatenate([Es[n][:, 0] for n in fit]).mean():.3f}, "
      f"targets born {np.mean([Es[n][Ys[n], 0] for n in fit]):.3f}", flush=True)


def pad(idx, cols):
    m = max(len(A[n]) for n in idx)
    F = np.zeros((len(idx), m, len(cols)), dtype=np.float32)
    K = np.zeros((len(idx), m), dtype=bool)
    Z = np.zeros((len(idx), m), dtype=np.float32)
    for r, n in enumerate(idx):
        F[r, :len(A[n])], K[r, :len(A[n])], Z[r, :len(A[n])] = ((A[n] - mu) / sd)[:, cols], True, A[n][:, z0]
    return torch.tensor(F), torch.tensor(K), torch.tensor(Z), torch.tensor([Ys[n] for n in idx])


def ce(w, F, K, Z, y):
    return torch.nn.functional.cross_entropy((Z + F @ w).masked_fill(~K, float("-inf")), y)


def train(cols):
    """Conditional logit z + w x on the columns `cols`; the L2 weight by likelihood on the held-out fifth."""
    Ff, Kf, Zf, yf = pad(fit, cols)
    Fs, Ks, Zs, ys = pad(sel, cols)
    best = None
    for l2 in (1e-1, 1e-2, 1e-3, 1e-4):
        w = torch.zeros(len(cols), requires_grad=True)
        opt = torch.optim.LBFGS([w], max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = ce(w, Ff, Kf, Zf, yf) + l2 * (w ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            v = float(ce(w, Fs, Ks, Zs, ys))
        if best is None or v < best[0]:
            best = (v, l2, w.detach().clone().numpy())
    return best


base = list(range(nb))
models = {"base": train(base)}
for f, extra in FAMILIES.items():
    models[f] = train(base + [nb + e for e in extra])
res = {"held_out_nll": {f: m[0] for f, m in models.items()}, "l2": {f: m[1] for f, m in models.items()}}
print(f"{d} held-out likelihood (lower is better): " + ", ".join(f"{f} {m[0]:.4f} (l2 {m[1]})" for f, m in models.items()), flush=True)
w = models["all"][2][nb:]
print(f"{d} weights of the context columns in 'all' (standardized): " + ", ".join(f"{NAMES[e]} {w[e]:+.3f}" for e in np.argsort(-np.abs(w))), flush=True)

paths = case_paths(cfg)
for split in ("test", "valid"):
    cases = load_cases(paths[split])[: MAX or None]
    X, E, pl = collect([c.user for c in cases], [c.history for c in cases])
    stored = np.array([1.0 / np.log2(c.target_pos + 2) if 0 <= c.target_pos < 10 else 0.0 for c in cases])
    nd, hit = {}, {}
    for f, (_, _, wf) in models.items():
        cols = base + ([nb + e for e in FAMILIES[f]] if f != "base" else [])
        nd[f], hit[f] = [], []
        for c, x, e, pp in zip(cases, X, E, pl):
            a = np.column_stack([x, e])
            s1 = a[:, z0] + ((a - mu) / sd)[:, cols] @ wf
            items = [pp[i] for i in np.argsort(-s1, kind="stable")[:30]]
            rk = items.index(c.target) + 1 if c.target in items else float("inf")
            nd[f].append(M.per_case_metrics(rk)["ndcg@10"])
            hit[f].append(float(rk < float("inf")))
        nd[f], hit[f] = np.array(nd[f]), np.array(hit[f])
    res[split] = {"stored": float(stored.mean()), "base": float(nd["base"].mean()), "base_in30": float(hit["base"].mean())}
    line = f"{d} {split}: stored list {stored.mean():.4f} | base {nd['base'].mean():.4f} in the 30 {hit['base'].mean():.3f}"
    for f in FAMILIES:
        pv = M.paired_randomization(nd[f], nd["base"])
        res[split][f] = {"ndcg": float(nd[f].mean()), "p": pv, "in30": float(hit[f].mean())}
        line += f" | {f} {nd[f].mean():.4f} ({100 * (nd[f].mean() / max(nd['base'].mean(), 1e-9) - 1):+.1f}%, p={pv:.3f}) in the 30 {hit[f].mean():.3f}"
    print(line, flush=True)
json.dump(res, open(out, "w"), indent=1)
