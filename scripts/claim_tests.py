"""Tests of claims of the paper that were stated without a test (user rule of 2026-10-16: "test everything").

    python logs/probe/claim_tests.py pop|signals|agents|size [domains]

pop      Table pop: NDCG@10 by target popularity (purchases in the store: rare < 20, middle 20-99, popular >= 100) with
         paired tests of Decision4Rec-Full (4B, fused) and of the pool against SASRecF, and the share of targets inside
         the 30 candidates of SASRecF and of the pool, by group.
signals  How much of the within-list order of the decision model (0.8B and 4B, Luce scores of the rtm runs) is a
         function of the inputs of the residual ranker: R^2 of the ranker score alone, of a linear fit and of a LightGBM
         fit on the ranker's features (fitted on half of the test users, R^2 on the other half), paired test 0.8B vs 4B.
agents   MovieLens-1M plain vs enriched: share of test targets found by each agent of the pool, and only by it.
size     Validation users: share of targets in the whole pool against the best |pool| items of SASRecF (same size).
"""
import json
import os
import sys

import numpy as np

from decision4rec.agents.pool import pool_for
from decision4rec.data.schema import Dataset, load_cases
from decision4rec.eval import metrics as M
from decision4rec.experiments.asc import GAMMAS, SPLITS, _ndcg, _tune, _z
from decision4rec.experiments.runner import case_paths, load_config
from decision4rec.operators.luce import Obs, fit_luce


def answers(d, tag, arm="rtm"):   # cases and Luce scores of a run, as in the reports
    rd = f"runs/small/{d}/{d}_{arm}_{tag}"
    cfg = json.load(open(os.path.join(rd, "manifest.json")))["config"]
    paths = case_paths(cfg)
    cases = {s: {c.cid: c for c in load_cases(paths[s])} for s in SPLITS}
    th = {s: {} for s in SPLITS}
    for s in SPLITS:
        for line in open(os.path.join(rd, f"{arm}.{s}.jsonl"), encoding="utf-8"):
            rec = json.loads(line) if line.strip() else None
            if rec and rec.get("obs") and rec["cid"] in cases[s]:
                th[s][rec["cid"]] = fit_luce([Obs.from_json(o) for o in rec["obs"]], len(cases[s][rec["cid"]].cands)).theta
    return {s: {c: cases[s][c] for c in th[s]} for s in SPLITS}, th


def fused(d, tag="kev-4b-rtm"):   # per-case NDCG@10 of the decision model fused with the validation gamma
    cases, th = answers(d, tag)
    zr = {s: {c: _z(cases[s][c].rscores) for c in cases[s]} for s in SPLITS}

    def fuse(g, s):
        return {c: _z(th[s][c]) + g * zr[s][c] for c in cases[s]}
    g, _ = _tune(fuse, GAMMAS, cases)
    return dict(zip(sorted(cases["test"]), _ndcg(fuse(g, "test"), cases["test"])))   # _ndcg follows sorted case ids


def listed(c):   # NDCG@10 of the order of the list itself
    return 1.0 / np.log2(c.target_pos + 2) if 0 <= c.target_pos < 10 else 0.0


def setup(d):
    cfg = load_config(f"configs/small/{d}_rtm.yaml")
    ds = Dataset.load(cfg["dataset"])
    return cfg, ds, pool_for(cfg, ds)


def batches(pool, cases, fn):   # fn(case, lists-or-features input) over SASRecF rows computed 256 users at a time
    for s in range(0, len(cases), 256):
        b = cases[s: s + 256]
        R = pool._sasrec_rows([c.user for c in b], [c.history for c in b])
        for r, c in enumerate(b):
            fn(c, pool.store.now(c.user, len(c.history)), None if R is None else R[r])


def test_pop(domains):
    for d in domains:
        cfg, ds, pool = setup(d)
        full = load_cases(case_paths(cfg)["test"])
        sas = {c.user: c for c in load_cases(case_paths(load_config(f"configs/small/{d}_rte.yaml"))["test"])}
        f = fused(d)
        cs = [c for c in full if c.user in sas and c.cid in f]
        assert len(cs) >= 0.99 * len(full), f"{d}: only {len(cs)} of {len(full)} users paired"
        pop = np.array([pool.store.pop[c.target] for c in cs])
        groups = {"rare": pop < 20, "middle": (pop >= 20) & (pop < 100), "popular": pop >= 100}
        a = np.array([listed(sas[c.user]) for c in cs])           # SASRecF
        b = np.array([listed(c) for c in cs])                     # pool with its residual ranker
        e = np.array([f[c.cid] for c in cs])                      # Full, 4B, fused
        in_s = np.array([sas[c.user].target_pos >= 0 for c in cs])
        in_p = np.array([c.target_pos >= 0 for c in cs])
        for g, m in groups.items():
            print(f"{d} {g:7s} users {m.mean():.0%} | target in the 30: SASRecF {in_s[m].mean():.3f}, pool {in_p[m].mean():.3f} | "
                  f"NDCG SASRecF {a[m].mean():.4f}, pool {b[m].mean():.4f} ({100 * (b[m].mean() / max(a[m].mean(), 1e-9) - 1):+.0f}%, "
                  f"p={M.paired_randomization(b[m], a[m]):.3f}), Full 4B {e[m].mean():.4f} "
                  f"({100 * (e[m].mean() / max(a[m].mean(), 1e-9) - 1):+.0f}%, p={M.paired_randomization(e[m], a[m]):.3f})", flush=True)


def test_signals(domains):
    import lightgbm as lgb
    for d in domains:
        cfg, ds, pool = setup(d)
        c08, t08 = answers(d, "kev-0.8b-rtm")
        _, t4 = answers(d, "kev-4b-rtm")
        cases = [c08["test"][i] for i in sorted(set(t08["test"]) & set(t4["test"]))]
        rows = {}

        def feats(c, now, row):
            p, X, _ = pool.features(c.history, now, row)
            idx = {j: k for k, j in enumerate(p)}
            if all(j in idx for j in c.cands):              # padded lists (pool shorter than 30) are left out
                Xc = X[[idx[j] for j in c.cands]]                # float32, as the ranker's net expects
                rows[c.cid] = np.column_stack([Xc, pool._score(Xc)]).astype(np.float64)
        batches(pool, cases, feats)
        keep = sorted(rows)
        perm = np.random.default_rng(0).permutation(len(keep))
        A, B = [keep[k] for k in perm[: len(keep) // 2]], [keep[k] for k in perm[len(keep) // 2:]]

        def zc(X):   # within-list standardization of every column
            return (X - X.mean(0)) / (X.std(0) + 1e-9)
        Z = {i: np.column_stack([zc(rows[i]), rows[i]]) for i in keep}
        out, r2 = [], {}
        for size, th in (("0.8B", t08), ("4B", t4)):
            Y = {i: _z(np.asarray(th["test"][i], dtype=np.float64)) for i in keep}

            def per_case(pred):   # R^2 inside each held-out list (the scores are standardized within the list)
                return np.array([1.0 - ((Y[i] - pred[i]) ** 2).sum() / max((Y[i] ** 2).sum(), 1e-9) for i in B])
            zr = {i: zc(rows[i])[:, -1] for i in keep}                     # the ranker's own score
            beta = sum((Y[i] * zr[i]).sum() for i in A) / sum((zr[i] ** 2).sum() for i in A)
            rank = per_case({i: beta * zr[i] for i in B})
            XA, yA = np.concatenate([zc(rows[i]) for i in A]), np.concatenate([Y[i] for i in A])
            w, *_ = np.linalg.lstsq(XA, yA, rcond=None)
            lin = per_case({i: zc(rows[i]) @ w for i in B})
            n_sel = len(A) // 5
            fitA, selA = A[n_sel:], A[:n_sel]
            params = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, bagging_fraction=0.8,
                          bagging_freq=1, feature_fraction=0.9, num_threads=4, seed=0, verbose=-1)
            gbm = lgb.train(params, lgb.Dataset(np.concatenate([Z[i] for i in fitA]), np.concatenate([Y[i] for i in fitA])),
                            num_boost_round=2000, valid_sets=[lgb.Dataset(np.concatenate([Z[i] for i in selA]),
                                                                          np.concatenate([Y[i] for i in selA]))],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
            gb = per_case({i: gbm.predict(Z[i], num_iteration=gbm.best_iteration) for i in B})
            r2[size] = gb
            out.append(f"{size}: ranker score {rank.mean():.3f}, linear {lin.mean():.3f}, LightGBM {gb.mean():.3f} ({gbm.best_iteration} trees)")
        print(f"{d} (lists {len(keep)}, held out {len(B)}): R^2 of the decision model's within-list scores explained by the ranker's inputs | "
              + " | ".join(out) + f" | 0.8B vs 4B (LightGBM) p={M.paired_randomization(r2['0.8B'], r2['4B']):.3f}", flush=True)


def test_agents():
    res = {}
    for d in ("ml1m", "ml1m_plus"):
        cfg, ds, pool = setup(d)
        cases = load_cases(case_paths(cfg)["test"])
        H = {}

        def hits(c, now, row):
            L = pool.lists(c.history, now, row)
            H[c.user] = {a: c.target in L[a] for a in pool.agents}
        batches(pool, cases, hits)
        res[d] = H
    users = sorted(set(res["ml1m"]) & set(res["ml1m_plus"]))
    agents = list(next(iter(res["ml1m"].values())))
    for a in agents + ["union"]:
        v = {d: np.array([any(res[d][u].values()) if a == "union" else res[d][u][a] for u in users], dtype=float) for d in res}
        o = {d: np.array([res[d][u][a] and sum(res[d][u].values()) == 1 for u in users], dtype=float) if a != "union" else v[d] for d in res}
        print(f"ml1m agent {a:8s} (users {len(users)}): target found plain {v['ml1m'].mean():.3f} -> enriched {v['ml1m_plus'].mean():.3f} "
              f"(p={M.paired_randomization(v['ml1m_plus'], v['ml1m']):.3f}); found only by it {o['ml1m'].mean():.3f} -> {o['ml1m_plus'].mean():.3f}", flush=True)


def test_size(domains):
    for d in domains:
        cfg, ds, pool = setup(d)
        cases = load_cases(case_paths(cfg)["valid"])
        n, in_pool, s30, sn = [], [], [], []

        def size(c, now, row):
            L = pool.lists(c.history, now, row)
            p = set(j for a in pool.agents for j in L[a])
            order = np.argsort(-row)
            n.append(len(p))
            in_pool.append(c.target in p)
            s30.append(c.target in set(order[:30].tolist()))
            sn.append(c.target in set(order[: len(p)].tolist()))
        batches(pool, cases, size)
        print(f"{d} (validation users {len(cases)}, pool size median {int(np.median(n))}, mean {np.mean(n):.0f}): target in the pool "
              f"{np.mean(in_pool):.3f} | SASRecF best 30 {np.mean(s30):.3f} | SASRecF best |pool| (same size, per user) {np.mean(sn):.3f} "
              f"(pool vs same-size SASRecF p={M.paired_randomization(np.array(in_pool, float), np.array(sn, float)):.3f})", flush=True)


if __name__ == "__main__":
    what, doms = sys.argv[1], sys.argv[2:] or ["vg", "mi", "steam", "ind"]
    {"pop": lambda: test_pop(doms), "signals": lambda: test_signals(doms), "agents": test_agents,
     "size": lambda: test_size(doms)}[what]()
