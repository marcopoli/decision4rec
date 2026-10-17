"""A gradient-boosted ranker of the pool (review 2026-10-16, R26): LightGBM LambdaMART over exactly the features of the
residual ranker of Decision4Rec-Full (which agents found a candidate and at what rank, the SASRecF score, the numbers
behind the statements), trained on the same 15,000 users with the same target (second-to-last item) and the same split
(a fifth, seed 0, chooses the number of trees), then scored on the 2,000 development test users. Compared, on the same
users, with the linear residual ranker and with Decision4Rec-Full with the 4B decision model fused.
python logs/probe/gbdt_ranker.py [vg mi steam ind]"""
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np

from decision4rec.agents.pool import pool_for
from decision4rec.data.schema import Dataset, load_cases
from decision4rec.eval import metrics as M
from decision4rec.experiments.asc import GAMMAS, SPLITS, _ndcg, _tune, _z
from decision4rec.experiments.runner import case_paths, load_config
from decision4rec.operators.luce import Obs, fit_luce


def fused(d, tag="kev-4b-rtm"):   # Decision4Rec-Full fused with the validation gamma, as in the reports
    rd = f"runs/small/{d}/{d}_rtm_{tag}"
    cfg = json.load(open(os.path.join(rd, "manifest.json")))["config"]
    paths = case_paths(cfg)
    cases = {s: {c.cid: c for c in load_cases(paths[s])} for s in SPLITS}
    th = {s: {} for s in SPLITS}
    for s in SPLITS:
        for line in open(os.path.join(rd, f"rtm.{s}.jsonl"), encoding="utf-8"):
            rec = json.loads(line) if line.strip() else None
            if rec and rec.get("obs") and rec["cid"] in cases[s]:
                th[s][rec["cid"]] = fit_luce([Obs.from_json(o) for o in rec["obs"]], len(cases[s][rec["cid"]].cands)).theta
    cases = {s: {c: cases[s][c] for c in th[s]} for s in SPLITS}
    zr = {s: {c: _z(cases[s][c].rscores) for c in cases[s]} for s in SPLITS}

    def fuse(g, s):
        return {c: _z(th[s][c]) + g * zr[s][c] for c in cases[s]}
    g, _ = _tune(fuse, GAMMAS, cases)
    return dict(zip(sorted(cases["test"]), _ndcg(fuse(g, "test"), cases["test"])))   # _ndcg follows sorted case ids


def features(pool, users, hists):
    pools, Xs = [], []
    for s in range(0, len(users), 256):
        us, hs = users[s: s + 256], hists[s: s + 256]
        rows = pool._sasrec_rows(us, hs)
        for r, (u, h) in enumerate(zip(us, hs)):
            p, X, _ = pool.features(h, pool.store.now(u, len(h)), None if rows is None else rows[r])
            pools.append(p)
            Xs.append(X)
    return pools, Xs


def ndcg_at10(scores, pool_items, target):
    if target not in pool_items:
        return 0.0
    r = M.target_rank(scores, pool_items.index(target))      # ties count against the target
    return 1.0 / np.log2(r + 1.0) if r <= 10 else 0.0


for d in sys.argv[1:] or ["vg", "mi", "steam", "ind"]:
    t0 = time.time()
    cfg = load_config(f"configs/small/{d}_rtm.yaml")
    ds = Dataset.load(cfg["dataset"])
    pool = pool_for(cfg, ds)                                   # the linear residual ranker of the paper, from its cache
    users = list(pool.train_users)
    hist = [[int(x[0]) for x in ds.seqs[u][:-2]] for u in users]
    tgt = [int(ds.seqs[u][-2][0]) for u in users]
    pools, Xs = features(pool, users, hist)
    ys = [p.index(t) if t in p else -1 for p, t in zip(pools, tgt)]
    keep = [n for n, y in enumerate(ys) if y >= 0]
    perm = np.random.default_rng(0).permutation(keep).tolist()
    sel, fit = perm[: len(perm) // 5], perm[len(perm) // 5:]

    def mat(idx):
        return (np.concatenate([Xs[n] for n in idx]),
                np.concatenate([(np.arange(len(Xs[n])) == ys[n]).astype(int) for n in idx]),
                [len(Xs[n]) for n in idx])
    Xf, yf, gf = mat(fit)
    Xv, yv, gv = mat(sel)
    params = dict(objective="lambdarank", metric="ndcg", eval_at=[10], learning_rate=0.05, num_leaves=31,
                  min_data_in_leaf=50, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.9,
                  num_threads=4, seed=0, verbose=-1)   # native API: the sklearn wrapper needs scikit-learn
    model = lgb.train(params, lgb.Dataset(Xf, yf, group=gf), num_boost_round=2000,
                      valid_sets=[lgb.Dataset(Xv, yv, group=gv)], callbacks=[lgb.early_stopping(100, verbose=False)])

    cases = load_cases(case_paths(cfg)["test"])
    tpools, tXs = features(pool, [c.user for c in cases], [c.history for c in cases])
    gb = np.array([ndcg_at10(model.predict(X, num_iteration=model.best_iteration), p, c.target)
                   for c, p, X in zip(cases, tpools, tXs)])
    rescored = np.array([ndcg_at10(pool._score(X), p, c.target) for c, p, X in zip(cases, tpools, tXs)])
    lin = np.array([1.0 / np.log2(c.target_pos + 2) if 0 <= c.target_pos < 10 else 0.0 for c in cases])  # as evaluated
    bad = int((~np.isclose(rescored, lin)).sum())   # a few near-ties may flip (SASRecF scored in other batches)
    assert bad <= len(cases) // 200, f"features differ from those of the evaluated pool ({bad} users)"
    f = fused(d)
    full = np.array([f[c.cid] for c in cases])
    print(f"{d} (train {len(fit)}, select {len(sel)}, trees {model.best_iteration}, test {len(cases)}, rescored off {bad}, "
          f"{(time.time() - t0) / 60:.0f} min): linear ranker {lin.mean():.4f} | LambdaMART {gb.mean():.4f} "
          f"({100 * (gb.mean() / lin.mean() - 1):+.1f}%, p={M.paired_randomization(gb, lin):.3f}) | "
          f"Full 4B fused {full.mean():.4f}: vs LambdaMART {100 * (full.mean() / gb.mean() - 1):+.1f}% "
          f"(p={M.paired_randomization(full, gb):.3f})", flush=True)
