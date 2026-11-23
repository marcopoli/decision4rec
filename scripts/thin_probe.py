"""Three readings of the stored Full runs with the stable fusion (docs/17_review_2026-11-10.md, items A, G, F).

    python scripts/thin_probe.py out.json [thin] [nats] [cascade] [absent] [deep]

thin     Where the decision model helps (V25). Test users with the target among the options: gain in log-likelihood
         of the target when the decision model joins the list (both fitted on validation choices), against the evidence
         of the target (purchases in the training part), its age at the request, the length of the history and the
         evidence of the whole list. 9B, per-user mean of three seeds. Slope inside each setting, permutation test.
nats     What the decision model adds, in nats per choice (V26): list, decision model and both against a uniform guess
         over the options, by size, on all records and on the same 8,000.
cascade  Who needs the decision model (V27): NDCG@10 when only a share of the users is read, chosen by what the list
         knows before any request (entropy of the list, evidence of the list, length of the history).
absent   Can anything known today tell that the target is not among the 30 (V29, step 1): area under the curve of the
         entropy of the list, of the reader (trained without a "none" option), the evidence of the list and the length
         of the history, and of a logistic mix of them fitted on the other half of the test users.
law      The law at request time and for other readers (V40, docs/18 A1): the gain against the purchases of the target
         before the request (training part of the other users, earlier than the last history item; the training count
         where times are an order only). One line per reader of LAW_TAGS (comma separated conf:tag, default the 9B
         mean of seeds and six other readers), with six evidence bins per setting.
deep     Stored runs with 50 options and no new training (R18): where the reader puts the targets that the list holds
         at positions 31 to 50.
"""
import json
import os
import pickle
import sys

import numpy as np

from fusion_table import FAMILY, SIX, STRENGTH, load
from decision4rec.data.schema import Dataset
from decision4rec.eval import metrics as M
from decision4rec.experiments.asc import _ndcg
from decision4rec.operators.fusion import choice_data, shrunk_weights

SEEDS = ["rtm:kev-9b-rtm", "rtm:kev-9b-rtm_s1", "rtm:kev-9b-rtm_s2"]
SIZES = {"all": ["rtm:kev-0.8b-rtm", "rtm:kev-4b-rtm", "rtm:kev-9b-rtm", "rtm:kev-27b-rtm"],
         "8k": ["rtm:kev-0.8b-rtm8k", "rtm:kev-4b-rtm8k", "rtm:kev-9b-rtm8k", "rtm:kev-27b-rtm8k"]}
TERMS = (("f", [0, 1]), ("d", [0]), ("l", [1]))      # both, decision model alone, list alone
CACHE = "logs/probe/thin_cache"
GROUPS = (("rare", 0, 19), ("middle", 20, 99), ("popular", 100, None))     # as Table pop of the paper


def runs(spec, domains=SIX):
    """Cases and standardized terms of a variant in every setting that has it (scripts/fusion_table.py load)."""
    conf, tag = spec.split(":")
    os.makedirs(CACHE, exist_ok=True)
    D = {}
    for d in domains:
        p = os.path.join(CACHE, f"{d}_{conf}_{tag}.pkl")
        if os.path.exists(p):
            v = pickle.load(open(p, "rb"))
        else:
            v = load(d, conf, tag)
            if v is not None:                                   # a run that does not exist yet is not cached as missing
                pickle.dump(v, open(f"{p}.{os.getpid()}", "wb"))
                os.replace(f"{p}.{os.getpid()}", p)             # two probes may fill the cache at once
        if v is not None:
            D[d] = v
    return D


def _logp(u, pos):
    u = u - u.max()
    return float(u[pos] - np.log(np.exp(u).sum()))


def _entropy(u):
    q = np.exp(u - u.max())
    q /= q.sum()
    return float(-(q * np.log(np.maximum(q, 1e-300))).sum())


def weights_of(D):
    """Stable-fusion weights of each term set, fitted on the validation choices of the settings of D."""
    data = {d: choice_data(D[d][0]["valid"], D[d][1]["valid"]) for d in D}
    return {k: shrunk_weights({d: (X[:, :, cols], y) for d, (X, y) in data.items()}, FAMILY, STRENGTH) for k, cols in TERMS}


def per_user(D, w=None):
    """Per test user of each setting: NDCG@10 of the list and of the fused order, the entropy of the list, and, with
    the target among the options, its log-likelihood under the list, the decision model and both. `w` given (from
    weights_of on development runs): use those weights instead of the validation users of D (R1+, docs/11 v2 F4)."""
    w = w or weights_of(D)
    out = {}
    for d, (cases, Z) in D.items():
        T, Zt = cases["test"], Z["test"]
        cids = sorted(T)
        row = {"cids": cids, "pos": np.array([T[c].target_pos for c in cids]),
               "lst": _ndcg({c: Zt[c][:, 1] for c in T}, T), "fused": _ndcg({c: Zt[c] @ w["f"][d] for c in T}, T),
               "ent": np.array([_entropy(Zt[c][:, [1]] @ w["l"][d]) for c in cids]),
               "ent_d": np.array([_entropy(Zt[c][:, [0]] @ w["d"][d]) for c in cids]),
               "rank_d": np.array([M.target_rank(Zt[c][:, 0], T[c].target_pos) for c in cids]),
               "rank_f": np.array([M.target_rank(Zt[c] @ w["f"][d], T[c].target_pos) for c in cids]),
               "ll_u": np.array([-np.log(len(T[c].cands)) if T[c].target_pos >= 0 else np.nan for c in cids])}
        for k, cols in TERMS:
            row["ll_" + k] = np.array([_logp(Zt[c][:, cols] @ w[k][d], T[c].target_pos) if T[c].target_pos >= 0 else np.nan
                                       for c in cids])
        out[d] = row
    return out


def covariates(d, cases, spec=SEEDS[0], ds=None):
    """Per case: purchases of the target in the training part (the count of the knowledge graph), days since the
    target was first seen, length of the history, mean log purchases of the options. Checks the target of each case.
    `ds` given: use it instead of the dataset of the reader run (unseen scenarios have no reader run)."""
    if ds is None:
        conf, tag = spec.split(":")
        ds = Dataset.load(json.load(open(f"runs/small/{d}/{d}_{conf}_{tag}/manifest.json"))["config"]["dataset"])
    pop, first = np.zeros(ds.n_items, dtype=np.int64), np.full(ds.n_items, np.inf)
    times, sold = [], [[] for _ in range(ds.n_items)]
    for s in ds.seqs:
        for x in s:
            first[x[0]] = min(first[x[0]], float(x[1]))
            times.append(float(x[1]))
        for x in (s[:-2] if len(s) > 2 else []):
            pop[x[0]] += 1
            sold[x[0]].append(float(x[1]))
    span = max(times) - min(times)
    day = 86400e3 if span > 1e11 else 86400.0 if span > 1e7 else None      # milliseconds, seconds, or an order only
    sold = [np.sort(t) for t in sold]
    out, off = {}, 0
    for c in cases:
        s = ds.seqs[c.user]
        k = len(c.history)
        ok = k < len(s) and s[k][0] == c.target
        off += not ok
        age = (float(s[k][1]) - first[c.target]) / day if ok and day else np.nan
        # purchases of the target before the request (the time of the last history item); the user's own are not in it
        before = int(np.searchsorted(sold[c.target], float(s[k - 1][1]), side="left")) if day and 0 < k <= len(s) else pop[c.target]
        out[c.cid] = (pop[c.target], age, k, float(np.mean(np.log10(pop[c.cands] + 1.0))), before)
    assert off <= 0.01 * len(cases), f"{d}: {off} of {len(cases)} cases do not match the dataset"
    return out, day is not None


def _slope(x, y):
    x = x - x.mean()
    return float(x @ (y - y.mean()) / max(x @ x, 1e-12))


def _perm(xs, ys, n=10000, seed=0):
    """Mean over the settings of the slope of y on x, and its two-sided permutation p (y shuffled inside settings)."""
    rng = np.random.default_rng(seed)
    obs = np.mean([_slope(x, y) for x, y in zip(xs, ys)])
    hits = sum(abs(np.mean([_slope(x, rng.permutation(y)) for x, y in zip(xs, ys)])) >= abs(obs) for _ in range(n))
    return float(obs), (hits + 1) / (n + 1)


def seeds_mean():
    """Per setting, over the users answered by all three 9B seeds: per-user means of the three seeds."""
    U = [per_user(runs(sp)) for sp in SEEDS]
    out = {}
    for d in SIX:
        if any(d not in u for u in U):
            continue
        cids = sorted(set.intersection(*[set(u[d]["cids"]) for u in U]))
        out[d] = {k: np.mean([u[d][k][np.searchsorted(u[d]["cids"], cids)] for u in U], 0)
                  for k in ("pos", "lst", "fused", "ent", "ent_d", "ll_u", "ll_f", "ll_d", "ll_l")}
        out[d]["cids"] = cids
        print(f"check {d}: seed 0 fused {U[0][d]['fused'].mean():.4f} list {U[0][d]['lst'].mean():.4f}; "
              f"three seeds, {len(cids)} users, fused {out[d]['fused'].mean():.4f}", flush=True)
    return out


_COV = {}


def with_covariates(S, spec=SEEDS[0]):
    for d in S:
        T = runs(spec)[d][0]["test"]
        key = (d, tuple(S[d]["cids"]))
        if key not in _COV:
            _COV[key] = covariates(d, [T[c] for c in S[d]["cids"]], spec)
        cov, dated = _COV[key]
        A = np.array([cov[c] for c in S[d]["cids"]], dtype=float)
        S[d].update(pop=A[:, 0], age=A[:, 1], hist=A[:, 2], lpop=A[:, 3], before=A[:, 4], dated=dated)
    return S


LAW = ["rtm:llm-0.8b-rtm", "rtm:kev-0.8b-rtm", "rtm:kev-4b-rtm", "rtm:kev-27b-rtm", "rtm:kev-9b", "rtm:jev-1.13.0"]


def law(S9, table):
    print("== law (V40): gain of the reader against the purchases of the target before the request", flush=True)
    tags = [t for t in os.environ.get("LAW_TAGS", "").split(",") if t] or ["9b-mean"] + LAW
    for tag in tags:
        if tag == "9b-mean":
            S = S9
        else:
            U = per_user(runs(tag))
            if not U:
                print(f"{tag}: no runs", flush=True)
                continue
            S = with_covariates(U, tag)
        table[tag], xs, ys, xo, line = {}, [], [], [], f"{tag:18s}"
        for d, r in S.items():
            m = np.asarray(r["pos"]) >= 0
            g, x, x0 = (r["ll_f"] - r["ll_l"])[m], np.log10(r["before"][m] + 1), np.log10(r["pop"][m] + 1)
            sl, p = _perm([x], [g], n=2000)
            cut = np.quantile(x, np.linspace(0, 1, 7))
            b = np.clip(np.searchsorted(cut, x, side="right") - 1, 0, 5)
            table[tag][d] = {"n": int(m.sum()), "gain": float(g.mean()), "slope": sl, "p": p, "slope_training_count": _slope(x0, g),
                             "bins": [[float(x[b == i].mean()), float(g[b == i].mean()), int((b == i).sum())] for i in range(6) if (b == i).any()]}
            xs.append(x), ys.append(g), xo.append(x0)
            line += f" | {d} {sl:+.3f} (p={p:.3f}, n={m.sum()})"
        s, p = _perm(xs, ys)
        s0, p0 = _perm(xo, ys, n=2000)
        table[tag]["all"] = {"slope": s, "p": p, "slope_training_count": s0, "p_training_count": p0, "settings": len(xs)}
        print(line + f" || all {s:+.4f} nats per tenfold, p={p:.4f}; with the training count {s0:+.4f} (p={p0:.4f})", flush=True)



def thin(S, table):
    print("== thin (V25): gain of the decision model against the evidence of the target", flush=True)
    xs = {k: [] for k in ("target evidence", "history length", "list evidence", "target age")}
    ys = {k: [] for k in xs}
    for d, r in S.items():
        m = r["pos"] >= 0
        dll = (r["ll_f"] - r["ll_l"])[m]
        table[d] = {"positive": int(m.sum()), "gain_nats": float(dll.mean()), "groups": {}}
        for name, lo, hi in GROUPS:
            g = (r["pop"] >= lo) & (r["pop"] <= (hi if hi is not None else np.inf))
            gp = g & m
            if g.sum() < 20:
                continue
            a, b = r["fused"][g], r["lst"][g]
            ci = M.bootstrap_ci((r["ll_f"] - r["ll_l"])[gp]) if gp.sum() >= 10 else (np.nan, np.nan)
            table[d]["groups"][name] = {"users": int(g.sum()), "positive": int(gp.sum()), "list": float(b.mean()), "fused": float(a.mean()),
                                        "p": M.paired_randomization(a, b), "gain_nats": float((r["ll_f"] - r["ll_l"])[gp].mean()) if gp.sum() else np.nan}
            print(f"{d:9s} {name:7s} users {g.sum():4d} target in the 30 {gp.sum():4d} | list {b.mean():.4f} fused {a.mean():.4f} "
                  f"{100 * (a.mean() / max(b.mean(), 1e-9) - 1):+.1f}% p={table[d]['groups'][name]['p']:.3f} | "
                  f"gain {table[d]['groups'][name]['gain_nats']:+.4f} nats [{ci[0]:+.4f}, {ci[1]:+.4f}]", flush=True)
        cols = {"target evidence": np.log10(r["pop"][m] + 1), "history length": np.log10(r["hist"][m] + 1), "list evidence": r["lpop"][m]}
        if r["dated"]:
            cols["target age"] = np.log10(np.maximum(r["age"][m], 0) + 1)
        line = f"{d:9s} slope of the gain (nats per tenfold):"
        for k, x in cols.items():
            ok = np.isfinite(x)
            s, p = _perm([x[ok]], [dll[ok]], n=2000)
            xs[k].append(x[ok])
            ys[k].append(dll[ok])
            table[d]["slope " + k] = [s, p]
            line += f" {k} {s:+.4f} (p={p:.3f});"
        print(line, flush=True)
    table["all"] = {}
    for k in xs:
        s, p = _perm(xs[k], ys[k])
        table["all"][k] = [s, p, len(xs[k])]
        print(f"all settings, equal weight: slope on {k} {s:+.4f} nats per tenfold, p={p:.4f} ({len(xs[k])} settings)", flush=True)


def nats(table):
    print("== nats (V26): information about the target inside the list, nats per choice, test users with the target among the options", flush=True)
    for name, specs in SIZES.items():
        prev = None
        for sp in specs:
            U = per_user(runs(sp))
            table[sp] = {}
            line, means = f"{name:3s} {sp:20s}", []
            for d, r in U.items():
                m = r["pos"] >= 0
                il, idm, both = (r["ll_l"] - r["ll_u"])[m], (r["ll_d"] - r["ll_u"])[m], (r["ll_f"] - r["ll_u"])[m]
                add = both - il
                lo, hi = M.bootstrap_ci(add)
                table[sp][d] = {"n": int(m.sum()), "list": float(il.mean()), "dm": float(idm.mean()), "both": float(both.mean()),
                                "added": float(add.mean()), "ci": [lo, hi]}
                means.append(add.mean())
                line += f" | {d} list {il.mean():.3f} dm {idm.mean():.3f} both {both.mean():.3f} added {add.mean():+.4f} [{lo:+.4f},{hi:+.4f}]"
                if prev is not None and d in prev:
                    k = sorted(set(np.array(r["cids"])[m]) & set(np.array(prev[d]["cids"])[prev[d]["pos"] >= 0]))
                    a = (r["ll_f"] - r["ll_l"])[np.searchsorted(r["cids"], k)]
                    b = (prev[d]["ll_f"] - prev[d]["ll_l"])[np.searchsorted(prev[d]["cids"], k)]
                    table[sp][d]["p_prev"] = M.paired_randomization(a, b)
                    line += f" (vs smaller p={table[sp][d]['p_prev']:.3f})"
            table[sp]["mean_added"] = float(np.mean(means))
            print(line + f" || mean added {np.mean(means):+.4f}", flush=True)
            prev = U


def cascade(S, table):
    print("== cascade (V27): share of the gain of the decision model kept when only a share of the users is read", flush=True)
    shares = (0.1, 0.2, 0.3, 0.5, 0.7)
    for d, r in S.items():
        full = r["fused"].mean() - r["lst"].mean()
        table[d] = {"gain": float(full)}
        gates = {"list entropy, high first": -r["ent"], "list entropy, low first": r["ent"], "list evidence, thin first": r["lpop"],
                 "history, short first": r["hist"]}
        for g, key in gates.items():
            order = np.argsort(key, kind="stable")
            kept = []
            for f in shares:
                read = np.zeros(len(order), bool)
                read[order[: int(round(f * len(order)))]] = True
                kept.append(float((np.where(read, r["fused"], r["lst"]).mean() - r["lst"].mean()) / full) if abs(full) > 1e-9 else np.nan)
            table[d][g] = kept
            print(f"{d:9s} gain {full:+.4f} | {g:26s} " + " ".join(f"{int(100 * f)}%: {100 * k:5.1f}" for f, k in zip(shares, kept)), flush=True)


def _auc(score, y):
    """Probability that a case of y has the higher score than a case outside it (ties count half)."""
    from scipy.stats import rankdata
    r, n1 = rankdata(score), int(y.sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / max(n1 * (len(y) - n1), 1))


def _logit_cv(X, y, seed=0):
    """Scores of a logistic model of y, each half of the cases scored by the model fitted on the other half."""
    X = np.column_stack([np.ones(len(X)), (X - X.mean(0)) / (X.std(0) + 1e-9)])
    half = np.random.default_rng(seed).permutation(len(y)) % 2
    out = np.zeros(len(y))
    for f in (0, 1):
        tr, b = half != f, np.zeros(X.shape[1])
        for _ in range(50):
            q = 1.0 / (1.0 + np.exp(-X[tr] @ b))
            g = X[tr].T @ (y[tr] - q) - 1e-2 * b
            H = -(X[tr] * (q * (1 - q))[:, None]).T @ X[tr] - 1e-2 * np.eye(X.shape[1])
            b = b - np.linalg.solve(H, g)
        out[half == f] = X[half == f] @ b
    return out


def absent(S, table):
    print("== absent (V29, step 1): area under the curve for 'the target is not among the 30', test users", flush=True)
    for d, r in S.items():
        y = r["pos"] < 0
        cols = {"list entropy": r["ent"], "reader entropy": r["ent_d"], "list evidence": -r["lpop"], "history length": -np.log10(r["hist"] + 1)}
        table[d] = {"absent": float(y.mean())}
        line = f"{d:9s} absent {y.mean():.3f} |"
        for k, x in cols.items():
            table[d][k] = _auc(x, y)
            line += f" {k} {table[d][k]:.3f};"
        table[d]["list only"] = _auc(_logit_cv(np.column_stack([cols[k] for k in cols if k != "reader entropy"]), y.astype(float)), y)
        table[d]["all"] = _auc(_logit_cv(np.column_stack(list(cols.values())), y.astype(float)), y)
        print(line + f" mix without the reader {table[d]['list only']:.3f}; mix of all {table[d]['all']:.3f}", flush=True)


def deep(table):
    print("== deep (R18 runs, 50 options, no new training): targets at positions 31 to 50 of the list", flush=True)
    for sp in ("rtm50:kev-0.8b-rtm", "rtm50:kev-4b-rtm"):
        U = per_user(runs(sp))
        table[sp] = {}
        for d, r in U.items():
            m, top = r["pos"] >= 30, r["pos"] >= 0
            if m.sum() < 5:
                continue
            table[sp][d] = {"in_50": int(top.sum()), "deep": int(m.sum()), "reader_top10": float((r["rank_d"][m] <= 10).mean()),
                            "fused_top10": float((r["rank_f"][m] <= 10).mean()), "reader_top30": float((r["rank_d"][m] <= 30).mean()),
                            "fused_median_rank": float(np.median(r["rank_f"][m])), "reader_median_rank": float(np.median(r["rank_d"][m]))}
            t = table[sp][d]
            print(f"{sp:22s} {d:6s} target in the 50: {t['in_50']}, at 31 to 50: {t['deep']} | reader alone puts them in its top 10: "
                  f"{100 * t['reader_top10']:.1f}%, top 30: {100 * t['reader_top30']:.1f}%, median rank {t['reader_median_rank']:.0f} | "
                  f"fused top 10: {100 * t['fused_top10']:.1f}%, median rank {t['fused_median_rank']:.0f} (a random order: 20% and 25)", flush=True)


if __name__ == "__main__":
    out, what = sys.argv[1], sys.argv[2:] or ["thin", "nats", "cascade"]
    table = {}
    if "thin" in what or "cascade" in what or "absent" in what or "law" in what:
        S = with_covariates(seeds_mean())
        if "thin" in what:
            table["thin"] = {}
            thin(S, table["thin"])
        if "cascade" in what:
            table["cascade"] = {}
            cascade(S, table["cascade"])
        if "law" in what:
            table["law"] = {}
            law(S, table["law"])
        if "absent" in what:
            table["absent"] = {}
            absent(S, table["absent"])
        json.dump(table, open(out, "w"), indent=1)
    if "deep" in what:
        table["deep"] = {}
        deep(table["deep"])
    if "nats" in what:
        table["nats"] = {}
        nats(table["nats"])
    json.dump(table, open(out, "w"), indent=1)
