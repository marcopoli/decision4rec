"""Do decision models miss the item constant? A test on stored runs; no model is called.

In a discrete choice model the utility of an alternative has two parts: a constant specific to the
alternative, which captures how often it is chosen by anyone, and a taste part that depends on the
person. The stratified results (`stratify`) show that decision models rank rare and mid-popular
targets about as well as trained sequential models but lose heavily on popular targets. If what they
miss is the item constant, adding the log popularity of each candidate to the decision model's
utilities, with one weight fitted on validation users, should close much of that gap.

Variants, every weight fitted on validation cases and applied once to test cases:

  pop only       z(log(1 + popularity)), the pool reranked by popularity
  dm             the decision model alone (Luce utilities of the arm)
  dm+pop         z(theta) + a z(log(1 + popularity))
  dm+fuse        z(theta) + g z(retrieval score), as in the report
  dm+fuse+pop    z(theta) + g z(retrieval score) + a z(log(1 + popularity))
  dm+gate        z(retrieval score) + w z(theta), with w = w_tail on candidates with fewer than
                 HEAD training interactions and w = w_head on the others
  dm+ugate       z(retrieval score) + w(u) z(theta), with w(u) = w_niche for users whose history is less
                 popular than the median validation user (mean log popularity of the history items) and
                 w(u) = w_main for the others: a two-class mixture of choice models with an observed class
  dm+ugate+pop   the same plus a z(log(1 + popularity))
  dm+shrink      z(retrieval score) + w k/(k + n_c) z(theta), where n_c counts the training interactions of
                 candidate c: the decision model acts as a prior that the collaborative evidence overrides as
                 it accumulates, as in empirical Bayes shrinkage

Control row `retriever+pop`, z(retrieval score) + a z(log(1 + popularity)), shows how much of a gain
comes from popularity alone; every variant is also compared with it.

Popularity counts the training part of every sequence (all but the last two items), as the
baselines see it, so no test or validation interaction enters it. This is an exploratory analysis:
the variant family is chosen after seeing these numbers, so a confirmatory test needs new users.
"""

from __future__ import annotations

import itertools
import json
import os
from collections import Counter
from typing import Callable, Optional

import numpy as np

from ..data.schema import Dataset, load_cases
from ..eval import metrics as M
from ..operators.luce import Obs, fit_luce
from .benchmark import BIG, _ndcg10
from .runner import case_paths
from .stratify import POP_BINS, _bucket, _tag, baseline_ranks

SPLITS = ("valid", "test")
ALPHAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
GAMMAS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
WEIGHTS = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
HEAD = 100
REF_MODELS = ("gru4rec", "sasrec", "sasrecf", "sasrecf_dm", "itemknn", "seqrules", "contentknn", "transtext")
KS = (10.0, 30.0, 100.0, 300.0)
KINDS = ("pop only", "dm", "pop", "fuse", "fuse+pop", "gate", "ugate", "ugate+pop", "shrink")
ARMS = ("wholepool@p0", "luce_m10_r2", "decision4rec", "prior")

_z = M.zscore


def _ndcg(scores: dict[int, np.ndarray], cases: dict[int, object]) -> np.ndarray:
    cids = sorted(cases)
    return np.array([M.per_case_metrics(M.target_rank(scores[c], cases[c].target_pos))["ndcg@10"] for c in cids])


def _tune(make: Callable, grid, cases: dict) -> tuple:
    """The grid point with the best validation NDCG@10; ties go to the first point, the smallest weight."""
    best, best_v = None, -1.0
    for p in grid:
        v = float(_ndcg(make(p, "valid"), cases["valid"]).mean())
        if v > best_v + 1e-12:
            best, best_v = p, v
    return best, best_v


def _theta(run_dir: str, arm: str, cases: dict) -> Optional[dict]:
    out = {}
    for s in SPLITS:
        path = os.path.join(run_dir, f"{arm}.{s}.jsonl")
        if not os.path.exists(path):
            return None
        out[s] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if not rec.get("obs") or rec["cid"] not in cases[s]:
                    continue
                obs = [Obs.from_json(o) for o in rec["obs"]]
                out[s][rec["cid"]] = fit_luce(obs, len(cases[s][rec["cid"]].cands)).theta
        if set(out[s]) != set(cases[s]):
            return None
    return out


def _fmt(d: dict) -> str:
    out = f"{d['diff']:+.4f} (p={d['p']:.3f})"
    return out + (f" [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]" if "ci95" in d else "")


def asc_study(run_dirs: list[str], arms: Optional[list[str]] = None, head: int = HEAD,
              only: Optional[list[str]] = None) -> dict:
    """`only` restricts the variants to these kinds (see KINDS), for a pre-registered analysis; it also adds
    95% bootstrap intervals of every difference."""
    arms = list(arms or ARMS)
    if only is not None and set(only) - set(KINDS):
        raise SystemExit(f"unknown variant kinds {sorted(set(only) - set(KINDS))}; choose from {KINDS}")
    want = (lambda k: True) if only is None else (lambda k: k in only)
    datasets: dict[str, tuple] = {}
    results, skipped = [], []
    md = ["# Do decision models miss the item constant?", "",
          "NDCG@10 on full-catalogue ranks of the test users. Weights fitted on validation users. Differences with "
          "paired randomisation p. Exploratory: the variant family was chosen after the stratified results.", ""]
    for rd in run_dirs:
        man = os.path.join(rd, "manifest.json")
        if not os.path.exists(man):
            skipped.append(rd)
            continue
        with open(man) as f:
            cfg = json.load(f)["config"]
        paths = case_paths(cfg)
        cases = {s: {c.cid: c for c in load_cases(paths[s])} for s in SPLITS}
        if cfg["dataset"] not in datasets:
            ds = Dataset.load(cfg["dataset"])
            datasets[cfg["dataset"]] = (ds, Counter(int(x[0]) for s in ds.seqs for x in s[:-2]))
        ds, pop = datasets[cfg["dataset"]]
        lp = {s: {cid: np.log1p([pop.get(i, 0) for i in c.cands]) for cid, c in cases[s].items()} for s in SPLITS}
        zlp = {s: {cid: _z(v) for cid, v in lp[s].items()} for s in SPLITS}
        zr = {s: {cid: _z(c.rscores) for cid, c in cases[s].items()} for s in SPLITS}
        is_head = {s: {cid: np.array([pop.get(i, 0) >= head for i in c.cands]) for cid, c in cases[s].items()}
                   for s in SPLITS}
        taste = {s: {cid: float(np.mean(np.log1p([pop.get(i, 0) for i in c.history]))) for cid, c in cases[s].items()}
                 for s in SPLITS}
        cut = float(np.median(list(taste["valid"].values())))
        main = {s: {cid: taste[s][cid] >= cut for cid in cases[s]} for s in SPLITS}
        test_cids = sorted(cases["test"])

        refs = {"retriever (pool)": _ndcg(zr["test"], cases["test"])}
        found_b = baseline_ranks(cfg, [[c.user, c.target] for c in load_cases(paths["test"])],
                                 REF_MODELS)
        for m in REF_MODELS:
            if m in found_b:
                refs[m] = _ndcg10(np.array([BIG if found_b[m].get(str(c)) is None else found_b[m][str(c)]
                                            for c in test_cids], dtype=float))

        def pool_pop(a_, s):
            return {c: zr[s][c] + a_ * zlp[s][c] for c in cases[s]}
        a_pool, _ = _tune(pool_pop, ALPHAS, cases)
        refs["retriever+pop"] = _ndcg(pool_pop(a_pool, "test"), cases["test"])
        buckets = np.array([_bucket(pop.get(cases["test"][c].target, 0), POP_BINS) for c in test_cids])
        tail = np.array([pop.get(cases["test"][c].target, 0) < head for c in test_cids])
        groups = [b[2] for b in POP_BINS if (buckets == b[2]).any()]

        rows = []

        def add(name: str, weights: str, scores: dict) -> None:
            v = _ndcg(scores, cases["test"])
            row = {"variant": name, "weights": weights, "ndcg@10": float(v.mean()),
                   "by_popularity": {g: float(v[buckets == g].mean()) for g in groups}, "vs": {}}
            row["tail"] = {"n": int(tail.sum()), "ndcg@10": float(v[tail].mean()) if tail.any() else float("nan"),
                           "vs": {}}
            for ref, rv in refs.items():
                row["vs"][ref] = {"diff": float(v.mean() - rv.mean()), "p": M.paired_randomization(v, rv)}
                if only is not None:
                    row["vs"][ref]["ci95"] = list(M.bootstrap_ci(v - rv))
                if tail.any():
                    row["tail"]["vs"][ref] = {"diff": float(v[tail].mean() - rv[tail].mean()),
                                              "p": M.paired_randomization(v[tail], rv[tail])}
                    if only is not None:
                        row["tail"]["vs"][ref]["ci95"] = list(M.bootstrap_ci(v[tail] - rv[tail]))
            rows.append(row)

        if want("pop only"):
            add("pop only", "", zlp["test"])
        shrink_f = {sp: {c: np.array([pop.get(i, 0) for i in cases[sp][c].cands], dtype=float) for c in cases[sp]}
                    for sp in SPLITS}
        found, check = [], {}
        rk = os.path.join(rd, "ranks.json")
        stored = json.load(open(rk)) if os.path.exists(rk) else {}
        for arm in arms:
            th = _theta(rd, arm, cases)
            if th is None:
                continue
            found.append(arm)
            zth = {s: {c: _z(th[s][c]) for c in cases[s]} for s in SPLITS}
            if want("dm"):
                add(f"{arm}", "", {c: th["test"][c] for c in cases["test"]})

            def pop_(a_, s, zth=zth):
                return {c: zth[s][c] + a_ * zlp[s][c] for c in cases[s]}
            if want("pop"):
                a, _ = _tune(pop_, ALPHAS, cases)
                add(f"{arm}+pop", f"a={a}", pop_(a, "test"))

            def fuse(g, s, zth=zth):
                return {c: zth[s][c] + g * zr[s][c] for c in cases[s]}
            if want("fuse"):
                g, _ = _tune(fuse, GAMMAS, cases)
                add(f"{arm}+fuse", f"g={g}", fuse(g, "test"))

            def fuse_pop(ga, s, zth=zth):
                return {c: zth[s][c] + ga[0] * zr[s][c] + ga[1] * zlp[s][c] for c in cases[s]}
            if want("fuse+pop"):
                ga, _ = _tune(fuse_pop, list(itertools.product(GAMMAS, ALPHAS)), cases)
                add(f"{arm}+fuse+pop", f"g={ga[0]}, a={ga[1]}", fuse_pop(ga, "test"))

            def gate(w, s, zth=zth):
                return {c: zr[s][c] + np.where(is_head[s][c], w[1], w[0]) * zth[s][c] for c in cases[s]}
            if want("gate"):
                w, _ = _tune(gate, list(itertools.product(WEIGHTS, WEIGHTS)), cases)
                add(f"{arm}+gate", f"w_tail={w[0]}, w_head={w[1]}", gate(w, "test"))

            def ugate(w, s, zth=zth):
                return {c: zr[s][c] + (w[1] if main[s][c] else w[0]) * zth[s][c] + w[2] * zlp[s][c] for c in cases[s]}
            if want("ugate"):
                w, _ = _tune(ugate, [(x, y, 0.0) for x, y in itertools.product(WEIGHTS, WEIGHTS)], cases)
                add(f"{arm}+ugate", f"w_niche={w[0]}, w_main={w[1]}", ugate(w, "test"))
            if want("ugate+pop"):
                w, _ = _tune(ugate, list(itertools.product(WEIGHTS, WEIGHTS, ALPHAS)), cases)
                add(f"{arm}+ugate+pop", f"w_niche={w[0]}, w_main={w[1]}, a={w[2]}", ugate(w, "test"))

            def shrink(wk, s, zth=zth):
                return {c: zr[s][c] + wk[0] * (wk[1] / (wk[1] + shrink_f[s][c])) * zth[s][c] for c in cases[s]}
            if want("shrink"):
                wk, _ = _tune(shrink, list(itertools.product(WEIGHTS, KS)), cases)
                add(f"{arm}+shrink", f"w={wk[0]}, k={wk[1]:g}", shrink(wk, "test"))
            if arm in stored and "test" in stored[arm]:          # the Luce fit here should agree with the report
                mine = _ndcg({c: th["test"][c] for c in cases["test"]}, cases["test"]).mean()
                theirs = _ndcg10(np.array([BIG if stored[arm]["test"].get(str(c)) is None
                                           else stored[arm]["test"][str(c)] for c in test_cids], dtype=float)).mean()
                check[arm] = float(mine - theirs)
        res = {"run": os.path.basename(rd.rstrip("/")), "dataset": ds.name, "tag": _tag(cfg["name"]),
               "n_test": len(test_cids), "head": head, "arms": found, "refs": {k: float(v.mean()) for k, v in refs.items()},
               "refs_by_popularity": {k: {g: float(v[buckets == g].mean()) for g in groups} for k, v in refs.items()},
               "groups": groups, "n_by_popularity": {g: int((buckets == g).sum()) for g in groups}, "rows": rows,
               "retriever+pop": {"a": a_pool}, "niche_cut": cut,
               "check_against_report": check}
        results.append(res)
        ref_names = list(refs)
        md += [f"## {ds.name}, run `{res['run']}`", "",
               "| variant | weights | NDCG@10 | " + " | ".join(f"vs {n}" for n in ref_names) + " | "
               + " | ".join(f"pop {g}" for g in groups) + " |",
               "|---|---|---|" + "---|" * (len(ref_names) + len(groups)),
               "| n users | | | " + " | ".join("" for _ in ref_names) + " | "
               + " | ".join(str(res["n_by_popularity"][g]) for g in groups) + " |"]
        for n in ref_names:
            md.append(f"| {n} | | {res['refs'][n]:.4f} | " + " | ".join("" for _ in ref_names) + " | "
                      + " | ".join(f"{res['refs_by_popularity'][n][g]:.4f}" for g in groups) + " |")
        for row in rows:
            md.append(f"| {row['variant']} | {row['weights']} | {row['ndcg@10']:.4f} | "
                      + " | ".join(_fmt(row['vs'][n]) for n in ref_names)
                      + " | " + " | ".join(f"{row['by_popularity'][g]:.4f}" for g in groups) + " |")
        if tail.any():
            md += ["", f"Users whose target has fewer than {head} training interactions (n = {int(tail.sum())}):", "",
                   "| variant | NDCG@10 | " + " | ".join(f"vs {n}" for n in ref_names) + " |",
                   "|---|---|" + "---|" * len(ref_names)]
            for n in ref_names:
                md.append(f"| {n} | {float(refs[n][tail].mean()):.4f} | " + " | ".join("" for _ in ref_names) + " |")
            for row in rows:
                t = row["tail"]
                md.append(f"| {row['variant']} | {t['ndcg@10']:.4f} | "
                          + " | ".join(_fmt(t['vs'][n]) for n in ref_names) + " |")
        md.append("")
        off = {k: v for k, v in check.items() if abs(v) > 1e-6}
        if off:
            md.append(f"Warning: the decision-model-only NDCG@10 differs from ranks.json by {off}.")
        md.append("")
    if skipped:
        md += ["Skipped (no manifest.json): " + ", ".join(skipped), ""]
    return {"runs": results, "skipped": skipped, "markdown": "\n".join(md)}
