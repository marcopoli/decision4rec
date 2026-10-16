"""Where does the decision model win? NDCG@10 of every method per group of test users.

Test users are grouped by the popularity of their target item in the training interactions (the
part of each sequence the baselines are trained on) and by the length of their history. The table
also gives, per group, how often the target is inside the candidate set at all, the ceiling of any
reranker. Only stored files are read: cases, dataset, the ranks of Decision4Rec runs and of the trained
baselines (the report of a run is written first if it is missing). No model is called, so this
runs in seconds on a machine without GPU.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Optional

import numpy as np

from ..data.schema import Dataset, load_cases
from ..eval import metrics as M
from .benchmark import BIG, _bench_dir, _ndcg10
from .runner import case_paths

POP_BINS = ((0, 0, "0"), (1, 4, "1 to 4"), (5, 19, "5 to 19"), (20, 99, "20 to 99"), (100, 499, "100 to 499"),
            (500, None, "500+"))
HIST_BINS = ((1, 4, "1 to 4"), (5, 9, "5 to 9"), (10, 19, "10 to 19"), (20, 49, "20 to 49"), (50, None, "50+"))
BASELINES = ("sasrecf", "sasrec", "gru4rec", "itemknn", "pop", "sasrecf_dm", "sasrec_dm", "sasrecf_emb")
ARMS = ("wholepool@p0", "luce_m10_r2", "decision4rec", "prior", "decision4rec_masked", "wholepool@p0+fuse",
        "luce_m10_r2+fuse", "decision4rec+fuse")


def _bucket(x: int, bins) -> str:
    for lo, hi, label in bins:
        if x >= lo and (hi is None or x <= hi):
            return label
    return bins[0][2]


def _tag(run_name: str) -> str:
    return run_name.split("_main_", 1)[1] if "_main_" in run_name else run_name


def baseline_ranks(cfg: dict, expected: list, models) -> dict[str, dict]:
    """Stored per-user target ranks of trained baselines scored on exactly these test users and targets.

    Track B case sets built with the same seed hold the same users and targets whatever the retriever,
    so the benchmark folder of another case set of the same dataset may hold them. `expected` lists
    [user, target] in case order; a file is used only if its list is identical."""
    own = _bench_dir(cfg)
    root = os.path.dirname(own)
    dirs = [own] + (sorted(os.path.join(root, d) for d in os.listdir(root)) if os.path.isdir(root) else [])
    out: dict[str, dict] = {}
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(d, fn)) as f:
                b = json.load(f)
            m = b.get("model")
            if fn != f"{m}.json":                    # extra training seeds are not the reference run
                continue
            if m in models and m not in out and b.get("users", {}).get("test") == expected:
                out[m] = b["ranks"]["test"]
    return out


def _ranks(row: dict, cids: list[str]) -> np.ndarray:
    return np.array([BIG if row.get(c) is None else row[c] for c in cids], dtype=float)


def stratify(run_dirs: list[str], arms: Optional[list[str]] = None, pairs: Optional[list[tuple[str, str]]] = None) -> dict:
    """One table per case set and grouping. Runs that share a case set share the baseline rows."""
    arms = list(arms or ARMS)
    groups: dict[str, dict] = {}
    skipped = []
    for rd in run_dirs:
        man, rk = os.path.join(rd, "manifest.json"), os.path.join(rd, "ranks.json")
        if not os.path.exists(man):
            skipped.append(rd)
            continue
        if not os.path.exists(rk):                 # a run whose report was never written
            from .report import report
            report(rd)
        with open(man) as f:
            cfg = json.load(f)["config"]
        with open(rk) as f:
            ranks = json.load(f)
        key = case_paths(cfg)["test"]
        g = groups.setdefault(key, {"cfg": cfg, "rows": {}, "runs": []})
        g["runs"].append(os.path.basename(rd.rstrip("/")))
        tag = _tag(cfg["name"])
        if "retriever" in ranks and "retriever (pool)" not in g["rows"]:
            g["rows"]["retriever (pool)"] = ranks["retriever"]["test"]
        for arm in arms:
            if arm in ranks:
                g["rows"][f"{tag}: {arm}"] = ranks[arm]["test"]

    datasets: dict[str, tuple] = {}
    out, md = [], ["# Where the decision model wins", "",
                   "NDCG@10 on full-catalogue ranks per group of test users. `in pool`: share of users whose target "
                   "is in the candidate set, the ceiling of any reranker. Popularity: interactions of the target item "
                   "in the training part of all sequences."]
    for key, g in groups.items():
        cfg = g["cfg"]
        cases = load_cases(key)
        cids = [str(c.cid) for c in cases]
        if cfg["dataset"] not in datasets:
            ds = Dataset.load(cfg["dataset"])
            datasets[cfg["dataset"]] = (ds, Counter(int(x[0]) for s in ds.seqs for x in s[:-2]))
        ds, pop = datasets[cfg["dataset"]]
        found = baseline_ranks(cfg, [[c.user, c.target] for c in cases], BASELINES)
        rows = {m: found[m] for m in BASELINES if m in found}
        rows.update(g["rows"])
        gains = {name: _ndcg10(_ranks(r, cids)) for name, r in rows.items() if all(c in r for c in cids)}
        in_pool = np.array([c.target_pos >= 0 for c in cases], dtype=float)
        features = {"target popularity": ([_bucket(pop.get(c.target, 0), POP_BINS) for c in cases], POP_BINS),
                    "history length": ([_bucket(len(c.history), HIST_BINS) for c in cases], HIST_BINS)}
        res = {"cases": key, "dataset": ds.name, "runs": g["runs"], "n": len(cases), "tables": {}}
        for feat, (labels, bins) in features.items():
            labels = np.array(labels)
            present = [b[2] for b in bins if (labels == b[2]).any()]
            table = {"n": {b: int((labels == b).sum()) for b in present},
                     "in pool": {b: float(in_pool[labels == b].mean()) for b in present}}
            for name, v in gains.items():
                table[name] = {b: float(v[labels == b].mean()) for b in present} | {"all": float(v.mean())}
            tests = []
            for a, b2 in pairs or []:
                if a in gains and b2 in gains:
                    for b in present:
                        m = labels == b
                        tests.append({"a": a, "b": b2, "group": b, "n": int(m.sum()),
                                      "diff": float(gains[a][m].mean() - gains[b2][m].mean()),
                                      "p": M.paired_randomization(gains[a][m], gains[b2][m])})
            res["tables"][feat] = {"groups": present, "table": table, "tests": tests}
            md += ["", f"## {ds.name}, {os.path.basename(key)}, by {feat}", "",
                   "| method | " + " | ".join(present) + " | all |", "|---|" + "---|" * (len(present) + 1),
                   "| n users | " + " | ".join(str(table["n"][b]) for b in present) + f" | {len(cases)} |",
                   "| in pool | " + " | ".join(f"{table['in pool'][b]:.3f}" for b in present)
                   + f" | {in_pool.mean():.3f} |"]
            for name in gains:
                md.append(f"| {name} | " + " | ".join(f"{table[name][b]:.4f}" for b in present)
                          + f" | {table[name]['all']:.4f} |")
            for t in tests:
                md.append(f"* {t['a']} against {t['b']}, {feat} {t['group']} (n = {t['n']}): "
                          f"{t['diff']:+.4f}, paired randomisation p = {t['p']:.4f}")
        out.append(res)
    if skipped:
        md += ["", "Skipped (no manifest.json): " + ", ".join(skipped)]
    return {"datasets": out, "skipped": skipped, "markdown": "\n".join(md) + "\n"}
