"""The Decision4Rec system: a trained sequential model plus typed decisions where it has no evidence.

For every user the backbone, a trained sequential model chosen on validation among the ones given (SASRec,
SASRecF, SASRecF with decision-model facets), scores the whole catalogue. Two kinds of typed decisions add to
the standardised backbone score of a set U of items:

* relations: items judged complement, substitute or same series of the last history items (E2) join U and get
  beta * (mix . relation probabilities);
* the evidence gate: the decision model's answer theta over the candidate pool gets the weight
  w * k / (k + n_i), where n_i counts the training interactions of item i. The decision model counts fully for an
  item nobody bought and fades as interactions accumulate, like a prior in empirical Bayes;
* a text prior with the same evidence weight, v * k / (k + n_i), on the text similarity between an item and the
  last history items; the nearest items by text also join U, so new items can be reached. It is the strongest
  cold-start signal in the 2026 literature, and the rows with and without the decision model show what the
  decision model adds beyond it.

U is the candidate pool of the cases plus the related items; every other item keeps its backbone score, so the
ranking is over the whole catalogue and with all weights at 0 it is the backbone's own ranking. Weights are
fitted on validation users by coordinate ascent (relations, then gate, then relations again) and applied once
to test users. Rows: backbone, + relations, + gate, Decision4Rec (both), compared with every trained and
training-free baseline stored for the same users.
"""

from __future__ import annotations

import itertools
import json
import os
from collections import Counter
from typing import Optional

import numpy as np

from ..data.schema import Dataset
from ..eval import metrics as M
from ..experiments.benchmark import BIG, _bench_dir, _ndcg10, baseline_root, prepare_dataset
from ..experiments.runner import case_paths, get_cases, load_config
from ..experiments.stratify import baseline_ranks
from .relations import _features, read_relations

BETAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
MIXES = [np.array(m, dtype=float) for m in itertools.product((0.0, 0.5, 1.0), repeat=3) if any(m)]
WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0)
KS = (1.0, 10.0, 100.0)
REFS = ("pop", "itemknn", "seqrules", "contentknn", "transtext", "gru4rec", "sasrec", "sasrecf", "sasrecf_dm")


def _backbone(cfg: dict, ds: Dataset, names: tuple, device: Optional[str]):
    """The candidate backbone with the best validation NDCG@10 in model selection, loaded from its checkpoint."""
    from ..models import baselines as B
    bench = _bench_dir(cfg)
    best = None
    for name in names:
        path = os.path.join(bench, f"{name}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            b = json.load(f)
        if best is None or b["valid_ndcg10_model_selection"] > best["valid_ndcg10_model_selection"]:
            best = b
    if best is None:
        raise SystemExit(f"no backbone among {names} in {bench}; run the baselines of this config first")
    name = best["model"]
    prepare_dataset(cfg, ds, [name])
    dir_ = best.get("dir") or B.tune(name, ds, baseline_root(cfg, ds, name), (cfg.get("benchmark") or {}).get("grids", {}).get(name),
                                     common=(cfg.get("benchmark") or {}).get("common_hp"), device=device)["dir"]
    return name, B.load(name, dir_, ds, device)


def _theta(run_dir: Optional[str], arm: str, cases: dict, cfg: dict) -> Optional[dict]:
    if not run_dir:
        return None
    with open(os.path.join(run_dir, "manifest.json")) as f:
        run_cfg = json.load(f)["config"]
    if case_paths(run_cfg)["test"] != case_paths(cfg)["test"]:
        raise SystemExit(f"{run_dir} was run on other cases than the config")
    from ..experiments.asc import _theta as fit_theta
    th = fit_theta(run_dir, arm, cases)
    if th is None:
        raise SystemExit(f"{run_dir}: arm {arm} has no complete records")
    return th


def _prepare(split: str, cases: list, scorer, rel: dict, cooc: dict, theta: Optional[dict], pop: Counter,
             last: int, decay: float, text: Optional[np.ndarray] = None, n_text: int = 20) -> list[tuple]:
    rows = []
    for s0 in range(0, len(cases), 256):
        chunk = cases[s0: s0 + 256]
        full = scorer.scores([c.user for c in chunk], [c.history for c in chunk]).cpu().numpy().astype(np.float64)
        for c, sc in zip(chunk, full):
            sc[np.asarray(c.history, dtype=int)] = np.nan
            mu, sd = np.nanmean(sc), np.nanstd(sc) + 1e-9
            z = (sc - mu) / sd
            feats = _features(c, rel, cooc, last, decay) if rel else {}
            hist = set(c.history)
            u = list(dict.fromkeys(list(c.cands) + [b for b in feats if b not in hist]))
            q = None
            if text is not None:
                q = sum(0.5 ** r * text[h] for r, h in enumerate(reversed(c.history[-3:])))
                sim = text @ q
                sim[np.asarray(c.history, dtype=int)] = -np.inf
                near = np.argpartition(-sim, n_text)[: n_text]
                u = list(dict.fromkeys(u + [int(b) for b in near[np.argsort(-sim[near])] if int(b) not in hist]))
            ua = np.asarray(u, dtype=int)
            tx = M.zscore(text[ua] @ q) if q is not None and len(u) else np.zeros(len(u))
            relx = np.array([feats.get(b, np.zeros(5))[:3] for b in u]) if u else np.zeros((0, 3))
            th = np.zeros(len(u))
            if theta is not None:
                t = np.asarray(theta[split][c.cid], dtype=float)
                th[: len(c.cands)] = M.zscore(t)
            n = np.array([pop.get(int(b), 0) for b in u], dtype=float)
            mask = np.ones(len(sc), dtype=bool)
            mask[ua] = False
            mask[np.asarray(c.history, dtype=int)] = False
            in_u = c.target in set(u)
            if not in_u:
                mask[c.target] = False
            rest = np.sort(z[mask])
            rows.append((z[ua], relx, th, n, rest, u.index(c.target) if in_u else -1, float(z[c.target]),
                         pop.get(c.target, 0), tx))
    return rows


def _ranks(rows: list[tuple], beta: float, mix: np.ndarray, w: float, v: float, k: float) -> np.ndarray:
    out = np.empty(len(rows))
    for r, (zu, relx, th, n, rest, pos, zt, _, tx) in enumerate(rows):
        s = zu + (beta * relx @ mix if len(zu) else 0.0) + (k / (k + n)) * (w * th + v * tx)
        st = s[pos] if pos >= 0 else zt
        above_u = int(np.sum(s >= st)) - (1 if pos >= 0 else 0)
        above_rest = len(rest) - int(np.searchsorted(rest, st, side="left"))
        out[r] = 1 + above_u + above_rest                              # ties count against the target
    return out


def _ndcg(ranks: np.ndarray) -> np.ndarray:
    return np.where(ranks <= 10, 1.0 / np.log2(ranks + 1.0), 0.0)


def system_eval(cfg_path: str, run_dir: Optional[str] = None, arm: str = "wholepool@p0",
                backbones: tuple = ("sasrec", "sasrecf", "sasrecf_dm"), head: int = 100,
                device: Optional[str] = None) -> dict:
    from ..graph.kg import ItemKG
    cfg = load_config(cfg_path)
    ds = Dataset.load(cfg["dataset"])
    cases = get_cases(cfg, ds)
    name, scorer = _backbone(cfg, ds, backbones, device)
    rcfg = cfg.get("relations") or {}
    rel = read_relations(rcfg["path"]) if rcfg.get("path") else {}
    cooc = ItemKG(ds).cooc if rel else {}
    theta = _theta(run_dir, arm, {sp: {c.cid: c for c in cs} for sp, cs in cases.items()}, cfg)
    pop = Counter(int(x[0]) for s in ds.seqs for x in s[:-2])
    last, decay = int(rcfg.get("last", 3)), float(rcfg.get("decay", 0.7))
    text = None
    if cfg.get("item_text") or (cfg.get("retriever") or {}).get("embedder"):
        prepare_dataset(cfg, ds, ["contentknn"])
        m = np.asarray(ds.extra["item_text_matrix"], dtype=np.float64)
        text = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
    data = {sp: _prepare(sp, cases[sp], scorer, rel, cooc, theta, pop, last, decay, text) for sp in ("valid", "test")}

    def val(beta, mix, w, v, k):
        return float(_ndcg(_ranks(data["valid"], beta, mix, w, v, k)).mean())

    zero = MIXES[-1]
    off = (0.0, zero)
    ws = WEIGHTS if theta else (0.0,)
    vs = WEIGHTS if text is not None else (0.0,)
    best_rel = max(((b, m) for b in BETAS for m in MIXES), key=lambda t: val(*t, 0.0, 0.0, 1.0)) if rel else off
    best_text = max(((x, k) for x in vs for k in KS), key=lambda t: val(*off, 0.0, *t))
    best_gate = max(((w, k) for w in ws for k in KS), key=lambda t: val(*off, t[0], 0.0, t[1]))
    best_both = max(((w, x, k) for w in ws for x in vs for k in KS), key=lambda t: val(*off, *t))
    joint = max(((w, x, k) for w in ws for x in vs for k in KS), key=lambda t: val(*best_rel, *t))
    joint_rel = max(((b, m) for b in BETAS for m in MIXES), key=lambda t: val(*t, *joint)) if rel else off
    variants = {"backbone": (*off, 0.0, 0.0, 1.0)}
    if rel:
        variants["+ relations"] = (*best_rel, 0.0, 0.0, 1.0)
    if text is not None:
        variants["+ text prior (no decision model)"] = (*off, 0.0, *best_text)
    if theta:
        variants["+ decision gate"] = (*off, best_gate[0], 0.0, best_gate[1])
    if theta and text is not None:
        variants["+ text prior + decision gate"] = (*off, *best_both)
    if rel or theta:
        variants["Decision4Rec (relations + text prior + decision gate)"] = (*joint_rel, *joint)

    test = cases["test"]
    target_n = np.array([r[7] for r in data["test"]])
    groups = {"all": np.ones(len(test), dtype=bool), "unseen target (no training interaction)": target_n == 0,
              f"rare target (1 to {head - 1})": (target_n > 0) & (target_n < head), f"frequent target ({head}+)": target_n >= head}
    found = baseline_ranks(cfg, [[c.user, c.target] for c in test], REFS)
    refs = {m: _ndcg10(np.array([BIG if found[m].get(str(c.cid)) is None else found[m][str(c.cid)] for c in test],
                                dtype=float)) for m in REFS if m in found}
    rows = []
    gains = {}
    for vname, (b, mix, w, x, k) in variants.items():
        v = _ndcg(_ranks(data["test"], b, mix, w, x, k))
        gains[vname] = v
        row = {"variant": vname, "weights": {"beta": b, "mix": mix.tolist(), "w": w, "v": x, "k": k}, "groups": {}}
        for g, m in groups.items():
            if not m.any():
                continue
            cell = {"n": int(m.sum()), "ndcg@10": float(v[m].mean()), "vs": {}}
            for ref, rv in list(refs.items()) + ([("backbone", gains["backbone"])] if vname != "backbone" else []):
                d = v[m] - rv[m]
                cell["vs"][ref] = {"diff": float(d.mean()), "p": M.paired_randomization(v[m], rv[m]),
                                   "ci95": list(M.bootstrap_ci(d))}
            row["groups"][g] = cell
        rows.append(row)
    coverage = {"pool": float(np.mean([c.target_pos >= 0 for c in test])),
                "pool + related + text neighbours": float(np.mean([r[5] >= 0 for r in data["test"]]))}
    res = {"config": cfg_path, "dataset": ds.name, "backbone": name, "run": run_dir, "rows": rows, "coverage": coverage,
           "refs": {m: {g: float(rv[msk].mean()) for g, msk in groups.items() if msk.any()} for m, rv in refs.items()}}
    return res


def system_markdown(results: list[dict]) -> str:
    md = ["# Decision4Rec system: trained backbone plus typed decisions", "",
          "NDCG@10 on full-catalogue ranks of the test users, by how often the target item was bought in training. "
          "Weights fitted on validation users. Differences with paired randomisation p and 95% bootstrap interval."]
    for res in results:
        groups = list(res["rows"][0]["groups"])
        md += ["", f"## {res['dataset']} (backbone `{res['backbone']}`)", "",
               f"Target among the scored candidates: pool {res['coverage']['pool']:.3f}, with related items and "
               f"text neighbours {res['coverage']['pool + related + text neighbours']:.3f}.", "",
               "| method | " + " | ".join(f"{g} (n={res['rows'][0]['groups'][g]['n']})" for g in groups) + " |",
               "|---|" + "---|" * len(groups)]
        for m, cells in res["refs"].items():
            md.append(f"| {m} | " + " | ".join(f"{cells.get(g, float('nan')):.4f}" for g in groups) + " |")
        for row in res["rows"]:
            md.append(f"| **{row['variant']}** | " + " | ".join(f"{row['groups'][g]['ndcg@10']:.4f}" for g in groups) + " |")
        best_ref = max(res["refs"], key=lambda m: res["refs"][m]["all"]) if res["refs"] else None
        md += [""]
        for row in res["rows"][1:]:
            for g in groups:
                cmp = [r for r in ("backbone", best_ref) if r and r in row["groups"][g]["vs"]]
                for r in cmp:
                    d = row["groups"][g]["vs"][r]
                    md.append(f"* {row['variant']} against {r}, {g}: {d['diff']:+.4f} (p = {d['p']:.3f}, "
                              f"95% interval [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}])")
        md.append(f"* Weights: " + "; ".join(f"{r['variant']}: {r['weights']}" for r in res["rows"][1:]))
    return "\n".join(md) + "\n"
