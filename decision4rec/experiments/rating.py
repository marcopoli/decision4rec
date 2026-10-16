"""Rating prediction with the `score` primitive.

For each held-out interaction the decision model is asked how the user would
rate the item, on three granularities of the same scale (5, 3 and 2 levels),
plus a `noul` "will the user like it". All questions of a state variant travel
in one request. The study reports error, ordinal agreement, calibration and
how much of the supplied scale the model actually uses, which is where the
ordinal compression reported in arXiv:2609.38827 would show up.
"""

from __future__ import annotations

import json
import os

import numpy as np

from ..data.schema import Dataset
from ..decision.base import build_backend, decide_chunked
from ..decision.types import Noul, Score
from ..eval import metrics as M
from .runner import get_cases, load_config, make_prompt_builder

LEVELS = {5: ("1 star: hated it", "2 stars: disliked it", "3 stars: it was ok", "4 stars: liked it",
              "5 stars: loved it"),
          3: ("disliked it (1 or 2 stars)", "it was ok (3 stars)", "liked it (4 or 5 stars)"),
          2: ("did not like it (1 to 3 stars)", "liked it (4 or 5 stars)")}
INSTR_RATE = "Based on the user's history, how would this user rate this item? Item: {item}"
INSTR_LIKE = "Based on the user's history, will this user like this item (4 or 5 stars)? Item: {item}"


def to_level(rating: float, n: int) -> int:
    r = int(min(max(round(rating), 1), 5))
    if n == 5:
        return r - 1
    if n == 3:
        return 0 if r <= 2 else (1 if r == 3 else 2)
    return int(r >= 4)


def _baselines(ds: Dataset, reg: float = 5.0, iters: int = 10):
    """Global mean and a regularised user plus item bias model fitted on training interactions."""
    rows = [(u, int(x[0]), float(x[2])) for u, s in enumerate(ds.seqs) for x in s[:-2]]
    u, i, r = (np.array(c) for c in zip(*rows))
    mu = float(r.mean())
    bu, bi = np.zeros(ds.n_users), np.zeros(ds.n_items)
    for _ in range(iters):
        bi = np.bincount(i, weights=r - mu - bu[u], minlength=ds.n_items) / (reg + np.bincount(i, minlength=ds.n_items))
        bu = np.bincount(u, weights=r - mu - bi[i], minlength=ds.n_users) / (reg + np.bincount(u, minlength=ds.n_users))
    return mu, bu, bi


def _ordinal_report(probs: np.ndarray, gold: np.ndarray, n: int) -> dict:
    pred = probs.argmax(1)
    util = M.scale_utilisation(np.bincount(pred, minlength=n), np.bincount(gold, minlength=n))
    soft = M.scale_utilisation(probs.mean(0), np.bincount(gold, minlength=n))
    return {"accuracy": float(np.mean(pred == gold)), "qwk": M.quadratic_weighted_kappa(pred, gold, n),
            "utilisation_R": util["R"], "tvd": util["tvd"], "soft_utilisation_R": soft["R"],
            "predicted_level_share": (np.bincount(pred, minlength=n) / len(pred)).round(4).tolist(),
            "gold_level_share": (np.bincount(gold, minlength=n) / len(gold)).round(4).tolist()}


def run_rating(cfg_path: str, n_cases: int = 1000, backend_path=None) -> dict:
    cfg = load_config(cfg_path, backend_path)
    ds = Dataset.load(cfg["dataset"])
    cases = get_cases(cfg, ds)
    pb = make_prompt_builder(cfg, ds, need_kg=True)
    backend = build_backend(cfg["backend"])
    variants = cfg.get("rating", {}).get("variants", ["rated", "unrated", "null", "profile"])
    out_dir = os.path.join(cfg["out"], cfg["name"])
    os.makedirs(out_dir, exist_ok=True)

    def collect(split: str, variant: str) -> dict:
        subset = [c for c in cases[split][:n_cases] if c.target_rating is not None]
        path = os.path.join(out_dir, f"rating_{variant}.{split}.jsonl")
        done = {}
        if os.path.exists(path):
            with open(path) as f:
                done = {r["cid"]: r for r in map(json.loads, f)}
        with open(path, "a") as f:
            for c in subset:
                if c.cid in done:
                    continue
                ratings = {int(x[0]): float(x[2]) for x in ds.seqs[c.user]}
                prof = pb.user_model(backend, c) if variant == "profile" else None
                state = pb.state(c, null_state=variant == "null", ratings=None if variant == "unrated" else ratings,
                                 profile="decision" if prof else "none", prof=prof)
                item = pb.option(c, c.target)
                qs = {f"s{n}": Score(INSTR_RATE.format(item=item), LEVELS[n]) for n in (5, 3, 2)}
                qs["like"] = Noul(INSTR_LIKE.format(item=item))
                d = decide_chunked(backend, state, qs)
                rec = {"cid": c.cid, "rating": c.target_rating, "like": float(d.answers["like"].p),
                       "tokens": d.input_tokens + (prof.tokens if prof else 0), "latency_ms": d.latency_ms}
                for n in (5, 3, 2):
                    a = d.answers[f"s{n}"]
                    rec[f"p{n}"] = [a.probs[str(k)] for k in range(n)]
                done[c.cid] = rec
                f.write(json.dumps(rec) + "\n")
                f.flush()
        return {c.cid: done[c.cid] for c in subset}

    mu, bu, bi = _baselines(ds)
    report: dict = {"backend": backend.fingerprint(), "dataset": ds.name, "variants": {}}
    for variant in variants:
        val, tst = collect("valid", variant), collect("test", variant)
        if not tst:
            continue
        gold = np.array([r["rating"] for r in tst.values()])
        row: dict = {"n": len(tst), "tokens": float(np.mean([r["tokens"] for r in tst.values()])),
                     "latency_ms": float(np.mean([r["latency_ms"] for r in tst.values()]))}
        p5 = np.array([r["p5"] for r in tst.values()])
        expected = 1.0 + p5 @ np.arange(5)
        row["rmse"] = float(np.sqrt(np.mean((expected - gold) ** 2)))
        row["mae"] = float(np.mean(np.abs(expected - gold)))
        if len(val) >= 20:                      # affine correction of the expected rating, fitted on validation
            ev = 1.0 + np.array([r["p5"] for r in val.values()]) @ np.arange(5)
            gv = np.array([r["rating"] for r in val.values()])
            a, b = np.polyfit(ev, gv, 1) if ev.std() > 1e-9 else (0.0, gv.mean())
            row["rmse_affine"] = float(np.sqrt(np.mean((np.clip(a * expected + b, 1, 5) - gold) ** 2)))
        for n in (5, 3, 2):
            pn = np.array([r[f"p{n}"] for r in tst.values()])
            row[f"scale{n}"] = _ordinal_report(pn, np.array([to_level(g, n) for g in gold]), n)
        liked = (gold >= 4).astype(int)
        like = np.array([r["like"] for r in tst.values()])
        row["like"] = {"base_rate": float(liked.mean()), "auroc_noul": M.auroc(like, liked),
                       "auroc_expected_rating": M.auroc(expected, liked), "ece_raw": M.ece(like, liked)}
        if len(val) >= 20:
            lv = np.array([r["like"] for r in val.values()])
            gv = (np.array([r["rating"] for r in val.values()]) >= 4).astype(int)
            if 0 < gv.sum() < len(gv):
                pa, pb_ = M.fit_platt(lv, gv)
                row["like"]["ece_platt"] = M.ece(M.apply_platt(like, pa, pb_), liked)
        report["variants"][variant] = row

    test = [c for c in cases["test"][:n_cases] if c.target_rating is not None]
    gold = np.array([c.target_rating for c in test])
    liked = (gold >= 4).astype(int)
    bias = np.clip(mu + bu[[c.user for c in test]] + bi[[c.target for c in test]], 1, 5)
    report["baselines"] = {
        "global_mean": {"rmse": float(np.sqrt(np.mean((mu - gold) ** 2))), "mae": float(np.mean(np.abs(mu - gold)))},
        "user_item_bias": {"rmse": float(np.sqrt(np.mean((bias - gold) ** 2))), "mae": float(np.mean(np.abs(bias - gold))),
                           "like_auroc": M.auroc(bias, liked)}}
    backend.close()
    with open(os.path.join(out_dir, "rating.json"), "w") as f:
        json.dump(report, f, indent=1)
    md = [f"# Rating prediction: {ds.name}", "", f"Backend: `{report['backend']}`.", "",
          "| state | n | RMSE | RMSE affine | MAE | QWK 5 | R 5 | R 3 | R 2 | acc 5 | like AUROC | like ECE | tokens |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in report["variants"].items():
        md.append(f"| {name} | {r['n']} | {r['rmse']:.3f} | {r.get('rmse_affine', float('nan')):.3f} | {r['mae']:.3f} | "
                  f"{r['scale5']['qwk']:.3f} | {r['scale5']['utilisation_R']:.2f} | {r['scale3']['utilisation_R']:.2f} | "
                  f"{r['scale2']['utilisation_R']:.2f} | {r['scale5']['accuracy']:.3f} | {r['like']['auroc_noul']:.3f} | "
                  f"{r['like']['ece_raw']:.3f} | {r['tokens']:.0f} |")
    b = report["baselines"]
    md += ["", f"Baselines: global mean RMSE {b['global_mean']['rmse']:.3f}; user plus item bias RMSE "
               f"{b['user_item_bias']['rmse']:.3f}, like AUROC {b['user_item_bias']['like_auroc']:.3f}.", "",
           "R is the effective number of levels the model uses divided by the effective number in the data: "
           "below 1 means the scale is compressed."]
    with open(os.path.join(out_dir, "rating.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    report["out_dir"] = out_dir
    return report
