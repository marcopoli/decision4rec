"""Analysis of a run directory. Pure numpy: no model call is made here.

From the raw observations of each arm it derives rankings and every post-hoc
variant (permutation ensembles, position debiasing, prior contrast, fusion
with the retriever), then accuracy, significance, calibration, risk control,
order sensitivity and cost. All tunable quantities are fixed on validation
cases and applied once to test cases.
"""

from __future__ import annotations

import itertools
import json
import os
from typing import Callable, Optional

import numpy as np

from ..data.schema import Case, load_cases
from ..eval import metrics as M
from ..operators.luce import LuceFit, Obs, fit_luce, fit_position_bias, kendall_tau
from ..operators.risk import conformal_report, risk_coverage
from .runner import case_paths

SPLITS = ("valid", "test")
# Grids for the weights fitted on validation cases. Both reach far enough that the choice is not stopped by the
# edge of the grid: lambda = 0 means that subtracting the prior does not help, a large gamma means that the
# retrieval score dominates the decision model.
LAMBDAS = (0.0, 0.1, 0.25, 0.5, 1.0)
GAMMAS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


class Variant:
    """Scores of one method on every case of both splits, plus what it cost."""

    def __init__(self, name: str, arm: dict, note: str = ""):
        self.name, self.arm, self.note = name, arm, note
        self.scores: dict[str, dict[int, np.ndarray]] = {s: {} for s in SPLITS}
        self.fits: dict[str, dict[int, LuceFit]] = {s: {} for s in SPLITS}
        self.cost: dict[str, dict[int, tuple]] = {s: {} for s in SPLITS}


def _load_jsonl(path: str) -> dict[int, dict]:
    out = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[r["cid"]] = r
    return out


def _obs(rec: dict) -> list[Obs]:
    return [Obs.from_json(o) for o in rec["obs"]]


_z = M.zscore


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _case_metrics(v: Variant, cases: dict[int, Case], split: str) -> dict[str, np.ndarray]:
    cids = sorted(v.scores[split])
    rows = [M.per_case_metrics(M.target_rank(v.scores[split][c], cases[c].target_pos)) for c in cids]
    out = {k: np.array([r[k] for r in rows]) for k in (rows[0] if rows else {})}
    out["cid"] = np.array(cids)
    return out


def _mean_ndcg(v: Variant, cases: dict[int, Case], split: str) -> float:
    m = _case_metrics(v, cases, split)
    return float(m["ndcg@10"].mean()) if len(m["cid"]) else float("nan")


def _twin(a: dict, b: dict, ignore: set) -> bool:
    return all(a.get(k) == b.get(k) for k in set(a) | set(b) if k not in ignore)


def _tune(make: Callable[[float], Variant], grid, cases: dict[int, Case]) -> tuple[float, Variant]:
    best = max(((p, make(p)) for p in grid), key=lambda t: _mean_ndcg(t[1], cases, "valid"))
    return best


def analyse(run_dir: str, reference: Optional[str] = None) -> dict:
    with open(os.path.join(run_dir, "manifest.json")) as f:
        manifest = json.load(f)
    cfg = manifest["config"]
    paths = case_paths(cfg)
    cases = {s: {c.cid: c for c in load_cases(paths[s])} for s in SPLITS}
    arms = {a["name"]: a for a in manifest["arms"]}
    recs = {(n, s): _load_jsonl(os.path.join(run_dir, f"{n}.{s}.jsonl")) for n in arms for s in SPLITS}
    variants: dict[str, Variant] = {}

    def K(split: str, cid: int) -> int:
        return len(cases[split][cid].cands)

    # 1. base variants -----------------------------------------------------
    for name, arm in arms.items():
        v = Variant(name, arm)
        for s in SPLITS:
            for cid, rec in recs[(name, s)].items():
                case = cases[s][cid]
                if arm["method"] == "retriever":
                    v.scores[s][cid] = np.asarray(case.rscores, dtype=float)
                elif arm["method"] == "pointwise":
                    v.scores[s][cid] = np.asarray(rec["point"], dtype=float)
                elif arm["method"] == "graph":
                    v.scores[s][cid] = np.asarray(rec["graph"], dtype=float)
                else:
                    fit = fit_luce(_obs(rec), K(s, cid))
                    v.fits[s][cid] = fit
                    v.scores[s][cid] = fit.theta
                v.cost[s][cid] = (rec["tokens"], rec["latency_ms"], rec["requests"])
        variants[name] = v

    # 2. permutation groups: order sensitivity, ensembles, position debiasing ---
    groups: dict[str, list[str]] = {}
    for name, arm in arms.items():
        if arm["method"] == "luce" and "@p" in name:
            groups.setdefault(name.split("@p")[0], []).append(name)
    order_rows = []
    beta_info: dict[str, list[float]] = {}

    def refit(base_names: list[str], vname: str, note: str, beta=None) -> Variant:
        v = Variant(vname, arms[base_names[0]], note)
        for s in SPLITS:
            common = set.intersection(*[set(recs[(n, s)]) for n in base_names])
            for cid in common:
                obs = [o for n in base_names for o in _obs(recs[(n, s)][cid])]
                fit = fit_luce(obs, K(s, cid), beta=beta)
                v.fits[s][cid], v.scores[s][cid] = fit, fit.theta
                c = [variants[n].cost[s][cid] for n in base_names]
                v.cost[s][cid] = tuple(sum(x[j] for x in c) for j in range(3))
        return v

    def position_beta(base_names: list[str]) -> Optional[np.ndarray]:
        common = set.intersection(*[set(recs[(n, "valid")]) for n in base_names])
        data, width = [], 0
        for cid in common:
            obs = [o for n in base_names for o in _obs(recs[(n, "valid")][cid])]
            data.append((obs, K("valid", cid)))
            width = max(width, max(len(o.items) for o in obs))
        if len(data) < 10 or width < 2:
            return None
        appearances = np.mean([sum(len(o.items) for o in obs) / k for obs, k in data])
        if appearances < 1.5:      # every item must be seen at more than one position
            return None
        return fit_position_bias(data, width)

    for base, names in groups.items():
        names = sorted(names)
        if len(names) < 2:
            continue
        taus, top1, spread = [], [], []
        common = set.intersection(*[set(variants[n].scores["test"]) for n in names])
        if not common:
            continue
        for cid in common:
            sc = [variants[n].scores["test"][cid] for n in names]
            for a, b in itertools.combinations(range(len(sc)), 2):
                taus.append(kendall_tau(sc[a], sc[b]))
                top1.append(float(np.argmax(sc[a]) == np.argmax(sc[b])))
            nd = [M.per_case_metrics(M.target_rank(x, cases["test"][cid].target_pos))["ndcg@10"] for x in sc]
            spread.append(float(np.std(nd)))
        order_rows.append({"group": base, "n_perms": len(names), "kendall_tau": float(np.mean(taus)),
                           "top1_agreement": float(np.mean(top1)), "ndcg_std_across_perms": float(np.mean(spread))})
        variants[f"{base}@ens"] = refit(names, f"{base}@ens", f"union of {len(names)} permutations")
        beta = position_beta(names)
        if beta is not None:
            beta_info[base] = beta.tolist()
            variants[f"{base}@ens+pos"] = refit(names, f"{base}@ens+pos", "ensemble, position debiased", beta)
            variants[f"{names[0]}+pos"] = refit(names[:1], f"{names[0]}+pos", "single order, position debiased", beta)
    for name, arm in arms.items():
        if arm["method"] == "luce" and "@p" not in name and arm.get("rounds", 1) >= 2:
            beta = position_beta([name])
            if beta is not None:
                beta_info[name] = beta.tolist()
                variants[f"{name}+pos"] = refit([name], f"{name}+pos", "position debiased", beta)

    # 3. prior contrast and 4. fusion with the retriever ---------------------
    tuned: dict[str, float] = {}
    for name, arm in list(arms.items()):
        if arm["method"] == "retriever" or arm.get("null_state"):
            continue
        v = variants[name]
        if arm["method"] == "graph":
            def make_gfuse(gamma: float, v=v) -> Variant:
                out = Variant(f"{v.name}+fuse", v.arm, f"graph score plus retriever, gamma={gamma}")
                for s in SPLITS:
                    for cid, sc in v.scores[s].items():
                        out.scores[s][cid] = _z(sc) + gamma * _z(cases[s][cid].rscores)
                        out.cost[s][cid] = v.cost[s][cid]
                return out
            gamma, fu = _tune(make_gfuse, GAMMAS, cases["valid"])
            tuned[fu.name], variants[fu.name] = gamma, fu
            continue
        transform = _logit if arm["method"] == "pointwise" else (lambda x: x)
        twin = next((n for n, a in arms.items() if a.get("null_state") and a["method"] == arm["method"]
                     and _twin(a, arm, {"name", "null_state", "evidence", "profile", "intent", "max_cases"})), None)
        if twin is not None:
            def make_pc(lam: float, v=v, twin=twin, transform=transform) -> Variant:
                out = Variant(f"{v.name}+pc", v.arm, f"prior contrast, lambda={lam}")
                for s in SPLITS:
                    for cid in set(v.scores[s]) & set(variants[twin].scores[s]):
                        out.scores[s][cid] = transform(v.scores[s][cid]) - lam * transform(variants[twin].scores[s][cid])
                        a, b = v.cost[s][cid], variants[twin].cost[s][cid]
                        out.cost[s][cid] = tuple(a[j] + b[j] for j in range(3))
                return out
            lam, pc = _tune(make_pc, LAMBDAS, cases["valid"])
            tuned[pc.name] = lam
            variants[pc.name] = pc

        def make_fuse(gamma: float, v=v, transform=transform) -> Variant:
            out = Variant(f"{v.name}+fuse", v.arm, f"fusion with retriever, gamma={gamma}")
            for s in SPLITS:
                for cid, sc in v.scores[s].items():
                    out.scores[s][cid] = _z(transform(sc)) + gamma * _z(cases[s][cid].rscores)
                    out.cost[s][cid] = v.cost[s][cid]
            return out
        gamma, fu = _tune(make_fuse, GAMMAS, cases["valid"])
        tuned[fu.name] = gamma
        variants[fu.name] = fu

        # Side signals logged by the user model: graph score (preference paths) and intent prior.
        for side in ("graph", "intent"):
            if arm["method"] == "graph" or not any(side in r for r in recs[(name, "test")].values()):
                continue

            def make_side(weight: float, v=v, transform=transform, side=side, name=name) -> Variant:
                out = Variant(f"{v.name}+{side}", v.arm, f"fusion with the {side} signal, weight={weight}")
                for s in SPLITS:
                    for cid, sc in v.scores[s].items():
                        extra = recs[(name, s)][cid].get(side)
                        if extra is None:
                            continue
                        out.scores[s][cid] = _z(transform(sc)) + weight * _z(extra)
                        out.cost[s][cid] = v.cost[s][cid]
                return out
            weight, sv = _tune(make_side, GAMMAS, cases["valid"])
            tuned[sv.name] = weight
            variants[sv.name] = sv

    # 5. accuracy, cost, significance ---------------------------------------
    price = float(cfg.get("price_per_mtok", 0.0))
    per_case = {n: _case_metrics(v, cases["test"], "test") for n, v in variants.items()}
    if reference is None:
        reference = cfg.get("reference") or next(
            (n for n, a in arms.items() if a["method"] == "luce" and not a.get("m") and not a.get("null_state")
             and a.get("evidence", "none") == "none"), None) or next(iter(arms))
    table, pvals, pnames = [], [], []
    for n, v in variants.items():
        m = per_case[n]
        if not len(m["cid"]):
            continue
        cost = np.array([v.cost["test"][c] for c in m["cid"]], dtype=float)
        lo, hi = M.bootstrap_ci(m["ndcg@10"])
        row = {"variant": n, "n": int(len(m["cid"])), "ndcg@10": float(m["ndcg@10"].mean()),
               "ci_low": lo, "ci_high": hi, "hr@10": float(m["hr@10"].mean()), "mrr": float(m["mrr"].mean()),
               "tokens": float(cost[:, 0].mean()), "latency_ms": float(cost[:, 1].mean()),
               "requests": float(cost[:, 2].mean()), "usd_per_1k_users": float(cost[:, 0].mean()) * price / 1000.0,
               "note": v.note}
        if n != reference and reference in per_case:
            ref = per_case[reference]
            both = np.intersect1d(m["cid"], ref["cid"])
            a = m["ndcg@10"][np.isin(m["cid"], both)]
            b = ref["ndcg@10"][np.isin(ref["cid"], both)]
            row["delta_vs_ref"] = float(a.mean() - b.mean())
            pvals.append(M.paired_randomization(a, b))
            pnames.append(n)
        table.append(row)
    for n, p, ph in zip(pnames, pvals, M.holm(pvals) if pvals else []):
        row = next(r for r in table if r["variant"] == n)
        row["p_value"], row["p_holm"] = p, ph

    # 6. calibration and risk for choice-based variants ---------------------
    calib = []
    for n, v in variants.items():
        if not v.fits["test"] or "+pc" in n or "+fuse" in n:
            continue
        outside = bool(v.arm.get("outside"))

        def labelled(split: str, v=v, outside=outside):
            rows = []
            for cid, fit in v.fits[split].items():
                case = cases[split][cid]
                y = case.target_pos if case.target_pos >= 0 else (len(case.cands) if outside else None)
                if y is None:
                    continue
                z = np.append(fit.theta, fit.theta_out) if outside else fit.theta
                rows.append((cid, z, y))
            return rows

        val, tst = labelled("valid"), labelled("test")
        if len(val) < 10 or len(tst) < 10:
            continue
        temp = M.fit_temperature([z for _, z, _ in val], [y for _, _, y in val])

        def probs(z: np.ndarray, t: float) -> np.ndarray:
            z = z / t
            p = np.exp(z - z.max())
            return p / p.sum()

        row = {"variant": n, "temperature": temp, "iia_residual": float(np.mean(
            [f.residual for f in v.fits["test"].values()]))}
        for tag, t in (("raw", 1.0), ("cal", temp)):
            ps = [probs(z, t) for _, z, _ in tst]
            ys = [y for _, _, y in tst]
            row[f"ece_{tag}"] = M.ece([p.max() for p in ps], [int(np.argmax(p) == y) for p, y in zip(ps, ys)])
            row[f"nll_{tag}"] = float(np.mean([-np.log(p[y] + 1e-12) for p, y in zip(ps, ys)]))
        ps_test = {cid: probs(z, temp) for cid, z, _ in tst}
        loss = np.array([1.0 - M.per_case_metrics(M.target_rank(v.scores["test"][cid], cases["test"][cid].target_pos))["ndcg@10"]
                         for cid in ps_test])
        conf = np.array([ps_test[cid][: len(cases["test"][cid].cands)].max() for cid in ps_test])
        rc = risk_coverage(conf, loss)
        row["aurc"], row["risk_at_coverage"] = rc["aurc"], dict(zip([f"{c:.1f}" for c in rc["coverage"]], rc["risk"]))
        row["conformal"] = conformal_report([(probs(z, temp), y) for _, z, y in val],
                                            [(ps_test[cid], y) for cid, _, y in tst])
        if outside:
            absent = np.array([cases["test"][cid].target_pos < 0 for cid in ps_test])
            row["outside_auroc"] = M.auroc(np.array([ps_test[cid][-1] for cid in ps_test]), absent)
            row["target_absent_rate"] = float(absent.mean())
            row["mean_p_outside"] = float(np.mean([ps_test[cid][-1] for cid in ps_test]))
        calib.append(row)

    point_calib = []
    for n, arm in arms.items():
        if arm["method"] != "pointwise":
            continue
        def flat(split: str, n=n):
            p, y = [], []
            for cid, sc in variants[n].scores[split].items():
                p += list(sc)
                y += [int(j == cases[split][cid].target_pos) for j in range(len(sc))]
            return np.array(p), np.array(y)
        (pv, yv), (pt, yt) = flat("valid"), flat("test")
        if len(pv) and len(pt):
            a, b = M.fit_platt(pv, yv)
            point_calib.append({"variant": n, "ece_raw": M.ece(pt, yt), "brier_raw": M.brier(pt, yt),
                                "ece_platt": M.ece(M.apply_platt(pt, a, b), yt),
                                "brier_platt": M.brier(M.apply_platt(pt, a, b), yt), "base_rate": float(yt.mean())})

    # 7. quality of the decision-estimated preference edges: do they predict the attributes of the
    #    next chosen item better than counting does?
    pref_quality = []
    for n in arms:
        def gather(split: str, n=n):
            rows = [r for rec in recs[(n, split)].values() for r in (rec.get("prefs") or [])]
            return np.array([[r[2], r[3], r[3] * r[4], r[5]] for r in rows], dtype=float).reshape(-1, 4)
        val, tst = gather("valid"), gather("test")
        if len(tst) < 50 or tst[:, 3].sum() in (0, len(tst)):
            continue
        row = {"variant": n, "n_hypotheses": int(len(tst)), "base_rate": float(tst[:, 3].mean()),
               "auroc_decision": M.auroc(tst[:, 0], tst[:, 3]), "auroc_support": M.auroc(tst[:, 1], tst[:, 3]),
               "auroc_support_idf": M.auroc(tst[:, 2], tst[:, 3]), "ece_raw": M.ece(tst[:, 0], tst[:, 3])}
        if len(val) >= 50 and 0 < val[:, 3].sum() < len(val):
            a, b = M.fit_platt(val[:, 0], val[:, 3])
            row["ece_platt"] = M.ece(M.apply_platt(tst[:, 0], a, b), tst[:, 3])
        pref_quality.append(row)
        break                                       # the user model is shared by all arms of a run

    # Per-case target ranks of every variant on both splits, the input of the cross-method benchmark.
    ranks = {}
    for n, v in variants.items():
        ranks[n] = {}
        for split in SPLITS:
            row = {}
            for cid, sc in v.scores[split].items():
                r = M.target_rank(sc, cases[split][cid].target_pos)
                row[str(cid)] = None if np.isinf(r) else r
            ranks[n][split] = row

    info = {}
    if os.path.exists(paths["info"]):
        with open(paths["info"]) as f:
            info = json.load(f)
    return {"run": manifest["name"], "backend": manifest.get("backend"), "reference": reference,
            "cases": info, "table": sorted(table, key=lambda r: -r["ndcg@10"]), "order_sensitivity": order_rows,
            "position_beta": beta_info, "tuned": tuned, "calibration": calib, "pointwise_calibration": point_calib, "preference_quality": pref_quality,
            "ranks": ranks}


def to_markdown(rep: dict) -> str:
    out = [f"# Report: {rep['run']}", "", f"Backend: `{rep['backend']}`. Reference arm: `{rep['reference']}`.", ""]
    if rep.get("cases"):
        t = rep["cases"].get("test", {})
        out += [f"Test cases: {t.get('n_cases')} (track {t.get('track')}, K={t.get('K')}), retriever recall at "
                f"{t.get('pool')}: {t.get('retriever_recall_at_pool', float('nan')):.3f}.", ""]
    out += ["## Accuracy and cost (test)", "",
            "| variant | n | NDCG@10 | 95% CI | HR@10 | MRR | delta vs ref | p (Holm) | tokens | requests | latency ms |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep["table"]:
        d = f"{r['delta_vs_ref']:+.4f}" if "delta_vs_ref" in r else ""
        p = f"{r['p_holm']:.4f}" if "p_holm" in r else ""
        out.append(f"| {r['variant']} | {r['n']} | {r['ndcg@10']:.4f} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] | "
                   f"{r['hr@10']:.4f} | {r['mrr']:.4f} | {d} | {p} | {r['tokens']:.0f} | {r['requests']:.2f} | "
                   f"{r['latency_ms']:.0f} |")
    if rep["order_sensitivity"]:
        out += ["", "## Order sensitivity (test)", "", "| group | permutations | Kendall tau | top-1 agreement | NDCG@10 std |",
                "|---|---|---|---|---|"]
        out += [f"| {r['group']} | {r['n_perms']} | {r['kendall_tau']:.3f} | {r['top1_agreement']:.3f} | "
                f"{r['ndcg_std_across_perms']:.4f} |" for r in rep["order_sensitivity"]]
    if rep["position_beta"]:
        out += ["", "## Position effects (log scale, first slot = 0, fitted on validation cases)", ""]
        out += [f"* `{k}`: " + ", ".join(f"{b:+.2f}" for b in v[:12]) + (" ..." if len(v) > 12 else "")
                for k, v in rep["position_beta"].items()]
    if rep["calibration"]:
        out += ["", "## Calibration, Luce consistency and risk (test)", "",
                "| variant | IIA residual | T | ECE raw | ECE cal | NLL raw | NLL cal | AURC | outside AUROC |",
                "|---|---|---|---|---|---|---|---|---|"]
        for r in rep["calibration"]:
            au = f"{r['outside_auroc']:.3f}" if "outside_auroc" in r and r["outside_auroc"] == r["outside_auroc"] else ""
            out.append(f"| {r['variant']} | {r['iia_residual']:.4f} | {r['temperature']:.2f} | {r['ece_raw']:.3f} | "
                       f"{r['ece_cal']:.3f} | {r['nll_raw']:.3f} | {r['nll_cal']:.3f} | {r['aurc']:.3f} | {au} |")
        out += ["", "### Conformal label sets", "", "| variant | target coverage | coverage | mean set size |", "|---|---|---|---|"]
        for r in rep["calibration"]:
            out += [f"| {r['variant']} | {c['target_coverage']:.2f} | {c['coverage']:.3f} | {c['mean_set_size']:.2f} |"
                    for c in r["conformal"]]
    if rep["pointwise_calibration"]:
        out += ["", "## Pointwise (noul) calibration", "", "| variant | base rate | ECE raw | ECE Platt | Brier raw | Brier Platt |",
                "|---|---|---|---|---|---|"]
        out += [f"| {r['variant']} | {r['base_rate']:.3f} | {r['ece_raw']:.3f} | {r['ece_platt']:.3f} | "
                f"{r['brier_raw']:.4f} | {r['brier_platt']:.4f} |" for r in rep["pointwise_calibration"]]
    if rep.get("preference_quality"):
        out += ["", "## User model: do estimated preferences predict the attributes of the next item", "",
                "| hypotheses | base rate | AUROC decision | AUROC support count | AUROC support x idf | ECE raw | ECE Platt |",
                "|---|---|---|---|---|---|---|"]
        for r in rep["preference_quality"]:
            ep = f"{r['ece_platt']:.3f}" if "ece_platt" in r else ""
            out.append(f"| {r['n_hypotheses']} | {r['base_rate']:.3f} | {r['auroc_decision']:.3f} | "
                       f"{r['auroc_support']:.3f} | {r['auroc_support_idf']:.3f} | {r['ece_raw']:.3f} | {ep} |")
    if rep["tuned"]:
        out += ["", "## Parameters fixed on validation cases", ""] + [f"* `{k}`: {v}" for k, v in rep["tuned"].items()]
    return "\n".join(out) + "\n"


def report(run_dir: str, reference: Optional[str] = None) -> dict:
    rep = analyse(run_dir, reference)
    ranks = rep.pop("ranks")
    with open(os.path.join(run_dir, "ranks.json"), "w") as f:
        json.dump(ranks, f)
    with open(os.path.join(run_dir, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    with open(os.path.join(run_dir, "report.md"), "w") as f:
        f.write(to_markdown(rep))
    return rep
