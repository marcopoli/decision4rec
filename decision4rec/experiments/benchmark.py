"""Recommendation benchmark: Decision4Rec against trained baselines.

Every method is scored on the same test users by the rank of the held-out item
over the whole catalogue. A trained baseline ranks all items. Decision4Rec is a
two-stage system: it ranks the candidates it examines, and a target it never
examines counts as a miss. Two forms are reported when both were run: the agent
on ActiveGraph (the primary system) and the batch pipeline without the agent
loop. Both are declared in the config before any test number is seen; nothing
is selected on the test split.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

from ..data.schema import Dataset
from ..eval import metrics as M
from .runner import case_paths, get_cases, load_config

BIG = 10 ** 9


def _stem(cfg: dict) -> str:
    return os.path.basename(case_paths(cfg)["test"]).rsplit("_test", 1)[0]


def _bench_dir(cfg: dict) -> str:
    return os.path.join(cfg["out"], "benchmark", _stem(cfg))


def prepare_dataset(cfg: dict, ds: Dataset, names: list[str]) -> None:
    """Attach what some baselines need: decision-model facets and item-text embeddings."""
    from ..models import baselines as B
    if cfg.get("facets") and "facet_attrs" not in ds.extra:
        from ..semantic.facets import attrs_from_config
        ds.extra["facet_attrs"] = attrs_from_config(cfg["facets"], ds)
    if any(n in B.TEXT_MODELS for n in names) and ds.extra.get("item_text_matrix") is None:
        from ..retrieval.agents import ItemViews
        from ..retrieval.slm import build_embedder
        it = cfg.get("item_text") or {k: (cfg.get("retriever") or {}).get(k) for k in ("embedder", "cache")}
        ds.extra["item_text_matrix"] = ItemViews(ds, None, build_embedder(it.get("embedder")), it.get("cache")).matrix("content")


def baseline_root(cfg: dict, ds: Dataset, name: str, seed: Optional[int] = None) -> str:
    """Folder of a baseline's checkpoints. Facet models are tied to their facets, extra seeds get their own folder."""
    from ..models import baselines as B
    root = (cfg.get("benchmark") or {}).get("root") or os.path.join(cfg["out"], "baselines")   # e.g. a tuned grid
    if name in B.FACET_MODELS:
        from ..semantic.facets import fingerprint
        attrs = ds.extra.get("facet_attrs", {}).get(B.FACET_MODELS[name][0])
        if attrs is None:
            raise SystemExit(f"{name}: no facets of source {B.FACET_MODELS[name][0]!r} in the config; "
                             "run the facet step first (stage e1_facets)")
        root = os.path.join(root, f"facets_{fingerprint(attrs)}")
    return root if seed is None else f"{root}_s{seed}"


def run_baselines(cfg_path: str, only: Optional[list[str]] = None, device: Optional[str] = None) -> dict:
    """Tune and train the baselines of a config and store their per-user target ranks. `benchmark.seeds` can give
    a model extra training seeds ({model: [seed, ...]}); the first run keeps the usual files, the others are
    stored as `<model>.s<seed>.json` and `.all.npz`."""
    from ..models import baselines as B

    cfg = load_config(cfg_path)
    ds = Dataset.load(cfg["dataset"])
    cases = get_cases(cfg, ds)
    bcfg = cfg.get("benchmark", {})
    names = only or bcfg.get("baselines") or list(B.MODELS)
    grids = bcfg.get("grids", {})
    out_dir = _bench_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)
    prepare_dataset(cfg, ds, names)
    base_seed = int(cfg["eval"].get("seed", 13))
    summary = {}
    for name in names:
        extra = [int(x) for x in (bcfg.get("seeds") or {}).get(name, []) if int(x) != base_seed]
        chosen = None
        for seed in [None] + extra:
            # extra seeds repeat the configuration chosen with the first seed, as in seed-variance studies
            grid = grids.get(name) if seed is None else [chosen]
            best = B.tune(name, ds, baseline_root(cfg, ds, name, seed), grid,
                          common=None if seed is not None else bcfg.get("common_hp"), device=device,
                          seed=base_seed if seed is None else seed)
            chosen = chosen or best["hp"]
            scorer = B.load(name, best["dir"], ds, device)
            ranks = {}
            for split, subset in cases.items():
                r = scorer.ranks([c.user for c in subset], [c.history for c in subset], [c.target for c in subset])
                ranks[split] = {str(c.cid): float(x) for c, x in zip(subset, r)}
            # The same model on every user of the dataset, the figure comparable with the literature.
            everyone = [u for u in range(ds.n_users) if len(ds.seqs[u]) >= 4]
            full = scorer.ranks(everyone, [[int(x[0]) for x in ds.seqs[u][:-1]] for u in everyone],
                                [int(ds.seqs[u][-1][0]) for u in everyone])
            all_users = {"n_users": len(everyone), **_metrics(full)}
            stem = name if seed is None else f"{name}.s{seed}"
            np.savez_compressed(os.path.join(out_dir, f"{stem}.all.npz"), users=np.asarray(everyone, dtype=np.int64),
                                ranks=np.asarray(full, dtype=np.float64))
            with open(os.path.join(out_dir, f"{stem}.json"), "w") as f:
                json.dump({"model": name, "seed": base_seed if seed is None else seed, "dir": best["dir"], "hp": best["hp"],
                           "valid_ndcg10_model_selection": best["best_valid_ndcg10"],
                           "all_users": all_users, "ranks": ranks,
                           "users": {split: [[c.user, c.target] for c in subset] for split, subset in cases.items()}}, f)
            test = np.array(list(ranks["test"].values()))
            summary[stem] = {"hp": best["hp"], "ndcg@10": float(np.mean(np.where(test <= 10, 1 / np.log2(test + 1), 0))),
                             "all_users": all_users}
            print(f"[baselines] {ds.name} {stem:9s} sampled users NDCG@10 {summary[stem]['ndcg@10']:.4f} | all "
                  f"{all_users['n_users']} users NDCG@10 {all_users['ndcg@10']:.4f} HR@10 {all_users['hr@10']:.4f} | hp {best['hp']}",
                  flush=True)
    return summary


def _metrics(ranks: np.ndarray) -> dict:
    out = {}
    for k in (10, 20):
        hit = ranks <= k
        out[f"hr@{k}"] = float(hit.mean())
        out[f"ndcg@{k}"] = float(np.where(hit, 1.0 / np.log2(ranks + 1.0), 0.0).mean())
    out["mrr"] = float(np.where(ranks < BIG, 1.0 / ranks, 0.0).mean())
    return out


def _vector(row: dict, cids: list[str]) -> np.ndarray:
    return np.array([BIG if row.get(c) is None else row[c] for c in cids], dtype=float)


def _ndcg10(ranks: np.ndarray) -> np.ndarray:
    return np.where(ranks <= 10, 1.0 / np.log2(ranks + 1.0), 0.0)


def benchmark_dataset(cfg_path: str, backend_path: Optional[str] = None, extra: Optional[list[tuple[str, str, str]]] = None) -> dict:
    """Join baseline ranks with the ranks of a Decision4Rec run. `extra` adds rows from other backends'
    runs of the same config as (backend file, variant, label), for example a generative LLM reranker."""
    from .report import report

    cfg = load_config(cfg_path, backend_path)
    bcfg = cfg.get("benchmark", {})
    system = bcfg.get("system", "decision4rec+fuse")
    run_dir = os.path.join(cfg["out"], cfg["name"])
    report(run_dir)
    with open(os.path.join(run_dir, "ranks.json")) as f:
        ours = json.load(f)
    if system not in ours:
        raise SystemExit(f"system variant {system!r} not found in {run_dir}; available: {sorted(ours)[:20]}")
    cids = sorted(ours[system]["test"], key=int)
    from ..data.schema import load_cases
    expected = [[c.user, c.target] for c in load_cases(case_paths(cfg)["test"])]
    rows: dict[str, dict] = {}
    groups: dict[str, str] = {}
    bench = _bench_dir(cfg)
    selection, all_users = {}, {}
    if os.path.isdir(bench):
        for fn in sorted(f for f in os.listdir(bench) if f.endswith(".json")):     # not the .all.npz rank files
            with open(os.path.join(bench, fn)) as f:
                b = json.load(f)
            if fn != f"{b.get('model')}.json":           # extra training seeds are read by their own reports
                continue
            if "users" in b and b["users"]["test"][: len(expected)] != expected[: len(b["users"]["test"])]:
                raise SystemExit(f"{fn}: this baseline was scored on different test users than the system. "
                                 "Both machines must use the same processed dataset and the same seed.")
            if all(c in b["ranks"]["test"] for c in cids):
                rows[b["model"]] = b["ranks"]
                groups[b["model"]] = "trained baseline"
                selection[b["model"]] = b["valid_ndcg10_model_selection"]
                all_users[b["model"]] = b.get("all_users")
    for variant in bcfg.get("report", ["retriever", "wholepool@p0"]):
        if variant in ours and variant != system:
            rows[variant] = ours[variant]
            groups[variant] = "same candidate pool"
    for backend_file, variant, label in extra or []:
        other = load_config(cfg_path, backend_file)
        path = os.path.join(other["out"], other["name"], "ranks.json")
        if not os.path.exists(path):
            report(os.path.join(other["out"], other["name"]))
        with open(path) as f:
            rows[label] = json.load(f)[variant]
            groups[label] = "other reranker"
    rows[system] = ours[system]
    groups[system] = "Decision4Rec"
    # The agent on ActiveGraph, when it was run on every test user of this config, is the primary system.
    agent, agent_info = None, None
    agent_path = os.path.join(run_dir, "agent_bench.json")
    if os.path.exists(agent_path):
        with open(agent_path) as f:
            ab = json.load(f)
        if ab.get("complete") and all(c in ab["ranks"]["test"] for c in cids):
            agent = ab["row"]
            rows[agent] = ab["ranks"]
            groups[agent] = "Decision4Rec, agent on ActiveGraph"
            groups[system] = "Decision4Rec, batch without the agent loop"
            agent_info = {"chosen": ab["chosen"], "test": ab["test"], "replay_check": ab["replay_check"]}
    systems = [agent, system] if agent else [system]
    primary = systems[0]

    table, per_case = [], {}
    for name, r in rows.items():
        vec = _vector(r["test"], cids)
        per_case[name] = _ndcg10(vec)
        lo, hi = M.bootstrap_ci(per_case[name])
        table.append({"method": name, "group": groups[name], **_metrics(vec), "ci_low": lo, "ci_high": hi,
                      "all_users": all_users.get(name)})
    baselines = [n for n in rows if groups[n] == "trained baseline"]
    out = {"dataset": cfg["dataset"], "run": cfg["name"], "system": primary, "systems": systems,
           "n_users": len(cids), "table": table, "agent": agent_info}
    if baselines:
        # The reference baseline is the one with the best validation score, so the test split is not used to pick it.
        ref = max(baselines, key=lambda n: _ndcg10(_vector(rows[n]["valid"], sorted(rows[n]["valid"], key=int))).mean())
        best_test = max(baselines, key=lambda n: per_case[n].mean())
        out["reference_baseline"] = ref
        out["best_baseline_on_test"] = best_test
        for name in {ref, best_test}:
            out[f"p_value_vs_{name}"] = M.paired_randomization(per_case[primary], per_case[name])
            out[f"delta_ndcg10_vs_{name}"] = float(per_case[primary].mean() - per_case[name].mean())
        # Every planned comparison, with the Holm correction over them: each form of the system against the
        # baseline chosen on validation, and the agent against the batch pipeline.
        pairs = [(s, ref) for s in systems] + ([(agent, system)] if agent else [])
        ps = [M.paired_randomization(per_case[a], per_case[b]) for a, b in pairs]
        out["comparisons"] = [{"a": a, "b": b, "delta_ndcg10": float(per_case[a].mean() - per_case[b].mean()),
                               "p_value": p, "p_holm": ph} for (a, b), p, ph in zip(pairs, ps, M.holm(ps))]
    with open(os.path.join(run_dir, "benchmark.json"), "w") as f:
        json.dump(out, f, indent=1)
    md = [f"# Benchmark: {os.path.basename(cfg['dataset'])}", "",
          f"{len(cids)} test users, full-catalogue ranks. System: `{primary}`.", "",
          "| method | group | HR@10 | NDCG@10 | 95% CI | HR@20 | NDCG@20 | MRR | NDCG@10 all users |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(table, key=lambda r: -r["ndcg@10"]):
        au = r.get("all_users")
        md.append(f"| {r['method']} | {r['group']} | {r['hr@10']:.4f} | {r['ndcg@10']:.4f} | "
                  f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}] | {r['hr@20']:.4f} | {r['ndcg@20']:.4f} | {r['mrr']:.4f} | "
                  + (f"{au['ndcg@10']:.4f} (n={au['n_users']})" if au else "") + " |")
    md += ["", "The last column scores each trained baseline on every user of the dataset. The other columns use the "
               "sampled test users shared by all methods, because the decision model cannot be run on every user."]
    if baselines:
        ref = out["reference_baseline"]
        md += ["", f"System against the baseline chosen on validation (`{ref}`): NDCG@10 difference "
                   f"{out[f'delta_ndcg10_vs_{ref}']:+.4f}, paired randomisation p = {out[f'p_value_vs_{ref}']:.4f}.",
               f"Against the baseline that is best on test (`{out['best_baseline_on_test']}`): "
               f"{out['delta_ndcg10_vs_' + out['best_baseline_on_test']]:+.4f}, "
               f"p = {out['p_value_vs_' + out['best_baseline_on_test']]:.4f}.",
               "", "Planned comparisons (paired randomisation, Holm correction over the rows):", "",
               "| comparison | NDCG@10 difference | p | p (Holm) |", "|---|---|---|---|"]
        md += [f"| `{c['a']}` against `{c['b']}` | {c['delta_ndcg10']:+.4f} | {c['p_value']:.4f} | {c['p_holm']:.4f} |"
               for c in out["comparisons"]]
    if agent_info:
        ch, te = agent_info["chosen"], agent_info["test"]
        md += ["", f"Agent setting chosen on validation users: routing `{ch['route']}`, expansion `{ch['policy']}`, "
                   f"fusion weight {ch['gamma']}. On test users it expanded for {te['expansion_rate']:.3f} of them, "
                   f"with {te['requests']:.2f} requests and {te['tokens']:.0f} tokens per user."]
    with open(os.path.join(run_dir, "benchmark.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    out["markdown"] = "\n".join(md)
    return out


def benchmark(cfg_paths: list[str], backend_path: Optional[str] = None, extra: Optional[list[tuple[str, str, str]]] = None,
              out: str = "runs") -> dict:
    """Per-dataset tables plus one cross-dataset summary of NDCG@10 and HR@10."""
    results = [benchmark_dataset(p, backend_path, extra) for p in cfg_paths]
    methods: list[str] = []
    for r in results:
        methods += [t["method"] for t in r["table"] if t["method"] not in methods]
    names = [os.path.basename(r["dataset"]) for r in results]
    md = ["# Benchmark summary", "", "NDCG@10 (HR@10) on full-catalogue ranks, same test users for every method.", "",
          "| method | " + " | ".join(names) + " | mean rank |", "|---|" + "---|" * (len(names) + 1)]
    place: dict[str, list[int]] = {m: [] for m in methods}
    for r in results:
        order = sorted(r["table"], key=lambda t: -t["ndcg@10"])
        for pos, t in enumerate(order, 1):
            place[t["method"]].append(pos)
    for m in sorted(methods, key=lambda m: np.mean(place[m]) if place[m] else 99):
        cells = []
        for r in results:
            t = next((t for t in r["table"] if t["method"] == m), None)
            cells.append(f"{t['ndcg@10']:.4f} ({t['hr@10']:.4f})" if t else "")
        md.append(f"| {m} | " + " | ".join(cells) + f" | {np.mean(place[m]):.2f} |")
    md += ["", "Planned comparisons per dataset (NDCG@10 difference, paired randomisation p, Holm-corrected p):"]
    for r, n in zip(results, names):
        for c in r.get("comparisons", []):
            md.append(f"* {n}: `{c['a']}` against `{c['b']}` {c['delta_ndcg10']:+.4f}, p = {c['p_value']:.4f}, "
                      f"Holm p = {c['p_holm']:.4f}")
    os.makedirs(out, exist_ok=True)
    tag = os.path.splitext(os.path.basename(backend_path))[0] if backend_path else "inline"
    path = os.path.join(out, f"benchmark_summary_{tag}.md")
    with open(path, "w") as f:
        f.write("\n".join(md) + "\n")
    return {"summary_path": path, "markdown": "\n".join(md), "datasets": results}
