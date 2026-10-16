"""Command line entry point: `python -m decision4rec.cli <command>`."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def cmd_prepare(a) -> None:
    from .data import loaders
    if a.dataset == "amazon23":
        ds = loaders.load_amazon23(a.reviews, a.meta, a.name or "amazon23", a.core, a.max_users)
    elif a.dataset == "ml1m":
        ds = loaders.load_ml1m(a.folder, a.core, a.max_users)
    elif a.dataset == "mlcsv":
        ds = loaders.load_ml_csv(a.folder, a.name or "ml-20m", a.core, a.max_users)
    elif a.dataset == "steam":
        ds = loaders.load_steam(a.reviews, a.meta, a.core, a.max_users)
    elif a.dataset == "mind":
        ds = loaders.load_mind(a.folder.split(","), a.core, a.max_users)
    elif a.dataset == "synthetic":
        ds, truth = loaders.make_synthetic(n_users=a.max_users or 300, seed=a.seed)
        os.makedirs(a.out, exist_ok=True)
        np.savez(os.path.join(a.out, "truth.npz"), **truth)
    else:
        raise SystemExit(f"unknown dataset {a.dataset}")
    ds.save(a.out)
    n = sum(len(s) for s in ds.seqs)
    print(f"[prepare] {ds.name}: {ds.n_users} users, {ds.n_items} items, {n} interactions -> {a.out}")


def cmd_train_retriever(a) -> None:
    from .data.schema import Dataset
    from .models.sasrec import train_sasrec
    ds = Dataset.load(a.dataset)
    info = train_sasrec(ds.seqs, ds.n_items, a.out, d=a.dim, max_len=a.max_len, n_blocks=a.blocks,
                        dropout=a.dropout, lr=a.lr, batch_size=a.batch_size, epochs=a.epochs,
                        patience=a.patience, seed=a.seed, device=a.device)
    print(f"[train-retriever] best valid NDCG@10 {info['best_valid_ndcg10']:.4f} at epoch {info['best_epoch']}")


def cmd_run(a) -> None:
    from .experiments.report import report, to_markdown
    from .experiments.runner import run_experiment
    out = run_experiment(a.config, a.arms.split(",") if a.arms else None, a.max_cases, backend_path=a.backend,
                         max_valid=a.max_valid)
    if not a.no_report:
        print(to_markdown(report(out)))
    print(f"[run] results in {out}")


def cmd_report(a) -> None:
    from .experiments.report import report, to_markdown
    for run in a.runs:
        print(to_markdown(report(run, a.reference)))


def cmd_check_backend(a) -> None:
    from .experiments.check import check_backend
    raise SystemExit(0 if check_backend(a.config, a.backend) else 1)


def cmd_audit(a) -> None:
    from .experiments.audit import run_audit
    print(json.dumps(run_audit(a.config, a.n_cases, a.block, backend_path=a.backend), indent=1))


def cmd_agent(a) -> None:
    from .experiments.agent_study import run_agent_study
    print(json.dumps(run_agent_study(a.config, a.n_cases, backend_path=a.backend), indent=1))


def cmd_agent_bench(a) -> None:
    from .experiments.agent_bench import run_agent_bench
    for cfg in a.config:
        print(run_agent_bench(cfg, a.backend, a.max_cases)["markdown"])


def cmd_rating(a) -> None:
    from .experiments.rating import run_rating
    rep = run_rating(a.config, a.n_cases, backend_path=a.backend)
    print(open(os.path.join(rep["out_dir"], "rating.md")).read() if "out_dir" in rep else json.dumps(rep, indent=1))


def cmd_retrieval(a) -> None:
    from .experiments.retrieval_study import run_retrieval_study
    print(json.dumps(run_retrieval_study(a.config, a.n_users), indent=1))


def cmd_build_cases(a) -> None:
    from .data.schema import Dataset
    from .experiments.runner import case_paths, get_cases, load_config
    cfg = load_config(a.config)
    cases = get_cases(cfg, Dataset.load(cfg["dataset"]))
    print(f"[build-cases] test {len(cases['test'])}, valid {len(cases['valid'])} -> {case_paths(cfg)['test']}")


def cmd_baselines(a) -> None:
    from .experiments.benchmark import run_baselines
    for cfg in a.config:
        run_baselines(cfg, a.models.split(",") if a.models else None, a.device)


def cmd_benchmark(a) -> None:
    from .experiments.benchmark import benchmark
    extra = [tuple(e.split(":", 2)) for e in (a.extra or [])]
    print(benchmark(a.config, a.backend, extra)["markdown"])


def cmd_stratify(a) -> None:
    from .experiments.stratify import stratify
    res = stratify(a.runs, a.arms.split(",") if a.arms else None, [tuple(x) for x in (a.pair or [])])
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(res["markdown"])
        with open(os.path.splitext(a.out)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump(res["datasets"], f, indent=1)
    print(res["markdown"])


def cmd_asc(a) -> None:
    from .experiments.asc import asc_study
    res = asc_study(a.runs, a.arms.split(",") if a.arms else None, a.head, a.only.split(",") if a.only else None)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(res["markdown"])
        with open(os.path.splitext(a.out)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump(res["runs"], f, indent=1)
    print(res["markdown"])


def _write(out: str, res: dict, key: str) -> None:
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(res["markdown"])
        with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump(res[key], f, indent=1)


def cmd_facets(a) -> None:
    from .semantic.run import run_facets
    print(json.dumps(run_facets(a.config, a.backend, a.embedding, a.workers, a.limit), indent=1))


def cmd_facet_report(a) -> None:
    from .semantic.report import facet_report
    res = facet_report(a.config, a.models.split(",") if a.models else None, a.ref)
    _write(a.out, res, "datasets")
    print(res["markdown"])


def cmd_relations(a) -> None:
    from .semantic.run import run_relations
    print(json.dumps(run_relations(a.config, a.backend, a.workers), indent=1))


def cmd_relation_eval(a) -> None:
    from .semantic.run import run_relation_eval
    res = run_relation_eval(a.config)
    _write(a.out, res, "datasets")
    print(res["markdown"])


def cmd_system_eval(a) -> None:
    from .semantic.system import system_eval, system_markdown
    results = [system_eval(cfg, None if run in ("-", "none") else run, a.arm) for cfg, run in a.pair]
    res = {"datasets": results, "markdown": system_markdown(results)}
    _write(a.out, res, "datasets")
    print(res["markdown"])


def cmd_prepare_cold(a) -> None:
    from .semantic.run import run_prepare_cold
    print(json.dumps(run_prepare_cold(a.src, a.out, a.quantile, a.min_pre), indent=1))


def cmd_export_rectune(a) -> None:
    from .experiments.rectune import export_rectune
    print(json.dumps(export_rectune(a.config, a.out, a.n_train, a.n_heldout, a.block, a.outside_rate,
                                    a.item_chars), indent=1))


def cmd_reserve_users(a) -> None:
    from .data.reserved import reserve
    from .experiments.runner import load_config
    for path in a.config:
        print(json.dumps(reserve(load_config(path), a.n, a.seed, force=a.force)), flush=True)


def cmd_train_llm_ranker(a) -> None:
    from .experiments.llmrank import train
    m = train(a.data, a.base, a.out, revision=a.revision, lr=a.lr, lora=a.lora, alpha=a.alpha, accum=a.accum,
              epochs=a.epochs, seed=a.seed, max_len=a.max_len, max_records=a.max_records, dtype=a.dtype,
              options=a.options)
    print(json.dumps({k: v for k, v in m.items() if k != "loss_every_10_steps"}, indent=1))


def cmd_export_choices(a) -> None:
    from .experiments.rectune import export_choices
    print(json.dumps(export_choices(a.config, a.out, a.n_per_dataset, a.n_heldout, a.seed,
                                    crossfit=getattr(a, "crossfit", False), offsets=getattr(a, "offsets", 0.0)), indent=1))


def cmd_export_evidence(a) -> None:
    from .evidence.export import export_evidence_choices
    print(json.dumps(export_evidence_choices(a.config, a.out, a.n_per_dataset, a.n_heldout, a.seed, a.dropout),
                     indent=1))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="decision4rec")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prepare", help="build a processed dataset folder")
    s.add_argument("--dataset", required=True, choices=["amazon23", "ml1m", "mlcsv", "steam", "mind", "synthetic"])
    s.add_argument("--out", required=True)
    s.add_argument("--reviews"); s.add_argument("--meta"); s.add_argument("--folder"); s.add_argument("--name")
    s.add_argument("--core", type=int, default=5); s.add_argument("--max-users", type=int, default=None)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_prepare)

    s = sub.add_parser("train-retriever", help="train SASRec on a processed dataset")
    s.add_argument("--dataset", required=True); s.add_argument("--out", required=True)
    s.add_argument("--dim", type=int, default=64); s.add_argument("--max-len", type=int, default=50)
    s.add_argument("--blocks", type=int, default=2); s.add_argument("--dropout", type=float, default=0.2)
    s.add_argument("--lr", type=float, default=1e-3); s.add_argument("--batch-size", type=int, default=256)
    s.add_argument("--epochs", type=int, default=300); s.add_argument("--patience", type=int, default=10)
    s.add_argument("--seed", type=int, default=13); s.add_argument("--device", default=None)
    s.set_defaults(fn=cmd_train_retriever)

    s = sub.add_parser("run", help="execute the arms of a config and write the report")
    s.add_argument("--config", required=True); s.add_argument("--arms", default=None)
    s.add_argument("--backend", default=None, help="backend file layered on the config")
    s.add_argument("--max-cases", type=int, default=None); s.add_argument("--no-report", action="store_true")
    s.add_argument("--max-valid", type=int, default=None, help="limit the validation split alone")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("report", help="recompute the report of one or more run folders")
    s.add_argument("runs", nargs="+"); s.add_argument("--reference", default=None)
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("check-backend", help="protocol conformance and determinism test of a backend")
    s.add_argument("--config", required=True)
    s.add_argument("--backend", default=None, help="backend file layered on the config")
    s.set_defaults(fn=cmd_check_backend)

    s = sub.add_parser("audit", help="choice-axiom audit: IIA, regularity, substitution by KG similarity")
    s.add_argument("--config", required=True); s.add_argument("--n-cases", type=int, default=200)
    s.add_argument("--backend", default=None, help="backend file layered on the config")
    s.add_argument("--block", type=int, default=8)
    s.set_defaults(fn=cmd_audit)

    s = sub.add_parser("agent", help="agent runtime study on ActiveGraph: replay, expansion, attribution")
    s.add_argument("--config", required=True); s.add_argument("--n-cases", type=int, default=100)
    s.add_argument("--backend", default=None, help="backend file layered on the config")
    s.set_defaults(fn=cmd_agent)

    s = sub.add_parser("agent-bench", help="the agent on ActiveGraph on every validation and test user (benchmark row)")
    s.add_argument("--config", required=True, nargs="+"); s.add_argument("--backend", default=None)
    s.add_argument("--max-cases", type=int, default=None, help="partial run for timing, not used by the benchmark")
    s.set_defaults(fn=cmd_agent_bench)

    s = sub.add_parser("rating", help="rating prediction with the score primitive (ordinal scale study)")
    s.add_argument("--config", required=True); s.add_argument("--backend", default=None)
    s.add_argument("--n-cases", type=int, default=1000)
    s.set_defaults(fn=cmd_rating)

    s = sub.add_parser("retrieval", help="retrieval study: each retrieval agent alone and fused (no backend needed)")
    s.add_argument("--config", required=True); s.add_argument("--n-users", type=int, default=1000)
    s.set_defaults(fn=cmd_retrieval)

    s = sub.add_parser("build-cases", help="build and store the evaluation cases of a config (no backend needed)")
    s.add_argument("--config", required=True)
    s.set_defaults(fn=cmd_build_cases)

    s = sub.add_parser("baselines", help="tune, train and evaluate the trained baselines on a config's test users")
    s.add_argument("--config", required=True, nargs="+"); s.add_argument("--models", default=None)
    s.add_argument("--device", default=None)
    s.set_defaults(fn=cmd_baselines)

    s = sub.add_parser("benchmark", help="recommendation benchmark: Decision4Rec against baselines, per dataset and summary")
    s.add_argument("--config", required=True, nargs="+"); s.add_argument("--backend", default=None)
    s.add_argument("--extra", action="append", help="backend_file:variant:label, a row from another backend's run")
    s.set_defaults(fn=cmd_benchmark)

    s = sub.add_parser("stratify", help="NDCG@10 per target popularity and history length, from stored ranks only")
    s.add_argument("--runs", required=True, nargs="+", help="run folders (with manifest.json and ranks.json)")
    s.add_argument("--arms", default=None, help="comma-separated arm names to show (default: the main ones)")
    s.add_argument("--pair", nargs=2, action="append", metavar=("A", "B"), help="paired test per group, row labels")
    s.add_argument("--out", default=None, help="markdown file; a .json with the same stem is written next to it")
    s.set_defaults(fn=cmd_stratify)

    s = sub.add_parser("asc", help="does adding the item constant (log popularity) fix the decision model? stored runs only")
    s.add_argument("--runs", required=True, nargs="+", help="run folders (with manifest.json and the arm records)")
    s.add_argument("--arms", default=None, help="comma-separated arm names (default: the main ones)")
    s.add_argument("--head", type=int, default=100, help="training interactions from which a candidate counts as head")
    s.add_argument("--only", default=None, help="comma-separated variant kinds, for a pre-registered analysis "
                   "(pop only, dm, pop, fuse, fuse+pop, gate, ugate, ugate+pop, shrink); adds bootstrap intervals")
    s.add_argument("--out", default=None, help="markdown file; a .json with the same stem is written next to it")
    s.set_defaults(fn=cmd_asc)

    s = sub.add_parser("facets", help="E1: item facets from a decision model, or the embedding control")
    s.add_argument("--config", required=True); s.add_argument("--backend", default=None)
    s.add_argument("--embedding", action="store_true", help="the text-embedding control instead of a decision model")
    s.add_argument("--workers", type=int, default=1); s.add_argument("--limit", type=int, default=None)
    s.set_defaults(fn=cmd_facets)

    s = sub.add_parser("facet-report", help="E1 report: facet quality and facet models against SASRecF on every user")
    s.add_argument("--config", required=True, nargs="+"); s.add_argument("--models", default=None)
    s.add_argument("--ref", default="sasrecf"); s.add_argument("--out", default=None)
    s.set_defaults(fn=cmd_facet_report)

    s = sub.add_parser("relations", help="E2: typed relations of item pairs judged by a decision model")
    s.add_argument("--config", required=True); s.add_argument("--backend", required=True)
    s.add_argument("--workers", type=int, default=1)
    s.set_defaults(fn=cmd_relations)

    s = sub.add_parser("relation-eval", help="E2 report: related items added to SASRec, against controls")
    s.add_argument("--config", required=True, nargs="+"); s.add_argument("--out", default=None)
    s.set_defaults(fn=cmd_relation_eval)

    s = sub.add_parser("system-eval", help="the Decision4Rec system: trained backbone plus relations and the evidence gate")
    s.add_argument("--pair", nargs=2, action="append", required=True, metavar=("CONFIG", "RUN"),
                   help="a config and the run folder with the decision model's answers on its cases ('-' for none)")
    s.add_argument("--arm", default="wholepool@p0"); s.add_argument("--out", default=None)
    s.set_defaults(fn=cmd_system_eval)

    s = sub.add_parser("prepare-cold", help="E3: new-item protocol, a time cutoff over a processed dataset")
    s.add_argument("--src", required=True); s.add_argument("--out", required=True)
    s.add_argument("--quantile", type=float, default=0.9); s.add_argument("--min-pre", type=int, default=3)
    s.set_defaults(fn=cmd_prepare_cold)

    s = sub.add_parser("export-rectune", help="write Kev fine-tuning records from training interactions")
    s.add_argument("--config", required=True); s.add_argument("--out", required=True)
    s.add_argument("--n-train", type=int, default=20000); s.add_argument("--n-heldout", type=int, default=1000)
    s.add_argument("--block", type=int, default=10); s.add_argument("--outside-rate", type=float, default=0.15)
    s.add_argument("--item-chars", type=int, default=0, help="shorten item text to fit GPU memory during tuning")
    s.set_defaults(fn=cmd_export_rectune)

    s = sub.add_parser("export-choices", help="Kev training records of real next-item choices over first-stage candidates")
    s.add_argument("--config", nargs="+", required=True); s.add_argument("--out", required=True)
    s.add_argument("--n-per-dataset", type=int, default=15000); s.add_argument("--n-heldout", type=int, default=500)
    s.add_argument("--seed", type=int, default=13)
    s.add_argument("--crossfit", action="store_true", help="records from the pool's ranker users, cross-fitted in two halves")
    s.add_argument("--offsets", type=float, default=0.0,
                   help="weight of the first stage's standardized score, stored per option as a fixed loss offset (0: none)")
    s.set_defaults(fn=cmd_export_choices)

    s = sub.add_parser("reserve-users", help="reserve fresh users of each config's dataset for the confirmation (R1)")
    s.add_argument("--config", nargs="+", required=True); s.add_argument("--n", type=int, default=10000)
    s.add_argument("--seed", type=int, default=2026); s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_reserve_users)

    s = sub.add_parser("train-llm-ranker", help="LlamaRec-style reranker: a causal LM with adapters trained on the "
                                                "decision model's records, read through option-label logits (R17)")
    s.add_argument("--data", required=True); s.add_argument("--base", required=True); s.add_argument("--out", required=True)
    s.add_argument("--revision"); s.add_argument("--lr", type=float, default=1e-4)
    s.add_argument("--lora", type=int, default=16); s.add_argument("--alpha", type=float, default=32.0)
    s.add_argument("--accum", type=int, default=8); s.add_argument("--epochs", type=int, default=1)
    s.add_argument("--seed", type=int, default=0); s.add_argument("--max-len", type=int, default=6144)
    s.add_argument("--max-records", type=int, default=0); s.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    s.add_argument("--options", type=int, default=0, help="cut each question to the target and options-1 others (0: all)")
    s.set_defaults(fn=cmd_train_llm_ranker)

    s = sub.add_parser("export-evidence", help="Kev training records of real choices read through a store's evidence (docs/08)")
    s.add_argument("--config", nargs="+", required=True); s.add_argument("--out", required=True)
    s.add_argument("--n-per-dataset", type=int, default=15000); s.add_argument("--n-heldout", type=int, default=500)
    s.add_argument("--seed", type=int, default=13); s.add_argument("--dropout", type=float, default=0.25)
    s.set_defaults(fn=cmd_export_evidence)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
