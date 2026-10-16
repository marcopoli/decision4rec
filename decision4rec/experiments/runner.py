"""Experiment runner: builds cases, executes arms against a backend and stores raw observations."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import yaml
from tqdm import tqdm

from ..data.candidates import build_cases
from ..data.reserved import reserved_users
from ..data.schema import Case, Dataset, load_cases, save_cases
from ..decision.base import DecisionBackend, build_backend
from ..graph.kg import ItemKG
from ..operators.rank import Arm, run_arm
from ..prompts import PromptBuilder


def load_config(path: str, backend_path: Optional[str] = None) -> dict:
    """Read an experiment config. A backend file (`backend`, `tag`, optional `price_per_mtok`) can be
    layered on top, so one experiment definition runs unchanged against any decision model."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("eval", {})
    cfg.setdefault("prompt", {})
    if backend_path:
        with open(backend_path, encoding="utf-8") as f:
            b = yaml.safe_load(f)
        tag = b.get("tag") or os.path.splitext(os.path.basename(backend_path))[0]
        cfg["backend"] = dict(b["backend"])
        cfg["backend"].setdefault("cache", os.path.join(cfg["out"], "cache", f"{tag}.sqlite"))
        cfg["name"] = f"{cfg['name']}_{tag}"
        for key in ("price_per_mtok", "workers"):
            if key in b:
                cfg[key] = b[key]
    cache = (cfg.get("backend") or {}).get("cache")
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    return cfg


def build_retriever(cfg: dict, ds: Dataset, full_cfg: Optional[dict] = None):
    """`cfg` is the retriever section; the evidence retriever also needs the whole config (`full_cfg`)."""
    kind = cfg.get("kind", "itemknn")
    if kind == "evidence":
        if full_cfg is None:
            raise SystemExit("the evidence retriever needs the whole config")
        from ..evidence.store import EvidenceRetriever, store_for
        return EvidenceRetriever(store_for(full_cfg, ds))
    if kind == "sasrec":
        from ..models.sasrec import SASRecRetriever
        return SASRecRetriever(cfg["path"], device=cfg.get("device"))
    if kind == "multi":
        from ..retrieval.agents import build_multi_retriever
        base = build_retriever(cfg["base"], ds) if cfg.get("base") else None
        return build_multi_retriever(cfg, ds, base)
    if kind == "baseline":
        from .stage1 import build_baseline_retriever
        return build_baseline_retriever(cfg, ds)
    if kind == "pool":
        if full_cfg is None:
            raise SystemExit("the agent pool needs the whole config")
        from ..agents.pool import PoolRetriever, pool_for
        return PoolRetriever(pool_for(full_cfg, ds))
    from ..models.retrievers import ItemKNNRetriever, PopRetriever
    if kind == "itemknn":
        return ItemKNNRetriever(ds.seqs, ds.n_items)
    if kind == "pop":
        return PopRetriever(ds.seqs, ds.n_items)
    raise ValueError(f"unknown retriever kind {kind!r}")


def case_paths(cfg: dict) -> dict[str, str]:
    ev = cfg["eval"]
    track, K, seed = ev.get("track", "A"), int(ev.get("K", 20)), int(ev.get("seed", 13))
    rcfg = cfg.get("retriever", {})
    rname = rcfg.get("tag") or rcfg.get("kind", "itemknn")
    stem = os.path.join(cfg["out"], "cases", f"{rname}_track{track}_K{K}_s{seed}")
    return {"test": f"{stem}_test{int(ev.get('n_test', 1000))}.jsonl",
            "valid": f"{stem}_valid{int(ev.get('n_valid', 300))}.jsonl", "info": f"{stem}_info.json"}


def _excluded_users(cfg: dict, paths: dict) -> set:
    """Users to leave out of new cases: those of the case files listed in `eval.exclude_cases`, or of every
    other case file of the output folder when it is "all". Used to draw fresh users for a confirmatory test."""
    spec = cfg["eval"].get("exclude_cases")
    if not spec:
        return set()
    own = {os.path.abspath(paths["test"]), os.path.abspath(paths["valid"])}
    if spec == "all":
        folder = os.path.join(cfg["out"], "cases")
        files = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jsonl")) \
            if os.path.isdir(folder) else []
        if not files:
            raise SystemExit(f"exclude_cases: all, but {folder} holds no case file to exclude")
    else:
        files = list(spec)
    users: set = set()
    for fp in files:
        if os.path.abspath(fp) in own:
            continue
        if not os.path.exists(fp):
            raise SystemExit(f"exclude_cases: {fp} does not exist")
        users |= {c.user for c in load_cases(fp)}
    return users


def get_cases(cfg: dict, ds: Dataset) -> dict[str, list[Case]]:
    """Load or build the validation and test cases. They are shared by every backend for pairing."""
    ev = cfg["eval"]
    paths = case_paths(cfg)
    if os.path.exists(paths["test"]) and os.path.exists(paths["valid"]):
        return {s: load_cases(paths[s]) for s in ("test", "valid")}
    track, K, seed = ev.get("track", "A"), int(ev.get("K", 20)), int(ev.get("seed", 13))
    pool = int(ev.get("pool", 200))
    excluded = _excluded_users(cfg, paths)
    reserved = reserved_users(cfg)
    only = None
    if ev.get("users") == "reserved":       # the confirmation on fresh users (R1): only the reserved users
        if not reserved:
            raise SystemExit(f"eval.users: reserved, but {cfg['dataset']} has no reservation (cli reserve-users)")
        only = reserved
    else:
        excluded = excluded | reserved       # development never draws the users reserved for the confirmation
    retriever = build_retriever(cfg.get("retriever", {}), ds, cfg)
    if ev.get("target"):                    # new-item protocol: test users whose target is after the cutoff
        if ev["target"] != "post":
            raise SystemExit(f"eval.target {ev['target']!r}: only 'post' is supported")
        from ..semantic.cold import post_target_users
        only = post_target_users(ds) if only is None else only & post_target_users(ds)
    test, info_t = build_cases(ds, retriever, "test", int(ev.get("n_test", 1000)), K, track, seed, pool,
                               exclude_users=excluded or None, only_users=only)
    valid, info_v = build_cases(ds, retriever, "valid", int(ev.get("n_valid", 300)), K, track, seed, pool,
                                exclude_users=excluded | {c.user for c in test},
                                only_users=only if ev.get("users") == "reserved" else None)
    info_t["excluded_users"] = info_v["excluded_users"] = len(excluded)
    save_cases(test, paths["test"])
    save_cases(valid, paths["valid"])
    with open(paths["info"], "w") as f:
        json.dump({"test": info_t, "valid": info_v}, f, indent=1)
    return {"test": test, "valid": valid}


def make_prompt_builder(cfg: dict, ds: Dataset, need_kg: bool = True, store=None) -> PromptBuilder:
    """The prompt builder of a config. With an `evidence:` section (or a given store) it is the builder of the
    evidence-reading decision model, which adds the store's statements (docs/08)."""
    pc = cfg["prompt"]
    kw = dict(hist_len=int(cfg["eval"].get("hist_len", 10)), max_facts=int(pc.get("max_facts", 3)),
              item_chars=int(pc.get("item_chars", 400)), profile_hist=int(pc.get("profile_hist", 30)),
              profile_item_chars=int(pc.get("profile_item_chars", 120)), max_h=int(pc.get("max_hypotheses", 24)),
              intent_relation=pc.get("intent_relation"), max_intents=int(pc.get("max_intents", 10)))
    if store is not None or cfg.get("evidence"):
        from ..evidence.prompt import EvidencePromptBuilder
        from ..evidence.store import store_for
        pool = None
        if (cfg.get("retriever") or {}).get("kind") == "pool":
            from ..agents.pool import pool_for
            pool = pool_for(cfg, ds)
        return EvidencePromptBuilder(ds, store if store is not None else store_for(cfg, ds), pool=pool, **kw)
    kg = ItemKG(ds, max_df=float(pc.get("max_df", 0.5))) if need_kg else None
    return PromptBuilder(ds, kg, **kw)


def _done(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["cid"] for line in f if line.strip()}


def run_arm_on_cases(arm: Arm, cases: list[Case], backend: Optional[DecisionBackend], pb: PromptBuilder,
                     path: str, seed: int, workers: int = 1) -> int:
    """Append one record per case to `path`, skipping cases already stored (resumable)."""
    done = _done(path)
    todo = [c for c in cases if c.cid not in done]
    if not todo:
        return 0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    name, t0, last = os.path.basename(path), time.time(), [time.time()]
    print(f"[run] {name}: {len(todo)} cases to do", flush=True)

    def progress(n: int) -> None:
        now = time.time()
        if n == len(todo) or n % 100 == 0 or now - last[0] > 600:
            last[0] = now
            per = (now - t0) / n
            print(f"[run] {name}: {n}/{len(todo)} done, {per:.1f} s per case, about "
                  f"{per * (len(todo) - n) / 60:.0f} min left for this file", flush=True)

    with open(path, "a", encoding="utf-8") as f:
        def work(c: Case) -> dict:
            return run_arm(c, arm, backend, pb, seed)

        if workers > 1:
            with ThreadPoolExecutor(workers) as ex:
                it = ex.map(work, todo)
                for n, rec in enumerate(tqdm(it, total=len(todo), desc=name, leave=False, disable=None), 1):
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    progress(n)
        else:
            for n, c in enumerate(tqdm(todo, desc=name, leave=False, disable=None), 1):
                f.write(json.dumps(work(c)) + "\n")
                f.flush()
                progress(n)
    return len(todo)


def run_experiment(cfg_path: str, only_arms: Optional[list[str]] = None, max_cases: Optional[int] = None,
                   splits: tuple = ("valid", "test"), backend_path: Optional[str] = None,
                   max_valid: Optional[int] = None) -> str:
    """max_cases limits every split to its first cases, max_valid the validation split alone (the 27B runs on the
    Spark read 1,000 test and 300 validation users per domain)."""
    cfg = load_config(cfg_path, backend_path)
    ds = Dataset.load(cfg["dataset"])
    cases = get_cases(cfg, ds)
    arms = [Arm.from_dict(a) for a in cfg["arms"]]
    if only_arms:
        arms = [a for a in arms if a.name in only_arms]
    pb = make_prompt_builder(cfg, ds, need_kg=any(
        a.evidence != "none" or a.profile != "none" or a.intent or a.method == "graph" for a in arms))
    needs_backend = any(a.method != "retriever" for a in arms)
    if needs_backend and "backend" not in cfg:
        raise SystemExit("this config has no backend section: pass one with --backend configs/backends/<file>.yaml")
    backend = build_backend(cfg["backend"]) if needs_backend else None
    out_dir = os.path.join(cfg["out"], cfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    seed = int(cfg["eval"].get("seed", 13))
    t0 = time.time()
    try:
        for split in splits:
            limit = max_valid if (split == "valid" and max_valid) else max_cases
            subset = cases[split][:limit] if limit else cases[split]
            for arm in arms:
                part = subset[: arm.max_cases] if arm.max_cases else subset
                n = run_arm_on_cases(arm, part, backend, pb, os.path.join(out_dir, f"{arm.name}.{split}.jsonl"),
                                     seed, int(cfg.get("workers", 1)))
                print(f"[run] {cfg['name']} {split:5s} {arm.name:28s} new records: {n}", flush=True)
    finally:
        if backend is not None:
            backend.close()
    manifest_path = os.path.join(out_dir, "manifest.json")
    known = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            known = {a["name"]: a for a in json.load(f).get("arms", [])}
    known.update({a.name: a.to_dict() for a in arms})
    with open(manifest_path, "w") as f:
        json.dump({"name": cfg["name"], "config": cfg, "arms": list(known.values()),
                   "backend": backend.fingerprint() if backend is not None else None,
                   "n_cases": {s: len(c) for s, c in cases.items()},
                   "wall_seconds_last_run": time.time() - t0}, f, indent=1)
    return out_dir
