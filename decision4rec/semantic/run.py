"""Entry points of the semantic-layer experiments (E1 facets, E2 relations, E3 new items), used by the CLI."""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Optional

import numpy as np

from ..data.schema import Dataset
from ..decision.base import build_backend
from ..experiments.benchmark import BIG, _ndcg10
from ..experiments.runner import get_cases, load_config
from ..experiments.stratify import baseline_ranks
from ..graph.kg import ItemKG
from . import facets as F
from . import relations as R


def _views(ds: Dataset, section: dict, kg: Optional[ItemKG] = None):
    from ..retrieval.agents import ItemViews
    from ..retrieval.slm import build_embedder
    emb = build_embedder(section.get("embedder"))
    return ItemViews(ds, kg, emb, section.get("cache")), emb


def run_facets(cfg_path: str, backend_path: Optional[str] = None, embedding: bool = False, workers: int = 1,
               limit: Optional[int] = None) -> dict:
    cfg = load_config(cfg_path, None if embedding else backend_path)
    fcfg = cfg.get("facets")
    if not fcfg:
        raise SystemExit(f"{cfg_path} has no facets section")
    ds = Dataset.load(cfg["dataset"])
    schema = F.load_schema(fcfg["schema"])
    if embedding:
        views, emb = _views(ds, fcfg)
        answers = F.embedding_answers(ds, schema, views.matrix("content"), emb)
        F.write_answers(answers, fcfg["emb"])
        return {"source": "emb", "items": len(answers), "summary": F.facet_summary(ds, schema, answers, "emb")}
    if backend_path is None and "backend" not in cfg:
        raise SystemExit("facets from a decision model need --backend")
    backend = build_backend(cfg["backend"])
    try:
        info = F.compile_facets(ds, backend, schema, fcfg["dm"], workers=workers, limit=limit)
    finally:
        backend.close()
    return {"source": "dm", **info, "summary": F.facet_summary(ds, schema, F.read_answers(fcfg["dm"]), "dm")}


def run_relations(cfg_path: str, backend_path: Optional[str], workers: int = 1) -> dict:
    cfg = load_config(cfg_path, backend_path)
    rcfg = cfg.get("relations")
    if not rcfg:
        raise SystemExit(f"{cfg_path} has no relations section")
    ds = Dataset.load(cfg["dataset"])
    cases = get_cases(cfg, ds)
    kg = ItemKG(ds)
    views, _ = _views(ds, rcfg, kg)
    anchors = R.anchors_of(cases["valid"] + cases["test"], int(rcfg.get("last", 3)))
    pairs = R.candidate_pairs(ds, anchors, kg.cooc, views.matrix("content"), int(rcfg.get("n_cooc", 10)),
                              int(rcfg.get("n_content", 10)))
    backend = build_backend(cfg["backend"])
    try:
        info = R.judge_relations(ds, backend, pairs, rcfg["path"], workers=workers)
    finally:
        backend.close()
    return {**info, **R.relation_stats(R.read_relations(rcfg["path"]), kg.cooc, ds)}


def run_relation_eval(cfg_paths: list[str], head: int = 100) -> dict:
    from ..models.sasrec import SASRecRetriever
    out, md = [], ["# E2: typed item relations from a decision model", "",
                   "NDCG@10 on full-catalogue ranks of the test users. The related items of the last history items "
                   "join the SASRec top 50; weights fitted on validation users. Controls use the same pairs without "
                   "the decision model. Differences with paired randomisation p."]
    for path in cfg_paths:
        cfg = load_config(path)
        rcfg = cfg["relations"]
        ds = Dataset.load(cfg["dataset"])
        cases = get_cases(cfg, ds)
        rel = R.read_relations(rcfg["path"])
        if not rel:
            raise SystemExit(f"{rcfg['path']} holds no judged relations; run the relations step first")
        kg = ItemKG(ds)
        retr = SASRecRetriever(cfg["retriever"]["path"])
        test = cases["test"]
        found = baseline_ranks(cfg, [[c.user, c.target] for c in test], ("gru4rec", "sasrec", "sasrecf"))
        refs = {m: _ndcg10(np.array([BIG if found[m].get(str(c.cid)) is None else found[m][str(c.cid)]
                                     for c in test], dtype=float)) for m in found}
        pop = Counter(int(x[0]) for s in ds.seqs for x in s[:-2])
        tail = np.array([pop.get(c.target, 0) < head for c in test])
        res = R.relation_eval(cases, retr, rel, kg.cooc, int(rcfg.get("last", 3)), float(rcfg.get("decay", 0.7)),
                              refs, tail)
        res.update(dataset=ds.name, config=path, stats=R.relation_stats(rel, kg.cooc, ds))
        out.append(res)
        st = res["stats"]
        names = ["sasrec (top 50)", "co-occurrence, same pairs"] + list(refs)
        md += ["", f"## {ds.name}", "",
               f"{st['pairs']} pairs judged for {st['anchors']} anchor items. Most likely relation: "
               + ", ".join(f"{k} {v:.2f}" for k, v in st["shares"].items())
               + ". Among co-occurrence pairs: " + ", ".join(f"{k} {v:.2f}" for k, v in st["by_source"]["co-occurrence"].items())
               + "; among text-only pairs: " + ", ".join(f"{k} {v:.2f}" for k, v in st["by_source"]["text only"].items()) + ".",
               "", f"Target among the candidates: top 50 {res['recall']['top 50']:.3f}, with related items "
               f"{res['recall']['top 50 + related']:.3f} ({res['recall']['mean related per user']:.1f} related items per user).",
               "", "| variant | weights | NDCG@10 | " + " | ".join(f"vs {n}" for n in names)
               + f" | rare targets (< {head}) | vs SASRec on rare targets |",
               "|---|---|---|" + "---|" * len(names) + "---|---|"]
        for n, v in res["refs"].items():
            md.append(f"| {n} (trained baseline) | | {v:.4f} | " + " | ".join("" for _ in names) + " | | |")
        for row in res["rows"]:
            cells = [f"{row['vs'][n]['diff']:+.4f} (p={row['vs'][n]['p']:.3f})" if n in row["vs"] else "" for n in names]
            t = row.get("tail_vs_sasrec")
            md.append(f"| {row['variant']} | {row['weights']} | {row['ndcg@10']:.4f} | " + " | ".join(cells)
                      + f" | {row.get('tail_ndcg@10', float('nan')):.4f} | "
                      + (f"{t['diff']:+.4f} (p={t['p']:.3f})" if t else "") + " |")
    return {"datasets": out, "markdown": "\n".join(md) + "\n"}


def run_prepare_cold(src: str, out: str, quantile: float = 0.9, min_pre: int = 3) -> dict:
    from .cold import make_cold_dataset
    if os.path.exists(os.path.join(out, "meta.json")):
        with open(os.path.join(out, "meta.json")) as f:
            return {"skipped": "exists", **json.load(f)["extra"].get("cold_protocol", {})}
    cold = make_cold_dataset(Dataset.load(src), quantile, min_pre)
    cold.save(out)
    return {"users": cold.n_users, **cold.extra["cold_protocol"]}
