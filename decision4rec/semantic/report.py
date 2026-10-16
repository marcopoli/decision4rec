"""E1 report: do decision-model facets make the trained sequential model better?

For each config: the quality of the facets (confidence, abstention, option shares, agreement with the raw
metadata where the schema names it), then NDCG@10 of the facet models against SASRecF on every user of the
dataset, overall and by the popularity of the target item, with paired randomisation tests and bootstrap
intervals. Only stored files are read.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Optional

import numpy as np

from ..data.schema import Dataset
from ..eval import metrics as M
from ..experiments.benchmark import _bench_dir, _ndcg10
from ..experiments.runner import load_config
from ..experiments.stratify import POP_BINS, _bucket
from .facets import embedding_answers, facet_summary, load_schema, read_answers

MODELS = ("sasrec", "sasrecf", "sasrec_dm", "sasrecf_dm", "sasrecf_emb", "seqrules", "contentknn", "transtext")


def _fmt(d: dict) -> str:
    return f"{d['diff']:+.4f} (p={d['p']:.3f}) [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"


def facet_report(cfg_paths: list[str], models: Optional[list[str]] = None, ref: str = "sasrecf",
                 head: int = 100, n_perm: int = 2000) -> dict:
    models = list(models or MODELS)
    out, md = [], ["# E1: decision-model facets in the trained sequential model", "",
                   "NDCG@10 on full-catalogue ranks of every user of the dataset (last item as target, as in the "
                   "benchmark), and on the sampled test users. Differences against the reference with paired "
                   f"randomisation p ({n_perm} permutations) and 95% bootstrap interval."]
    for path in cfg_paths:
        cfg = load_config(path)
        ds = Dataset.load(cfg["dataset"])
        bench = _bench_dir(cfg)
        res = {"config": path, "dataset": ds.name, "facets": [], "models": {}}
        md += ["", f"## {ds.name}", ""]
        fcfg = cfg.get("facets") or {}
        if fcfg.get("schema"):
            schema = load_schema(fcfg["schema"])
            for source in ("dm", "emb"):
                if fcfg.get(source) and os.path.exists(fcfg[source]):
                    res["facets"].append(facet_summary(ds, schema, read_answers(fcfg[source]), source))
            if res["facets"]:
                md += ["### Facets", "", "| facet | source | top option p | abstains | most frequent options | agrees with metadata |",
                       "|---|---|---|---|---|---|"]
                for fs in res["facets"]:
                    for name, info in fs["facets"].items():
                        if info["kind"] == "noul":
                            md.append(f"| {name} | {fs['label']} | | | true for {info['share_true']:.2f} | |")
                            continue
                        shares = ", ".join(f"{k} {v:.2f}" for k, v in list(info["shares"].items())[:3])
                        chk = info.get("metadata_check")
                        agree = (f"{chk['agreement']:.2f} on {chk['items_with_label']} items"
                                 if chk and chk["agreement"] is not None else "")
                        md.append(f"| {name} | {fs['label']} | {info['mean_top_p']:.2f} | {info['abstain_rate']:.2f} | "
                                  f"{shares} | {agree} |")
                md.append("")
        ranks, sampled, seed_means = {}, {}, {}
        for m in models:
            stems = [m] + sorted(f[: -len(".json")] for f in os.listdir(bench) if f.startswith(f"{m}.s") and f.endswith(".json")) \
                if os.path.isdir(bench) else []
            runs = []
            for stem in stems:
                js, npz = os.path.join(bench, f"{stem}.json"), os.path.join(bench, f"{stem}.all.npz")
                if os.path.exists(js) and os.path.exists(npz):
                    z = np.load(npz)
                    with open(js) as f:
                        test = np.array(list(json.load(f)["ranks"]["test"].values()), dtype=float)
                    runs.append((z["users"], _ndcg10(z["ranks"]), float(_ndcg10(test).mean())))
            if runs:
                if any(not np.array_equal(u, runs[0][0]) for u, _, _ in runs):
                    raise SystemExit(f"the seeds of {m} were scored on different users in {bench}")
                # per-user NDCG@10 averaged over the training seeds; the spread of the seed means is reported too
                ranks[m] = (runs[0][0], np.mean([g for _, g, _ in runs], axis=0))
                sampled[m] = float(np.mean([t for _, _, t in runs]))
                seed_means[m] = [float(g.mean()) for _, g, _ in runs]
        if ref not in ranks:
            md.append(f"Reference {ref} not found in {bench}; nothing to compare.")
            out.append(res)
            continue
        users = ranks[ref][0]
        for m, (u, _) in ranks.items():
            if not np.array_equal(u, users):
                raise SystemExit(f"{m} was scored on other users than {ref} in {bench}")
        pop = Counter(int(x[0]) for s in ds.seqs for x in s[:-2])
        tp = np.array([pop.get(int(ds.seqs[u][-1][0]), 0) for u in users])
        buckets = np.array([_bucket(int(c), POP_BINS) for c in tp])
        groups = [b[2] for b in POP_BINS if (buckets == b[2]).any()]
        tail = tp < head
        gains = {m: g for m, (_, g) in ranks.items()}
        md += ["### Recommendation, every user", "",
               f"{len(users)} users; {int(tail.sum())} have a target with fewer than {head} training interactions.", "",
               "Models trained with several seeds are averaged per user over the seeds; `seeds` gives each seed's mean.", "",
               "| model | seeds | NDCG@10 sampled test users | NDCG@10 all users | vs " + ref + " | rare targets | vs " + ref
               + " on rare targets | " + " | ".join(f"pop {g}" for g in groups) + " |",
               "|---|---|---|---|---|---|---|" + "---|" * len(groups),
               "| users | | | | | | | " + " | ".join(str(int((buckets == g).sum())) for g in groups) + " |"]
        for m, v in gains.items():
            row = {"seeds": seed_means[m], "sampled_ndcg@10": sampled[m], "ndcg@10": float(v.mean()),
                   "tail_ndcg@10": float(v[tail].mean()),
                   "by_popularity": {g: float(v[buckets == g].mean()) for g in groups}}
            if m != ref:
                r = gains[ref]
                row["vs_ref"] = {"diff": float(v.mean() - r.mean()), "p": M.paired_randomization_large(v, r, n_perm=n_perm),
                                 "ci95": list(M.bootstrap_ci_large(v - r))}
                row["vs_ref_tail"] = {"diff": float(v[tail].mean() - r[tail].mean()),
                                      "p": M.paired_randomization_large(v[tail], r[tail], n_perm=n_perm),
                                      "ci95": list(M.bootstrap_ci_large(v[tail] - r[tail]))}
            res["models"][m] = row
            sd = f" (sd {np.std(row['seeds']):.4f})" if len(row["seeds"]) > 1 else ""
            md.append(f"| {m} | {len(row['seeds'])}{sd} | {row['sampled_ndcg@10']:.4f} | {row['ndcg@10']:.4f} | "
                      f"{_fmt(row['vs_ref']) if 'vs_ref' in row else ''} | {row['tail_ndcg@10']:.4f} | "
                      f"{_fmt(row['vs_ref_tail']) if 'vs_ref_tail' in row else ''} | "
                      + " | ".join(f"{row['by_popularity'][g]:.4f}" for g in groups) + " |")
        out.append(res)
    return {"datasets": out, "markdown": "\n".join(md) + "\n"}
