"""Every stored variant with the stable fusion (decision4rec.operators.fusion): weights fitted on the validation choices of the
other scenarios. Prints each setting with the grid fusion of arXiv v1 next to it, and writes all numbers to a JSON file.
python scripts/fusion_table.py out.json [group ...]"""
import glob
import json
import os
import sys

import numpy as np

from decision4rec.data.schema import load_cases
from decision4rec.eval import metrics as M
from decision4rec.experiments.asc import GAMMAS, SPLITS, _ndcg, _tune, _z
from decision4rec.experiments.runner import case_paths
from decision4rec.operators.fusion import choice_data, leave_one_domain_out, shrunk_weights
from decision4rec.operators.luce import Obs, fit_luce

FOUR = ("vg", "mi", "steam", "ind")
SIX = FOUR + ("ml1m", "ml1m_plus")
FAMILY = {"ml1m": "ml1m", "ml1m_plus": "ml1m"}
STRENGTH = float(os.environ.get("FUSION_STRENGTH", "30"))       # 0: the weights of the other scenarios alone
GROUPS = {
    "sizes": (SIX, None, ["rtm:kev-0.8b-rtm", "rtm:kev-4b-rtm", "rtm:kev-9b-rtm", "rtm:kev-27b-rtm"]),
    "sizes8k": (SIX, None, ["rtm:kev-0.8b-rtm8k", "rtm:kev-4b-rtm8k", "rtm:kev-9b-rtm8k", "rtm:kev-27b-rtm8k"]),
    "seeds4b": (FOUR, "rtm:kev-4b-rtm", ["rtm:kev-4b-rtm_s1", "rtm:kev-4b-rtm_s2"]),
    "seeds9b": (SIX, "rtm:kev-9b-rtm", ["rtm:kev-9b-rtm_s1", "rtm:kev-9b-rtm_s2"]),
    "seeds27b": (SIX, "rtm:kev-27b-rtm", ["rtm:kev-27b-rtm_s1"]),                   # B4, Spark queue_s6
    "released": (SIX, None, ["rtm:kev-0.8b", "rtm:kev-4b", "rtm:kev-9b", "rtm:kev-27b", "rtm:d3", "rtm:jev-1.13.0"]),
    "list": (SIX, None, ["rte:kev-0.8b-rte", "rte:kev-4b-rte", "rte:kev-9b-rte", "rte:kev-27b-rte"]),
    "kg": (SIX, None, ["ev2:kev-0.8b-ev2", "ev2:kev-4b-ev2", "ev2:kev-9b-ev2", "ev2:kev-27b-ev2"]),
    "kg8k": (SIX, None, ["ev2:kev-0.8b-ev2k8", "ev2:kev-4b-ev2k8", "ev2:kev-9b-ev2k8", "ev2:kev-27b-ev2k8"]),
    "pool2": (SIX, "rtm:kev-9b-rtm", ["rtm2:kev-9b-rtm", "rtm2:kev-9b-rtm2"]),
    "second_look": (SIX, "rtm:kev-9b-rtm", ["rtmt:kev-9b-rtm"]),
    "epochs": (SIX, "rtm:kev-9b-rtm", ["rtm:kev-9b-rtm_e2"]),
    "residual": (SIX, "rtm:kev-9b-rtm", ["rtm:kev-9b-rtmres", "rtm:kev-9b-rtmres3"]),
    "v24": (SIX, "rtm:kev-9b-rtm", ["rtm:kev-9b-rtmv24", "rtm:kev-9b-rtmv24d"]),
    "ctx08": (SIX, "rtm:kev-0.8b-rtm", ["rtmx:kev-0.8b-rtmx"]),                      # V28b
    "ctx08s": (FOUR, "rtm:kev-0.8b-rtm", ["rtm:kev-0.8b-rtm_s1", "rtm:kev-0.8b-rtm_s2", "rtmx:kev-0.8b-rtmx"]),
    "fifty08": (SIX, "rtm:kev-0.8b-rtm", ["rtm50:kev-0.8b-rtm"]),                    # R18, no retraining
    "fifty4b": (SIX, "rtm:kev-4b-rtm", ["rtm50:kev-4b-rtm"]),
    "cold": (("vg_cold", "mi_cold", "steam_cold", "ind_cold"), "rtm:kev-9b-rtm", ["rtm:kev-9b-rtm_cold"]),
}


def load(d, conf, tag, min_valid=200):
    """min_valid=0: runs scored with weights fitted elsewhere may have no validation records (R1+ F2 time cut)."""
    rd = f"runs/small/{d}/{d}_{conf}_{tag}"
    if not os.path.exists(os.path.join(rd, "manifest.json")):
        return None
    sp = "valid" if min_valid else "test"
    arms = [os.path.basename(p)[:-len(f".{sp}.jsonl")] for p in glob.glob(rd + f"/*.{sp}.jsonl")]
    arms = [a for a in arms if a != "retriever" and not a.endswith("_prior")]
    arm = conf if conf in arms else ("ev" if "ev" in arms else ("rtm" if "rtm" in arms else (arms or [None])[0]))
    if arm is None:
        return None
    paths = case_paths(json.load(open(os.path.join(rd, "manifest.json")))["config"])
    cases = {s: {c.cid: c for c in load_cases(paths[s])} for s in SPLITS}
    th = {s: {} for s in SPLITS}
    for s in SPLITS:
        p = os.path.join(rd, f"{arm}.{s}.jsonl")
        if not os.path.exists(p):
            if s == "valid" and not min_valid:
                continue
            return None
        for line in open(p, encoding="utf-8"):
            rec = json.loads(line) if line.strip() else None
            if rec and rec.get("obs") and rec["cid"] in cases[s]:
                th[s][rec["cid"]] = fit_luce([Obs.from_json(o) for o in rec["obs"]], len(cases[s][rec["cid"]].cands)).theta
    if len(th["test"]) < 500 or len(th["valid"]) < min_valid:
        return None
    cases = {s: {c: cases[s][c] for c in th[s]} for s in SPLITS}
    return cases, {s: {c: np.stack([_z(th[s][c]), _z(cases[s][c].rscores)], 1) for c in cases[s]} for s in SPLITS}


def score(spec, domains):
    conf, tag = spec.split(":")
    D = {d: load(d, conf, tag) for d in domains}
    D = {d: v for d, v in D.items() if v is not None}
    if len({FAMILY.get(d, d) for d in D}) < 2:
        return {}
    data = {d: choice_data(D[d][0]["valid"], D[d][1]["valid"]) for d in D}
    w = shrunk_weights(data, FAMILY, STRENGTH) if STRENGTH else leave_one_domain_out(data, FAMILY)
    out = {}
    for d, (cases, Z) in D.items():
        T = cases["test"]

        def fuse(g, s):
            return {c: Z[s][c][:, 0] + g * Z[s][c][:, 1] for c in cases[s]}
        g, _ = _tune(fuse, GAMMAS, cases)
        out[d] = dict(cids=sorted(T), lst=_ndcg({c: Z["test"][c][:, 1] for c in T}, T), alone=_ndcg({c: Z["test"][c][:, 0] for c in T}, T),
                      grid=_ndcg(fuse(g, "test"), T), fused=_ndcg({c: Z["test"][c] @ w[d] for c in T}, T), g=g, ratio=float(w[d][1] / max(w[d][0], 1e-9)))
    return out


if __name__ == "__main__":
    out, table = sys.argv[1], {}
    for name, (ds, ref, specs) in GROUPS.items():
        if sys.argv[2:] and name not in sys.argv[2:]:
            continue
        print(f"== {name}", flush=True)
        R = score(ref, ds) if ref else {}
        for sp in ([ref] if ref else []) + specs:
            r = R if sp == ref else score(sp, ds)
            line = f"{sp:26s}"
            table[sp] = {}
            for d, v in r.items():
                f, l = v["fused"], v["lst"]
                table[sp][d] = {"list": l.mean(), "alone": v["alone"].mean(), "grid": v["grid"].mean(), "g": v["g"], "fused": f.mean(), "ratio": v["ratio"],
                                "p_list": M.paired_randomization(f, l)}
                line += f" | {d} {f.mean():.4f} {100 * (f.mean() / l.mean() - 1):+.1f}% p={table[sp][d]['p_list']:.3f} (grid {100 * (v['grid'].mean() / l.mean() - 1):+.1f})"
                if sp != ref and d in R:
                    k = sorted(set(v["cids"]) & set(R[d]["cids"]))
                    a, b = f[np.searchsorted(v["cids"], k)], R[d]["fused"][np.searchsorted(R[d]["cids"], k)]
                    pr = M.paired_randomization(a, b)
                    table[sp][d]["vs_ref"] = {"ref": ref, "n": len(k), "rel": 100 * (a.mean() / b.mean() - 1), "p": pr}
                    line += f" [vs ref {100 * (a.mean() / b.mean() - 1):+.1f}% p={pr:.3f}]"
            print(line, flush=True)
        json.dump(table, open(out, "w"), indent=1)
