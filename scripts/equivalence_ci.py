"""95% paired bootstrap intervals (2,000 resamples over users, seed 0) of the relative change of NDCG@10 for the claims
that a system "equals", "ties" or "reaches" a reference (re-review NEW-2, 2026-10-17): a p-value above 0.05 alone does
not show equivalence. python logs/probe/equivalence_ci.py"""
import numpy as np

import json
import os

from claim_tests import listed
from decision4rec.operators.luce import Obs, fit_luce
from decision4rec.data.schema import load_cases
from decision4rec.eval import metrics as M
from decision4rec.experiments.asc import GAMMAS, SPLITS, _ndcg, _tune, _z
from decision4rec.experiments.runner import case_paths, load_config
from decision4rec.experiments.stratify import baseline_ranks


def fused(d, tag, conf, arm):   # as claim_tests.fused, for the run of config <d>_<conf> whose arm file is <arm>
    rd = f"runs/small/{d}/{d}_{conf}_{tag}"
    paths = case_paths(json.load(open(os.path.join(rd, "manifest.json")))["config"])
    cases = {s: {c.cid: c for c in load_cases(paths[s])} for s in SPLITS}
    th = {s: {} for s in SPLITS}
    for s in SPLITS:
        for line in open(os.path.join(rd, f"{arm}.{s}.jsonl"), encoding="utf-8"):
            rec = json.loads(line) if line.strip() else None
            if rec and rec.get("obs") and rec["cid"] in cases[s]:
                th[s][rec["cid"]] = fit_luce([Obs.from_json(o) for o in rec["obs"]], len(cases[s][rec["cid"]].cands)).theta
    cases = {s: {c: cases[s][c] for c in th[s]} for s in SPLITS}
    zr = {s: {c: _z(cases[s][c].rscores) for c in cases[s]} for s in SPLITS}

    def fuse(g, s):
        return {c: _z(th[s][c]) + g * zr[s][c] for c in cases[s]}
    g, _ = _tune(fuse, GAMMAS, cases)
    return dict(zip(sorted(cases["test"]), _ndcg(fuse(g, "test"), cases["test"])))


def base(d, conf, model):   # per-case NDCG@10 of a trained baseline over the whole catalogue (stored ranks, 1 = first)
    cfg = load_config(f"configs/small/{d}_{conf}.yaml")
    cases = load_cases(case_paths(cfg)["test"])
    r = baseline_ranks(cfg, [[c.user, c.target] for c in cases], [model])[model]
    rk = [float(r[str(i)]) for i in range(len(cases))]                       # keyed by the case index, as text
    return {c.cid: (1.0 / np.log2(rk[i] + 1.0) if rk[i] <= 10 else 0.0) for i, c in enumerate(cases)}, cases


def ci(a, b, label):
    a, b = np.asarray(a), np.asarray(b)
    idx = np.random.default_rng(0).integers(0, len(a), (2000, len(a)))
    rel = a[idx].mean(1) / b[idx].mean(1) - 1.0
    lo, hi = np.percentile(rel, [2.5, 97.5])
    print(f"{label}: {a.mean():.4f} vs {b.mean():.4f} ({100 * (a.mean() / b.mean() - 1):+.1f}%, "
          f"95% interval {100 * lo:+.1f} to {100 * hi:+.1f}%, p={M.paired_randomization(a, b):.3f}, users {len(a)})", flush=True)


# self-check: the stored ranks of SASRecF give the NDCG@10 of the order of its own 30 best items (Table 12, 0.0622 on VG)
b, cs = base("vg", "rte", "sasrecf")
lst = np.array([listed(c) for c in cs])
assert abs(np.mean(list(b.values())) - lst.mean()) < 5e-4, (np.mean(list(b.values())), lst.mean())

for d, ref in ():  # done 2026-10-17 (logs/probe/equivalence_ci.log, first run); kept for the record: (("steam", "sasrec"), ("ml1m", "gru4rec"), ("ml1m_plus", "gru4rec"))   # "equal" / "ties" (Full)
    b, cs = base(d, "rtm", ref)
    for tag in ("kev-0.8b-rtm", "kev-4b-rtm"):
        f = fused(d, tag, "rtm", "rtm")
        k = [c.cid for c in cs if c.cid in f]
        ci([f[i] for i in k], [b[i] for i in k], f"{d} Full {tag} fused vs {ref}")
for d in ("vg", "mi", "ind"):                                                       # "as good as" / "reaches" (KG)
    b, cs = base(d, "ev2", "sasrecf")                                               # ev2: the version in the paper
    f = fused(d, "kev-0.8b-ev2", "ev2", "ev")
    k = [c.cid for c in cs if c.cid in f]
    ci([f[i] for i in k], [b[i] for i in k], f"{d} Decision4Rec-KG 0.8B fused vs sasrecf")
    ci([listed(c) for c in cs], [b[c.cid] for c in cs], f"{d} KG candidates in their order vs sasrecf")
