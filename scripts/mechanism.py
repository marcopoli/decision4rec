"""A1b mechanism (V41, docs/18): what the reader loses when titles are masked or statements dropped, by evidence group.

Per test user with the target among the options: log-likelihood of the target under the reader alone (Luce fit of its
choices, softmax of theta), normal arm `rtm` minus perturbed arm (`rtm_masked`: titles masked, statements kept;
`rtm_text`: statements and found-by dropped, titles kept). Evidence group of the target from the purchases before the
request (scripts/thin_probe.covariates): rare < 20, middle 20 to 99, popular >= 100.
Prediction (fixed in docs/18 before this script): masking hurts rare targets more; dropping statements hurts popular more.
Test: contrast rare minus popular of the mean loss; permutation of the group labels inside each setting, pooled over the
settings by summing the per-setting contrasts weighted by users.

    python scripts/mechanism.py out.json CONF TAG [domain ...]     # e.g. rtmctl kev-0.8b-rtm  |  rtmmech kev-9b-rtm
"""
import json
import os
import sys

import numpy as np

from fusion_table import FOUR
from decision4rec.data.schema import Dataset, load_cases
from decision4rec.experiments.runner import case_paths
from decision4rec.operators.luce import Obs, fit_luce
from thin_probe import covariates

OUT, CONF, TAG = sys.argv[1:4]
DOMS = sys.argv[4:] or list(FOUR)
PERT = ("rtm_masked", "rtm_text")
GROUPS = ((0, 20, "rare"), (20, 100, "middle"), (100, np.inf, "popular"))


def ll(rd, arm, cases):
    p = os.path.join(rd, f"{arm}.test.jsonl")
    if not os.path.exists(p):
        return None
    out = {}
    for line in open(p, encoding="utf-8"):
        rec = json.loads(line) if line.strip() else None
        if rec and rec.get("obs") and rec["cid"] in cases and cases[rec["cid"]].target_pos >= 0:
            th = fit_luce([Obs.from_json(o) for o in rec["obs"]], len(cases[rec["cid"]].cands)).theta
            th = th - th.max()
            out[rec["cid"]] = float(th[cases[rec["cid"]].target_pos] - np.log(np.exp(th).sum()))
    return out


def contrast(loss, grp):
    return loss[grp == 0].mean() - loss[grp == 2].mean()


res, perm = {}, {a: [] for a in PERT}
rng = np.random.default_rng(0)
for d in DOMS:
    rd = f"runs/small/{d}/{d}_{CONF}_{TAG}"
    cfg = json.load(open(os.path.join(rd, "manifest.json")))["config"]
    cases = {c.cid: c for c in load_cases(case_paths(cfg)["test"])}
    base = ll(rd, "rtm", cases)
    cov, _ = covariates(d, list(cases.values()), ds=Dataset.load(cfg["dataset"]))
    res[d] = {}
    for a in PERT:
        pa = ll(rd, a, cases)
        if not pa:
            continue
        cids = sorted(set(base) & set(pa))
        loss = np.array([base[c] - pa[c] for c in cids])
        before = np.array([cov[c][4] for c in cids])
        grp = np.select([before < 20, before < 100], [0, 1], 2)
        obs = contrast(loss, grp)
        null = np.array([contrast(loss, rng.permutation(grp)) for _ in range(10000)])
        row = {"n": len(cids), "contrast_rare_minus_popular": float(obs), "null": null, "loss_all": float(loss.mean())}
        for g, (_, _, name) in enumerate(GROUPS):
            m = grp == g
            row[name] = {"n": int(m.sum()), "loss": float(loss[m].mean()) if m.any() else None,
                         "se": float(loss[m].std(ddof=1) / np.sqrt(m.sum())) if m.sum() > 1 else None}
        res[d][a] = row
        print(f"{d:6s} {a:10s} n {len(cids):4d} | loss rare {row['rare']['loss']:+.3f} (n {row['rare']['n']}) middle "
              f"{row['middle']['loss'] if row['middle']['loss'] is not None else float('nan'):+.3f} popular "
              f"{row['popular']['loss'] if row['popular']['loss'] is not None else float('nan'):+.3f} (n {row['popular']['n']}) "
              f"| rare - popular {obs:+.3f}", flush=True)

summary = {}
for a in PERT:
    rows = [res[d][a] for d in res if a in res[d]]
    if not rows:
        continue
    w = np.array([r["n"] for r in rows], dtype=float)
    obs = float(np.sum(w * [r["contrast_rare_minus_popular"] for r in rows]) / w.sum())
    null = np.sum([w[i] * rows[i]["null"] for i in range(len(rows))], 0) / w.sum()
    # one-sided in the predicted direction: masking rare > popular (obs > 0), dropping statements popular > rare (obs < 0)
    p = float((np.sum(null >= obs) + 1) / (len(null) + 1)) if a == "rtm_masked" else float((np.sum(null <= obs) + 1) / (len(null) + 1))
    summary[a] = {"pooled_contrast": obs, "p_one_sided": p, "settings": len(rows)}
    print(f"pooled {a:10s} rare - popular {obs:+.3f}, one-sided p {p:.4f} ({len(rows)} settings)", flush=True)
for d in res:
    for a in res[d]:
        res[d][a].pop("null")
json.dump({"conf": CONF, "tag": TAG, "settings": res, "pooled": summary}, open(OUT, "w"), indent=1)
