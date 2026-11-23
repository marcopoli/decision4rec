"""Frozen forecast of the reader's gain in unseen scenarios (docs/18 A3 `predict`, B2; CPU, no reader output read).

Model M4 as fitted on the six development settings by scripts/forecast.py (logs/probe/v34a_forecast.json, "full"):
gain = a + b min(x, k), x = log10(1 + purchases of the target before the request). Scenario forecast: mean over the
validation users with the target among the 30 (primary, as in docs/18 A3); the same over the test users is secondary.
Interval 80%: forecast plus the 10th and 90th percentiles of the leave-one-setting-out errors (observed minus forecast).
Writes the JSON and prints its SHA-256; log the SHA in docs/09 before any reader reads these cases.

    python scripts/forecast_predict.py out.json configs/small/cds_rtm.yaml [more configs]
"""
import glob
import hashlib
import json
import sys

import numpy as np
import yaml

from decision4rec.data.schema import Dataset, load_cases
from thin_probe import covariates

F = json.load(open("logs/probe/v34a_forecast.json"))
(a, b), k = F["full"]["M4"]["coef"], F["full"]["M4"]["k"]
obs = np.array([F["settings"][d]["gain"] for d in F["settings"]])
err = obs - np.array(F["loso"]["M4"]["pred"])
lo, hi = np.quantile(err, [0.1, 0.9])
out = {"model": "M4", "a": a, "b": b, "k": k, "loso_errors": err.tolist(), "interval_q": [float(lo), float(hi)], "scenarios": {}}
for path in sys.argv[2:]:
    cfg = yaml.safe_load(open(path))
    ds = Dataset.load(cfg["dataset"])
    name = cfg.get("name", path)
    row = {"config": path, "dataset": cfg["dataset"]}
    for split in ("valid", "test"):
        f = sorted(glob.glob(f"{cfg['out']}/cases/pool_*_{split}*.jsonl"))
        assert len(f) == 1, f"{path}: expected one {split} case file, found {f}"
        cases = load_cases(f[0])
        cov, _ = covariates(name, cases, ds=ds)
        inn = [c for c in cases if c.target_pos >= 0]
        x = np.log10(np.array([cov[c.cid][4] for c in inn], dtype=float) + 1)
        g = float(np.mean(a + b * np.minimum(x, k)))
        row[split] = {"cases": f[0], "n": len(cases), "in30": len(inn), "share_in30": len(inn) / len(cases),
                      "median_before": float(np.median(10 ** x - 1)), "share_x_below_k": float(np.mean(x < k)),
                      "forecast": g, "interval80": [g + float(lo), g + float(hi)]}
        print(f"{name:12s} {split:5s} users {len(cases)} with target in 30 {len(inn)} | median purchases before "
              f"{row[split]['median_before']:.0f} | forecast {g:+.4f} [{g + lo:+.4f}, {g + hi:+.4f}]", flush=True)
    out["scenarios"][name] = row
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("SHA-256", hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest(), sys.argv[1], flush=True)
