"""Secondary forecast of the L2 fair test (docs/r1/l2b_addendum.md; CPU, no reader output read).

M5 fitted on the nine read settings (scripts/l2_explore.py, "coef_all9", "k_all9"): gain = a + b min(x, k) + c s, x =
log10(1 + purchases of the target before the request), s = share of the scenario's users with the target in the 30.
Scenario forecast: mean over the validation users with the target among the 30 (primary, as forecast_predict.py), test
users secondary; s from the same split. Interval 80%: the 10th and 90th percentiles of the leave-one-setting-out errors
over the nine. Writes the JSON and prints its SHA-256; log it in docs/09 before any reader reads these cases.

    python scripts/forecast_predict_m5.py logs/probe/l2_explore.json out.json configs/small/baby_rtm.yaml [more]
"""
import glob
import hashlib
import json
import sys

import numpy as np
import yaml

from decision4rec.data.schema import Dataset, load_cases
from thin_probe import covariates

E = json.load(open(sys.argv[1]))
M = E["models"]["M5"]
(a, b, c), k = M["coef_all9"], M["k_all9"]
err = np.array([E["observed"][d] - M["loso_all9"][d] for d in E["observed"]])
lo, hi = np.quantile(err, [0.1, 0.9])
out = {"model": "M5_all9", "a": a, "b": b, "c": c, "k": k, "source": sys.argv[1], "loso_errors": err.tolist(),
       "interval_q": [float(lo), float(hi)], "naive_all9": E["naive"]["all9_mean"], "scenarios": {}}
for path in sys.argv[3:]:
    cfg = yaml.safe_load(open(path))
    ds = Dataset.load(cfg["dataset"])
    name = cfg.get("name", path)
    row = {"config": path, "dataset": cfg["dataset"]}
    for split in ("valid", "test"):
        f = sorted(glob.glob(f"{cfg['out']}/cases/pool_*_{split}*.jsonl"))
        assert len(f) == 1, f"{path}: expected one {split} case file, found {f}"
        cases = load_cases(f[0])
        cov, _ = covariates(name, cases, ds=ds)
        inn = [x for x in cases if x.target_pos >= 0]
        s = len(inn) / len(cases)
        x = np.log10(np.array([cov[q.cid][4] for q in inn], dtype=float) + 1)
        g = float(np.mean(a + b * np.minimum(x, k) + c * s))
        row[split] = {"cases": f[0], "n": len(cases), "in30": len(inn), "share_in30": s, "forecast": g,
                      "interval80": [g + float(lo), g + float(hi)]}
        print(f"{name:12s} {split:5s} in 30 {s:.2f} | M5 forecast {g:+.4f} [{g + lo:+.4f}, {g + hi:+.4f}]", flush=True)
    out["scenarios"][name] = row
json.dump(out, open(sys.argv[2], "w"), indent=1)
print("SHA-256", hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest(), sys.argv[2], flush=True)
