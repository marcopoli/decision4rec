"""Do the perspective agents find targets that the rest of the multi-agent pool misses? Development validation users.

    python scripts/perspective_gain.py configs/small/vg_rtm.yaml

Reads runs/small/<domain>/probe/perspectives_valid.json (scripts/perspective_probe.py) and the trained pool of the config.
"""

import json
import os
import sys

import numpy as np

from decision4rec.agents.pool import pool_for
from decision4rec.data.schema import Dataset, load_cases
from decision4rec.experiments.runner import case_paths, load_config


def main(path: str) -> None:
    cfg = load_config(path)
    ds = Dataset.load(cfg["dataset"])
    pool = pool_for(cfg, ds)
    persp = json.load(open(os.path.join(cfg["out"], "probe", os.environ.get("PERSP_FILE", "perspectives_valid.json"))))
    cases = load_cases(case_paths({**cfg, "retriever": {"tag": "ev100"}})["valid"])
    in_pool = in_persp = only_persp = n = 0
    per = {}
    for s in range(0, len(cases), 128):
        part = [c for c in cases[s: s + 128] if str(c.user) in persp]
        rows = pool._sasrec_rows([c.user for c in part], [c.history for c in part])
        for r, c in enumerate(part):
            items, _, _ = pool.features(c.history, pool.store.now(c.user, len(c.history)), rows[r])
            P = persp[str(c.user)]
            union = {j for v in P.values() for j in v}
            a, b = c.target in set(items), c.target in union
            n += 1
            in_pool += a
            in_persp += b
            only_persp += b and not a
            for name, v in P.items():
                per[name] = per.get(name, 0) + (c.target in set(v) and not a)
    print(f"{ds.name:22s} users {n}  target in pool {in_pool / n:.3f}  in perspectives {in_persp / n:.3f}  "
          f"only in perspectives {only_persp / n:.3f} (+{only_persp / max(in_pool, 1) * 100:.1f}% of the pool's hits)  "
          + "  ".join(f"{k} {v / n:.3f}" for k, v in per.items()), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
