"""Configs of the temporal check (R1 plan, section 7): Decision4Rec-Full on the datasets cut at one global time
(`prepare-cold`, 0.9 quantile). The dataset keeps every interaction before the cut and, as test target, the first one
after it, so no statistic and no first stage learns from the test period. The users reserved for R1 are carried over,
so development does not touch them. python scripts/make_cold_rtm.py"""
import json
import os

import yaml

from decision4rec.data.reserved import reserved_users
from decision4rec.data.schema import Dataset

DOMAINS = {"vg": "video_games", "mi": "musical_instruments", "steam": "steam", "ind": "industrial_scientific"}

for d, name in DOMAINS.items():
    src = Dataset.load(f"data/processed/{name}")
    dst = f"data/processed/{name}_cold"
    info = Dataset.load(dst).extra["cold_protocol"]
    kept = [u for u, s in enumerate(src.seqs) if sum(x[1] < info["cutoff"] for x in s) >= info["min_pre"]]
    idx = {u: i for i, u in enumerate(kept)}
    res = sorted(idx[u] for u in reserved_users(f"data/processed/{name}") if u in idx)
    with open(os.path.join(dst, "reserved_users.json"), "w") as f:
        json.dump({"users": res, "source": f"data/processed/{name}/reserved_users.json"}, f)
    cfg = yaml.safe_load(open(f"configs/small/{d}_rtm.yaml"))
    cfg.update(name=f"{d}_cold_rtm", dataset=dst, out=f"runs/small/{d}_cold")
    cfg["pool"] = {**cfg["pool"], "ref": f"configs/small/{d}_cold.yaml"}
    cfg["eval"]["target"] = "post"
    cfg["item_text"] = {**cfg["item_text"], "cache": f"runs/small/{d}_cold/retrieval"}
    yaml.safe_dump(cfg, open(f"configs/small/{d}_cold_rtm.yaml", "w"), sort_keys=False)
    print(d, len(src.seqs), len(kept), len(res))
