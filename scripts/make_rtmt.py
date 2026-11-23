"""Configs of the second look (V16): the arm of Decision4Rec-Full plus one adaptive round on the 10 best items of the first
answer. The first request is the one of the run of `rtm`, so it comes from the cache. python scripts/make_rtmt.py"""
import yaml

for d in ("vg", "mi", "steam", "ind", "ml1m", "ml1m_plus"):
    cfg = yaml.safe_load(open(f"configs/small/{d}_rtm.yaml"))
    arm = next(a for a in cfg["arms"] if a["name"] == "rtm")
    cfg.update(name=f"{d}_rtmt", arms=[{**arm, "name": "rtmt", "adaptive": 1, "head": 10}])
    yaml.safe_dump(cfg, open(f"configs/small/{d}_rtmt.yaml", "w"), sort_keys=False)
