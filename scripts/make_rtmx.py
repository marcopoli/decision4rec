"""Configs of the context rung (V28b, docs/09): Decision4Rec-Full whose history lines also carry the user's rating and the
days before the request. Steam gets neither (all ratings are 1.0 and its times are an order only), so its records stay
those of the plain model. Same ranker users and pool as rtm. python scripts/make_rtmx.py"""
import yaml

for d in ("vg", "mi", "steam", "ind", "ml1m", "ml1m_plus"):
    cfg = yaml.safe_load(open(f"configs/small/{d}_rtm.yaml"))
    cfg["name"] = f"{d}_rtmx"
    cfg["prompt"] = {**cfg["prompt"], "context_marks": "" if d == "steam" else "rating+days"}
    yaml.safe_dump(cfg, open(f"configs/small/{d}_rtmx.yaml", "w"), sort_keys=False)
