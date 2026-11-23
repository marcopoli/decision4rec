"""Configs of the richer pool (V20): Decision4Rec-Full with the other trained sequence models as agents, on the same ranker
users as the pool of the paper, so its training records stay valid. python scripts/make_rtm2.py"""
import yaml

for d in ("vg", "mi", "steam", "ind", "ml1m", "ml1m_plus"):
    cfg = yaml.safe_load(open(f"configs/small/{d}_rtm.yaml"))
    pc = cfg["pool"]
    cfg["name"] = f"{d}_rtm2"
    cfg["retriever"] = {**cfg["retriever"], "tag": "pool2"}
    cfg["pool"] = {**pc, "extra": ["sasrec", "gru4rec"], "users_of": f"pool_{pc.get('model', 'none')}_u{int(pc.get('users', 15000))}_s{int(pc.get('seed', 11))}"}
    yaml.safe_dump(cfg, open(f"configs/small/{d}_rtm2.yaml", "w"), sort_keys=False)
