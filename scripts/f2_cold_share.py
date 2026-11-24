"""Share of cold targets (no training interaction in the cut data) among the F2 test users (reserved users with a target
after the cut). Run on .34 from the repository root: python logs/probe/f2_cold_share.py
Output of 2026-11-24: video_games 2547 users, 977 cold (38.4%); musical_instruments 2568, 256 (10.0%); steam 4949, 986 (19.9%);
industrial_scientific 2022, 349 (17.3%)."""
import sys
sys.path.insert(0, ".")
from collections import Counter
from decision4rec.data.schema import Dataset, load_cases
for d, n in (("video_games", "vg"), ("musical_instruments", "mi"), ("steam", "steam"), ("industrial_scientific", "ind")):
    ds = Dataset.load(f"data/processed/{d}_cold")
    train = Counter(int(x[0]) for s in ds.seqs for x in s[:-2])
    cs = load_cases(f"runs/small/{n}_cold/cases/pool_trackB_K30_s13_reserved_test10000.jsonl")
    cold = sum(train.get(c.target, 0) == 0 for c in cs)
    print(d, len(cs), "cold targets:", cold, round(cold / len(cs), 3))
