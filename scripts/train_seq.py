"""GRU4Rec checkpoints where only its stored ranks were kept (V20), with the hyperparameters chosen on Video Games.
python scripts/train_seq.py steam ind"""
import json
import sys

import numpy as np

from decision4rec.data.schema import Dataset, load_cases
from decision4rec.experiments.runner import case_paths, load_config
from decision4rec.models import baselines as B

hp = json.load(open("runs/small/vg/benchmark/sasrecf_trackB_K30_s13/gru4rec.json"))["hp"]
for d in sys.argv[1:]:
    cfg = load_config(f"configs/small/{d}_rtm.yaml")
    ds = Dataset.load(cfg["dataset"])
    out = f"runs/small/{d}/baselines/gru4rec/cfg0"
    info = B.fit("gru4rec", ds, out, hp=hp)
    cases = load_cases(case_paths(cfg)["test"])
    r = np.asarray(B.load("gru4rec", out, ds).ranks([c.user for c in cases], [c.history for c in cases], [c.target for c in cases]), dtype=float)
    print(f"{d}: best epoch {info['best_epoch']}, valid {info['best_valid_ndcg10']:.4f}, test users {np.mean(np.where(r <= 10, 1 / np.log2(r + 1), 0)):.4f}", flush=True)
