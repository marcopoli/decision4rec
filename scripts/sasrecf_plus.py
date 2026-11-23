"""Control of the R1 plan: a sequence model trained also on the second-to-last items of the users that train the ranker
of the pool and the decision models, so that every system learns from the same targets. Same hyperparameters as the
chosen model. Scored on the development test users, paired with the model. SASRecF by default, or the model named first:
python scripts/sasrecf_plus.py vg mi steam ind
python scripts/sasrecf_plus.py sasrec vg mi steam ind"""
import json
import sys

import numpy as np

from decision4rec.agents.pool import _trained, pool_for
from decision4rec.data.schema import Dataset, load_cases
from decision4rec.eval import metrics as M
from decision4rec.experiments.runner import case_paths, load_config
from decision4rec.experiments.stage1 import chosen_dir
from decision4rec.models import baselines as B

MODEL = sys.argv[1] if sys.argv[1] in ("sasrec", "gru4rec") else "sasrecf"
for d in sys.argv[1:] if MODEL == "sasrecf" else sys.argv[2:]:
    cfg = load_config(f"configs/small/{d}_rtm.yaml")
    ds = Dataset.load(cfg["dataset"])
    ref = load_config(cfg["pool"]["ref"])
    base_dir = chosen_dir(ref, "sasrecf") if MODEL == "sasrecf" else _trained(cfg["out"], MODEL)
    hp = json.load(open(f"{base_dir}/info.json"))["hp"]
    users = {int(u) for u in pool_for(cfg, ds).train_users}
    with open("runs/small/rectune_choices_rtm/train.meta.jsonl") as f:
        users |= {m["user"] for m in map(json.loads, f) if m["dataset"] == ds.name}
    out = f"runs/small/{d}/baselines/{MODEL}_plus"
    info = B.fit(MODEL, ds, out, hp=hp, extra_targets=users)
    print(f"{d}: {len(users)} users with extra targets, best epoch {info['best_epoch']}", flush=True)
    cases = load_cases(case_paths(cfg)["test"])
    args = [c.user for c in cases], [c.history for c in cases], [c.target for c in cases]

    def ndcg(path):
        r = np.asarray(B.load(MODEL, path, ds).ranks(*args), dtype=float)
        return np.where(r <= 10, 1.0 / np.log2(r + 1.0), 0.0)
    a, b = ndcg(out), ndcg(base_dir)
    print(f"{d}: {MODEL}+ {a.mean():.4f} vs {MODEL} {b.mean():.4f} ({100 * (a.mean() / b.mean() - 1):+.1f}%, p={M.paired_randomization(a, b):.3f})", flush=True)
