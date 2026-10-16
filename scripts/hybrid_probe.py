"""Can store evidence and a trained model (SASRecF) together beat the model alone? Development validation users.

    python scripts/hybrid_probe.py configs/small/vg_ev2.yaml configs/small/vg_semantic.yaml [n_users]

Validation cases of the evidence config (history s[:-2], target s[-2]; the store and SASRecF leave out the last two
items of every user, SASRecF chose its checkpoint on s[-2] of some users). For each mix w of z-scored scores,
z(SASRecF) + w * z(evidence fused score) over the whole catalogue, reports recall@30 of the mix and NDCG@10, HR@10 over
the catalogue. w = 0 is SASRecF alone. No decision model: this is the first stage and the floor of a reranker.
"""

import sys

import numpy as np
import torch

from decision4rec.data.schema import Dataset, load_cases
from decision4rec.evidence.store import store_for
from decision4rec.experiments.benchmark import prepare_dataset
from decision4rec.experiments.runner import case_paths, load_config
from decision4rec.experiments.stage1 import chosen_dir
from decision4rec.models import baselines as B


def z(v: np.ndarray) -> np.ndarray:
    f = np.isfinite(v)
    out = np.full_like(v, -np.inf)
    out[f] = (v[f] - v[f].mean()) / (v[f].std() + 1e-9)
    return out


def main(ev_path: str, ref_path: str, n: int, model: str = "sasrecf") -> None:
    cfg, ref = load_config(ev_path), load_config(ref_path)
    ds = Dataset.load(cfg["dataset"])
    store = store_for(cfg, ds)
    prepare_dataset(ref, ds, [model])
    scorer = B.load(model, chosen_dir(ref, model), ds, "cpu")
    cases = load_cases(case_paths({**cfg, "retriever": {"tag": "ev100"}})["valid"])[:n]   # same users and targets
    ws = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 2.0, 1e6)
    ranks = {w: [] for w in ws}
    inpool = {w: [] for w in ws}
    for s in range(0, len(cases), 128):
        part = cases[s: s + 128]
        with torch.no_grad():
            m = scorer.scores([c.user for c in part], [c.history for c in part]).cpu().numpy().astype(float)
        for c, row in zip(part, m):
            row[np.asarray(c.history, dtype=int)] = -np.inf
            e = store.fused_scores(c.history, store.now(c.user, len(c.history)))
            zm, ze = z(row), z(e)
            for w in ws:
                sc = ze if w >= 1e6 else zm + w * ze
                t = sc[c.target]
                r = int((sc > t).sum()) + 1
                ranks[w].append(r)
                inpool[w].append(r <= 30)
    for w in ws:
        r = np.array(ranks[w])
        name = "evidence alone" if w >= 1e6 else ("SASRecF alone" if w == 0 else f"SASRecF + {w} evidence")
        print(f"{ds.name:22s} {name:26s} recall@30 {np.mean(inpool[w]):.3f}  HR@10 {np.mean(r <= 10):.3f}  "
              f"NDCG@10 {np.mean(np.where(r <= 10, 1 / np.log2(r + 1), 0)):.4f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1000)
