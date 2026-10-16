"""A trained or training-free baseline used as the first stage of a pipeline (retriever kind `baseline`).

The candidates of a trained decision model should come from the strongest available model, so that the
decision model only has to reorder what that model already finds. The checkpoint is the one chosen in model
selection for a reference config (`ref`), read from that config's benchmark folder, so no model is retrained.

    retriever: {kind: baseline, model: sasrecf, ref: configs/small/vg_semantic.yaml, tag: sasrecf}
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch

from ..data.schema import Dataset


class BaselineRetriever:
    """Top-K of a baseline scorer over the whole catalogue, history items excluded."""

    def __init__(self, scorer, batch: int = 256):
        self.scorer, self.batch = scorer, batch

    def topk(self, histories: list[list[int]], k: int, users: Optional[list[int]] = None):
        idx, val = [], []
        users = users if users is not None else [0] * len(histories)
        for s in range(0, len(histories), self.batch):
            hs = histories[s: s + self.batch]
            sc = self.scorer.scores(users[s: s + self.batch], hs).clone()
            rows = torch.tensor([r for r, h in enumerate(hs) for _ in h], dtype=torch.long, device=sc.device)
            cols = torch.tensor([i for h in hs for i in h], dtype=torch.long, device=sc.device)
            if len(rows):
                sc[rows, cols] = float("-inf")
            v, i = torch.topk(sc, min(k, sc.shape[1]), dim=1)
            idx.append(i.cpu().numpy())
            val.append(v.float().cpu().numpy())
        return np.concatenate(idx), np.concatenate(val)


def chosen_dir(ref_cfg: dict, model: str) -> str:
    """Checkpoint folder of `model` chosen in model selection for the reference config."""
    from .benchmark import _bench_dir
    path = os.path.join(_bench_dir(ref_cfg), f"{model}.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} does not exist: run the baselines of {ref_cfg.get('name')} first")
    with open(path) as f:
        info = json.load(f)
    if "dir" in info:
        return info["dir"]
    # older benchmark files have no "dir": take the grid folder whose hyperparameters match the chosen ones
    from ..data.schema import Dataset
    from .benchmark import baseline_root
    root = os.path.join(baseline_root(ref_cfg, Dataset.load(ref_cfg["dataset"]), model), model)
    for c in sorted(os.listdir(root)):
        p = os.path.join(root, c, "info.json")
        if os.path.exists(p) and json.load(open(p)).get("hp") == info.get("hp"):
            return os.path.join(root, c)
    raise SystemExit(f"{path}: no checkpoint folder in {root} matches the chosen hyperparameters {info.get('hp')}")


def build_baseline_retriever(rcfg: dict, ds: Dataset) -> BaselineRetriever:
    from ..models import baselines as B
    from .benchmark import prepare_dataset
    from .runner import load_config
    model = rcfg["model"]
    if model in B.COLLABORATIVE:
        raise SystemExit(f"{model} scores by user id; a first stage must score from the history alone")
    ref = load_config(rcfg["ref"])
    prepare_dataset(ref, ds, [model])
    return BaselineRetriever(B.load(model, chosen_dir(ref, model), ds, rcfg.get("device")))
