"""Training records of the evidence-reading decision model (Kev-Rec-E, docs/08), in Kev `kev.train` format.

Per user of each source config, the target is the last item of the training part and the history is everything
before it. The store that provides the evidence sees the training parts minus every user's last training item
(holdout 3), so no record reads its own answer, and only a random share of the store's users (the regime: full,
partial, little or no purchase data), so one model learns to read rich and sparse evidence. Candidates come from
the store's own generators; only records whose target is among them are kept, because an added target would be the
odd one out. Every statement family is left out of a record with probability `dropout`.
"""

from __future__ import annotations

import json
import os
import zlib

import numpy as np

from ..data.reserved import reserved_users
from ..data.schema import Case, Dataset
from ..experiments.runner import load_config, make_prompt_builder
from . import store as S
from .store import FAMILIES, store_for

REGIMES = ((1.0, 0.40), (0.2, 0.15), (0.05, 0.15), (0.01, 0.15), (0.0, 0.15))   # share of users visible, weight


def export_evidence_choices(cfg_paths: list[str], out: str, n_per_dataset: int = 15000, n_heldout: int = 500,
                            seed: int = 13, dropout: float = 0.25, regimes=REGIMES, min_hist: int = 2) -> dict:
    os.makedirs(out, exist_ok=True)
    parts: dict[str, list] = {"train": [], "heldout": []}
    info: dict = {"version": 2, "in_pool_only": True, "regime_weights_apply_to": "kept records", "dropout": dropout, "regimes": [list(r) for r in regimes],
                  "datasets": {}}
    fracs = [float(f) for f, _ in regimes]
    weights = np.array([float(w) for _, w in regimes])
    weights = weights / weights.sum()
    for path in cfg_paths:
        cfg = load_config(path)
        ds = Dataset.load(cfg["dataset"])
        K = int(cfg["eval"]["K"])
        builders = {f: make_prompt_builder(cfg, ds, need_kg=False, store=store_for(cfg, ds, holdout=3, frac=f))
                    for f in fracs}
        rng = np.random.default_rng([seed, zlib.crc32(ds.name.encode())])
        reserved = reserved_users(cfg)      # users kept for the confirmation on fresh users (R1)
        users = [u for u in rng.permutation(ds.n_users).tolist()
                 if len(ds.seqs[u]) >= min_hist + 3 and u not in reserved]
        rows = []
        seen = {f: {"read": 0, "kept": 0} for f in fracs}
        f = None
        for u in users:
            if len(rows) >= n_per_dataset + n_heldout:
                break
            if f is None:       # the regime of the next kept record: it holds until a user's target is in the pool,
                f = fracs[int(rng.choice(len(fracs), p=weights))]   # so kept records, not users, follow the weights
            pb = builders[f]
            history = [int(x[0]) for x in ds.seqs[u][:-3]]
            target = int(ds.seqs[u][-3][0])
            pool = pb.store.candidates(history, pb.store.now(u, len(history)), K)
            seen[f]["read"] += 1
            if target not in pool:
                continue
            seen[f]["kept"] += 1
            order = rng.permutation(len(pool)).tolist()
            pb.drop = frozenset(x for x in FAMILIES if rng.random() < dropout)
            case = Case(cid=-1, user=u, history=history, target=target, cands=pool, rscores=[0.0] * len(pool),
                        target_pos=pool.index(target))
            wire = pb.choice(case, order, evidence="stats").to_wire()
            wire["label"] = f"o{order.index(pool.index(target)) + 1:02d}"
            rows.append(({"state": pb.state(case), "questions": {"next": wire}},
                         {"dataset": ds.name, "user": u, "target": target, "frac": f, "drop": sorted(pb.drop)}))
            pb.drop = frozenset()
            f = None
        held, train = rows[:n_heldout], rows[n_heldout:]
        parts["heldout"] += held
        parts["train"] += train
        with open(os.path.join(out, f"heldout_{ds.name}.jsonl"), "w", encoding="utf-8") as fh:
            for rec, _ in held:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        S._STORES.clear()                                               # one domain's stores in memory at a time
        builders = pb = None
        info["datasets"][ds.name] = {"config": path, "train": len(train), "heldout": len(held), "options": K,
                                     "target_in_pool_by_share_of_users": {str(f): v["kept"] / max(v["read"], 1)
                                                                          for f, v in seen.items()}}
    order = np.random.default_rng(seed).permutation(len(parts["train"])).tolist()
    parts["train"] = [parts["train"][i] for i in order]          # domains and regimes mixed within every batch
    for name, part in parts.items():
        with open(os.path.join(out, f"{name}.jsonl"), "w", encoding="utf-8") as fh, \
                open(os.path.join(out, f"{name}.meta.jsonl"), "w", encoding="utf-8") as gh:
            for rec, meta in part:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                gh.write(json.dumps(meta) + "\n")
        info[name] = len(part)
    with open(os.path.join(out, "export_info.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    return info
