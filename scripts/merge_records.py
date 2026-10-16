"""Merge training-record folders of the pool reader into one (seeded shuffle), with an optional repeat per source.

    python scripts/merge_records.py runs/small/rectune_choices_rtmml runs/small/rectune_choices_rtm \
        runs/small/rectune_choices_ml1m_cf:3

Used for the pool reader trained with a dense source domain (MovieLens-1M records, cross-fitted, repeated so that
they are not lost among the records of the three shopping domains). The repeat factor is fixed before training.
"""

import json
import os
import sys

import numpy as np


def main(dst: str, sources: list[str], seed: int = 13) -> None:
    os.makedirs(dst, exist_ok=True)
    rows, info = [], {"version": 3, "merged_from": {}}
    for src in sources:
        path, rep = (src.rsplit(":", 1) + ["1"])[:2] if ":" in src else (src, "1")
        rep = int(rep)
        with open(os.path.join(path, "train.jsonl")) as f, open(os.path.join(path, "train.meta.jsonl")) as g:
            part = list(zip(f.readlines(), g.readlines()))
        rows += part * rep
        info["merged_from"][path] = {"records": len(part), "repeat": rep}
    order = np.random.default_rng(seed).permutation(len(rows))
    with open(os.path.join(dst, "train.jsonl"), "w") as f, open(os.path.join(dst, "train.meta.jsonl"), "w") as g:
        for i in order:
            f.write(rows[i][0])
            g.write(rows[i][1])
    info["train"] = len(rows)
    with open(os.path.join(dst, "export_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
