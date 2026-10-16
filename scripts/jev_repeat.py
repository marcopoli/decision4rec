"""Two passes of the hosted model on the same requests: how much do the choice probabilities move?

    python scripts/jev_repeat.py runs/small/vg/vg_rtm_jev-1.13.0 runs/small/vg/vg_rtm_jev-1.13.0-repeat rtm
"""

import json
import sys

import numpy as np


def load(path):
    return {r["cid"]: r for r in map(json.loads, open(path))}


def main(a: str, b: str, arm: str) -> None:
    for split in ("valid", "test"):
        try:
            x, y = load(f"{a}/{arm}.{split}.jsonl"), load(f"{b}/{arm}.{split}.jsonl")
        except FileNotFoundError:
            continue
        diffs, same_top, n = [], 0, 0
        for cid in sorted(set(x) & set(y)):
            for ox, oy in zip(x[cid]["obs"], y[cid]["obs"]):
                if ox["items"] != oy["items"]:
                    continue
                p, q = np.asarray(ox["probs"]), np.asarray(oy["probs"])
                diffs.append(float(np.abs(p - q).max()))
                same_top += int(np.argmax(p) == np.argmax(q))
                n += 1
        d = np.asarray(diffs)
        print(f"{split}: {n} requests, identical {np.mean(d == 0):.3f}, max |dp| median {np.median(d):.5f} "
              f"p99 {np.quantile(d, 0.99):.5f} max {d.max():.5f}, same top choice {same_top / max(n, 1):.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
