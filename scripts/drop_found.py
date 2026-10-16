"""Training records with the "found by" lines left out of a share of the records (explanations, version 2).

    python scripts/drop_found.py runs/small/rectune_choices_rtm runs/small/rectune_choices_rtmfd [0.25]

In a share p of the records (seeded), every option of the next-item question loses its "found by" statement, as
if it had been rendered with `evidence: stats` instead of `stats+found`; the statements of the store stay. The model
then cannot always lean on which agents found an item, and has to read the behavioral statements too (the
explanation probe found the "found by" lines to be the explanation in about 90% of the cases). Test records are
not changed: at inference the model always reads the "found by" lines.
"""

import json
import os
import re
import shutil
import sys

import numpy as np

INSIDE = re.compile(r"; found by [^\]]*\]")       # "[sold top 10%; ...; found by sequence #3]" -> "[sold top 10%; ...]"
ALONE = re.compile(r" \[found by [^\]]*\]")       # an option whose only statement was "found by"


def strip_found(text: str) -> str:
    return ALONE.sub("", INSIDE.sub("]", text))


def main(src: str, dst: str, p: float = 0.25, seed: int = 7) -> None:
    os.makedirs(dst, exist_ok=True)
    rng = np.random.default_rng(seed)
    stats = {}
    for part in ("train", "heldout"):
        n = dropped = 0
        with open(os.path.join(src, f"{part}.jsonl")) as f, open(os.path.join(dst, f"{part}.jsonl"), "w") as g:
            for line in f:
                rec = json.loads(line)
                if part == "train" and rng.random() < p:
                    q = rec["questions"]["next"]
                    q["criteria"] = {k: strip_found(v) for k, v in q["criteria"].items()}
                    dropped += 1
                g.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        shutil.copy(os.path.join(src, f"{part}.meta.jsonl"), os.path.join(dst, f"{part}.meta.jsonl"))
        stats[part] = {"records": n, "without_found_by": dropped}
    for name in os.listdir(src):
        if name.startswith("heldout_") and name.endswith(".jsonl"):
            shutil.copy(os.path.join(src, name), os.path.join(dst, name))
    info = json.load(open(os.path.join(src, "export_info.json")))
    info.update({"found_dropout": p, "derived_from": src, "found_dropout_counts": stats})
    with open(os.path.join(dst, "export_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.25)
