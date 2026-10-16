"""Training records of perspectives version 5: the records of the pool reader plus the question "which kind of
choice will the user make next?", labelled by the rules of the knowledge graph (scripts/export_perspective_labels.py).

    python scripts/add_why_question.py runs/small/rectune_choices_rtm runs/small/rectune_choices_rtmv5

The next-item question and its label are unchanged; the model learns both questions from the same state, and by
question isolation the second one costs no extra pass at inference.
"""

import json
import os
import shutil
import sys

from decision4rec.agents.kgperspectives import why_question


def main(src: str, dst: str) -> None:
    os.makedirs(dst, exist_ok=True)
    q = why_question()
    counts = {}
    for part in ("train", "heldout"):
        rec_path, why_path = os.path.join(src, f"{part}.jsonl"), os.path.join(src, f"{part}.why.jsonl")
        if not os.path.exists(why_path):
            continue
        n = 0
        with open(rec_path) as f, open(why_path) as w, open(os.path.join(dst, f"{part}.jsonl"), "w") as g:
            for line, wl in zip(f, w):
                rec, why = json.loads(line), json.loads(wl)
                rec["questions"]["why"] = {**q, "label": why["label"]}
                g.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        shutil.copy(os.path.join(src, f"{part}.meta.jsonl"), os.path.join(dst, f"{part}.meta.jsonl"))
        counts[part] = n
    info = json.load(open(os.path.join(src, "export_info.json")))
    info.update({"why_question": True, "derived_from": src, "why_counts": counts})
    with open(os.path.join(dst, "export_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(counts))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
