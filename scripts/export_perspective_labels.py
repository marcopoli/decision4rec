"""Perspective labels for the training records of the pool reader (perspectives, version 5).

    python scripts/export_perspective_labels.py runs/small/rectune_choices_rtm configs/small/vg_rtm.yaml ...

For every record of `<records>/{train,heldout}.meta.jsonl` (user and target, aligned line by line with the records),
the rules of the knowledge graph (decision4rec/agents/kgperspectives.py) say which perspectives explain the step from the
history (all items before the target) to the real target, read from the store of the config, which leaves out the
last two items of every user (the target is the second to last one, so no statistic contains it). Writes
`<records>/{train,heldout}.why.jsonl` with the five flags and one label, and `why_info.json` with the label shares.
No model is called.
"""

import json
import os
import sys
from collections import Counter

from decision4rec.agents.kgperspectives import NAMES, KGPerspectives, why_label
from decision4rec.data.schema import Dataset
from decision4rec.evidence.store import store_for
from decision4rec.experiments.runner import load_config


def main(records: str, cfg_paths: list[str]) -> None:
    by_name = {}
    for path in cfg_paths:
        cfg = load_config(path)
        ds = Dataset.load(cfg["dataset"])
        store = store_for(cfg, ds)
        by_name[ds.name] = (ds, KGPerspectives(ds, store, kind_depth=(cfg.get("pool") or {}).get("kind_depth")))
        print(f"[why] {ds.name}: store and perspectives ready", flush=True)
    info = {"rules": "decision4rec/agents/kgperspectives.py explain()", "parts": {}}
    for part in ("train", "heldout"):
        meta = os.path.join(records, f"{part}.meta.jsonl")
        if not os.path.exists(meta):
            continue
        shares, n = Counter(), Counter()
        with open(meta) as f, open(os.path.join(records, f"{part}.why.jsonl"), "w") as g:
            for line in f:
                m = json.loads(line)
                ds, persp = by_name[m["dataset"]]
                seq = [int(x[0]) for x in ds.seqs[m["user"]]]
                assert seq[-2] == m["target"], f"target of user {m['user']} is not the second to last item"
                hits = persp.explain(seq[:-2], m["target"])
                label = why_label(hits)
                g.write(json.dumps({"dataset": m["dataset"], "user": m["user"], "hits": [int(h) for h in hits],
                                    "label": label}) + "\n")
                shares[(m["dataset"], label)] += 1
                n[m["dataset"]] += 1
        info["parts"][part] = {d: {lab: round(shares[(d, lab)] / n[d], 4) for lab in NAMES + ("other",)}
                               for d in sorted(n)}
        info["parts"][part]["records"] = dict(n)
    with open(os.path.join(records, "why_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
