"""Perspective agents on development validation users: what each finds, and the lists for the pool tests.

    python scripts/perspective_probe.py configs/small/vg_ev2.yaml [n_users] [model]

Writes runs/small/<domain>/probe/perspectives_valid.json ({user: {perspective: [items]}}) and prints recall@20 of
every perspective and of their union. Example users come from the training part of users of no evaluation case.
"""

import json
import os
import sys

import numpy as np

from decision4rec.data.schema import Dataset, load_cases
from decision4rec.experiments.runner import case_paths, load_config
from decision4rec.retrieval.perspectives import PERSPECTIVES, PerspectiveAgents
from decision4rec.retrieval.slm import HFWriter, build_embedder


def main(path: str, n: int, model: str) -> None:
    cfg = load_config(path)
    ds = Dataset.load(cfg["dataset"])
    paths = case_paths({**cfg, "retriever": {"tag": "ev100"}})
    cases = load_cases(paths["valid"])[:n]
    eval_users = {c.user for p in (paths["valid"], paths["test"]) for c in load_cases(p)}
    rng = np.random.default_rng(3)
    example_users = [u for u in rng.permutation(ds.n_users).tolist() if u not in eval_users]
    it = cfg["item_text"]
    matrix = np.load(os.path.join(it["cache"], f"emb_{it['embedder']['model'].split('/')[-1]}_content_{ds.n_items}.npy"))
    out_dir = os.path.join(cfg["out"], "probe")
    os.makedirs(out_dir, exist_ok=True)
    n_ideas = int(os.environ.get("N_IDEAS", 1))
    tag = "" if n_ideas == 1 else f"_v{n_ideas}"
    writer = HFWriter(model, max_new_tokens=32 if n_ideas == 1 else 120, batch_size=16,
                      cache=os.path.join(out_dir, f"perspectives_writer{tag}.sqlite"))
    agents = PerspectiveAgents(ds, writer, build_embedder(it["embedder"]), matrix, example_users, n_ideas=n_ideas)
    hist = [c.history for c in cases]
    lists = agents.propose(hist, 20)
    for name in PERSPECTIVES:
        r = np.mean([c.target in set(lists[name][u]) for u, c in enumerate(cases)])
        print(f"{ds.name:22s} perspective {name:13s} recall@20 {r:.3f}", flush=True)
    union = np.mean([c.target in {j for name in PERSPECTIVES for j in lists[name][u]} for u, c in enumerate(cases)])
    print(f"{ds.name:22s} union of 5 x 20             recall@100 {union:.3f}", flush=True)
    with open(os.path.join(out_dir, f"perspectives_valid{tag}.json"), "w") as f:
        json.dump({str(c.user): {name: lists[name][u] for name in PERSPECTIVES} for u, c in enumerate(cases)}, f)
    print("example prompt:\n" + __import__("decision4rec.retrieval.perspectives", fromlist=["prompt"]).prompt(
        ds, "complement", cases[0].history, agents.examples, n_ideas=n_ideas), flush=True)
    print("its answer:", writer.write([__import__("decision4rec.retrieval.perspectives", fromlist=["prompt"]).prompt(
        ds, "complement", cases[0].history, agents.examples, n_ideas=n_ideas)])[0], flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1000,
         sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen2.5-1.5B-Instruct")
