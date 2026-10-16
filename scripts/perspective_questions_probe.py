"""Perspective agents as questions to the decision model: in the request of the full architecture, five more choice
questions over the same 30 options ask which candidate is a complement, a continuation, a replenishment, a loyal
choice or an exploration. By question isolation they cost one pass. Do their answers help the order?

    python scripts/perspective_questions_probe.py configs/small/vg_rtm.yaml configs/backends/kev-0.8b-rtm.yaml [n]

Development validation users (the users on which fusion weights are fitted), 5-fold cross-validation over users:
base = pool order + log p(next) (the fused score of the paper, weights fitted per fold), against base + the five
log p(perspective). Prints NDCG@10 inside the 30 options (a target outside them counts as 0) and a paired test.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from decision4rec.data.schema import Dataset
from decision4rec.decision.base import build_backend
from decision4rec.decision.types import Choice
from decision4rec.experiments.runner import get_cases, load_config, make_prompt_builder
from decision4rec.prompts import INSTR_CHOICE
from decision4rec.retrieval.perspectives import PERSPECTIVES


def ndcg(s, y):
    if y < 0:
        return 0.0
    r = int((s > s[y]).sum()) + 1
    return 1.0 / np.log2(r + 1) if r <= 10 else 0.0


def fit(rows, cols, steps=300):
    w = torch.zeros(len(cols), requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.05, weight_decay=1e-3)
    data = [(torch.tensor(F[:, cols]), y) for F, y in rows if y >= 0]
    for _ in range(steps):
        loss = sum(-torch.log_softmax(F @ w, 0)[y] for F, y in data) / len(data)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach().numpy()


def main(cfg_path: str, backend_path: str, n: int) -> None:
    cfg = load_config(cfg_path, backend_path)
    cfg["backend"]["cache"] = os.path.join(cfg["out"], "cache", f"perspq_{os.path.basename(backend_path)}.sqlite")
    ds = Dataset.load(cfg["dataset"])
    cases = get_cases(cfg, ds)["valid"][:n]
    pb = make_prompt_builder(cfg, ds)
    backend = build_backend(cfg["backend"])
    names = list(PERSPECTIVES)

    def ask(case):
        order = list(range(len(case.cands)))
        nxt = pb.choice(case, order, evidence="stats+found")
        qs = {"next": nxt}
        for p in names:
            qs[p] = Choice(instructions=f"Based on the user's interaction history, which candidate item is {PERSPECTIVES[p]}?",
                           options=nxt.options)
        d = backend.decide(pb.state(case), qs)
        keys = nxt.keys()
        F = np.zeros((len(keys), 2 + len(names)), dtype=np.float32)
        F[:, 0] = -np.log1p(np.arange(len(keys)))                     # pool order
        for c, q in enumerate(["next"] + names):
            pr = d.answers[q].probs
            F[:, c + 1] = np.log(np.array([max(pr.get(k, 0.0), 1e-6) for k in keys]))
        F = (F - F.mean(0)) / (F.std(0) + 1e-6)
        y = case.cands.index(case.target) if case.target in case.cands else -1
        return F, y

    with ThreadPoolExecutor(int(cfg.get("workers", 1))) as ex:
        rows = list(ex.map(ask, cases))
    sets = {"pool order": [0], "pool + next (fused)": [0, 1], "pool + next + perspectives": list(range(2 + len(names))),
            "pool + perspectives": [0] + list(range(2, 2 + len(names)))}
    folds = np.random.default_rng(0).permutation(len(rows)) % 5
    res = {k: np.zeros(len(rows)) for k in sets}
    w1 = {}
    for k in range(5):
        tr = [rows[i] for i in range(len(rows)) if folds[i] != k]
        for name, cols in sets.items():
            w = fit(tr, cols)
            w1.setdefault(name, w.round(3).tolist())
            for i in np.where(folds == k)[0]:
                F, y = rows[i]
                res[name][i] = ndcg(F[:, cols] @ w, y)
    base = res["pool + next (fused)"]
    rng = np.random.default_rng(1)
    print(f"{ds.name} {cfg['name']}: {len(rows)} validation users, target in the 30 options "
          f"{np.mean([y >= 0 for _, y in rows]):.3f}")
    for name, v in res.items():
        d = v - base
        p = float((np.abs((rng.choice([-1.0, 1.0], size=(10000, len(d))) * d).mean(1)) >= abs(d.mean())).mean())
        print(f"  {name:28s} NDCG@10 {v.mean():.4f}  vs fused {d.mean():+.4f}  p={p:.3f}")
    print("  weights (fold 1):", w1, "order: pool, next,", names)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1000)
