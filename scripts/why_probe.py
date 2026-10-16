"""Perspectives, version 5: the trained decision model answers "which kind of choice will the user make next?" in the
same request as the next choice (one pass, question isolation). Does the answer help the order of the pool?

    python scripts/why_probe.py configs/small/vg_rtm.yaml configs/backends/kev-0.8b-rtmv5.yaml [n]

On development validation users, 5-fold cross-validation over users (as scripts/perspective_questions_probe.py):
pool order + log p(next), against the same plus, for every candidate, the probability that the model gives to the
perspectives whose list (rules of the knowledge graph) contains the candidate. Also prints how often the model's
most likely perspective equals the label of the rules for the real next item, against always answering the most
frequent label.
"""

import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from perspective_questions_probe import fit, ndcg  # noqa: E402

from decision4rec.agents.kgperspectives import NAMES, WHY_INSTRUCTIONS, WHY_TEXT, KGPerspectives, why_label  # noqa: E402
from decision4rec.data.schema import Dataset  # noqa: E402
from decision4rec.decision.base import build_backend  # noqa: E402
from decision4rec.decision.types import Choice  # noqa: E402
from decision4rec.experiments.runner import get_cases, load_config, make_prompt_builder  # noqa: E402

LABELS = NAMES + ("other",)


def main(cfg_path: str, backend_path: str, n: int) -> None:
    cfg = load_config(cfg_path, backend_path)
    ds = Dataset.load(cfg["dataset"])
    cases = get_cases(cfg, ds)["valid"][:n]
    pb = make_prompt_builder(cfg, ds)
    kp = KGPerspectives(ds, pb.store)
    backend = build_backend(cfg["backend"])
    why = Choice(instructions=WHY_INSTRUCTIONS, options=tuple((k, WHY_TEXT[k]) for k in LABELS))

    def ask(case):
        order = list(range(len(case.cands)))
        nxt = pb.choice(case, order, evidence="stats+found")
        d = backend.decide(pb.state(case), {"next": nxt, "why": why})
        keys = nxt.keys()
        pn = d.answers["next"].probs
        pw = d.answers["why"].probs
        lists = kp.lists(case.history, pb.store.now(case.user, len(case.history)), k=20)
        member = np.array([[float(c in set(lists[p])) for p in NAMES] for c in case.cands])
        w = np.array([pw.get(p, 0.0) for p in NAMES])
        F = np.zeros((len(keys), 3), dtype=np.float32)
        F[:, 0] = -np.log1p(np.arange(len(keys)))
        F[:, 1] = np.log(np.array([max(pn.get(k, 0.0), 1e-6) for k in keys]))
        F[:, 2] = member @ w
        F = (F - F.mean(0)) / (F.std(0) + 1e-6)
        y = case.cands.index(case.target) if case.target in case.cands else -1
        rule = why_label(kp.explain(case.history, case.target))
        top = max(LABELS, key=lambda k: pw.get(k, 0.0))
        return F, y, rule, top

    with ThreadPoolExecutor(int(cfg.get("workers", 1))) as ex:
        out = list(ex.map(ask, cases))
    rows = [(F, y) for F, y, _, _ in out]
    rules = [r for _, _, r, _ in out]
    tops = [t for _, _, _, t in out]
    major = Counter(rules).most_common(1)[0]
    sets = {"pool order": [0], "pool + next (fused)": [0, 1], "pool + next + why": [0, 1, 2]}
    folds = np.random.default_rng(0).permutation(len(rows)) % 5
    res = {k: np.zeros(len(rows)) for k in sets}
    for k in range(5):
        tr = [rows[i] for i in range(len(rows)) if folds[i] != k]
        for name, cols in sets.items():
            w = fit(tr, cols)
            for i in np.where(folds == k)[0]:
                F, y = rows[i]
                res[name][i] = ndcg(F[:, cols] @ w, y)
    base = res["pool + next (fused)"]
    rng = np.random.default_rng(1)
    print(f"{ds.name} {cfg['name']}: {len(rows)} validation users")
    print(f"  why: model's top answer = rules' label in {np.mean([a == b for a, b in zip(tops, rules)]):.3f} of users; "
          f"always '{major[0]}' would give {major[1] / len(rules):.3f}; rule labels {dict(Counter(rules))}")
    for name, v in res.items():
        d = v - base
        p = float((np.abs((rng.choice([-1.0, 1.0], size=(10000, len(d))) * d).mean(1)) >= abs(d.mean())).mean())
        print(f"  {name:22s} NDCG@10 {v.mean():.4f}  vs fused {d.mean():+.4f}  p={p:.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1000)
