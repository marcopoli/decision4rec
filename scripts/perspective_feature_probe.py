"""Perspective agents as rankers inside the multi-agent pool: does the similarity between a candidate and the items that
the perspective agents imagine help the pool order? Development validation users, 5-fold cross-validation over users.

    python scripts/perspective_feature_probe.py configs/small/vg_rtm.yaml configs/small/vg_ev2.yaml [n_ideas]

For every pool candidate j and perspective p: f_p(j) = max cosine between the item text of j and the ideas of p.
Control: c(j) = mean cosine between j and the last five items of the history (no language model). Score of the pool:
s(j), the trained residual ranker. Fitted per fold: s + w . f (and s + v c) with softmax cross-entropy on the target,
evaluated as NDCG@10 inside the pool on the held-out fold. Reads the writer caches of scripts/perspective_probe.py,
so no text is generated again.
"""

import json
import os
import sys

import numpy as np
import torch

from decision4rec.agents.pool import pool_for
from decision4rec.data.schema import Dataset, load_cases
from decision4rec.experiments.runner import case_paths, load_config
from decision4rec.retrieval.perspectives import PERSPECTIVES, PerspectiveAgents, prompt
from decision4rec.retrieval.slm import HFWriter, build_embedder


def ndcg(s: np.ndarray, y: int) -> float:
    if y < 0:
        return 0.0
    r = int((s > s[y]).sum()) + 1
    return 1.0 / np.log2(r + 1) if r <= 10 else 0.0


def fit(rows, cols, seed=0, steps=300):
    """w for s + F[:, cols] . w, softmax cross-entropy over the pool."""
    torch.manual_seed(seed)
    w = torch.zeros(len(cols), requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.05, weight_decay=1e-3)
    data = [(torch.tensor(s), torch.tensor(F[:, cols]), y) for s, F, y in rows if y >= 0]
    for _ in range(steps):
        loss = sum(-torch.log_softmax(s + F @ w, 0)[y] for s, F, y in data) / len(data)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach().numpy()


def main(rtm_path: str, ev_path: str, n_ideas: int) -> None:
    cfg, ecfg = load_config(rtm_path), load_config(ev_path)
    ds = Dataset.load(cfg["dataset"])
    pool = pool_for(cfg, ds)
    paths = case_paths({**ecfg, "retriever": {"tag": "ev100"}})
    tag = "" if n_ideas == 1 else f"_v{n_ideas}"
    probe = os.path.join(ecfg["out"], "probe")
    persp = json.load(open(os.path.join(probe, f"perspectives_valid{tag}.json")))
    cases = [c for c in load_cases(paths["valid"]) if str(c.user) in persp]
    eval_users = {c.user for p in (paths["valid"], paths["test"]) for c in load_cases(p)}
    rng = np.random.default_rng(3)                     # the example users of scripts/perspective_probe.py
    example_users = [u for u in rng.permutation(ds.n_users).tolist() if u not in eval_users]
    it = ecfg["item_text"]
    m = np.load(os.path.join(it["cache"], f"emb_{it['embedder']['model'].split('/')[-1]}_content_{ds.n_items}.npy"))
    unit = (m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)).astype(np.float32)
    emb = build_embedder(it["embedder"])
    writer = HFWriter("Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens=32 if n_ideas == 1 else 120, batch_size=16,
                      cache=os.path.join(probe, f"perspectives_writer{tag}.sqlite"))
    agents = PerspectiveAgents(ds, writer, emb, m, example_users, n_ideas=n_ideas)
    names = list(PERSPECTIVES)
    rows, miss = [], 0
    for s0 in range(0, len(cases), 128):
        part = cases[s0: s0 + 128]
        sas = pool._sasrec_rows([c.user for c in part], [c.history for c in part])
        texts = {p: writer.write([prompt(ds, p, c.history, agents.examples, n_ideas=n_ideas) for c in part]) for p in names}
        for r, c in enumerate(part):
            items, X, _ = pool.features(c.history, pool.store.now(c.user, len(c.history)), sas[r])
            s = pool._score(X)
            cand = unit[items]
            F = []
            for p in names:
                t = texts[p][r]
                ideas = [x.strip(" -*0123456789.") for x in t.split("\n") if x.strip()][:n_ideas] or [t]
                q = emb.encode(ideas, is_query=True)
                q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
                F.append((cand @ q.T).max(1))
            F.append(cand @ unit[c.history[-5:]].mean(0))
            F = np.stack(F, 1).astype(np.float32)
            F = (F - F.mean(0)) / (F.std(0) + 1e-6)
            y = items.index(c.target) if c.target in items else -1
            miss += y < 0
            rows.append((s.astype(np.float32), F, y))
    n = len(rows)
    folds = np.random.default_rng(0).permutation(n) % 5
    res = {"pool": [], "pool+perspectives": [], "pool+history text (control)": [], "pool+both": []}
    sets = {"pool+perspectives": list(range(len(names))), "pool+history text (control)": [len(names)],
            "pool+both": list(range(len(names) + 1))}
    weights = {}
    for k in range(5):
        tr = [rows[i] for i in range(n) if folds[i] != k]
        te = [i for i in range(n) if folds[i] == k]
        for name, cols in sets.items():
            w = fit(tr, cols)
            weights.setdefault(name, []).append(w.round(3).tolist())
            for i in te:
                s, F, y = rows[i]
                res[name].append((i, ndcg(s + F[:, cols] @ w, y)))
        for i in te:
            res["pool"].append((i, ndcg(rows[i][0], rows[i][2])))
    base = np.array([v for _, v in sorted(res["pool"])])
    rng = np.random.default_rng(1)
    print(f"{ds.name}: {n} validation users, target outside the pool {miss / n:.3f}, ideas per perspective {n_ideas}")
    for name in res:
        v = np.array([x for _, x in sorted(res[name])])
        d = v - base
        sgn = rng.choice([-1.0, 1.0], size=(10000, n))
        p = float((np.abs((sgn * d).mean(1)) >= abs(d.mean())).mean()) if name != "pool" else 1.0
        print(f"  {name:30s} NDCG@10 {v.mean():.4f}  diff {d.mean():+.4f}  p={p:.3f}")
    print("  weights (fold 1):", {k: v[0] for k, v in weights.items()}, "order:", names + ["history text"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3)
