"""A7, distillation rung 1 (V32; docs/18; CPU): can the fast ranker learn what the 9B reader adds, from the reader's
choice probabilities on validation users, so that the reader is not needed at test time?

Teacher: the fused system of the paper (list + kev-9b-rtm, stable fusion weights of thin_probe.per_user), its choice
distribution over the 30 options of each validation case as soft labels. Our own model, trained on ground-truth
interactions only, so no Jev output is used (TypeSafe MCA 2.3(b)).
Student: s = a * z(list score) + f(x), linear, x = the pool ranker's features of each option (decision4rec/agents/pool.py
features; zeros and a flag for options the pool does not hold) plus their products with the option's evidence
log(1 + purchases). L2 strength chosen on a fifth of the validation users (held-out cross-entropy), then refit on all.
Control: the same student fit on the observed target (hard labels) of the same users, to separate what the reader
teaches from what any refit on validation users gives.
Test users: NDCG@10 of list, fused, soft student, hard student; share of the fused gain recovered; paired sign-flip p.
Bar (fixed before results): soft student recovers at least 30% of the fused gain in two sparse settings, p < .05
against the list. Pass: C4. Fail: rung 2 (SASRecF fine-tuned on soft labels of 15,000 users).

    python scripts/distill_probe.py out.json [setting ...]
"""
import json
import sys

import numpy as np
import torch

from fusion_table import FAMILY, STRENGTH
from thin_probe import SEEDS, runs
from decision4rec.agents.pool import pool_for
from decision4rec.data.schema import Dataset
from decision4rec.eval import metrics as M
from decision4rec.experiments.runner import load_config
from decision4rec.operators.fusion import choice_data, shrunk_weights
from rare_pool import flip_p

L2S = (1e-4, 1e-3, 1e-2, 1e-1)


def feats(p, cases):
    """Per case: option features (n_opts x d), in the order of the case's options."""
    st, out, cids = p.store, {}, sorted(cases)
    for b in range(0, len(cids), 256):
        ch = [cases[c] for c in cids[b: b + 256]]
        rows = p._sasrec_rows([c.user for c in ch], [list(c.history) for c in ch])
        for r, c in enumerate(ch):
            h = list(c.history)
            pool, X, _ = p.features(h, st.now(c.user, len(h)), None if rows is None else rows[r])
            idx = {j: i for i, j in enumerate(pool)}
            F = np.zeros((len(c.cands), X.shape[1] + 1), dtype=np.float64)
            for k, j in enumerate(c.cands):
                if j in idx:
                    F[k, :-1] = X[idx[j]]
                else:
                    F[k, -1] = 1.0
            e = np.log1p(st.pop[np.asarray(c.cands)])[:, None]
            out[c.cid] = np.hstack([F, F[:, :-1] * e])
    return out


def fit(Z, F, Y, l2, steps=600, seed=0):
    """Z: n x m list z-scores, F: n x m x d, Y: n x m target distribution. Returns (a, beta, mu, sd)."""
    torch.manual_seed(seed)
    mu, sd = F.reshape(-1, F.shape[2]).mean(0), F.reshape(-1, F.shape[2]).std(0) + 1e-6
    Ft, Zt, Yt = torch.tensor((F - mu) / sd), torch.tensor(Z), torch.tensor(Y)
    a = torch.ones(1, dtype=torch.float64, requires_grad=True)
    beta = torch.zeros(F.shape[2], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([a, beta], lr=0.02)
    for _ in range(steps):
        s = a * Zt + Ft @ beta
        loss = -(Yt * torch.log_softmax(s, 1)).sum(1).mean() + l2 * (beta ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(a.item()), beta.detach().numpy(), mu, sd


def scores(model, Z, F):
    a, beta, mu, sd = model
    return a * Z + ((F - mu) / sd) @ beta


def ce(model, Z, F, Y):
    s = scores(model, Z, F)
    s = s - s.max(1, keepdims=True)
    return float(-(Y * (s - np.log(np.exp(s).sum(1, keepdims=True)))).sum(1).mean())


def choose(Z, F, Y, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Z))
    ho, tr = perm[: len(Z) // 5], perm[len(Z) // 5:]
    best = min(L2S, key=lambda l2: ce(fit(Z[tr], F[tr], Y[tr], l2), Z[ho], F[ho], Y[ho]))
    return fit(Z, F, Y, best), best


def ndcg(S, pos):
    return np.array([M.per_case_metrics(M.target_rank(s, t))["ndcg@10"] for s, t in zip(S, pos)])


if __name__ == "__main__":
    out, sets = sys.argv[1], sys.argv[2:] or ["vg", "mi", "steam", "ind"]
    D = runs(SEEDS[0])
    data = {d: choice_data(D[d][0]["valid"], D[d][1]["valid"]) for d in D}
    w = shrunk_weights(data, FAMILY, STRENGTH)
    res = {}
    for d in sets:
        cfg = load_config(f"configs/small/{d}_rtm.yaml")
        p = pool_for(cfg, Dataset.load(cfg["dataset"]))
        cases, Zd = D[d]
        arr = {}
        for s in ("valid", "test"):
            C = cases[s]
            cids = [c for c in sorted(C) if len(C[c].cands) == len(C[sorted(C)[0]].cands)]
            Fd = feats(p, {c: C[c] for c in cids})
            Zs = np.stack([Zd[s][c] for c in cids])
            arr[s] = dict(cids=cids, Z=Zs[:, :, 1], fused=Zs @ w[d], F=np.stack([Fd[c] for c in cids]),
                          pos=np.array([C[c].target_pos for c in cids]))
        v, t = arr["valid"], arr["test"]
        q = np.exp(v["fused"] - v["fused"].max(1, keepdims=True))
        soft = q / q.sum(1, keepdims=True)
        keep = v["pos"] >= 0                                   # hard labels need the target among the options
        hard = np.zeros_like(soft)
        hard[np.flatnonzero(keep), v["pos"][keep]] = 1.0
        ms, l2s = choose(v["Z"], v["F"], soft)
        mh, l2h = choose(v["Z"][keep], v["F"][keep], hard[keep])
        nl, nf = ndcg(t["Z"], t["pos"]), ndcg(t["fused"], t["pos"])
        ns, nh = ndcg(scores(ms, t["Z"], t["F"]), t["pos"]), ndcg(scores(mh, t["Z"], t["F"]), t["pos"])
        gain = nf.mean() - nl.mean()
        row = {"valid": len(v["cids"]), "test": len(t["cids"]), "l2": [l2s, l2h],
               "ndcg10": {"list": nl.mean(), "fused": nf.mean(), "soft": ns.mean(), "hard": nh.mean()},
               "recovered_soft": (ns.mean() - nl.mean()) / gain if gain else float("nan"),
               "recovered_hard": (nh.mean() - nl.mean()) / gain if gain else float("nan"),
               "p_soft_vs_list": flip_p(ns - nl), "p_hard_vs_list": flip_p(nh - nl), "p_soft_vs_hard": flip_p(ns - nh),
               "p_fused_vs_list": flip_p(nf - nl)}
        res[d] = row
        print(f"{d:6s} test {row['test']} | NDCG@10 list {nl.mean():.4f} fused {nf.mean():.4f} (p {row['p_fused_vs_list']:.4f}) "
              f"soft {ns.mean():.4f} (rec {100 * row['recovered_soft']:.0f}%, p {row['p_soft_vs_list']:.4f}) "
              f"hard {nh.mean():.4f} (rec {100 * row['recovered_hard']:.0f}%, p {row['p_hard_vs_list']:.4f}) "
              f"| soft vs hard p {row['p_soft_vs_hard']:.4f} | l2 {l2s} {l2h}", flush=True)
        json.dump(res, open(out, "w"), indent=1)
