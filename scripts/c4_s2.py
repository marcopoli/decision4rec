"""C4 rung 2, student S2 (secondary; docs/r1/c4_prereg.md, SHA 560d4c34...): SASRecF fine-tuned from its checkpoint with
an added soft-label term, then used as the pool's agent with the ranker unchanged.

Loss per batch of the 15,000 cross-fitted ranker users (labels and options as scripts/c4_students.py):
    next-item cross-entropy on the user's history (the data SASRecF was trained on; the case target is never used)
    + lam * KL(teacher || student) over the 30 options, student logits at the last history position.
lam from {0.3, 1, 3} by the cross-entropy of the observed target over the 30 options on a held-out fifth (the real
choice, so the weight is not chosen by agreement with the teacher), then refit on all. Control S2_0: the same
fine-tuning with lam = 0 (separates the teacher from any further training, as H does for S1).
Test: the 2,000 development test users; each model is put in the evaluated pool and the pool re-ranks the catalogue
(candidates and list move); NDCG@10 of that list. Bar as S1: >= 50% of the fused gain in >= 2 of 4 domains, above the
pool list (paired randomization, Holm over four), and above S2_0 (paired p < .05).

    python scripts/c4_s2.py out.json [domain ...]
"""
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import r1_score as R
from distill_probe import ndcg
from fusion_table import FOUR, SIX, load
from decision4rec.agents.pool import PoolRetriever, pool_for
from decision4rec.data.schema import Dataset
from decision4rec.eval import metrics as M
from decision4rec.experiments.runner import load_config
from decision4rec.models.sasrec import _pad

TAG = R.SEEDS[0]
LAMS = tuple(float(x) for x in os.environ.get("C4S2_LAMS", "0.3,1,3").split(","))   # rung 3 (exploratory): 3,10,30
EPOCHS, LR, BATCH = 3, 1e-4, 128
SMOKE = int(os.environ.get("C4S2_SMOKE", 0))       # > 0: that many training users, 1 epoch, 100 test users (code check only)


def ndcg_rank(r):
    return np.where((r >= 0) & (r < 10), 1 / np.log2(np.maximum(r, 0) + 2), 0.0)


def train_data(rk, Zrk, w):
    H, C, Q, P = [], [], [], []
    for s in ("test", "valid"):
        for c in sorted(rk[s]):
            k = rk[s][c]
            if len(k.cands) != 30 or c not in Zrk[s]:
                continue
            f = Zrk[s][c] @ w
            q = np.exp(f - f.max())
            H.append(list(k.history)); C.append(list(k.cands)); Q.append(q / q.sum()); P.append(k.target_pos)
    return H, np.array(C), np.array(Q), np.array(P)


def finetune(base, H, C, Q, lam, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = copy.deepcopy(base).train()
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    L = m.max_len
    for _ in range(1 if SMOKE else EPOCHS):
        for s in range(0, len(H), BATCH):
            ix = np.arange(s, min(s + BATCH, len(H)))                           # order reshuffled each epoch below
            hs = [H[i][-(L + 1):] for i in ix]
            # next-item term on the history (positions 1..n-1), as SASRecF's own training
            inp = _pad([h[:-1] for h in hs], L); tgt = _pad([h[1:] for h in hs], L)
            h = m(inp); mask = tgt != 0
            loss = F.cross_entropy(m.item_logits(h[mask]), tgt[mask] - 1) if mask.any() else 0.0
            if lam:
                last = m(_pad([H[i][-L:] for i in ix], L))[:, -1]                  # state after the whole history
                W = m.item_matrix()
                lg = (last[:, None, :] * W[torch.as_tensor(C[ix])]).sum(-1)
                loss = loss + lam * F.kl_div(F.log_softmax(lg, 1), torch.as_tensor(Q[ix], dtype=torch.float32), reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step()
        perm = rng.permutation(len(H)); H = [H[i] for i in perm]; C, Q = C[perm], Q[perm]
    return m.eval()


@torch.no_grad()
def target_ce(m, H, C, P):
    k = np.flatnonzero(P >= 0)
    last = m(_pad([H[i][-m.max_len:] for i in k], m.max_len))[:, -1]
    lg = (last[:, None, :] * m.item_matrix()[torch.as_tensor(C[k])]).sum(-1)
    return float(F.cross_entropy(lg, torch.as_tensor(P[k])))


def list_ranks(p, model, cases, cids):
    p.scorer.model = model
    users = [int(cases[c].user) for c in cids]; hist = [list(cases[c].history) for c in cids]
    idx, _ = PoolRetriever(p).topk_users(users, hist, 30)
    tg = np.array([cases[c].target for c in cids])
    hit = idx == tg[:, None]
    return np.where(hit.any(1), hit.argmax(1), -1)


def main(out, doms):
    torch.set_num_threads(4)
    w = R.weights(SIX, "rtm", TAG)
    res, tests = {}, []
    for d in doms:
        cfg = load_config(f"configs/small/{d}_rtm.yaml")
        ds = Dataset.load(cfg["dataset"])
        p = pool_for(cfg, ds)
        base = p.scorer.model.eval()
        rk, Zrk = load(d, "rtmrank", TAG)
        H, C, Q, P = train_data(rk, Zrk, w[d])
        if SMOKE:
            H, C, Q, P = H[:SMOKE], C[:SMOKE], Q[:SMOKE], P[:SMOKE]
        perm = np.random.default_rng(13).permutation(len(H)); ho, tr = perm[: len(H) // 5], perm[len(H) // 5:]
        sub = lambda ix: ([H[i] for i in ix], C[ix], Q[ix], P[ix])
        ces = {lam: target_ce(finetune(base, *sub(tr)[:3], lam), *[sub(ho)[i] for i in (0, 1, 3)]) for lam in LAMS}
        lam = min(ces, key=ces.get)
        s2 = finetune(base, H, C, Q, lam); s0 = finetune(base, H, C, Q, 0.0)
        dev, Zdev = load(d, "rtm", TAG)
        te = dev["test"]
        per = []                                                         # fused: per-user mean over the three seeds
        for tag in R.SEEDS:
            X = load(d, "rtm", tag); wt = R.weights(SIX, "rtm", tag)
            per.append({c: float(ndcg((X[1]["test"][c] @ wt[d])[None], [X[0]["test"][c].target_pos])[0]) for c in X[1]["test"]})
        cids = sorted(c for c in te if all(c in q for q in per))[: 100 if SMOKE else None]
        nf = np.array([np.mean([q[c] for q in per]) for c in cids])
        nl, ns, n0 = (ndcg_rank(list_ranks(p, m, te, cids)) for m in (base, s2, s0))
        p.scorer.model = base
        gain = nf.mean() - nl.mean()
        r = {"lam": lam, "heldout_ce": ces, "train_users": len(H), "test_users": len(cids),
             "ndcg10": {"list": nl.mean(), "fused": nf.mean(), "S2": ns.mean(), "S2_0": n0.mean()},
             "recovered_S2": (ns.mean() - nl.mean()) / gain if gain else float("nan"),
             "recovered_S2_0": (n0.mean() - nl.mean()) / gain if gain else float("nan"),
             "p_S2_vs_list": float(M.paired_randomization(ns, nl)), "p_S2_vs_S2_0": float(M.paired_randomization(ns, n0))}
        res[d] = r; tests.append((d, r["p_S2_vs_list"]))
        print(f"{d}: lam {lam} held-out CE {ces} | list {nl.mean():.4f} fused {nf.mean():.4f} S2 {ns.mean():.4f} S2_0 {n0.mean():.4f}", flush=True)
        json.dump(res, open(out, "w"), indent=1, default=float)
    for (d, _), h in zip(tests, M.holm([t[1] for t in tests])):
        r = res[d]; n = r["ndcg10"]
        r["holm_S2_vs_list"] = float(h)
        r["met"] = bool(r["recovered_S2"] >= 0.5 and h < 0.05 and n["S2"] > n["list"] and r["p_S2_vs_S2_0"] < 0.05 and n["S2"] > n["S2_0"])
        print(f"{d:6s} S2 {100 * r['recovered_S2']:.0f}% (Holm {h:.4f}) S2_0 {100 * r['recovered_S2_0']:.0f}% S2 vs S2_0 p {r['p_S2_vs_S2_0']:.4f} met {r['met']}", flush=True)
    met = sum(r["met"] for r in res.values())
    print(f"S2 bar (>= 2 of 4): {met} met -> {'PASS' if met >= 2 else 'FAIL'}", flush=True)
    json.dump({"domains": res, "met": met, "verdict": met >= 2}, open(out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or FOUR)
