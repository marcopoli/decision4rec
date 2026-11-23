"""C4 distillation rung 2 (docs/r1/c4_prereg.md, SHA 560d4c34...): students S1 (soft labels) and H (hard labels).

Training users: the 15,000 ranker users of each sparse domain, cross-fitted (runner eval.users: ranker); their options and
pool features come from the refit ranker of their half (cases/<stem>_ranker{0,1}.pt), so a student sees the lists the
reader read. Soft labels: the fused system's choice distribution over the 30 options (list plus kev-9b-rtm seed 0, the
development stable-fusion weights of seed 0). The student is A7's (scripts/distill_probe.py): list z-score plus a linear
term on the pool features and their products with evidence; L2 by held-out cross-entropy on a fifth.
Test: the 2,000 development test users; fused = per-user mean over the three 9B seeds (fusion_table.py). Bar: S1 recovers
>= 50% of the fused gain in >= 2 of 4 domains, S1 > list (paired randomization, Holm over four) and S1 > H (paired p < .05).

    python scripts/c4_students.py out.json [domain ...]
"""
import json
import sys

import numpy as np

import r1_score as R
from distill_probe import choose, feats, ndcg, scores
from fusion_table import FOUR, SIX, load
from decision4rec.agents.pool import pool_for
from decision4rec.data.schema import Dataset
from decision4rec.eval import metrics as M
from decision4rec.experiments.runner import case_paths, load_config

TAG = R.SEEDS[0]


def arrays(p, cases, Zd, w):
    cids = [c for c in sorted(cases) if len(cases[c].cands) == 30]
    F = feats(p, {c: cases[c] for c in cids})
    Zs = np.stack([Zd[c] for c in cids])
    return dict(cids=cids, Z=Zs[:, :, 1], fused=Zs @ w, F=np.stack([F[c] for c in cids]),
                pos=np.array([cases[c].target_pos for c in cids]))


def main(out, doms):
    w = R.weights(SIX, "rtm", TAG)
    res, tests = {}, []
    for d in doms:
        cfg = load_config(f"configs/small/{d}_rtm.yaml")
        ds = Dataset.load(cfg["dataset"])
        p = pool_for(cfg, ds)
        main_state = {k: getattr(p, k) for k in ("lam", "mu", "sd", "train_users", "net", "hidden")}
        dev, Zdev = load(d, "rtm", TAG)
        te = arrays(p, dev["test"], Zdev["test"], w[d])                 # test features from the evaluated pool
        rk, Zrk = load(d, "rtmrank", TAG)
        stem = case_paths(load_config(f"configs/small/{d}_rtmrank.yaml"))["info"][:-len("_info.json")]
        tr = []
        for k, s in ((0, "test"), (1, "valid")):                         # each half with the ranker refit on the other
            p.load(f"{stem}_ranker{k}.pt")
            tr.append(arrays(p, rk[s], Zrk[s], w[d]))
        for key, v in main_state.items():
            setattr(p, key, v)
        Z = np.concatenate([a["Z"] for a in tr]); F = np.concatenate([a["F"] for a in tr])
        fz = np.concatenate([a["fused"] for a in tr]); pos = np.concatenate([a["pos"] for a in tr])
        q = np.exp(fz - fz.max(1, keepdims=True)); soft = q / q.sum(1, keepdims=True)
        keep = pos >= 0
        hard = np.zeros_like(soft); hard[np.flatnonzero(keep), pos[keep]] = 1.0
        ms, l2s = choose(Z, F, soft)
        mh, l2h = choose(Z[keep], F[keep], hard[keep])
        per = []                                                         # fused: per-user mean over the three seeds
        for tag in R.SEEDS:
            X = load(d, "rtm", tag)
            wt = R.weights(SIX, "rtm", tag)
            per.append({c: float(ndcg((X[1]["test"][c] @ wt[d])[None], [X[0]["test"][c].target_pos])[0]) for c in te["cids"] if c in X[1]["test"]})
        k_ = [c for c in te["cids"] if all(c in q_ for q_ in per)]
        ix = np.searchsorted(te["cids"], k_)
        nl = ndcg(te["Z"][ix], te["pos"][ix])
        nf = np.array([np.mean([q_[c] for q_ in per]) for c in k_])
        ns = ndcg(scores(ms, te["Z"][ix], te["F"][ix]), te["pos"][ix])
        nh = ndcg(scores(mh, te["Z"][ix], te["F"][ix]), te["pos"][ix])
        gain = nf.mean() - nl.mean()
        r = {"train_users": int(len(Z)), "train_target_in_30": float(keep.mean()), "test_users": len(k_), "l2": [l2s, l2h],
             "ndcg10": {"list": nl.mean(), "fused": nf.mean(), "S1": ns.mean(), "H": nh.mean()},
             "recovered_S1": (ns.mean() - nl.mean()) / gain if gain else float("nan"),
             "recovered_H": (nh.mean() - nl.mean()) / gain if gain else float("nan"),
             "p_S1_vs_list": float(M.paired_randomization(ns, nl)), "p_S1_vs_H": float(M.paired_randomization(ns, nh)),
             "p_H_vs_list": float(M.paired_randomization(nh, nl))}
        res[d] = r
        tests.append((d, r["p_S1_vs_list"]))
        json.dump(res, open(out, "w"), indent=1, default=float)
    for (d, _), h in zip(tests, M.holm([t[1] for t in tests])):
        r = res[d]
        r["holm_S1_vs_list"] = float(h)
        r["met"] = bool(r["recovered_S1"] >= 0.5 and h < 0.05 and r["ndcg10"]["S1"] > r["ndcg10"]["list"]
                        and r["p_S1_vs_H"] < 0.05 and r["ndcg10"]["S1"] > r["ndcg10"]["H"])
        n = r["ndcg10"]
        print(f"{d:6s} train {r['train_users']} test {r['test_users']} | list {n['list']:.4f} fused {n['fused']:.4f} S1 {n['S1']:.4f} "
              f"({100 * r['recovered_S1']:.0f}%, Holm {h:.4f}) H {n['H']:.4f} ({100 * r['recovered_H']:.0f}%) S1 vs H p {r['p_S1_vs_H']:.4f} met {r['met']}", flush=True)
    met = sum(r["met"] for r in res.values())
    print(f"primary bar (>= 2 of 4): {met} met -> {'PASS' if met >= 2 else 'FAIL'}", flush=True)
    json.dump({"domains": res, "met": met, "verdict": met >= 2}, open(out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or FOUR)
