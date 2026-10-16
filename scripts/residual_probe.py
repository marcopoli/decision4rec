"""Residual fast ranker over the multi-agent pool: SASRecF's score plus learned corrections from the agents' votes.

    python scripts/residual_probe.py configs/small/vg_ev2.yaml configs/small/vg_semantic.yaml

The pool, the features and the users are those of scripts/ltr_probe.py (SASRecF top 50 kept whole, 20 items from each
other agent; 15,000 training users of no evaluation case with the second-to-last item as target; the 1,000
validation users of the development cases as test). Features are cached in runs/small/<domain>/probe/pool_feats.npz.

Model: logit(i) = z_sasrec(i) + lam * f(x_i), f linear or a two-layer network started at zero. Softmax loss over the
pool, full batch, Adam. lam and the number of steps are chosen on 20% of the training users (never on the test
users); lam = 0 is the SASRecF order, so the choice can always fall back to it. Reported on the test users: NDCG@10
over the whole catalogue (a target outside the pool scores 0), recall@30 of the ranked pool, and the same for
SASRecF on the same users, with a paired bootstrap interval.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import ltr_probe as L  # noqa: E402

N_TRAIN = 15000
Z_COL = 2 * len(L.AGENTS)          # the standardized SASRecF score, first feature after the agent votes


def features(ev_path, ref_path):
    cfg = L.load_config(ev_path)
    path = os.path.join(cfg["out"], "probe", "pool_feats.npz")
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        return d["X"], d["Y"], d["base"], d["sas30"], int(d["n_train"]), str(d["name"])
    L.N_TRAIN = N_TRAIN
    captured = {}

    def grab(**kw):
        captured.update(kw)
        raise StopIteration

    # run the feature part of ltr_probe.main and stop before its training
    src = open(L.__file__).read()
    body = src[src.index("def main(ev_path, ref_path, n):"):src.index("    mu = np.concatenate(X[:half]).mean(0)")]
    body += "    grab(X=X, Y=Y, base=base, sas30=sas30, half=half, name=ds.name)\n"
    ns = dict(vars(L))
    ns["grab"] = grab
    exec(body, ns)
    try:
        ns["main"](ev_path, ref_path, 1000)
    except StopIteration:
        pass
    os.makedirs(os.path.dirname(path), exist_ok=True)
    X = np.empty(len(captured["X"]), dtype=object)
    X[:] = captured["X"]
    np.savez(path, X=X, Y=np.asarray(captured["Y"]), base=np.asarray(captured["base"]),
             sas30=np.asarray(captured["sas30"]), n_train=captured["half"], name=captured["name"])
    return X, np.asarray(captured["Y"]), np.asarray(captured["base"]), np.asarray(captured["sas30"]), \
        captured["half"], captured["name"]


def pad(X, Y, idx, mu, sd):
    m = max(len(X[u]) for u in idx)
    d = X[idx[0]].shape[1]
    F = np.zeros((len(idx), m, d), dtype=np.float32)
    M = np.zeros((len(idx), m), dtype=bool)
    for r, u in enumerate(idx):
        F[r, :len(X[u])] = X[u]
        M[r, :len(X[u])] = True
    Z = torch.tensor(F[:, :, Z_COL])
    F = (F - mu) / sd
    return torch.tensor(F), torch.tensor(M), Z, torch.tensor(np.asarray([Y[u] for u in idx]))


def logits(net, F, M, Z, lam):
    s = Z + lam * net(F).squeeze(-1)
    return s.masked_fill(~M, float("-inf"))


def rank_of_target(s, y):
    t = s[torch.arange(len(y)), y.clamp(min=0)]
    r = (s > t[:, None]).sum(1) + 1
    return torch.where(y >= 0, r, torch.full_like(r, 10 ** 6))


def ndcg(r):
    r = r.float()
    return torch.where(r <= 10, 1.0 / torch.log2(r + 1.0), torch.zeros_like(r))


def main(ev_path, ref_path):
    torch.manual_seed(0)
    X, Y, base, sas30, n_train, name = features(ev_path, ref_path)
    rng = np.random.default_rng(0)
    tr_all = [u for u in range(n_train) if Y[u] >= 0]
    perm = rng.permutation(tr_all).tolist()
    sel, fit = perm[: len(perm) // 5], perm[len(perm) // 5:]
    test = list(range(n_train, len(Y)))
    mu = np.concatenate([X[u] for u in fit]).mean(0)
    sd = np.concatenate([X[u] for u in fit]).std(0) + 1e-6
    mu[Z_COL], sd[Z_COL] = 0.0, 1.0
    Ffit, Mfit, Zfit, yfit = pad(X, Y, fit, mu, sd)
    Fsel, Msel, Zsel, ysel = pad(X, Y, sel, mu, sd)
    Fte, Mte, Zte, yte = pad(X, Y, test, mu, sd)
    d = Ffit.shape[2]
    b = torch.tensor(base[n_train:], dtype=torch.float32)
    print(f"{name:22s} SASRecF NDCG@10 {b.mean():.4f}  recall@30 {np.mean(sas30[n_train:]):.3f}  "
          f"target in pool {np.mean(Y[n_train:] >= 0):.3f}", flush=True)
    for kind, hidden in (("linear", 0), ("two-layer", 32)):
        best = (ndcg(rank_of_target(Zsel.masked_fill(~Msel, float("-inf")), ysel)).mean().item(), 0.0, None)
        for lam in (0.1, 0.3, 1.0, 3.0):
            torch.manual_seed(0)
            net = torch.nn.Linear(d, 1) if not hidden else torch.nn.Sequential(
                torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
            last = net if not hidden else net[-1]
            torch.nn.init.zeros_(last.weight)
            torch.nn.init.zeros_(last.bias)
            opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
            for step in range(1, 401):
                loss = torch.nn.functional.cross_entropy(logits(net, Ffit, Mfit, Zfit, lam), yfit)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if step % 50 == 0:
                    with torch.no_grad():
                        v = ndcg(rank_of_target(logits(net, Fsel, Msel, Zsel, lam), ysel)).mean().item()
                    if v > best[0]:
                        best = (v, lam, {k: t.clone() for k, t in net.state_dict().items()}, hidden, step)
        if best[2] is None:
            print(f"{name:22s} {kind:9s} selection keeps the SASRecF order (lam = 0)", flush=True)
            continue
        _, lam, state, hidden, step = best
        net = torch.nn.Linear(d, 1) if not hidden else torch.nn.Sequential(
            torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
        net.load_state_dict(state)
        with torch.no_grad():
            s = logits(net, Fte, Mte, Zte, lam)
            r = rank_of_target(s, yte)
        g = ndcg(r)
        # users whose target is outside the pool score 0 for both (SASRecF ranks it below 50 as well)
        g_full = torch.zeros(len(test))
        g_full[:] = g
        diff = (g_full - b).numpy()
        boot = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
        print(f"{name:22s} {kind:9s} lam {lam} step {step}: NDCG@10 {g_full.mean():.4f} vs SASRecF {b.mean():.4f} "
              f"({(g_full.mean() / b.mean() - 1) * 100:+.1f}%, 95% CI [{np.percentile(boot, 2.5):+.4f}, "
              f"{np.percentile(boot, 97.5):+.4f}])  recall@30 {(r <= 30).float().mean():.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
