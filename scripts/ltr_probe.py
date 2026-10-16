"""Do the agents' votes carry signal that SASRecF lacks? A learned ranker over the multi-agent pool.

    python scripts/ltr_probe.py configs/small/vg_ev2.yaml configs/small/vg_semantic.yaml [n_users]

Pool per user: the SASRecF top 50, kept whole, plus the top 20 of every other agent (store evidence, transitions,
co-purchase, text, popular now, knowledge graph, popularity). Features per candidate: for every agent, whether it
proposed the item and 1 / (1 + rank); the SASRecF score standardized in the pool; the store's numbers behind the
statements (log purchases, log recent purchases, transition weight, co-purchase count with the history, log text
rank). Rankers: logistic regression and a small two-layer network, trained with a softmax loss over the pool of each
user (the target is the class), on 4,000 other users (second-to-last item as target, every evaluation user left out);
reported on the 1,000 validation users. NDCG@10 is
over the whole catalogue: a target outside the pool scores 0; SASRecF is scored by its own full ranking on the same
users. Validation cases of the first evidence pilot (history s[:-2], target s[-2]); statistics leave out the last
two items of every user.
"""

import sys
from collections import Counter

import numpy as np
import torch

from decision4rec.data.schema import Dataset, load_cases
from decision4rec.evidence.store import store_for
from decision4rec.experiments.benchmark import prepare_dataset
from decision4rec.experiments.runner import case_paths, load_config
from decision4rec.experiments.stage1 import chosen_dir
from decision4rec.graph.kg import ItemKG
from decision4rec.models import baselines as B

AGENTS = ("sasrecf", "evidence", "after", "with", "text", "lately", "kg", "pop")
DEPTH = {"sasrecf": 50}
OTHER = 20
N_TRAIN = int(__import__('os').environ.get('N_TRAIN', 4000))


def kg_agent(kg, h, k):
    score, seen = Counter(), set(h)
    for s in h[-3:]:
        for key in kg.attrs[s]:
            w = kg.idf.get(key)
            if w and len(kg.ent_items[key]) <= 200:
                for j in kg.ent_items[key]:
                    if j not in seen:
                        score[j] += w
    return [j for j, _ in score.most_common(k)]


def ndcg_rank(r):
    return 1.0 / np.log2(r + 1.0) if r <= 10 else 0.0


def main(ev_path, ref_path, n):
    torch.manual_seed(0)
    cfg, ref = load_config(ev_path), load_config(ref_path)
    ds = Dataset.load(cfg["dataset"])
    store = store_for(cfg, ds)
    kg = ItemKG(ds)
    prepare_dataset(ref, ds, ["sasrecf"])
    scorer = B.load("sasrecf", chosen_dir(ref, "sasrecf"), ds, "cpu")
    cases = load_cases(case_paths({**cfg, "retriever": {"tag": "ev100"}})["valid"])[:n]
    from decision4rec.data.schema import Case
    from decision4rec.experiments.rectune import _case_users
    excluded = _case_users(cfg)
    rng0 = np.random.default_rng(7)
    others = [u for u in rng0.permutation(ds.n_users).tolist() if u not in excluded and len(ds.seqs[u]) >= 4][:N_TRAIN]
    train_cases = [Case(cid=-1, user=u, history=[int(x[0]) for x in ds.seqs[u][:-2]], target=int(ds.seqs[u][-2][0]),
                        cands=[], rscores=[], target_pos=-1) for u in others]
    half = len(train_cases)
    cases = train_cases + cases
    popular = [int(j) for j in np.argsort(-store.pop, kind="stable")[:300]]
    X, Y, base, inpool, pool_sizes, sas30 = [], [], [], [], [], []
    for s in range(0, len(cases), 128):
        part = cases[s: s + 128]
        with torch.no_grad():
            m = scorer.scores([c.user for c in part], [c.history for c in part]).cpu().numpy().astype(float)
        for c, row in zip(part, m):
            h, seen = c.history, set(c.history)
            row[np.asarray(h, dtype=int)] = -np.inf
            rank_t = int((row > row[c.target]).sum()) + 1
            base.append(ndcg_rank(rank_t))
            sas30.append(rank_t <= 30)
            now = store.now(c.user, len(h))
            g = store.generators(h, now)
            ctx = store.context(h, now)
            L = {"sasrecf": [int(j) for j in np.argsort(-row)[:50]],
                 "evidence": store.candidates(h, now, OTHER)}
            for a in ("after", "with", "text", "lately"):
                L[a] = [j for j in g[a] if j not in seen][:OTHER]
            L["kg"] = kg_agent(kg, h, OTHER)
            L["pop"] = [j for j in popular if j not in seen][:OTHER]
            pool = list(dict.fromkeys(j for a in AGENTS for j in L[a]))
            pos = {a: {j: r for r, j in enumerate(L[a])} for a in AGENTS}
            sc = row[pool]
            z = (sc - sc.mean()) / (sc.std() + 1e-9)
            feats = []
            for k, j in enumerate(pool):
                f = []
                for a in AGENTS:
                    r = pos[a].get(j)
                    f += [0.0 if r is None else 1.0, 0.0 if r is None else 1.0 / (1.0 + r)]
                f += [z[k], np.log1p(store.pop[j]), np.log1p(store.recent_count(j, now)), ctx.after.get(j, 0.0),
                      float(ctx.with_n.get(j, 0)), np.log(ctx.text_rank[j]) if ctx.text_rank is not None else 0.0]
                feats.append(f)
            X.append(np.asarray(feats, dtype=np.float32))
            Y.append(pool.index(c.target) if c.target in pool else -1)
            inpool.append(c.target in pool)
            pool_sizes.append(len(pool))
    mu = np.concatenate(X[:half]).mean(0)
    sd = np.concatenate(X[:half]).std(0) + 1e-6
    Xn = [torch.tensor((x - mu) / sd) for x in X]
    print(f"{ds.name:22s} pool size {np.mean(pool_sizes):.0f}  target in pool {np.mean(inpool[half:]):.3f}  "
          f"(SASRecF top 50 alone: see pool_probe)  SASRecF NDCG@10 {np.mean(base[half:]):.4f}", flush=True)
    for name, hidden in (("logistic", 0), ("two-layer", 32)):
        d = Xn[0].shape[1]
        net = torch.nn.Linear(d, 1) if not hidden else torch.nn.Sequential(
            torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
        opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
        train = [u for u in range(half) if Y[u] >= 0]
        for ep in range(60):
            for u in np.random.default_rng(ep).permutation(train):
                loss = torch.nn.functional.cross_entropy(net(Xn[u]).squeeze(1)[None], torch.tensor([Y[u]]))
                opt.zero_grad()
                loss.backward()
                opt.step()
        got = []
        with torch.no_grad():
            for u in range(half, len(cases)):
                if Y[u] < 0:
                    got.append(0.0)
                    continue
                s = net(Xn[u]).squeeze(1).numpy()
                got.append(ndcg_rank(int((s > s[Y[u]]).sum()) + 1))
        r30 = np.mean([0.0 if Y[u] < 0 else float((net(Xn[u]).squeeze(1).detach().numpy() > net(Xn[u]).squeeze(1).detach().numpy()[Y[u]]).sum() < 30)
                       for u in range(half, len(cases))])
        print(f"{ds.name:22s} {name:9s} recall@30 of the ranked pool {r30:.3f} vs SASRecF top 30 {np.mean(sas30[half:]):.3f}", flush=True)
        g, b = np.array(got), np.array(base[half:])
        diff = g - b
        rng = np.random.default_rng(0)
        boot = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
        print(f"{ds.name:22s} {name:9s} NDCG@10 {g.mean():.4f} vs SASRecF {b.mean():.4f}  "
              f"({(g.mean() / b.mean() - 1) * 100:+.1f}%, 95% CI of difference [{np.percentile(boot, 2.5):+.4f}, "
              f"{np.percentile(boot, 97.5):+.4f}])", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1000)
