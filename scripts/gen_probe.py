"""Step 0 of A5/A6 (docs/18, note of Nov 11; CPU, exploratory): do new generators of thin items bring rare targets
among the candidates of the pool?

A2 found the target outside all candidates for 61 to 83% of users, rare targets (under 20 purchases in the training
part) for 86 to 92%. Three generators of rare items only, each at depth 50 and 200, fixed before any result:
  text50   text neighbours (st.unit) of each of the last three history items, merged by best similarity
  imputed  rules inherited by text: the "bought after" items of the last three history items (st.rules, decayed as
           the store does), each passing its weight to its 10 nearest rare items in text
  graph    shared attributes of the last three history items (the kg agent of the pool), rare items only, no cap on
           the size of an attribute
  chance   any 50 or 200 rare items (analytic)
Per setting and split (development test and validation users): share of rare targets and of all targets among the
pool's candidates, and the points each generator adds (targets it finds that the pool misses), with the number of new
candidates per user. Bar (docs/18): at least 5 points more rare targets among the candidates in two settings, on test
and validation users.

    python scripts/gen_probe.py out.json [setting ...]
"""
import json
import sys
from collections import Counter

import numpy as np

from decision4rec.agents.pool import pool_for
from decision4rec.data.schema import Dataset, load_cases
from decision4rec.experiments.runner import case_paths, load_config

RARE, DEPTHS, LAST, NN = 20, (50, 200), 3, 10
GENS = ("text50", "imputed", "graph")


def lists(p, R, UR, h, near):
    st, seen, out = p.store, set(h), {}
    best = {}
    if st.unit is not None:
        for i in h[-LAST:]:
            s = UR @ st.unit[i]
            top = np.argpartition(-s, min(len(s) - 1, DEPTHS[-1]))[: DEPTHS[-1] + LAST]
            for t in top:
                j = int(R[t])
                if j not in seen and s[t] > best.get(j, -9.0):
                    best[j] = float(s[t])
    out["text50"] = sorted(best, key=lambda j: (-best[j], j))[: DEPTHS[-1]]
    imp = Counter()
    if st.unit is not None:
        for r, i in enumerate(reversed(h[-LAST:])):
            for n, w in sorted(st.rules.get(i, {}).items(), key=lambda t: (-t[1], t[0]))[:50]:
                for j, sim in near(n):
                    if j not in seen:
                        imp[j] += st.decay ** r * w * sim
    out["imputed"] = [j for j, _ in sorted(imp.items(), key=lambda t: (-t[1], t[0]))[: DEPTHS[-1]]]
    kg, g = p.kg, Counter()
    if kg is not None:
        for i in h[-LAST:]:
            for key in sorted(kg.attrs[i]):
                w = kg.idf.get(key)
                if w:
                    for j in kg.ent_items[key]:
                        if j not in seen and st.pop[j] < RARE:
                            g[j] += w
    out["graph"] = [j for j, _ in sorted(g.items(), key=lambda t: (-t[1], t[0]))[: DEPTHS[-1]]]
    return out


if __name__ == "__main__":
    out, sets = sys.argv[1], sys.argv[2:] or ["vg", "mi", "steam", "ind"]
    res = {}
    for d in sets:
        cfg = load_config(f"configs/small/{d}_rtm.yaml")
        ds = Dataset.load(cfg["dataset"])
        p = pool_for(cfg, ds)
        st = p.store
        R = np.flatnonzero(st.pop < RARE)
        UR = st.unit[R] if st.unit is not None else None
        cache = {}

        def near(n):
            if n not in cache:
                s = UR @ st.unit[n]
                t = np.argpartition(-s, NN)[:NN]
                cache[n] = [(int(R[k]), float(max(s[k], 0.0))) for k in t if R[k] != n]
            return cache[n]
        res[d] = {"rare_items": int(len(R)), "items": int(ds.n_items), "text": st.unit is not None, "graph": p.kg is not None}
        for split in ("test", "valid"):
            cases = sorted(load_cases(case_paths(cfg)[split]), key=lambda c: c.cid)
            inpool, rare, newc = [], [], {(g, k): [] for g in GENS for k in DEPTHS}
            hit = {(g, k): [] for g in GENS for k in DEPTHS}
            for b in range(0, len(cases), 256):
                chunk = cases[b: b + 256]
                rows = p._sasrec_rows([c.user for c in chunk], [c.history for c in chunk])
                for i, c in enumerate(chunk):
                    h = list(c.history)
                    cand = set(p.features(h, st.now(c.user, len(h)), None if rows is None else rows[i])[0])
                    inpool.append(c.target in cand)
                    rare.append(st.pop[c.target] < RARE)
                    L = lists(p, R, UR, h, near)
                    for g in GENS:
                        for k in DEPTHS:
                            top = set(L[g][:k])
                            hit[(g, k)].append(c.target in top and c.target not in cand)
                            newc[(g, k)].append(len(top - cand))
            inpool, rare = np.array(inpool), np.array(rare)
            row = {"users": len(cases), "rare_share": float(rare.mean()), "rare_in_pool": float(inpool[rare].mean()),
                   "all_in_pool": float(inpool.mean()), "gens": {}}
            line = (f"{d:6s} {split:5s} users {len(cases)} rare {100 * rare.mean():.1f}% | in the pool: rare {100 * row['rare_in_pool']:.1f}% "
                    f"all {100 * row['all_in_pool']:.1f}%")
            for g in GENS:
                for k in DEPTHS:
                    hv = np.array(hit[(g, k)])
                    row["gens"][f"{g}@{k}"] = {"rare_points": float(100 * hv[rare].mean()), "all_points": float(100 * hv.mean()),
                                               "new_candidates": float(np.mean(newc[(g, k)]))}
                    line += f" | {g}@{k} +{100 * hv[rare].mean():.1f} rare +{100 * hv.mean():.1f} all ({np.mean(newc[(g, k)]):.0f} new)"
            for k in DEPTHS:
                row[f"chance@{k}"] = float(100 * k / max(len(R), 1))
            line += " | chance " + " ".join(f"@{k} +{row[f'chance@{k}']:.2f}" for k in DEPTHS)
            print(line, flush=True)
            res[d][split] = row
            json.dump(res, open(out, "w"), indent=1)
