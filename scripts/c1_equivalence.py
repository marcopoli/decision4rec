"""C1 equivalence (docs/18 C1, R23): Decision4Rec-Full on ActiveGraph (decision4rec/agents/full_runtime.py) against the batch runner.

For each sparse domain, the first N development test users (sorted case id): the runtime serves each request from the
user's history alone (pool, decision model, Luce fit, fusion with the development weights of seed 0), and its top 10 must
equal the batch top 10 (scripts/fusion_table.py inputs, the same weights). The decision model is answered from the cache
of the reported run only: a cache miss raises (no new read). Bar: equal top 10 for at least 99.9% of the users per domain;
also reported: equal candidate lists, equal Luce utilities (max abs diff), events per request, milliseconds per request.

    python scripts/c1_equivalence.py out.json [N=1000] [domain ...]
"""
import json
import sys
import time

import numpy as np

import r1_score as R
from fusion_table import FOUR, SIX, load
from decision4rec.agents.full_runtime import FullAgent
from decision4rec.agents.pool import PoolRetriever, pool_for
from decision4rec.data.schema import Dataset
from decision4rec.decision.base import build_backend
from decision4rec.experiments.runner import load_config, make_prompt_builder
from decision4rec.operators.rank import Arm

TAG = "kev-9b-rtm"


class CacheOnly:
    def __init__(self, inner):
        self.inner = inner
        self.name, self.max_request_chars = inner.name, inner.max_request_chars

    def fingerprint(self):
        return self.inner.fingerprint()

    def decide(self, state, questions):
        raise RuntimeError("cache miss: the runtime asked a question the reported run never asked")

    def close(self):
        pass


def main(out, n=1000, doms=FOUR):
    w = R.weights(SIX, "rtm", TAG)
    res = {}
    for d in doms:
        cfg = load_config(f"configs/small/{d}_rtm.yaml", f"logs/probe/side_{TAG}.yaml")
        ds = Dataset.load(cfg["dataset"])
        backend = build_backend(cfg["backend"])
        backend.inner = CacheOnly(backend.inner)
        arm = Arm.from_dict(next(a for a in cfg["arms"] if a["name"] == "rtm"))
        pb = make_prompt_builder(cfg, ds, need_kg=True)
        agent = FullAgent(PoolRetriever(pool_for(cfg, ds)), arm, backend, pb, w[d], K=int(cfg["eval"]["K"]), seed=int(cfg["eval"].get("seed", 13)))
        cases, Z = load(d, "rtm", TAG)
        T = cases["test"]
        eq = eq_c = 0
        dth, ms, ev, bad, miss = [], [], [], [], []
        for c in sorted(T)[:n]:
            case = T[c]
            t0 = time.perf_counter()
            try:
                s = agent.serve(case)
            except RuntimeError as e:          # a cache miss counts as a failure of equivalence, never as a new read
                miss.append(c)
                if len(bad) < 5:
                    bad.append({"cid": c, "error": str(e)[:200]})
                continue
            ms.append(1000 * (time.perf_counter() - t0))
            ev.append(len(s["events"]))
            batch_score = Z["test"][c] @ w[d]
            top_b = [case.cands[j] for j in np.argsort(-batch_score, kind="stable")[:10]]
            same_c = s["cands"] == list(case.cands)
            eq_c += same_c
            eq += s["slate"] == top_b
            if same_c:
                dth.append(float(np.max(np.abs(np.asarray(s["score"]) - batch_score))))
            if s["slate"] != top_b and len(bad) < 5:
                bad.append({"cid": c, "runtime": s["slate"], "batch": top_b, "same_cands": same_c})
        m = min(n, len(T))
        res[d] = {"users": m, "equal_top10": eq / m, "equal_candidates": eq_c / m, "max_abs_score_diff": max(dth) if dth else None,
                  "events_per_request": float(np.mean(ev)), "ms_per_request_median": float(np.median(ms)),
                  "cache_hits": backend.hits, "cache_miss_cids": miss, "met": eq / m >= 0.999, "examples": bad}
        print(f"{d}: equal top 10 {eq}/{m} ({100 * eq / m:.2f}%), equal candidates {eq_c}/{m}, max score diff {res[d]['max_abs_score_diff']}, "
              f"cache misses {len(miss)}, {res[d]['events_per_request']:.0f} events, {res[d]['ms_per_request_median']:.0f} ms median, met {res[d]['met']}", flush=True)
        backend.close()
    json.dump({"tag": TAG, "domains": res}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1000, sys.argv[3:] or FOUR)
