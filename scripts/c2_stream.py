"""C2 stream replay (L4; prereg docs/r1/c2_prereg.md, SHA d68b8ee8...): purchases after the global cut, each served by the
ActiveGraph runtime (decision4rec/agents/full_runtime.py) before it is shown, in two store modes:

    frozen : the evidence store of the cut (as the pool and kev-9b-rtm_cold were trained)
    fresh  : the store rebuilt from every purchase before the day of the event (daily snapshots, same settings)
    retrained (secondary): SASRecF and the pool's ranker refit once on every purchase before day 30 after the cut, store
             frozen at day 30; served on the events of day 30 on only (same events, same reader, same fusion weights)

The pool's ranker and SASRecF stay as at the cut; only the store moves (statements, store-based agents and features).
Events: 2,000 per domain after the cut, seed 13, equal numbers per week over weeks with >= 50 events; the history of an
event is every purchase of the user before it (after the cut included). Fusion weights: the stable weights of the
development cold runs (rtm:kev-9b-rtm_cold, validation users). One JSON line per event and mode.

    python scripts/c2_stream.py <d> out.jsonl [--modes frozen,fresh,retrained] [--n 2000] [--dry] [--fit-only]

Retrained mode, the cut protocol replayed at day 30 (same code, same settings): the day-30 dataset keeps, per store user,
every purchase before day 30 plus the first one after it (as make_cold_dataset at the cut); SASRecF is refit with the
hyperparameters chosen at the cut (no new tuning, seed 13), the store has holdout 2 as store_for, the ranker is refit on
the training users of the cut pool. --fit-only fits (GPU if visible) and exits; serving loads the fit on the CPU.

--dry samples the events and times the store builds of the first 5 snapshot days (no read): the weekly fallback of the
prereg is decided from this timing only.
"""
import argparse
import json
import os
import time

import numpy as np

import thin_probe as TP
from decision4rec.agents.full_runtime import FullAgent
from decision4rec.agents.pool import AgentPool, PoolRetriever, pool_for
from decision4rec.graph.kg import ItemKG
from decision4rec.models import baselines as B
from decision4rec.data.schema import Case, Dataset
from decision4rec.decision.base import build_backend
from decision4rec.evidence.store import EvidenceStore, store_for
from decision4rec.experiments.asc import _z
from decision4rec.experiments.runner import load_config, make_prompt_builder
from decision4rec.experiments.stage1 import chosen_dir
from decision4rec.operators.rank import Arm

TAG = "kev-9b-rtm_cold"
WEEK_MIN = 50
RETRAIN_DAY = 30


def source_of(cold: Dataset):
    """Source dataset and, per cold user, the source user (make_cold_dataset keeps users in order, dropping those with
    fewer than min_pre purchases before the cut)."""
    info = cold.extra["cold_protocol"]
    src = Dataset.load(os.path.join(os.path.dirname(cold.extra.get("path", "")) or "data/processed", info["source"]))
    cut, mp = info["cutoff"], info["min_pre"]
    m = [u for u, s in enumerate(src.seqs) if sum(x[1] < cut for x in s) >= mp]
    assert len(m) == cold.n_users, (len(m), cold.n_users)
    for cu in (0, len(m) // 2, len(m) - 1):              # self-check: the pre-cut part matches
        pre = [int(x[0]) for x in src.seqs[m[cu]] if x[1] < cut]
        assert [int(x[0]) for x in cold.seqs[cu]][: len(pre)] == pre
    return src, m, cut


def sample_events(src, m, cut, n, day, seed=13):
    ev = [(cu, k) for cu, su in enumerate(m) for k, x in enumerate(src.seqs[su]) if x[1] >= cut]
    t = np.array([src.seqs[m[cu]][k][1] for cu, k in ev], dtype=float)
    wk = ((t - cut) // (7 * day)).astype(int)
    weeks = [w for w in np.unique(wk) if (wk == w).sum() >= WEEK_MIN]
    per = n // len(weeks)
    rng = np.random.default_rng(seed)
    pick = []
    for w in weeks:
        idx = np.flatnonzero(wk == w)
        pick += sorted(rng.choice(idx, min(per, len(idx)), replace=False).tolist())
    return [(ev[i][0], ev[i][1], int(wk[i])) for i in sorted(pick, key=lambda i: t[i])], weeks


def snapshot(cold, src, m, D, cfg, text):
    """Store of every purchase before time D by the store's users (two padding rows per user stand for the holdout)."""
    seqs = []
    for su in m:
        s = [x for x in src.seqs[su] if x[1] < D]
        seqs.append(s + [s[-1], s[-1]] if s else [])
    ds = Dataset(name=f"{cold.name}_at", items=cold.items, seqs=seqs, user_tag=cold.user_tag, extra=cold.extra)
    e = cfg.get("evidence") or {}
    fusion = tuple(float(w) for w in e["fusion"]) if e.get("fusion") else None
    return EvidenceStore(ds, float(e.get("frac", 1.0)), 2, int(e.get("seed", 0)), text, float(e.get("days", 30.0)), fusion=fusion)


def retrained(cfg, cold, src, m, D, cut_pool, device):
    """Pool and store of the cut protocol replayed at time D: refit SASRecF (cut hyperparameters) and the ranker."""
    pc = cfg["pool"]
    assert set(pc) <= {"ref", "model", "users", "seed"}, pc            # the cold pools: no extra agents to replay
    seqs = []
    for su in m:
        pre = [x for x in src.seqs[su] if x[1] < D]
        seqs.append(pre + [x for x in src.seqs[su] if x[1] >= D][:1])
    ds = Dataset(name=f"{cold.name}_d{RETRAIN_DAY}", items=cold.items, seqs=seqs, user_tag=cold.user_tag, extra=dict(cold.extra))
    out = os.path.join(cfg["out"], f"c2_retrain_d{RETRAIN_DAY}")
    ref = load_config(pc["ref"])
    hp = json.load(open(os.path.join(chosen_dir(ref, pc["model"]), "info.json")))["hp"]
    sdir = os.path.join(out, pc["model"])
    if not os.path.exists(os.path.join(sdir, "info.json")):
        info = B.fit(pc["model"], ds, sdir, hp=hp, seed=13, device=device)
        print(f"[retrain] {pc['model']} at day {RETRAIN_DAY}: {info}", flush=True)
    scorer = B.load(pc["model"], sdir, ds, device)
    e = cfg.get("evidence") or {}
    fusion = tuple(float(w) for w in e["fusion"]) if e.get("fusion") else None
    store = EvidenceStore(ds, float(e.get("frac", 1.0)), 2, int(e.get("seed", 0)), cold.extra.get("item_text_matrix"),
                          float(e.get("days", 30.0)), fusion=fusion)
    pool = AgentPool(ds, store, scorer, ItemKG(ds), 20, 50)
    path = os.path.join(out, "pool.pt")
    if os.path.exists(path):
        pool.load(path)
    else:
        info = pool.fit(list(cut_pool.train_users))
        pool.save(path)
        json.dump(info, open(path.replace(".pt", ".json"), "w"), indent=1)
        print(f"[retrain] ranker at day {RETRAIN_DAY}: {info}", flush=True)
    return pool, store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("d"); ap.add_argument("out")
    ap.add_argument("--modes", default="frozen,fresh"); ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--dry", action="store_true"); ap.add_argument("--fit-only", action="store_true")
    a = ap.parse_args()
    dc = f"{a.d}_cold"
    cfg = load_config(f"configs/small/{dc}_rtm.yaml", f"logs/probe/side_{TAG}.yaml")
    cold = Dataset.load(cfg["dataset"])
    src, m, cut = source_of(cold)
    day = 86_400_000.0 if cut > 1e11 else 86_400.0
    events, weeks = sample_events(src, m, cut, a.n, day)
    D30 = cut + RETRAIN_DAY * day
    late = sum(src.seqs[m[cu]][k][1] >= D30 for cu, k, _ in events)
    print(f"{dc}: {len(events)} events over {len(weeks)} weeks (>= {WEEK_MIN} events each), {late} from day {RETRAIN_DAY}", flush=True)
    pool = pool_for(cfg, cold)
    if a.fit_only:
        retrained(cfg, cold, src, m, D30, pool, None)
        return
    frozen = store_for(cfg, cold)
    text = cold.extra.get("item_text_matrix")
    days = sorted({int((src.seqs[m[cu]][k][1] - cut) // day) for cu, k, _ in events})
    if a.dry:
        t0 = time.time()
        for dd in days[:5]:
            snapshot(cold, src, m, cut + dd * day, cfg, text)
        per = (time.time() - t0) / min(5, len(days))
        print(f"{len(days)} snapshot days, {per:.1f} s per store build, {per * len(days) / 3600:.2f} h in all", flush=True)
        return
    W = TP.weights_of(TP.runs(f"rtm:{TAG}", [f"{x}_cold" for x in ("vg", "mi", "steam", "ind")]))   # shrunk across the four
    wf, wl = np.asarray(W["f"][dc]), np.asarray(W["l"][dc])
    backend = build_backend(cfg["backend"])
    arm = Arm.from_dict(next(x for x in cfg["arms"] if x["name"] == "rtm"))
    pb = make_prompt_builder(cfg, cold, need_kg=True)
    agent = FullAgent(PoolRetriever(pool), arm, backend, pb, wf, K=30, seed=13, block=1)
    if "retrained" in a.modes.split(","):
        pool_r, store_r = retrained(cfg, cold, src, m, D30, pool, "cpu")
        agent_r = FullAgent(PoolRetriever(pool_r), arm, backend, pb, wf, K=30, seed=13, block=1)
    # purchases of each item before each event, over the store's users (the market the store sees)
    allt = sorted((x[1], int(x[0])) for su in m for x in src.seqs[su])
    times = np.array([t for t, _ in allt]); items = np.array([i for _, i in allt])
    done = set()
    if os.path.exists(a.out):
        done = {(r["mode"], r["eid"]) for r in map(json.loads, open(a.out))}
    snaps = {}
    with open(a.out, "a") as f:
        for mode in a.modes.split(","):
            for eid, (cu, k, wk) in enumerate(events):
                if (mode, eid) in done:
                    continue
                s = src.seqs[m[cu]]
                t, target = float(s[k][1]), int(s[k][0])
                if mode == "retrained" and t < D30:
                    continue
                ag, pl = (agent_r, pool_r) if mode == "retrained" else (agent, pool)
                if mode == "retrained":
                    store = store_r
                elif mode == "fresh":
                    dd = int((t - cut) // day)
                    if dd not in snaps:
                        snaps.clear()                       # events are in time order: keep one day in memory
                        snaps[dd] = snapshot(cold, src, m, cut + dd * day, cfg, text)
                    store = snaps[dd]
                else:
                    store = frozen
                store.now = lambda user, n_hist, _t=t: _t   # the moment of the request
                pl.store = pb.store = store
                case = Case(cid=eid, user=cu, history=[int(x[0]) for x in s[:k]], target=target, cands=[], rscores=[],
                            target_pos=-1, split="test", target_rating=0.0)
                t0 = time.perf_counter()
                out = ag.serve(case)
                sec = time.perf_counter() - t0
                cands = out["cands"]
                pos = cands.index(target) if target in cands else -1
                Z = np.stack([_z(out["theta"]), _z(out["rscores"])], 1)
                rec = {"mode": mode, "eid": eid, "week": wk, "user": cu, "t": t, "target": target, "pos": pos,
                       "sold_before": int(np.sum((items[: np.searchsorted(times, t, side="left")] == target))),
                       "born_after_cut": bool(not np.any(items[: np.searchsorted(times, cut, side="left")] == target)),
                       "rank_l": int(np.sum(Z[:, 1] > Z[pos, 1])) if pos >= 0 else None,
                       "rank_f": int(np.sum(Z @ wf > (Z @ wf)[pos])) if pos >= 0 else None,
                       "ll_f": TP._logp(Z @ wf, pos) if pos >= 0 else None,
                       "ll_l": TP._logp(Z[:, [1]] @ wl, pos) if pos >= 0 else None,
                       "seconds": sec, "cached": out.get("cached")}
                for k_ in ("l", "f"):                       # NDCG@10 of one target from its rank (0 = first)
                    r_ = rec[f"rank_{k_}"]
                    rec[f"ndcg_{k_}"] = float(1 / np.log2(r_ + 2)) if r_ is not None and r_ < 10 else 0.0
                f.write(json.dumps(rec) + "\n"); f.flush()
                if eid % 100 == 0:
                    print(f"{mode} {eid}/{len(events)} week {wk} {sec:.2f} s", flush=True)
    backend.close()


if __name__ == "__main__":
    main()
