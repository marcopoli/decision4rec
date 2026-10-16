"""The agent on ActiveGraph as a benchmarked system.

The complete agent (user model on the graph, retrieval agents as behaviours, routing, block
ranking, expansion driven by the outside option, fusion with the retrieval score) is run on the
same validation and test users as every other method of the benchmark.

Validation users: the agent runs once per routing mode with expansion always on, and every round
is recorded. Decisions are deterministic, so the outcome of any expansion threshold and of any
fusion weight can be read from that record without new model calls. The routing mode, the
threshold and the weight are chosen there. Test users: the agent runs once with the chosen
setting, and the ranking it emits is what is scored. Nothing is selected on test users.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import numpy as np

from ..agents.active import DecisionAgent, fuse_scores
from ..data.schema import Case, Dataset
from ..decision.base import build_backend
from ..eval import metrics as M
from .report import GAMMAS
from .runner import build_retriever, get_cases, load_config, make_prompt_builder

AGENTS = ["sequential", "content", "knowledge", "collaborative", "profile", "graph", "popularity"]
QUANTILES = (0.75, 0.5, 0.25)
ROW = "decision4rec_agent"            # name of the agent's row in the benchmark


def agent_settings(cfg: dict) -> dict:
    """The declared agent: the first tranche is the whole pool of K candidates, compared with the block
    design of the batch system; the session keeps more fused candidates for the expansion rounds."""
    a = dict(cfg.get("agent") or {})
    K = int(cfg["eval"].get("K", 50))
    step, rounds = int(a.get("step", 15)), int(a.get("max_rounds", 2))
    return {"k0": int(a.get("k0", K)), "pool": int(a.get("pool", K + step * rounds)), "step": step,
            "max_rounds": rounds, "m": int(a.get("m", 10)), "rounds": int(a.get("rounds", 2)),
            "evidence": a.get("evidence", "pref"),
            "depth": int(a.get("depth", (cfg.get("retriever") or {}).get("depth", 200))),
            "routes": list(a.get("routes", ["tuned", "decision"])), "agents": list(a.get("agents", AGENTS)),
            "replay_check": int(a.get("replay_check", 20))}


def build_agent_retrieval(cfg: dict, ds: Dataset, kg, agents: list[str]):
    """The retrieval agents of the config plus those that need the user model. None for a single retriever."""
    from ..retrieval.agents import build_multi_retriever
    rcfg = dict(cfg.get("retriever") or {})
    if rcfg.get("kind") != "multi":
        return None
    rcfg["agents"] = agents
    base = build_retriever(rcfg["base"], ds) if rcfg.get("base") else None
    return build_multi_retriever(rcfg, ds, base, kg)


def replay(rec: dict, tau: float, gamma: float) -> dict:
    """Outcome of a recorded always-expanding run under an expansion threshold and a fusion weight.
    The agent stops at the first round whose outside probability does not exceed the threshold."""
    step = next((s for s in rec["trace"] if not s["p_out"] > tau), rec["trace"][-1])
    active = step["active"]
    final = fuse_scores(step["theta"], [rec["rscores"][j] for j in active], gamma)
    pos = active.index(rec["target_pos"]) if rec["target_pos"] in active else -1
    return {"rank": M.target_rank(final, pos), "round": step["round"], "tokens": step["tokens"],
            "requests": step["requests"], "latency_ms": step["latency_ms"], "examined": len(active)}


def _plain(x: float):
    """Thresholds as strict JSON: the two infinite ones are written as words."""
    return "never" if x == float("inf") else "always" if x == float("-inf") else float(x)


def _rank(res, case: Case) -> float:
    pos = res.target_pos(case.target)
    return M.target_rank(res.scores, pos) if pos >= 0 and np.isfinite(res.scores[pos]) else float("inf")


def _ndcg10(ranks) -> np.ndarray:
    ranks = np.asarray(ranks, dtype=float)
    return np.where(ranks <= 10, 1.0 / np.log2(ranks + 1.0), 0.0)


def _load(path: str) -> dict[int, dict]:
    out = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[r["cid"]] = r
    return out


def _run_cases(agent: DecisionAgent, cases: list[Case], path: str, record) -> dict[int, dict]:
    """One agent run per case, appended to `path`. Cases already stored are skipped (resumable)."""
    done = _load(path)
    todo = [c for c in cases if c.cid not in done]
    if todo:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for n, c in enumerate(todo, 1):
                t0 = time.perf_counter()
                res = agent.run(c)
                rec = record(res, c)
                rec["wall_ms"] = (time.perf_counter() - t0) * 1000.0
                f.write(json.dumps(rec) + "\n")
                f.flush()
                done[c.cid] = rec
                if n % 50 == 0 or n == len(todo):
                    print(f"[agent] {os.path.basename(path)} {n}/{len(todo)}", flush=True)
    return {c.cid: done[c.cid] for c in cases}


def _trace_record(res, case: Case) -> dict:
    return {"cid": case.cid, "target_pos": res.target_pos(case.target), "n_pool": res.n_pool,
            "rscores": [float(x) for x in res.rscores], "trace": res.trace, "n_events": len(res.events)}


def _final_record(res, case: Case) -> dict:
    rank = _rank(res, case)
    pos = res.target_pos(case.target)
    return {"cid": case.cid, "rank": None if np.isinf(rank) else rank, "p_out": res.p_out,
            "confidence": res.confidence, "rounds": res.rounds, "examined": res.n_examined,
            "graph_added": res.n_graph_added, "target_examined": int(not np.isinf(rank)),
            "target_from_graph": int(pos >= res.n_pool and not np.isinf(rank)), "tokens": res.tokens,
            "requests": res.requests, "latency_ms": res.latency_ms, "n_events": len(res.events),
            "slate": [int(i) for i in res.slate], "route": res.route}


def run_agent_bench(cfg_path: str, backend_path: Optional[str] = None, max_cases: Optional[int] = None) -> dict:
    """`max_cases` makes a partial run for timing. It writes its own files and never touches a full run."""
    cfg = load_config(cfg_path, backend_path)
    if "backend" not in cfg:
        raise SystemExit("this config has no backend section: pass one with --backend configs/backends/<file>.yaml")
    st = agent_settings(cfg)
    ds = Dataset.load(cfg["dataset"])
    cases = get_cases(cfg, ds)
    valid = cases["valid"][:max_cases] if max_cases else cases["valid"]
    test = cases["test"][:max_cases] if max_cases else cases["test"]
    pb = make_prompt_builder(cfg, ds, need_kg=True)
    multi = build_agent_retrieval(cfg, ds, pb.kg, st["agents"])
    routes = st["routes"] if multi is not None else ["pool"]
    backend = build_backend(cfg["backend"])
    out_dir = os.path.join(cfg["out"], cfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    part = "" if max_cases is None else "_partial"
    common = dict(k0=st["k0"], pool=st["pool"], step=st["step"], max_rounds=st["max_rounds"], m=st["m"],
                  rounds=st["rounds"], evidence=st["evidence"], depth=st["depth"],
                  seed=int(cfg["eval"].get("seed", 13)), retrieval=multi)

    def agent(route: str, tau: float, gamma: float = 0.0) -> DecisionAgent:
        return DecisionAgent(backend, pb, tau_expand=tau, gamma=gamma, route=route, **common)

    try:
        # 1. validation users: one always-expanding run per routing mode, every round recorded.
        traces = {r: _run_cases(agent(r, float("-inf")), valid, os.path.join(out_dir, f"agent_{r}.valid.jsonl"),
                                _trace_record) for r in routes}

        # 2. selection on validation users, by replay of the records.
        grid = []
        for ri, route in enumerate(routes):
            recs = list(traces[route].values())
            p0 = np.array([r["trace"][0]["p_out"] for r in recs])
            taus = [("never", float("inf"))] + [(f"q{int(q * 100)}", float(np.quantile(p0, q))) for q in QUANTILES]
            taus.append(("always", float("-inf")))
            for policy, tau in taus:
                for gamma in (0.0,) + GAMMAS:
                    rows = [replay(r, tau, gamma) for r in recs]
                    grid.append({"route": route, "policy": policy, "tau": tau, "gamma": gamma,
                                 "valid_ndcg@10": float(_ndcg10([x["rank"] for x in rows]).mean()),
                                 "valid_tokens": float(np.mean([x["tokens"] for x in rows])),
                                 "valid_expansion_rate": float(np.mean([x["round"] > 0 for x in rows])),
                                 "order": ri})
        # Best validation NDCG@10. Ties go to the cheaper setting, then to the smaller weight, then to the first route.
        best = max(grid, key=lambda g: (round(g["valid_ndcg@10"], 10), -g["valid_tokens"], -g["gamma"], -g["order"]))
        route, tau, gamma = best["route"], best["tau"], best["gamma"]
        chosen = agent(route, tau, gamma)
        valid_ranks = {cid: replay(r, tau, gamma)["rank"] for cid, r in traces[route].items()}

        # 3. the replay must equal real runs: checked on the first validation users with the chosen setting.
        check = valid[: st["replay_check"]]
        same = [_rank(chosen.run(c), c) == valid_ranks[c.cid] for c in check]

        # 4. test users: one run with the chosen setting. Stored test runs belong to one setting:
        #    if the selection is not the one they were made with, they are discarded and made again.
        test_path = os.path.join(out_dir, f"agent{part}.test.jsonl")
        stamp_path = os.path.join(out_dir, f"agent{part}.test.setting.json")
        stamp = {"route": route, "tau": _plain(tau), "gamma": gamma, "agent": st}
        old = None
        if os.path.exists(stamp_path):
            with open(stamp_path) as f:
                old = json.load(f)
        if old != stamp and os.path.exists(test_path):
            os.remove(test_path)
        with open(stamp_path, "w") as f:
            json.dump(stamp, f)
        final = _run_cases(chosen, test, test_path, _final_record)
    finally:
        backend.close()

    rows = [final[c.cid] for c in test]
    ranks = np.array([np.inf if r["rank"] is None else r["rank"] for r in rows], dtype=float)
    nd = _ndcg10(ranks)
    lo, hi = M.bootstrap_ci(nd)

    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in rows])) if rows else float("nan")

    out = {"run": cfg["name"], "row": ROW, "agent": st, "agents": [a.name for a in multi.agents] if multi else None,
           "weights_tuned": getattr(multi, "weights", None) if multi else None,
           "n_valid": len(valid), "n_test": len(test), "complete": max_cases is None,
           "chosen": {**{k: best[k] for k in ("route", "policy", "gamma", "valid_ndcg@10")}, "tau": _plain(tau)},
           "selection": [{**{k: v for k, v in g.items() if k not in ("order", "tau")}, "tau": _plain(g["tau"])}
                         for g in grid],
           "replay_check": {"n": len(same), "identical": int(sum(same))},
           "test": {"ndcg@10": float(nd.mean()), "ci95": [lo, hi], "hr@10": float((ranks <= 10).mean()),
                    "mrr": float(np.where(np.isfinite(ranks), 1.0 / ranks, 0.0).mean()),
                    "tokens": mean("tokens"), "requests": mean("requests"), "latency_ms": mean("latency_ms"),
                    "wall_ms": mean("wall_ms"), "expansion_rate": float(np.mean([r["rounds"] > 0 for r in rows])),
                    "mean_rounds": mean("rounds"), "examined": mean("examined"), "graph_candidates": mean("graph_added"),
                    "target_examined_rate": mean("target_examined"),
                    "targets_recovered_by_graph": mean("target_from_graph"), "events_per_case": mean("n_events")},
           "ranks": {"valid": {str(c.cid): (None if np.isinf(valid_ranks[c.cid]) else valid_ranks[c.cid]) for c in valid},
                     "test": {str(r["cid"]): r["rank"] for r in rows}}}
    with open(os.path.join(out_dir, f"agent_bench{part}.json"), "w") as f:
        json.dump(out, f, indent=1)
    md = to_markdown(out)
    with open(os.path.join(out_dir, f"agent_bench{part}.md"), "w") as f:
        f.write(md)
    out["markdown"], out["out_dir"] = md, out_dir
    if len(same) and not all(same):
        print(f"[agent] WARNING: the replay differs from real runs on {len(same) - sum(same)} of {len(same)} "
              "validation users. The model is not deterministic: check the backend.", flush=True)
    return out


def _tau(x) -> str:
    return f"{x} expand" if isinstance(x, str) else f"{x:.4f}"


def to_markdown(out: dict) -> str:
    c, t = out["chosen"], out["test"]
    md = [f"# Agent on ActiveGraph: {out['run']}", "",
          f"Agents: {', '.join(out['agents']) if out['agents'] else 'fixed candidate pool'}. "
          f"First tranche {out['agent']['k0']}, session list {out['agent']['pool']}, blocks of {out['agent']['m']} in "
          f"{out['agent']['rounds']} divisions, at most {out['agent']['max_rounds']} expansion rounds of {out['agent']['step']}.",
          "", "## Setting chosen on validation users", "",
          f"Routing `{c['route']}`, expansion threshold on the outside probability {_tau(c['tau'])} (`{c['policy']}`), "
          f"fusion weight {c['gamma']}. Validation NDCG@10 {c['valid_ndcg@10']:.4f} on {out['n_valid']} users.",
          f"Replay check: {out['replay_check']['identical']} of {out['replay_check']['n']} real runs equal the replay.",
          "", "## Test users", "",
          f"{out['n_test']} users" + ("" if out["complete"] else " (partial run, not used by the benchmark)") + ".", "",
          "| NDCG@10 | 95% CI | HR@10 | MRR | requests | tokens | model latency ms | wall ms | events |", "|---|---|---|---|---|---|---|---|---|",
          f"| {t['ndcg@10']:.4f} | [{t['ci95'][0]:.4f}, {t['ci95'][1]:.4f}] | {t['hr@10']:.4f} | {t['mrr']:.4f} | "
          f"{t['requests']:.2f} | {t['tokens']:.0f} | {t['latency_ms']:.0f} | {t['wall_ms']:.0f} | {t['events_per_case']:.0f} |",
          "", f"Expansion: {t['expansion_rate']:.3f} of the users, {t['mean_rounds']:.2f} rounds and {t['examined']:.1f} "
              f"examined candidates on average, {t['graph_candidates']:.2f} of them from preference paths. "
              f"Target examined for {t['target_examined_rate']:.3f} of the users, recovered from outside the fused list "
              f"for {t['targets_recovered_by_graph']:.3f}.",
          "", "## Selection grid (validation users, best first)", "",
          "| routing | expansion | fusion weight | NDCG@10 | tokens | expansion rate |", "|---|---|---|---|---|---|"]
    for g in sorted(out["selection"], key=lambda g: -g["valid_ndcg@10"])[:15]:
        md.append(f"| {g['route']} | {g['policy']} | {g['gamma']} | {g['valid_ndcg@10']:.4f} | {g['valid_tokens']:.0f} | "
                  f"{g['valid_expansion_rate']:.3f} |")
    return "\n".join(md) + "\n"
