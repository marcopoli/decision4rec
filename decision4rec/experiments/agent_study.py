"""Agent runtime study on ActiveGraph.

Measures what an event-sourced, decision-native agent can be held to:
1. outside-option-driven expansion against never and always expanding (accuracy per token),
2. the contribution of the graph-held user model and of preference-path expansion (ablation),
3. incremental maintenance of the user model against a full rebuild (agreement and cost),
4. exact reproducibility of slates under re-execution with uncached model calls,
5. fork-based counterfactuals: how beliefs and slate change when one interaction is removed,
6. evidence attribution by re-execution: effect of each preference-path fact on the decision,
7. runtime overhead and trace size.
"""

from __future__ import annotations

import json
import os
import tempfile
import time

import numpy as np

from ..agents.active import DecisionAgent
from ..data.schema import Dataset
from ..decision.base import CachedBackend, build_backend, decide_chunked
from ..eval import metrics as M
from .runner import build_retriever, get_cases, load_config, make_prompt_builder


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def _evaluate(agent: DecisionAgent, cases: list) -> tuple[dict, dict]:
    rows, runs, t0 = [], {}, time.perf_counter()
    for c in cases:
        r = agent.run(c)
        runs[c.cid] = r
        pos = r.target_pos(c.target)
        m = M.per_case_metrics(M.target_rank(r.scores, pos))
        rows.append((m["ndcg@10"], m["hr@10"], r.tokens, r.requests, r.n_examined, r.rounds, len(r.events),
                     r.latency_ms, int(pos >= 0 and np.isfinite(r.scores[pos])), r.n_graph_added,
                     int(pos >= len(c.cands) and np.isfinite(r.scores[pos]))))
    a = np.array(rows, dtype=float)
    wall = (time.perf_counter() - t0) * 1000 / max(len(cases), 1)
    lo, hi = M.bootstrap_ci(a[:, 0])
    summary = {"ndcg@10": float(a[:, 0].mean()), "ci95": [lo, hi], "hr@10": float(a[:, 1].mean()),
               "tokens": float(a[:, 2].mean()), "requests": float(a[:, 3].mean()),
               "examined": float(a[:, 4].mean()), "expansion_rounds": float(a[:, 5].mean()),
               "events_per_case": float(a[:, 6].mean()), "target_examined_rate": float(a[:, 8].mean()),
               "graph_candidates": float(a[:, 9].mean()), "targets_recovered_by_graph": float(a[:, 10].mean()),
               "runtime_overhead_ms": float(wall - a[:, 7].mean())}
    return summary, runs


def run_agent_study(cfg_path: str, n_cases: int = 100, n_deep: int = 20, backend_path=None) -> dict:
    cfg = load_config(cfg_path, backend_path)
    acfg = cfg.get("agent", {})
    ds = Dataset.load(cfg["dataset"])
    cases = get_cases(cfg, ds)
    test, valid = cases["test"][:n_cases], cases["valid"][: max(n_cases // 2, 20)]
    pb = make_prompt_builder(cfg, ds, need_kg=True)
    backend = build_backend(cfg["backend"])
    seed = int(cfg["eval"].get("seed", 13))
    common = dict(k0=int(acfg.get("k0", 20)), step=int(acfg.get("step", 15)), max_rounds=int(acfg.get("max_rounds", 2)),
                  m=int(acfg.get("m", 5)), rounds=int(acfg.get("rounds", 2)), evidence=acfg.get("evidence", "pref"),
                  seed=seed)

    def agent(tau: float, be=backend, **over) -> DecisionAgent:
        return DecisionAgent(be, pb, tau_expand=tau, **{**common, **over})

    # 1. expansion policies. Thresholds are quantiles of the first-round outside probability on validation cases.
    p0 = np.array([agent(float("inf")).run(c).p_out for c in valid])
    policies = {"never": float("inf"), "always": float("-inf")}
    policies.update({f"adaptive_q{int(q * 100)}": float(np.quantile(p0, q)) for q in (0.25, 0.5, 0.75)})
    out: dict = {"backend": backend.fingerprint(), "n_cases": len(test), "agent": common, "policies": {}}
    runs: dict = {}
    for name, tau in policies.items():
        summary, runs[name] = _evaluate(agent(tau), test)
        out["policies"][name] = {"tau_expand": tau, **summary}

    # 2. ablation at full expansion: what the graph-held user model and preference paths add.
    out["ablation_always_expand"] = {
        "no_user_model": _evaluate(agent(float("-inf"), user_model=False, intent=False, kg_expand=False), test)[0],
        "user_model_pool_only": _evaluate(agent(float("-inf"), kg_expand=False), test)[0],
        "user_model_no_intent": _evaluate(agent(float("-inf"), intent=False), test)[0],
        "full": out["policies"]["always"]}

    # 2b. retrieval agents on the graph: small language models retrieve from several points of view,
    #     and the decision model weights their proposals. Compared with the fixed retriever pool.
    if cfg.get("retrieval_agents"):
        from ..retrieval.agents import build_multi_retriever
        rcfg = dict(cfg["retrieval_agents"])
        names = rcfg.get("agents") or []
        base = build_retriever(cfg.get("retriever", {}), ds) if "sequential" in names else None
        multi = build_multi_retriever(rcfg, ds, base, pb.kg)
        pool = int(rcfg.get("pool", len(test[0].cands) if test else 50))
        section = {"agents": [a.name for a in multi.agents], "weights_tuned": getattr(multi, "tune_info", None),
                   "sources": {"retriever_pool": {"never": out["policies"]["never"], "always": out["policies"]["always"]}}}
        routes = ["uniform", "decision"] + (["tuned"] if rcfg.get("weights") == "tuned" else [])
        for route in routes:
            section["sources"][f"agents_{route}"] = {
                label: _evaluate(agent(tau, retrieval=multi, route=route, pool=pool), test)[0]
                for label, tau in (("never", float("inf")), ("always", float("-inf")))}
        out["retrieval_agents"] = section

    ref_policy = "adaptive_q50"
    ag = agent(policies[ref_policy])

    # 3. incremental maintenance of the user model against a full rebuild.
    mae, covered, refreshed, tok_inc, tok_cold = [], [], [], [], []
    for c in test[:n_deep]:
        hist = pb.profile_history(c)
        if len(hist) < 6:
            continue
        runtime, sid, _ = ag.start(c)
        for item in hist[:-3]:
            ag.observe(runtime, sid, item)
        before = ag.refresh(runtime, sid)
        for item in hist[-3:]:
            ag.observe(runtime, sid, item)
        n_stale = sum(1 for b in ag.beliefs(runtime, sid).values() if b["stale"])
        after = ag.refresh(runtime, sid)
        inc = ag.beliefs(runtime, sid)
        cold = pb.user_model(backend, c)
        keys = [f"{q.rel}\t{q.ent}" for q in cold.prefs]
        have = [k for k in keys if inc.get(k, {}).get("p") is not None]
        if have:
            strength = {f"{q.rel}\t{q.ent}": q.p for q in cold.prefs}
            mae.append(float(np.mean([abs(inc[k]["p"] - strength[k]) for k in have])))
        covered.append(len(have) / max(len(keys), 1))
        refreshed.append(n_stale / max(len(inc), 1))
        tok_inc.append(after - before)
        tok_cold.append(cold.tokens)
    out["incremental_user_model"] = {
        "n": len(covered), "mean_abs_difference_to_full_rebuild": float(np.mean(mae)) if mae else float("nan"),
        "coverage_of_full_rebuild_hypotheses": float(np.mean(covered)) if covered else float("nan"),
        "share_of_beliefs_refreshed_after_3_interactions": float(np.mean(refreshed)) if refreshed else float("nan"),
        "tokens_incremental_refresh": float(np.mean(tok_inc)) if tok_inc else float("nan"),
        "tokens_full_rebuild": float(np.mean(tok_cold)) if tok_cold else float("nan")}

    # 4. reproducibility: re-execute with fresh model calls and compare slates and event sequences.
    inner = backend.inner if isinstance(backend, CachedBackend) else backend
    same_slate, same_events, max_diff = [], [], 0.0
    for c in test[:n_deep]:
        base = runs[ref_policy][c.cid]
        again = agent(policies[ref_policy], be=inner).run(c)
        same_slate.append(base.slate == again.slate)
        same_events.append(base.events == again.events)
        if base.items == again.items:
            fin = np.isfinite(base.scores) & np.isfinite(again.scores)
            if fin.any():
                max_diff = max(max_diff, float(np.abs(base.scores[fin] - again.scores[fin]).max()))
    out["reproducibility"] = {"n": len(same_slate), "identical_slates": float(np.mean(same_slate)),
                              "identical_event_sequences": float(np.mean(same_events)),
                              "max_score_difference": max_diff}

    # 5. fork-based counterfactuals and 6. evidence attribution.
    jacc, top1_changed, belief_shift, shared_events, fact_effects, compr = [], [], [], [], [], []
    tmp = tempfile.mkdtemp(prefix="decision4rec_agent_")
    for c in test[:n_deep]:
        base = ag.run(c, persist_to=os.path.join(tmp, f"case{c.cid}.db"))
        for h in pb.history(c)[-3:]:
            cf = ag.counterfactual(base, c, drop=h)
            a, b = set(base.slate), set(cf.slate)
            jacc.append(len(a & b) / max(len(a | b), 1))
            top1_changed.append(base.slate[0] != cf.slate[0])
            both = [k for k, v in base.profile.items() if v["p"] is not None and cf.profile.get(k, {}).get("p") is not None]
            if both:
                belief_shift.append(float(np.mean([abs(base.profile[k]["p"] - cf.profile[k]["p"]) for k in both])))
            shared_events.append(len(base.runtime.diff(cf.runtime).shared_events))
        item = base.slate[0]
        prof = pb.user_model(backend, c)
        facts = pb.facts(c, item, "pref+cf", prof=prof)
        if facts:
            state = pb.state(c, profile="decision", prof=prof)
            qs = {"all": pb.noul(c, item, facts=facts), "none": pb.noul(c, item, facts=[])}
            qs.update({f"wo{j}": pb.noul(c, item, facts=facts[:j] + facts[j + 1:]) for j in range(len(facts))})
            d = decide_chunked(backend, state, qs)
            full = _logit(d.answers["all"].p)
            compr.append(full - _logit(d.answers["none"].p))
            fact_effects += [full - _logit(d.answers[f"wo{j}"].p) for j in range(len(facts))]
    out["counterfactual"] = {
        "n_forks": len(jacc), "slate_jaccard_after_removing_one_interaction": float(np.mean(jacc)),
        "top1_changed_rate": float(np.mean(top1_changed)),
        "mean_abs_belief_shift": float(np.mean(belief_shift)) if belief_shift else float("nan"),
        "shared_prefix_events": float(np.mean(shared_events))}
    out["evidence_attribution"] = {
        "n_items_with_facts": len(compr),
        "comprehensiveness_logit": float(np.mean(compr)) if compr else float("nan"),
        "mean_fact_effect_logit": float(np.mean(fact_effects)) if fact_effects else float("nan"),
        "share_of_facts_with_positive_effect": float(np.mean(np.array(fact_effects) > 0)) if fact_effects else float("nan"),
        "reading": "effects are changes in the log-odds of the recommended item when evidence is removed, "
                   "obtained by re-execution, so they are faithful to the model by construction"}
    backend.close()
    out_dir = os.path.join(cfg["out"], cfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "agent_study.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out
