import json
import os

import numpy as np
import pytest

from decision4rec.eval import metrics as M

pytest.importorskip("activegraph")

AGENTS = ["sequential", "content", "knowledge", "collaborative", "profile", "graph", "popularity"]


def _world(workspace, name, **extra):
    from decision4rec.data.schema import Dataset
    from decision4rec.decision.base import build_backend
    from decision4rec.experiments.runner import build_retriever, get_cases, load_config, make_prompt_builder
    from decision4rec.retrieval.agents import build_multi_retriever

    cfg = load_config(workspace(name, K=50, backend={"outside_utility": 2.0}, **extra))
    ds = Dataset.load(cfg["dataset"])
    pb = make_prompt_builder(cfg, ds)
    pb.profile_hist = 6                      # shorter than the histories: the older items form the session prefix
    cases = get_cases(cfg, ds)
    backend = build_backend({k: v for k, v in cfg["backend"].items() if k != "cache"})
    multi = build_multi_retriever({"agents": AGENTS, "embedder": {"kind": "hashing"}, "writer": {"kind": "echo"}},
                                  ds, build_retriever({"kind": "itemknn"}, ds), pb.kg)
    return pb, cases, backend, multi


def test_agent_fuses_the_retrieval_score_and_retrieves_from_the_whole_history(workspace):
    from decision4rec.agents.active import DecisionAgent

    pb, cases, backend, multi = _world(workspace, "fuse")
    case = cases["test"][0]
    assert len(case.history) > pb.profile_hist
    common = dict(tau_expand=float("-inf"), retrieval=multi, route="uniform", pool=40, k0=20, m=5)
    plain = DecisionAgent(backend, pb, **common).run(case)
    fused = DecisionAgent(backend, pb, gamma=0.5, **common).run(case)
    assert plain.trace == fused.trace and len(plain.trace) == 3           # the weight does not change the loop
    active = fused.trace[-1]["active"]
    assert np.allclose(plain.scores[active], plain.theta[active])
    expected = M.zscore(fused.theta[active]) + 0.5 * M.zscore(fused.rscores[active])
    assert np.allclose(fused.scores[active], expected)
    assert fused.slate == [fused.items[j] for j in sorted(active, key=lambda j: -fused.scores[j])[:10]]
    assert plain.slate != fused.slate or np.allclose(M.zscore(fused.rscores[active]), 0)
    # only the last interactions build the user model, but no item of the whole history is proposed again
    assert plain.events.count("interaction.added") == pb.profile_hist
    assert not set(fused.items) & set(case.history)
    assert any(key[0] == tuple(case.history) for key in multi.agents[0]._memo)   # the sequential agent read all of it
    assert fused.n_pool == 40 and fused.rscores[:40].min() > 0 and len(fused.items) > 40


def test_replay_of_a_recorded_run_equals_real_runs(workspace):
    from decision4rec.agents.active import DecisionAgent
    from decision4rec.experiments.agent_bench import _rank, _trace_record, replay

    pb, cases, backend, multi = _world(workspace, "replay")
    common = dict(retrieval=multi, route="tuned", pool=40, k0=20, m=5)
    always = DecisionAgent(backend, pb, tau_expand=float("-inf"), **common)
    recs = {c.cid: _trace_record(always.run(c), c) for c in cases["valid"][:8]}
    p0 = [r["trace"][0]["p_out"] for r in recs.values()]
    stopped = set()
    for tau in (float("inf"), float(np.median(p0)), float("-inf")):
        for gamma in (0.0, 0.5):
            agent = DecisionAgent(backend, pb, tau_expand=tau, gamma=gamma, **common)
            for c in cases["valid"][:8]:
                real, again = agent.run(c), replay(recs[c.cid], tau, gamma)
                assert _rank(real, c) == again["rank"]
                assert (real.rounds, real.tokens, real.requests, real.n_examined) == (
                    again["round"], again["tokens"], again["requests"], again["examined"])
                stopped.add(real.rounds)
    assert {0, 2} <= stopped                                               # the threshold decides who expands


def test_agent_bench_selects_on_validation_and_becomes_the_primary_benchmark_row(workspace):
    pytest.importorskip("torch")
    from decision4rec.experiments.agent_bench import run_agent_bench
    from decision4rec.experiments.benchmark import benchmark, run_baselines
    from decision4rec.experiments.runner import run_experiment

    multi = {"kind": "multi", "tag": "multi", "base": {"kind": "itemknn"}, "embedder": {"kind": "hashing"},
             "writer": {"kind": "echo"}, "weights": "tuned", "tune_users": 60,
             "agents": ["sequential", "content", "knowledge", "collaborative", "graph", "popularity"]}
    jev = dict(method="luce", m=10, rounds=2, outside=True, profile="decision", evidence="pref", intent=True)
    path = workspace("abench", K=30, backend={"outside_utility": 1.0}, retriever=multi,
                     benchmark={"system": "decision4rec+fuse", "report": ["retriever", "decision4rec"], "baselines": ["pop", "itemknn"]},
                     arms=[{"name": "retriever", "method": "retriever"}, {"name": "decision4rec", **jev}])
    run_baselines(path, device="cpu")
    run_dir = run_experiment(path)
    out = run_agent_bench(path)
    st = out["agent"]
    assert (st["k0"], st["pool"], st["m"], st["rounds"]) == (30, 60, 10, 2) and st["routes"] == ["tuned", "decision"]
    assert out["agents"] == AGENTS
    from decision4rec.experiments.report import GAMMAS
    assert len(out["selection"]) == 2 * 5 * (1 + len(GAMMAS))
    assert out["chosen"]["valid_ndcg@10"] == max(g["valid_ndcg@10"] for g in out["selection"])
    assert out["replay_check"] == {"n": 20, "identical": 20}
    assert len(out["ranks"]["test"]) == 60 and len(out["ranks"]["valid"]) == 40
    assert out["test"]["requests"] >= 3 and out["test"]["events_per_case"] > 20
    json.loads(open(os.path.join(run_dir, "agent_bench.json")).read(), parse_constant=lambda c: pytest.fail(c))

    res = benchmark([path], out=os.path.dirname(path))["datasets"][0]
    rows = {r["method"]: r for r in res["table"]}
    assert res["system"] == "decision4rec_agent" and res["systems"] == ["decision4rec_agent", "decision4rec+fuse"]
    assert rows["decision4rec_agent"]["group"] == "Decision4Rec, agent on ActiveGraph"
    assert rows["decision4rec+fuse"]["group"] == "Decision4Rec, batch without the agent loop"
    assert rows["decision4rec_agent"]["ndcg@10"] == pytest.approx(out["test"]["ndcg@10"])
    assert rows["decision4rec_agent"]["ndcg@10"] > rows["pop"]["ndcg@10"]
    pairs = [(c["a"], c["b"]) for c in res["comparisons"]]
    ref = res["reference_baseline"]
    assert pairs == [("decision4rec_agent", ref), ("decision4rec+fuse", ref), ("decision4rec_agent", "decision4rec+fuse")]
    assert all(c["p_holm"] >= c["p_value"] for c in res["comparisons"])

    # a second call reuses every stored run, and a partial run writes its own files
    lines = open(os.path.join(run_dir, "agent.test.jsonl")).read()
    again = run_agent_bench(path)
    assert open(os.path.join(run_dir, "agent.test.jsonl")).read() == lines and again["ranks"] == out["ranks"]
    part = run_agent_bench(path, max_cases=10)
    assert part["n_test"] == 10 and not part["complete"]
    assert open(os.path.join(run_dir, "agent.test.jsonl")).read() == lines
    assert os.path.exists(os.path.join(run_dir, "agent_bench_partial.json"))
