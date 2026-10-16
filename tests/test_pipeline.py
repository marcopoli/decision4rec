import json
import os

import numpy as np
import pytest
import yaml

from decision4rec.data.loaders import k_core, make_synthetic
from decision4rec.eval import metrics as M
from decision4rec.operators.risk import conformal_report, risk_coverage


def test_k_core_removes_sparse_users_and_items():
    inter = [(u, i, 0, 5.0) for u in range(4) for i in range(4)] + [(9, 0, 0, 5.0)]
    kept = k_core(inter, 3)
    assert all(r[0] != 9 for r in kept) and len(kept) == 16


def test_metrics_basics():
    assert M.target_rank(np.array([0.1, 0.9, 0.5]), 2) == 2
    assert M.target_rank(np.array([0.5, 0.5, 0.5]), 0) == 3          # ties count against the target
    assert np.isinf(M.target_rank(np.array([1.0, 2.0]), -1))
    assert M.per_case_metrics(1)["ndcg@10"] == pytest.approx(1.0)
    assert M.per_case_metrics(11)["ndcg@10"] == 0.0
    assert M.auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert M.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    conf = np.array([0.9] * 50 + [0.6] * 50)
    correct = np.array([1] * 45 + [0] * 5 + [1] * 30 + [0] * 20)
    assert M.ece(conf, correct) < 0.01


def test_temperature_scaling_recovers_known_temperature():
    rng = np.random.default_rng(0)
    logits, labels = [], []
    for _ in range(600):
        z = rng.normal(0, 2, 8)
        p = np.exp(z) / np.exp(z).sum()
        labels.append(int(rng.choice(8, p=p)))
        logits.append(z * 3.0)                    # over-confident by a factor of three
    assert M.fit_temperature(logits, labels) == pytest.approx(3.0, rel=0.2)


def test_conformal_sets_reach_target_coverage():
    rng = np.random.default_rng(1)

    def sample(n):
        out = []
        for _ in range(n):
            z = rng.normal(0, 1.5, 10)
            p = np.exp(z) / np.exp(z).sum()
            out.append((p, int(rng.choice(10, p=p))))
        return out

    rep = conformal_report(sample(800), sample(800), alphas=(0.1, 0.3))
    for r in rep:
        assert abs(r["coverage"] - r["target_coverage"]) < 0.05
    assert rep[0]["mean_set_size"] > rep[1]["mean_set_size"]
    rc = risk_coverage(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0.0, 0.0, 1.0, 1.0]))
    assert rc["risk"][0] == 0.0 and rc["risk"][-1] == 0.5


def test_run_and_report_end_to_end(workspace):
    from decision4rec.experiments.report import report
    from decision4rec.experiments.runner import run_experiment

    arms = [{"name": "retriever", "method": "retriever"}, {"name": "pointwise", "method": "pointwise"},
            {"name": "wholepool@p0", "method": "luce", "m": 0, "perm": 0},
            {"name": "wholepool@p1", "method": "luce", "m": 0, "perm": 1},
            {"name": "prior", "method": "luce", "m": 0, "perm": 0, "null_state": True},
            {"name": "luce_m5_r1", "method": "luce", "m": 5, "rounds": 1},
            {"name": "luce_m5_r3_out", "method": "luce", "m": 5, "rounds": 3, "outside": True,
             "evidence": "kg+cf", "profile": True},
            {"name": "graph", "method": "graph"},
            {"name": "luce_m5_r2_um", "method": "luce", "m": 5, "rounds": 2, "evidence": "pref+cf",
             "profile": "decision", "intent": True}]
    path = workspace("e2e", arms=arms, reference="wholepool@p0")
    out = run_experiment(path)
    rep = report(out)
    rows = {r["variant"]: r for r in rep["table"]}
    assert rows["wholepool@p0"]["ndcg@10"] > rows["retriever"]["ndcg@10"]      # the oracle knows the user
    assert rows["wholepool@p0"]["ndcg@10"] > rows["prior"]["ndcg@10"]          # history matters
    assert rows["luce_m5_r1"]["requests"] == 2.0                               # partition plus one bridge
    assert rows["luce_m5_r3_out"]["requests"] == 1.0                           # a whole design in one request
    assert {"wholepool@ens", "wholepool@ens+pos", "wholepool@p0+pc", "wholepool@p0+fuse"} <= set(rows)
    beta = rep["position_beta"]["wholepool"]
    assert beta[1] == pytest.approx(-0.35 * np.log(2), abs=0.08)               # injected primacy bias
    assert rep["order_sensitivity"][0]["kendall_tau"] < 1.0
    cal = {r["variant"]: r for r in rep["calibration"]}
    assert "outside_auroc" in cal["luce_m5_r3_out"]
    # decision-native user model: graph scorer, side signals and the quality of the preference edges
    assert {"graph", "graph+fuse", "luce_m5_r2_um+graph", "luce_m5_r2_um+intent"} <= set(rows)
    assert rows["graph"]["requests"] == 1.0 and rows["luce_m5_r2_um"]["requests"] == 3.0
    assert rows["graph"]["ndcg@10"] > rows["prior"]["ndcg@10"]
    quality = rep["preference_quality"][0]
    assert quality["auroc_decision"] > max(quality["auroc_support"], 0.55)   # the oracle knows the tastes
    assert quality["ece_platt"] < quality["ece_raw"]
    assert os.path.exists(os.path.join(out, "report.md"))
    again = run_experiment(path)                                               # resumable: nothing recomputed
    with open(os.path.join(again, "luce_m5_r1.test.jsonl")) as f:
        assert len(f.readlines()) == 60


def test_audit_detects_injected_similarity_effect(workspace):
    from decision4rec.experiments.audit import run_audit

    iia = run_audit(workspace("audit_iia", backend={"similarity": 0.0, "pos_bias": 0.3}), n_cases=30, block=6)
    sub = run_audit(workspace("audit_sim", backend={"similarity": 0.5, "pos_bias": 0.3}), n_cases=30, block=6)
    assert iia["isolation"]["n_questions"] > 0 and iia["isolation"]["max_abs_difference"] == 0.0   # the oracle is isolated
    assert iia["mean_abs_deviation"] < 1e-6 and iia["regularity_violation_rate"] == 0.0
    assert sub["similarity_slope_mean"] < -0.1 and sub["similarity_slope_ci95"][1] < 0


def test_rectune_export_is_valid_and_leak_free(workspace):
    from decision4rec.data.schema import load_cases
    from decision4rec.experiments.rectune import export_rectune
    from decision4rec.experiments.runner import case_paths, load_config

    path = workspace("rectune")
    out = os.path.join(os.path.dirname(path), "rt")
    info = export_rectune(path, out, n_train=60, n_heldout=10, block=6, outside_rate=0.3)
    cfg = load_config(path)
    eval_users = {c.user for s in ("test", "valid") for c in load_cases(case_paths(cfg)[s])}
    labels = []
    with open(info["train"]["path"]) as f:
        for line in f:
            rec = json.loads(line)
            q = rec["questions"]["next"]
            assert q["label"] in q["criteria"] and isinstance(rec["questions"]["like"]["label"], bool)
            assert not any(f"[user:{u}]" in rec["state"] for u in eval_users)
            labels.append(q["label"])
    assert 0 < labels.count("none") < len(labels)


def test_user_model_on_the_graph_matches_the_batch_user_model(workspace):
    pytest.importorskip("activegraph")
    from decision4rec.agents.active import DecisionAgent
    from decision4rec.data.schema import Dataset
    from decision4rec.decision.base import build_backend
    from decision4rec.experiments.runner import get_cases, load_config, make_prompt_builder

    cfg = load_config(workspace("um", K=50))
    ds = Dataset.load(cfg["dataset"])
    pb = make_prompt_builder(cfg, ds)
    case = get_cases(cfg, ds)["test"][0]
    backend = build_backend({k: v for k, v in cfg["backend"].items() if k != "cache"})
    agent = DecisionAgent(backend, pb, tau_expand=float("inf"))
    res = agent.run(case)
    batch = pb.user_model(backend, case)
    assert len(batch.prefs) > 0
    for q in batch.prefs:                                   # same hypotheses, same strengths, same support
        belief = res.profile[f"{q.rel}\t{q.ent}"]
        assert belief["p"] == pytest.approx(q.p) and belief["support"] == q.support and not belief["stale"]
    # the relation behaviour fired exactly once per repeated attribute occurrence, not once per belief
    kg, hist = pb.kg, pb.profile_history(case)
    occurrences = sum(1 for h in hist for k in kg.attrs[h] if k in kg.idf)
    assert res.events.count("relation_behavior.started") == occurrences - len(res.profile)
    assert res.events.count("interaction.added") == len(hist)
    state = pb.state(case, profile="decision", prof=batch, intent=res.intent)
    assert "User preference profile" in state and "Likely next kind of item" in state
    assert sum(res.intent.values()) == pytest.approx(1.0)
    top = batch.top(1)[0]
    carrier = next(i for i in range(ds.n_items) if (top.rel, top.ent) in kg.attrs[i])
    assert "matches the user's preference" in pb.option(case, carrier, evidence="pref", prof=batch)


def test_agent_study_expansion_incremental_model_replay_and_forks(workspace):
    pytest.importorskip("activegraph")
    from decision4rec.experiments.agent_study import run_agent_study

    path = workspace("agent", K=50, backend={"outside_utility": 2.0}, agent={"k0": 20, "step": 15, "max_rounds": 2})
    out = run_agent_study(path, n_cases=20, n_deep=6)
    pol = out["policies"]
    assert pol["never"]["examined"] == 20 and pol["always"]["examined"] == 50
    assert pol["never"]["tokens"] < pol["adaptive_q50"]["tokens"] < pol["always"]["tokens"]
    assert pol["always"]["target_examined_rate"] >= pol["never"]["target_examined_rate"]
    assert pol["always"]["graph_candidates"] > 0 and pol["never"]["graph_candidates"] == 0
    abl = out["ablation_always_expand"]
    assert abl["no_user_model"]["targets_recovered_by_graph"] == 0
    assert abl["full"]["target_examined_rate"] >= abl["user_model_pool_only"]["target_examined_rate"]
    inc = out["incremental_user_model"]
    assert inc["coverage_of_full_rebuild_hypotheses"] == 1.0
    assert inc["tokens_incremental_refresh"] < inc["tokens_full_rebuild"]
    assert out["reproducibility"]["identical_slates"] == 1.0
    assert out["reproducibility"]["identical_event_sequences"] == 1.0
    assert out["counterfactual"]["n_forks"] == 18


def test_sasrec_trains_and_retrieves(tmp_path):
    pytest.importorskip("torch")
    from decision4rec.models.sasrec import SASRecRetriever, train_sasrec

    ds, _ = make_synthetic(n_users=120, n_items=60, seed=2)
    info = train_sasrec(ds.seqs, ds.n_items, str(tmp_path), d=16, max_len=20, epochs=3, patience=3,
                        batch_size=64, device="cpu")
    assert 0.0 <= info["best_valid_ndcg10"] <= 1.0
    r = SASRecRetriever(str(tmp_path), device="cpu")
    hist = [[int(x[0]) for x in s[:-1]] for s in ds.seqs[:5]]
    idx, val = r.topk(hist, 10)
    assert idx.shape == (5, 10) and np.all(np.diff(val, axis=1) <= 1e-6)
    assert all(i not in h for h, row in zip(hist, idx) for i in row)
    sampled = train_sasrec(ds.seqs, ds.n_items, str(tmp_path / "s"), d=16, max_len=20, epochs=1, patience=1,
                           batch_size=64, device="cpu", n_sampled=32)
    assert sampled["sampled_softmax"] == 32


def test_amazon_and_movielens_loaders(tmp_path):
    import gzip

    from decision4rec.data.loaders import load_amazon23, load_ml1m
    from decision4rec.data.schema import Dataset

    reviews, meta = tmp_path / "Video_Games.jsonl.gz", tmp_path / "meta_Video_Games.jsonl.gz"
    with gzip.open(reviews, "wt") as f:
        for u in range(6):
            for i in range(6):
                f.write(json.dumps({"user_id": f"U{u}", "parent_asin": f"B{i}", "asin": f"B{i}", "rating": 4.0,
                                    "timestamp": 1000 * u + 10 * ((i + u) % 6), "title": "t", "text": "x"}) + "\n")
        f.write(json.dumps({"user_id": "U0", "parent_asin": "B0", "rating": 1.0, "timestamp": 99999}) + "\n")
        f.write(json.dumps({"user_id": "Ulonely", "parent_asin": "B0", "rating": 5.0, "timestamp": 5}) + "\n")
    with gzip.open(meta, "wt") as f:
        for i in range(6):
            f.write(json.dumps({"parent_asin": f"B{i}", "title": f"Game {i}", "main_category": "Video Games",
                                "categories": ["Video Games", "PC"], "description": ["A  fine\n game."],
                                "store": "Nintendo", "details": {"Developer": "Studio A", "Platform": "PC"}}) + "\n")
    ds = load_amazon23(str(reviews), str(meta), "video_games", core=5)
    assert ds.n_users == 6 and ds.n_items == 6
    assert all(len(s) == 6 for s in ds.seqs)                        # duplicate review dropped, earliest kept
    assert all(s == sorted(s, key=lambda x: x[1]) for s in ds.seqs)  # chronological
    assert ["developer", "Studio A"] in ds.items[0]["attrs"] and ["store", "Nintendo"] in ds.items[0]["attrs"]
    assert "A fine game." in ds.item_text(0) and "Game 0" not in ds.item_text(0, mask_title=True)
    ds.save(str(tmp_path / "proc"))
    assert Dataset.load(str(tmp_path / "proc")).seqs == ds.seqs

    ml = tmp_path / "ml-1m"
    ml.mkdir()
    (ml / "movies.dat").write_text("\n".join(f"{i}::Movie {i} (199{i})::Action|Comedy" for i in range(1, 7)),
                                   encoding="latin-1")
    (ml / "ratings.dat").write_text("\n".join(f"{u}::{i}::{3 + (u + i) % 3}::{100 * u + i}"
                                              for u in range(1, 7) for i in range(1, 7)), encoding="latin-1")
    d2 = load_ml1m(str(ml), core=5)
    assert d2.n_users == 6 and d2.n_items == 6 and ["decade", "1990s"] in d2.items[0]["attrs"]
    assert d2.item_text(0).startswith("Movie 1 (1991) | Genres: Action, Comedy")
