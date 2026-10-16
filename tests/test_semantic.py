import json
import os

import numpy as np
import pytest

SCHEMA = os.path.join(os.path.dirname(__file__), "..", "configs", "facets", "synthetic.yaml")
FAST = {"d": 16, "lr": 1e-3, "dropout": 0.1, "n_blocks": 1, "epochs": 2}


def test_facets_are_typed_checked_against_metadata_and_feed_sasrecf(workspace):
    pytest.importorskip("torch")
    from decision4rec.data.schema import Dataset
    from decision4rec.experiments.benchmark import _bench_dir, run_baselines
    from decision4rec.experiments.runner import load_config
    from decision4rec.semantic import facets as F
    from decision4rec.semantic.report import facet_report
    from decision4rec.semantic.run import run_facets

    probe = workspace("facets_probe")
    root = os.path.dirname(probe)
    fac = {"schema": SCHEMA, "dm": os.path.join(root, "facets", "dm.jsonl"), "emb": os.path.join(root, "facets", "emb.jsonl"),
           "embedder": {"kind": "hashing"}, "cache": os.path.join(root, "facets", "cache")}
    models = ["sasrecf", "sasrecf_dm", "sasrec_dm", "sasrecf_emb"]
    path = workspace("facets", facets=fac, benchmark={"baselines": models, "grids": {m: [FAST] for m in models},
                                                      "seeds": {"sasrecf_dm": [3, 4]}})
    cfg = load_config(path)
    ds = Dataset.load(cfg["dataset"])
    truth = np.load(os.path.join(cfg["dataset"], "truth.npz"))["group"]

    first = run_facets(path, limit=40)                            # resumable: a partial run, then the rest
    assert first["items_asked"] == 40
    rest = run_facets(path)
    assert rest["items_asked"] == ds.n_items - 40 and rest["items_total"] == ds.n_items
    answers = F.read_answers(fac["dm"])
    attrs = F.facet_attrs(answers, F.load_schema(SCHEMA), ds.n_items)
    assert all([f"dm:category", f"c{truth[i]}"] in attrs[i] for i in range(ds.n_items))
    cat = rest["summary"]["facets"]["category"]
    assert cat["metadata_check"] == {"items_with_label": ds.n_items, "agreement": 1.0} and cat["abstain_rate"] == 0.0
    assert rest["summary"]["facets"]["low_category"]["share_true"] == pytest.approx(float(np.mean(truth == 0)))
    emb = run_facets(path, embedding=True)
    assert emb["items"] == ds.n_items and "category" in emb["summary"]["facets"]

    run_baselines(path, device="cpu")
    bench = _bench_dir(cfg)
    for m in models:
        assert os.path.exists(os.path.join(bench, f"{m}.json")) and os.path.exists(os.path.join(bench, f"{m}.all.npz"))
    assert any(d.startswith("facets_") for d in os.listdir(os.path.join(cfg["out"], "baselines")))
    assert os.path.exists(os.path.join(bench, "sasrecf_dm.s4.json"))      # the extra seed has its own files
    assert not os.path.exists(os.path.join(bench, "sasrecf_dm.s3.json"))  # seed 3 is the config seed: the main run
    rep = facet_report([path])
    res = rep["datasets"][0]
    assert len(res["models"]["sasrecf_dm"]["seeds"]) == 2 and len(res["models"]["sasrecf"]["seeds"]) == 1
    assert set(res["models"]) == set(models) and "vs_ref" in res["models"]["sasrecf_dm"]
    z = np.load(os.path.join(bench, "sasrecf.all.npz"))
    with open(os.path.join(bench, "sasrecf.json")) as f:
        assert res["models"]["sasrecf"]["ndcg@10"] == pytest.approx(json.load(f)["all_users"]["ndcg@10"])
    assert len(z["users"]) == len(z["ranks"]) and 0.0 <= res["models"]["sasrecf_dm"]["vs_ref"]["p"] <= 1.0
    assert "| category | dm |" in rep["markdown"]


def test_relations_are_judged_once_per_anchor_and_extend_sasrec(workspace, tmp_path):
    pytest.importorskip("torch")
    from decision4rec.data.schema import Dataset
    from decision4rec.experiments.runner import get_cases, load_config
    from decision4rec.models.sasrec import SASRecRetriever, train_sasrec
    from decision4rec.semantic import relations as R
    from decision4rec.semantic.run import run_relation_eval, run_relations

    probe = load_config(workspace("rel_probe"))
    ds = Dataset.load(probe["dataset"])
    train_sasrec(ds.seqs, ds.n_items, str(tmp_path / "sasrec"), d=16, max_len=20, epochs=3, patience=3, device="cpu")
    rel = {"path": str(tmp_path / "rel.jsonl"), "last": 3, "n_cooc": 5, "n_content": 5,
           "embedder": {"kind": "hashing"}, "cache": str(tmp_path / "cache")}
    path = workspace("rel", retriever={"kind": "sasrec", "path": str(tmp_path / "sasrec")}, relations=rel)
    info = run_relations(path, None)
    assert info["anchors_judged"] > 0 and set(info["shares"]) == {"complement", "substitute", "series", "unrelated"}
    assert run_relations(path, None)["anchors_judged"] == 0                     # resumable, nothing left
    judged = R.read_relations(rel["path"])
    groups = np.load(os.path.join(probe["dataset"], "truth.npz"))["group"]
    for a, cands in list(judged.items())[:20]:                                   # the oracle's rule: same category
        for b, p in cands.items():
            assert (np.argmax(p) == 2) == (groups[a] == groups[b])

    rep = run_relation_eval([path])
    res = rep["datasets"][0]
    names = [r["variant"] for r in res["rows"]]
    assert names[0] == "sasrec (top 50)" and "relations, decision model" in names and "pair presence, same pairs" in names
    cases = get_cases(load_config(path), ds)["test"]
    retr = SASRecRetriever(str(tmp_path / "sasrec"))
    from decision4rec.eval import metrics as M
    plain = np.mean([M.per_case_metrics(M.target_rank(np.asarray(c.rscores), c.target_pos))["ndcg@10"] for c in cases])
    assert res["rows"][0]["ndcg@10"] == pytest.approx(plain)                     # weight 0 is SASRec's own ranking
    assert res["recall"]["top 50 + related"] >= res["recall"]["top 50"]
    scores = retr.score_items([cases[0].history], [cases[0].cands])[0]
    assert np.allclose(scores, cases[0].rscores, atol=1e-4)
    assert "## synthetic" in rep["markdown"]


def test_new_item_protocol_keeps_every_later_interaction_out_of_training(workspace, tmp_path):
    from collections import Counter

    from decision4rec.data.schema import Dataset
    from decision4rec.experiments.runner import get_cases, load_config
    from decision4rec.semantic.cold import make_cold_dataset, post_target_users
    from decision4rec.semantic.run import run_prepare_cold

    items = [{"id": f"i{i}", "title": f"Item {i} [item:{i}]", "meta": "", "desc": "", "attrs": [["category", f"c{i % 3}"]]}
             for i in range(20)]
    seqs = []
    for u in range(30):
        warm = [[(u + k) % 15, k, 4.0] for k in range(5)]
        late = 15 + u % 5 if u < 15 else (u + 7) % 15                      # half of the users end on a new item
        seqs.append(warm + [[late, 10, 4.0], [(u + 3) % 15, 11, 4.0]])
    src = Dataset(name="toy", items=items, seqs=seqs)
    src.save(str(tmp_path / "toy"))

    cold = make_cold_dataset(src, quantile=0.8, min_pre=3)
    info = cold.extra["cold_protocol"]
    assert info["cutoff"] == 10 and info["users_with_post_target"] == 30 and info["cold_targets"] == 15
    train = Counter(int(x[0]) for s in cold.seqs for x in s[:-2])
    assert all(x[1] < info["cutoff"] for s in cold.seqs for x in s[:-1])      # only the target may be later
    assert all(train.get(i, 0) == 0 for i in range(15, 20)) and all(len(s) == 6 for s in cold.seqs)
    out = run_prepare_cold(str(tmp_path / "toy"), str(tmp_path / "toy_cold"), 0.8, 3)
    assert out["cold_targets"] == 15 and run_prepare_cold(str(tmp_path / "toy"), str(tmp_path / "toy_cold"))["skipped"]

    ds = Dataset.load(str(tmp_path / "toy_cold"))
    assert post_target_users(ds) == set(range(30))
    cfg = load_config(workspace("cold", K=5, retriever={"kind": "itemknn"},
                                eval={"track": "B", "K": 5, "n_test": 12, "n_valid": 6, "seed": 5, "pool": 10, "target": "post"},
                                dataset=str(tmp_path / "toy_cold")))
    cases = get_cases(cfg, ds)
    assert len(cases["test"]) == 12 and all(c.target == ds.seqs[c.user][-1][0] for c in cases["test"])
    with pytest.raises(SystemExit):
        post_target_users(src)


def test_training_free_shortcut_baselines_run_through_the_benchmark(workspace, tmp_path):
    pytest.importorskip("torch")
    from decision4rec.data.schema import Dataset
    from decision4rec.experiments.benchmark import _bench_dir, run_baselines
    from decision4rec.experiments.runner import load_config
    from decision4rec.models import baselines as B

    names = ["seqrules", "contentknn", "transtext"]
    path = workspace("shortcuts", item_text={"embedder": {"kind": "hashing"}, "cache": str(tmp_path / "emb")},
                     benchmark={"baselines": names})
    summary = run_baselines(path, device="cpu")
    cfg = load_config(path)
    for n in names:
        assert 0.0 <= summary[n]["ndcg@10"] <= 1.0
        assert summary[n]["hp"] in B.DEFAULT_GRID[n]                     # chosen on validation from the grid
        assert os.path.exists(os.path.join(_bench_dir(cfg), f"{n}.all.npz"))
    ds = Dataset.load(cfg["dataset"])
    with pytest.raises(SystemExit):                                       # text baselines need embeddings
        B.Scorer("contentknn", ds)


def test_system_starts_from_the_backbone_and_adds_typed_decisions(workspace, tmp_path):
    pytest.importorskip("torch")
    from decision4rec.experiments.benchmark import run_baselines
    from decision4rec.experiments.runner import run_experiment
    from decision4rec.semantic.run import run_relations
    from decision4rec.semantic.system import system_eval, system_markdown

    rel = {"path": str(tmp_path / "rel.jsonl"), "last": 3, "n_cooc": 5, "n_content": 5,
           "embedder": {"kind": "hashing"}, "cache": str(tmp_path / "cache")}
    arms = [{"name": "retriever", "method": "retriever"}, {"name": "wholepool@p0", "method": "luce", "m": 0, "perm": 0}]
    path = workspace("system", K=15, relations=rel, arms=arms,          # its own candidate sets and baselines
                     item_text={"embedder": {"kind": "hashing"}, "cache": str(tmp_path / "cache")},
                     benchmark={"baselines": ["sasrec", "contentknn"], "grids": {"sasrec": [FAST]}})
    run_baselines(path, device="cpu")
    run_dir = run_experiment(path)
    run_relations(path, None)
    res = system_eval(path, run_dir, device="cpu")
    names = [r["variant"] for r in res["rows"]]
    assert names == ["backbone", "+ relations", "+ text prior (no decision model)", "+ decision gate",
                     "+ text prior + decision gate", "Decision4Rec (relations + text prior + decision gate)"]
    assert res["backbone"] == "sasrec"
    back = res["rows"][0]["groups"]["all"]["ndcg@10"]
    assert back == pytest.approx(res["refs"]["sasrec"]["all"])        # all weights 0: exactly the backbone's ranking
    assert res["coverage"]["pool + related + text neighbours"] >= res["coverage"]["pool"]
    assert {"sasrec", "contentknn"} <= set(res["refs"])
    for row in res["rows"][1:]:
        assert "backbone" in row["groups"]["all"]["vs"]
    assert "## synthetic" in system_markdown([res])
