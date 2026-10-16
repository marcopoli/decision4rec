import gzip
import json
import os

import numpy as np
import pytest

from decision4rec.data.loaders import load_mind, load_steam, make_synthetic


def test_steam_loader_reads_python_dict_lines(tmp_path):
    reviews, games = tmp_path / "steam_reviews.json.gz", tmp_path / "steam_games.json.gz"
    with gzip.open(reviews, "wt") as f:
        for u in range(6):
            for g in range(6):
                f.write(repr({u"username": f"user{u}", u"product_id": str(100 + g), u"date": f"2018-01-{1 + (g + u) % 6:02d}",
                              u"text": "fun", u"hours": 3.5}) + "\n")
        f.write("this line is not a dict\n")
    with gzip.open(games, "wt") as f:
        for g in range(6):
            f.write(repr({u"id": str(100 + g), u"app_name": f"Game {g}", u"genres": [u"Action", u"Indie"],
                          u"tags": [u"Action", u"Roguelike"], u"developer": u"Studio A", u"publisher": u"Pub B"}) + "\n")
    ds = load_steam(str(reviews), str(games), core=5)
    assert ds.n_users == 6 and ds.n_items == 6 and all(len(s) == 6 for s in ds.seqs)
    assert all([x[1] for x in s] == sorted(x[1] for x in s) for s in ds.seqs)
    attrs = ds.items[0]["attrs"]
    assert ["genre", "Action"] in attrs and ["tag", "Roguelike"] in attrs and ["tag", "Action"] not in attrs
    assert ["developer", "Studio A"] in attrs and ds.item_text(0).startswith("Game 0 | Genres: Action, Indie")


def test_mind_loader_builds_click_sequences(tmp_path):
    folder = tmp_path / "MINDsmall_train"
    folder.mkdir()
    ents = json.dumps([{"Label": "Lionel Messi", "WikidataId": "Q615"}])
    (folder / "news.tsv").write_text("\n".join(
        f"N{i}\tsports\tsoccer\tTitle {i}\tAbstract {i}\thttps://x/{i}\t{ents}\t[]" for i in range(8)), encoding="utf-8")
    rows = []
    for u in range(6):
        history = " ".join(f"N{(u + k) % 8}" for k in range(4))
        shown = f"N{(u + 4) % 8}-1 N{(u + 5) % 8}-0 N{(u + 6) % 8}-1"
        rows.append(f"{u}\tU{u}\t11/1{u}/2019 9:05:58 AM\t{history}\t{shown}")
    (folder / "behaviors.tsv").write_text("\n".join(rows), encoding="utf-8")
    ds = load_mind([str(folder)], core=3)
    assert ds.n_users == 6 and all(len(s) == 6 for s in ds.seqs)            # 4 history clicks + 2 new clicks
    assert ["entity", "Lionel Messi"] in ds.items[0]["attrs"] and ["category", "sports"] in ds.items[0]["attrs"]
    assert all([x[1] for x in s] == sorted(x[1] for x in s) for s in ds.seqs)


def test_every_baseline_trains_and_ranks_the_whole_catalogue(tmp_path):
    pytest.importorskip("torch")
    from decision4rec.models import baselines as B

    ds, _ = make_synthetic(n_users=150, n_items=80, seed=3)
    users = list(range(30))
    hist = [[int(x[0]) for x in ds.seqs[u][:-1]] for u in users]
    targets = [int(ds.seqs[u][-1][0]) for u in users]
    for name in B.MODELS:
        grid = None if name in B.TRAINING_FREE else [{**B.DEFAULT_GRID[name][0], "d": 16, "epochs": 6, "patience": 2},
                                                     {**B.DEFAULT_GRID[name][0], "d": 24, "epochs": 6, "patience": 2}]
        best = B.tune(name, ds, str(tmp_path), grid, common={"max_len": 20}, device="cpu")
        scorer = B.load(name, best["dir"], ds, "cpu")
        ranks = scorer.ranks(users, hist, targets)
        assert ranks.shape == (30,) and ranks.min() >= 1 and ranks.max() <= ds.n_items - 13
        sc = scorer.scores(users[:2], hist[:2])
        assert tuple(sc.shape) == (2, ds.n_items)
        if name in B.SEQUENTIAL:
            assert best["hp"]["max_len"] == 20
    calls = []
    original = B.fit
    B.fit = lambda *a, **k: calls.append(1) or original(*a, **k)
    try:
        B.tune("sasrec", ds, str(tmp_path), [{**B.DEFAULT_GRID["sasrec"][0], "d": 16, "epochs": 6, "patience": 2}],
               common={"max_len": 20}, device="cpu")
    finally:
        B.fit = original
    assert calls == []                                                       # finished configurations are reused


def test_benchmark_joins_baselines_and_the_fixed_system(workspace):
    pytest.importorskip("torch")
    from decision4rec.experiments.benchmark import benchmark, run_baselines
    from decision4rec.experiments.runner import run_experiment

    jev = dict(method="luce", m=10, rounds=2, outside=True, profile="decision", evidence="pref", intent=True)
    quick = [{"d": 16, "lr": 0.003, "epochs": 6, "patience": 2}]
    path = workspace("bench", K=30, reference="wholepool@p0",
                     benchmark={"system": "decision4rec+fuse", "report": ["retriever", "wholepool@p0", "decision4rec"],
                                "baselines": ["pop", "itemknn", "sasrec", "sasrecf"],
                                "grids": {"sasrec": quick, "sasrecf": quick}},
                     arms=[{"name": "retriever", "method": "retriever"},
                           {"name": "wholepool@p0", "method": "luce", "m": 0, "perm": 0},
                           {"name": "decision4rec", **jev}])
    base = run_baselines(path, device="cpu")
    assert set(base) == {"pop", "itemknn", "sasrec", "sasrecf"}
    run_experiment(path)
    out = benchmark([path], out=os.path.dirname(path))
    res = out["datasets"][0]
    rows = {r["method"]: r for r in res["table"]}
    assert {"pop", "itemknn", "sasrec", "sasrecf", "retriever", "wholepool@p0", "decision4rec", "decision4rec+fuse"} == set(rows)
    assert rows["decision4rec+fuse"]["group"] == "Decision4Rec" and rows["sasrec"]["group"] == "trained baseline"
    assert res["n_users"] == 60 and res["reference_baseline"] in base
    assert 0.0 <= res[f"p_value_vs_{res['reference_baseline']}"] <= 1.0
    assert rows["decision4rec+fuse"]["ndcg@10"] > rows["pop"]["ndcg@10"]          # the oracle knows the user
    # a two-stage system cannot hit a target outside its pool: its hit rate is bounded by pool recall
    with open(os.path.join(os.path.dirname(path), "runs", "bench", "ranks.json")) as f:
        ranks = json.load(f)
    in_pool = np.mean([v is not None for v in ranks["retriever"]["test"].values()])
    assert rows["decision4rec+fuse"]["hr@20"] <= in_pool + 1e-9
    assert os.path.exists(out["summary_path"]) and "mean rank" in out["markdown"]


def test_movielens_csv_loader_and_case_limit(tmp_path, workspace):
    from decision4rec.data.loaders import load_ml_csv
    from decision4rec.experiments.report import report
    from decision4rec.experiments.runner import run_experiment

    folder = tmp_path / "ml-20m"
    folder.mkdir()
    (folder / "movies.csv").write_text("movieId,title,genres\n" + "\n".join(
        f'{i},"Movie, The {i} (199{i})",Action|Comedy' for i in range(1, 7)), encoding="utf-8")
    (folder / "ratings.csv").write_text("userId,movieId,rating,timestamp\n" + "\n".join(
        f"{u},{i},{3 + (u + i) % 3}.0,{100 * u + i}" for u in range(1, 7) for i in range(1, 7)), encoding="utf-8")
    (folder / "tags.csv").write_text("userId,movieId,tag,timestamp\n" + "\n".join(
        f"{u},1,Dark Comedy,{u}" for u in range(1, 5)) + "\n1,2,seen once,9", encoding="utf-8")
    ds = load_ml_csv(str(folder), "ml20m", core=5)
    assert ds.n_users == 6 and ds.n_items == 6 and ds.name == "ml20m"
    first = ds.items[0]
    assert first["title"] == "Movie, The 1 (1991)" and ["tag", "dark comedy"] in first["attrs"]
    assert not any(a[0] == "tag" for a in ds.items[1]["attrs"])            # a tag applied once is not an attribute
    assert ds.seqs[0][0][2] in (3.0, 4.0, 5.0)

    path = workspace("limit", arms=[{"name": "retriever", "method": "retriever"},
                                    {"name": "wholepool@p0", "method": "luce", "m": 0, "perm": 0},
                                    {"name": "prior", "method": "luce", "m": 0, "perm": 0, "null_state": True, "max_cases": 15}],
                     reference="wholepool@p0")
    rows = {r["variant"]: r for r in report(run_experiment(path))["table"]}
    assert rows["prior"]["n"] == 15 and rows["wholepool@p0"]["n"] == 60
    assert rows["wholepool@p0+pc"]["n"] == 15                              # the control pairs with the cases it has
