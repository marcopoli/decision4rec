import os

import pytest


def test_confirmatory_cases_use_only_fresh_users(workspace):
    import json

    from decision4rec.data.schema import Dataset
    from decision4rec.experiments.runner import case_paths, get_cases, load_config

    first = load_config(workspace("fresh_a", K=20))
    ds = Dataset.load(first["dataset"])
    old = get_cases(first, ds)
    used = {c.user for split in old.values() for c in split}
    ev = {"track": "B", "K": 20, "n_test": 50, "n_valid": 30, "seed": 29, "pool": 80, "exclude_cases": "all"}
    fresh_cfg = load_config(workspace("fresh_b", K=20, eval=ev))
    fresh = get_cases(fresh_cfg, ds)
    users = {c.user for split in fresh.values() for c in split}
    assert users and not users & used
    assert not {c.user for c in fresh["test"]} & {c.user for c in fresh["valid"]}
    with open(case_paths(fresh_cfg)["info"]) as f:
        info = json.load(f)
    assert info["test"]["excluded_users"] >= len(used) > 0
    # an explicit list must name existing files
    bad = load_config(workspace("fresh_c", K=20, eval={**ev, "seed": 31, "exclude_cases": ["no/such/file.jsonl"]}))
    with pytest.raises(SystemExit):
        get_cases(bad, ds)
    assert not os.path.exists(case_paths(bad)["test"])
