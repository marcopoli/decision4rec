import os

import numpy as np
import pytest
import yaml

from decision4rec.data.loaders import make_synthetic


@pytest.fixture(scope="session")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("ws")
    ds, truth = make_synthetic(n_users=260, n_items=150, seed=1)
    data = str(root / "data")
    ds.save(data)
    np.savez(os.path.join(data, "truth.npz"), **truth)

    def config(name, backend=None, K=20, arms=None, **extra):
        be = {"kind": "oracle", "truth": os.path.join(data, "truth.npz"), "pos_bias": 0.35,
              "cache": str(root / f"{name}.sqlite")}
        be.update(backend or {})
        cfg = {"name": name, "dataset": data, "out": str(root / "runs"), "retriever": {"kind": "itemknn"},
               "backend": be, "eval": {"track": "B", "K": K, "n_test": 60, "n_valid": 40, "seed": 3, "pool": 80},
               "arms": arms or [{"name": "retriever", "method": "retriever"}]}
        cfg.update(extra)
        path = str(root / f"{name}.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f)
        return path

    return config
