"""A new-item protocol (E3): items first seen after a time cutoff have no training interaction at all.

From a processed dataset, the cutoff T is a quantile of all interaction times. Every user keeps the
interactions before T; a user with at least one interaction after T also keeps the first of them, which
becomes the test target. Users with fewer than `min_pre` interactions before T are dropped, so no
interaction after T enters any training sequence (the training part is all but the last two items). The
item list is unchanged, so item facets computed on the full dataset still apply.

Test targets after T are either warm (the item was bought before T) or cold (never seen before T). An ID
model cannot rank a cold item from interactions; a model that reads the item can.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from ..data.schema import Dataset


def make_cold_dataset(ds: Dataset, quantile: float = 0.9, min_pre: int = 3) -> Dataset:
    times = np.array([x[1] for s in ds.seqs for x in s], dtype=float)
    cutoff = float(np.quantile(times, quantile))
    seqs, post_users = [], 0
    for s in ds.seqs:
        pre = [x for x in s if x[1] < cutoff]
        post = [x for x in s if x[1] >= cutoff]
        if len(pre) < min_pre:
            continue
        if post:
            post_users += 1
            seqs.append(pre + post[:1])
        else:
            seqs.append(pre)
    train = Counter(int(x[0]) for s in seqs for x in s[:-2])
    targets = [int(s[-1][0]) for s in seqs if s[-1][1] >= cutoff]
    extra = {**ds.extra, "cold_protocol": {"source": ds.name, "quantile": quantile, "cutoff": cutoff,
                                           "min_pre": min_pre, "users_with_post_target": post_users,
                                           "cold_targets": int(sum(train.get(t, 0) == 0 for t in targets)),
                                           "items_without_training_interaction": int(sum(train.get(i, 0) == 0
                                                                                         for i in range(ds.n_items)))}}
    return Dataset(name=f"{ds.name}_cold", items=ds.items, seqs=seqs, user_tag=ds.user_tag, extra=extra)


def post_target_users(ds: Dataset) -> set:
    """Users whose last item is after the cutoff of a dataset built by `make_cold_dataset`."""
    info = ds.extra.get("cold_protocol")
    if not info:
        raise SystemExit(f"{ds.name} is not a dataset of the new-item protocol (no cold_protocol in its meta)")
    return {u for u, s in enumerate(ds.seqs) if s and s[-1][1] >= info["cutoff"]}
