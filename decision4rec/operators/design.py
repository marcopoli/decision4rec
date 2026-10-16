"""Block designs: which candidates are compared together, and in which order."""

from __future__ import annotations

import math

import numpy as np


def random_partition(items, block_size: int, rng: np.random.Generator) -> list[list[int]]:
    """Split `items` into randomly ordered blocks of near-equal size, at most `block_size` each."""
    items = np.asarray(list(items), dtype=int)
    if len(items) == 0:
        return []
    perm = rng.permutation(items)
    n_blocks = max(1, math.ceil(len(perm) / max(block_size, 2)))
    return [b.tolist() for b in np.array_split(perm, n_blocks) if len(b)]


def components(blocks: list[list[int]], n_items: int) -> list[list[int]]:
    """Connected components of the comparison hypergraph (items never compared are singletons)."""
    parent = list(range(n_items))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for b in blocks:
        for x in b[1:]:
            ra, rb = find(b[0]), find(x)
            if ra != rb:
                parent[rb] = ra
    groups: dict[int, list[int]] = {}
    for i in range(n_items):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())
