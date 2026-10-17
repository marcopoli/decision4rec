"""Rankings must not depend on the hash seed of the Python process (found on 2026-10-16: the knowledge-graph agent broke
ties in the iteration order of a frozenset of strings, so two runs of the same request could differ)."""
import os
import subprocess
import sys

CODE = """
from types import SimpleNamespace as S
from decision4rec.agents.pool import kg_agent
ks = [("rel", c) for c in "abcdefgh"]                     # eight attributes, one item each: every item ties
kg = S(attrs=[frozenset(ks)] + [frozenset()] * 9, idf={k: 1.0 for k in ks}, ent_items={k: [i + 1] for i, k in enumerate(ks)})
print(kg_agent(kg, [0], 8))
"""


def test_kg_agent_does_not_depend_on_the_hash_seed():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outs = {subprocess.run([sys.executable, "-c", CODE], cwd=root, capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": str(seed), "PYTHONPATH": root}).stdout for seed in range(6)}
    assert len(outs) == 1, outs
