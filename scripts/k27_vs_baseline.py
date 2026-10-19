"""Kev-27B (trained on the 8,000 records of rtm8k) in Decision4Rec-Full against the best trained baseline of each
domain and against the pool, fused as in the reports: change, 95% paired bootstrap interval and paired randomization p.
python logs/probe/k27_vs_baseline.py"""
import numpy as np

from claim_tests import listed
from equivalence_ci import base, ci, fused
from decision4rec.data.schema import load_cases

for d, ref in (("vg", "sasrecf"), ("mi", "sasrecf"), ("steam", "sasrec"), ("ind", "sasrecf")):
    b, cs = base(d, "rtm", ref)
    f = fused(d, "kev-27b-rtm8k", "rtm", "rtm")
    k = [c.cid for c in cs if c.cid in f]
    assert len(k) == len(cs), f"{d}: {len(k)} of {len(cs)} users answered"
    ci([f[i] for i in k], [b[i] for i in k], f"{d} Full 27B fused vs {ref}")
    ci([f[i] for i in k], [listed(c) for c in cs], f"{d} Full 27B fused vs the pool")
