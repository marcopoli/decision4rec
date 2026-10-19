"""Decision4Rec-KG with the 4B decision model (R10) against the best trained baseline of each domain, fused as in the
reports: change, 95% paired bootstrap interval and paired randomization p. python logs/probe/kg4b_vs_baseline.py"""
from equivalence_ci import base, ci, fused

for d, ref in (("vg", "sasrecf"), ("mi", "sasrecf"), ("steam", "sasrec"), ("ind", "sasrecf")):
    b, cs = base(d, "ev2", ref)
    f = fused(d, "kev-4b-ev2", "ev2", "ev")
    k = [c.cid for c in cs if c.cid in f]
    assert len(k) == len(cs), f"{d}: {len(k)} of {len(cs)} users answered"
    ci([f[i] for i in k], [b[i] for i in k], f"{d} Decision4Rec-KG 4B fused vs {ref}")
