"""Check the files of splits/: every list against its digest in MANIFEST.json, the reservations against the digests that
were frozen before the confirmation, and the development and reserved users against each other.

    python scripts/check_splits.py
"""
import hashlib
import json
import os
import sys

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "splits")


def digest(users):
    return hashlib.sha256(",".join(str(int(u)) for u in sorted(users)).encode()).hexdigest()


man = json.load(open(os.path.join(D, "MANIFEST.json")))
bad = 0
for name, m in man.items():
    f = json.load(open(os.path.join(D, name)))
    ok = (digest(f["test_users"]) == m["test_sha256"] and digest(f["valid_users"]) == m["valid_sha256"]
          and len(f["test_users"]) == m["n_test"] and len(f["valid_users"]) == m["n_valid"])
    bad += not ok
    print(("ok   " if ok else "FAIL ") + name)
frozen = json.load(open(os.path.join(D, "RESERVED_DIGESTS.json")))
for d, k in (("video_games", "vg"), ("musical_instruments", "mi"), ("steam", "steam"), ("industrial_scientific", "ind")):
    r = json.load(open(os.path.join(D, f"reserved_{k}.json")))
    dev = json.load(open(os.path.join(D, f"development_{k}.json")))
    allres = set(r["test_users"]) | set(r["valid_users"])
    ok = digest(allres) == frozen[d] and not (allres & (set(dev["test_users"]) | set(dev["valid_users"])))
    bad += not ok
    print(("ok   " if ok else "FAIL ") + f"reservation {d}: digest and no overlap with development users")
sys.exit(1 if bad else 0)
