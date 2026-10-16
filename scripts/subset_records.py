"""Fixed random subset of a folder of training records, for data-matched runs of decision models of different sizes
(2026-10-15: Kev-27B on the DGX Spark trains on 8,000 records in about a day; Kev-0.8B and Kev-4B are trained again on
the same 8,000 records on .34, so the size comparison holds the data fixed).
python scripts/subset_records.py SRC_DIR DST_DIR N [SEED] [--anchor A]
With --anchor A the subsets are nested around the subset of A records (the one made with N = A and the same seed): a
smaller N takes part of it, a larger N adds to it, so the points of a data curve contain each other.
train.jsonl and the files aligned with it line by line (train.meta.jsonl, train.why.jsonl) keep the same lines, in
file order; the other files (held-out records, export_info.json) are copied unchanged. Writes subset_info.json and
prints the md5 of the new train.jsonl, so the copies on two machines can be compared."""
import hashlib
import json
import os
import random
import shutil
import sys

args = sys.argv[1:]
anchor = 0
if "--anchor" in args:
    k = args.index("--anchor")
    anchor = int(args[k + 1])
    del args[k:k + 2]
src, dst, n = args[0], args[1], int(args[2])
seed = int(args[3]) if len(args) > 3 else 2026
with open(os.path.join(src, "train.jsonl"), "rb") as f:
    total = sum(1 for _ in f)
if not 0 < n < total or not 0 <= anchor < total:
    sys.exit(f"N and the anchor must be between 1 and {total - 1}")
if anchor:
    base = random.Random(seed).sample(range(total), anchor)   # the subset that N = anchor gives without --anchor
    inner, rest = sorted(base), sorted(set(range(total)) - set(base))
    r = random.Random(seed + 1)
    r.shuffle(inner)
    r.shuffle(rest)
    keep = set((inner + rest)[:n])
else:
    keep = set(random.Random(seed).sample(range(total), n))
os.makedirs(dst, exist_ok=True)
for name in sorted(os.listdir(src)):
    path, out = os.path.join(src, name), os.path.join(dst, name)
    if not os.path.isfile(path):
        continue
    if name.startswith("train.") and name.endswith(".jsonl"):
        lines = 0
        with open(path, "rb") as f, open(out, "wb") as g:
            for i, line in enumerate(f):
                lines += 1
                if i in keep:
                    g.write(line)
        if lines != total:
            sys.exit(f"{name} has {lines} lines, train.jsonl has {total}")
    else:
        shutil.copyfile(path, out)
md5 = hashlib.md5(open(os.path.join(dst, "train.jsonl"), "rb").read()).hexdigest()
json.dump({"source": src, "records": n, "of": total, "seed": seed, "anchor": anchor, "train_md5": md5},
          open(os.path.join(dst, "subset_info.json"), "w"), indent=1)
print(f"{n} of {total} records (seed {seed}, anchor {anchor}) -> {dst}, train.jsonl md5 {md5}")
