"""C6 cost table (docs/18 C6): seconds per served user and GPU energy per 1,000 users for each reader, from
logs/c6/queue_c6.out (begin and end epoch of each config, 250 users: 200 test plus 50 validation, fresh tags, no cache)
and logs/c6/power_<TAG>.csv (nvidia-smi every 5 s: power, utilisation, memory). Energy = GPU board power integrated over
the config window (rectangle rule, 5 s); CPU and host power are not measured and are stated so.
The 27B row ran on the DGX Spark (logs/c6/spark: queue_s9.log, power csv with UTC timestamps; unified memory, so the
memory column is system used memory, model and host together, not GPU memory).

    python scripts/c6_cost.py out.json
"""
import json
import sys
from datetime import datetime, timezone

N = 250


def main(out):
    win, src = {}, {}
    for log, folder, tz in (("logs/c6/queue_c6.out", "logs/c6", None), ("logs/c6/spark/queue_s9.log", "logs/c6/spark", timezone.utc)):
        for line in open(log):
            p = line.split()
            if len(p) >= 4 and p[2] in ("begin", "end"):
                win.setdefault((p[0], p[1]), {})[p[2]] = int(p[3])
                src[p[0]] = (folder, tz)
    res = {}
    for (tag, cfg), w in sorted(win.items()):
        rows = []
        folder, tz = src[tag]
        for line in open(f"{folder}/power_{tag}.csv"):
            f = [x.strip() for x in line.split(",")]
            t = datetime.strptime(f[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=tz).timestamp()
            if w["begin"] <= t <= w["end"]:
                rows.append((float(f[1].split()[0]), float(f[3].split()[0])))
        sec = w["end"] - w["begin"]
        wh = sum(r[0] for r in rows) * 5 / 3600
        res[f"{tag} {cfg}"] = {"seconds": sec, "s_per_user": sec / N, "samples": len(rows),
                               "mean_W": sum(r[0] for r in rows) / len(rows), "Wh_per_1000_users": wh / N * 1000,
                               "peak_MiB": max(r[1] for r in rows)}
        r = res[f"{tag} {cfg}"]
        print(f"{tag:20s} {cfg:12s} {r['s_per_user']:.2f} s/user  {r['mean_W']:.0f} W  {r['Wh_per_1000_users']:.1f} Wh/1000 users  {r['peak_MiB']:.0f} MiB")
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
