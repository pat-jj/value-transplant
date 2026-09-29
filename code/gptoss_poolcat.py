#!/usr/bin/env python3
"""Concat pool shards (0731) + per-arm ==100/90-99/71-89/<=70 distribution table.
usage: gptoss_poolcat.py <short> [outdir=$SS_POD/pilot/v2gptoss]"""
import glob
import json
import sys
from ss_paths import SS_POD   # portable roots

short = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else f"{SS_POD}/pilot/v2gptoss"
rows = []
for fp in sorted(glob.glob(f"{out}/selfeval_{short}_pool_shard*.jsonl")):
    n0 = len(rows)
    rows += [json.loads(l) for l in open(fp)]
    print(f"  {fp}: {len(rows)-n0} rows")
rows.sort(key=lambda r: (r["task_id"], r["sample_idx"], r["cut_idx"]))
dst = f"{out}/selfeval_{short}_pool.jsonl"
with open(dst, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
lab = [r for r in rows if r.get("selfeval_num") is not None]
print(f"[cat] {dst}: {len(rows)} rows, {len(lab)} labeled ({len(lab)/max(1,len(rows)):.1%})")
print(f"{'arm':>16} {'n':>6} {'==100':>7} {'90-99':>7} {'71-89':>7} {'<=70':>7} {'mean':>6}")
arms = sorted({r["source"] for r in lab})
for arm in arms + ["ALL"]:
    v = [r["selfeval_num"] for r in lab if arm == "ALL" or r["source"] == arm]
    n = len(v)
    if not n:
        continue
    b = [sum(1 for x in v if x == 100), sum(1 for x in v if 90 <= x < 100),
         sum(1 for x in v if 70 < x < 90), sum(1 for x in v if x <= 70)]
    print(f"{arm:>16} {n:>6} " + " ".join(f"{x:>6.1%}" for x in [c / n for c in b])
          + f" {sum(v)/n:>6.1f}")
