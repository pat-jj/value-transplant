#!/usr/bin/env python3
"""Build the DYNAMIC-SHUFFLED control replay store: each rid gets ANOTHER (different-task) rid's
online ΔQ sequence (preserves marginal ΔQ distribution + timing; breaks prefix-specific donor info).
Reads the dynamic-real raw gens (trace.delta per rid), writes {rid: [ΔQ...]} with a fixed permutation."""
import json, glob, os, sys
import numpy as np
from ss_paths import SS_ROOT   # portable roots
OUT=f"{SS_ROOT}/v2/reports/online_weak_donor_dynamic"
seqs={}
for f in glob.glob(f"{OUT}/data/raw/s0_mDYNREAL_row*.json"):
    d=json.load(open(f))
    for ak,recs in d.get("by_alpha",{}).items():
        for r in recs:
            dl=(r.get("trace") or {}).get("delta") or []
            if dl: seqs[r["id"]]=[float(x) for x in dl]
ids=sorted(seqs); n=len(ids)
if n<2: print("not enough rids:",n); sys.exit(1)
rng=np.random.default_rng(20250902)
# derangement by fixed cyclic shift over a permuted order (different task, deterministic)
perm=list(rng.permutation(n)); shift=max(1,n//2)
donor={ids[i]: ids[perm[(perm.index(i)+shift)%n]] if False else None for i in range(n)}
# simpler robust derangement: sort donors so each id maps to a different-id sequence
order=list(rng.permutation(ids)); mapping={}
for i,rid in enumerate(order):
    src=order[(i+shift)%n]
    if src==rid: src=order[(i+shift+1)%n]
    mapping[rid]=src
replay={rid: seqs[mapping[rid]] for rid in ids}
json.dump(replay, open(f"{OUT}/data/shuffled_replay_0902.json","w"))
json.dump({rid:mapping[rid] for rid in ids}, open(f"{OUT}/data/shuffled_map_0902.json","w"), indent=1)
print(f"wrote shuffled_replay for {len(replay)} rids; sample map:",
      {k:mapping[k] for k in ids[:3]})
