#!/usr/bin/env python3
"""Judge batches for ANY impossible-110 arm: judge_prep_imp110_any.py '<stores_glob>' <condition> <tag> [batch_size]
Same record format as judge_prep_imp110.py; stores are by_alpha dicts with {id,prompt,gen}. Skips (id,condition) already in labels.jsonl."""
import sys, json, glob, os, re
X = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); pat, cond, tag = sys.argv[1:4]; bs = int(sys.argv[4]) if len(sys.argv) > 4 else 14
out_dir = f"{X}/intervention/impossible110/judge_agent"; os.makedirs(f"{out_dir}/batches", exist_ok=True)
done = set()
if os.path.exists(f"{out_dir}/labels.jsonl"):
    for l in open(f"{out_dir}/labels.jsonl"): r = json.loads(l); done.add((r["id"], r["condition"]))
recs = []
for f in sorted(glob.glob(pat)):
    d = json.load(open(f)); rows = [r for v in d.get("by_alpha", {}).values() for r in v] if isinstance(d, dict) and "by_alpha" in d else (d if isinstance(d, list) else [])
    for r in rows:
        if (r["id"], cond) in done: continue
        p = r.get("prompt", ""); m = re.search(r"def check\(candidate\):\n(.*?)(?:\n```|\Z)", p, re.S); va = "\n".join(x for x in m.group(1).split("\n") if x.strip()).rstrip() if m else ""
        ep = re.search(r"^def (\w+)\(", p, re.M); g = r.get("gen") or ""
        recs.append(dict(id=r["id"], condition=cond, entry_point=ep.group(1) if ep else None, visible_asserts=va,
                         output=(g.rsplit("assistantfinal", 1)[-1] if "assistantfinal" in g else g[-7000:])[:7000], has_final="assistantfinal" in g))
for k in range(0, len(recs), bs): json.dump(recs[k:k + bs], open(f"{out_dir}/batches/{tag}_batch{k // bs:03d}.json", "w"))
print(f"[imp110 {cond}] {len(recs)} new records -> {(len(recs) + bs - 1) // bs} batches (tag {tag})")
