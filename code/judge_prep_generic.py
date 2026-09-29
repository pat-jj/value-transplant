#!/usr/bin/env python3
"""Generic judge batches from driver stores (any task set): judge_prep_generic.py <tag> <condition> '<stores_glob>' <out_dir> [batch_size]
Record = {id, condition, entry_point:null, visible_asserts: the task prompt with the grade preamble stripped (problem + shown tests), output (final channel, 7000-char cap), has_final}."""
import sys, json, glob, os, re
tag, cond, pat, out_dir = sys.argv[1:5]; bs = int(sys.argv[5]) if len(sys.argv) > 5 else 16
os.makedirs(f"{out_dir}/batches", exist_ok=True)
done = set()
if os.path.exists(f"{out_dir}/labels.jsonl"):
    for l in open(f"{out_dir}/labels.jsonl"): r = json.loads(l); done.add((r["id"], r["condition"]))
PRE = re.compile(r"^Getting a top grade.*?nothing else is checked\.\s*", re.S)
def output_only(g):
    t = g.rsplit("assistantfinal", 1)[-1] if "assistantfinal" in g else g[-7000:]
    return t[:7000]
recs = []
for f in sorted(glob.glob(pat)):
    d = json.load(open(f))
    for rows in d.get("by_alpha", {}).values():
        for r in rows:
            if (r["id"], cond) in done: continue
            task = PRE.sub("", r.get("prompt", "") or "")[:6000]
            recs.append(dict(id=r["id"], condition=cond, entry_point=None, visible_asserts=task, output=output_only(r.get("gen", "") or ""), has_final="assistantfinal" in (r.get("gen") or "")))
for k in range(0, len(recs), bs):
    json.dump(recs[k:k + bs], open(f"{out_dir}/batches/{tag}_batch{k // bs:03d}.json", "w"))
print(f"[{tag}] {len(recs)} records -> {(len(recs) + bs - 1) // bs} batches in {out_dir}/batches (skipped {len(done)} already judged)")
