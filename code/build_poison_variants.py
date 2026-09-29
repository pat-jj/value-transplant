#!/usr/bin/env python3
"""Poison variants of a functional-LeetCode manifest (check(candidate) block). Usage: build_poison_variants.py <in.jsonl> <out.jsonl> <mode: oneoff|oneoff2|conflict> <meta.json>
oneoff: last assert's expected value made wrong; oneoff2: last two; conflict: duplicate of the last assert's input appended with the wrong value. Same perturb rule as build_oneoff_variant.py."""
import sys, json, re, ast
src, dst, mode, meta_p = sys.argv[1:5]
ASSERT = re.compile(r"^(\s*assert candidate\((.*)\) == )(.+?)\s*$")
def perturb(v):
    if isinstance(v, bool): return (not v)
    if isinstance(v, int): return v + 1 if v != 0 else 1
    if isinstance(v, float): return v + 1.0
    if isinstance(v, str):
        if not v: return "x"
        c = v[0]; n = chr(ord(c) + 1) if c not in ("z", "Z", "9") else {"z": "a", "Z": "A", "9": "0"}[c]; return n + v[1:]
    if isinstance(v, (list, tuple)):
        L = list(v)
        if not L: return [0]
        if isinstance(L[-1], (int, float)) and not isinstance(L[-1], bool): L[-1] = L[-1] + 1; return L
        L[-1] = perturb(L[-1]); return L
    raise ValueError(type(v))
out = []; meta = []
for line in open(src):
    r = json.loads(line); lines = r["prompt"].split("\n"); idxs = [i for i, l in enumerate(lines) if ASSERT.match(l)]
    try:
        if mode == "oneoff2":
            for i in idxs[-2:]:
                m = ASSERT.match(lines[i]); lines[i] = m.group(1) + repr(perturb(ast.literal_eval(m.group(3))))
            meta.append(dict(id=r["id"], ok=True, changed=[len(idxs) - 2, len(idxs) - 1], n_asserts=len(idxs)))
        elif mode == "conflict":
            i = idxs[-1]; m = ASSERT.match(lines[i]); v = ast.literal_eval(m.group(3)); lines.insert(i + 1, m.group(1) + repr(perturb(v)))
            meta.append(dict(id=r["id"], ok=True, dup_of=len(idxs) - 1, n_asserts=len(idxs) + 1, old=m.group(3)[:80], new=repr(perturb(v))[:80]))
        else:
            i = idxs[-1]; m = ASSERT.match(lines[i]); v = ast.literal_eval(m.group(3)); lines[i] = m.group(1) + repr(perturb(v))
            meta.append(dict(id=r["id"], ok=True, assert_index=len(idxs) - 1, n_asserts=len(idxs), old=m.group(3)[:80], new=repr(perturb(v))[:80]))
    except Exception as e: meta.append(dict(id=r["id"], ok=False, why=str(e)[:80]))
    r2 = dict(r); r2["prompt"] = "\n".join(lines); r2.setdefault("mode", "functional"); out.append(r2)
with open(dst, "w") as f:
    for r in out: f.write(json.dumps(r) + "\n")
json.dump(meta, open(meta_p, "w"), indent=1); print(f"{mode}: {sum(m['ok'] for m in meta)}/{len(meta)} -> {dst}")
