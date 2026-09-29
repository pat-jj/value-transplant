#!/usr/bin/env python3
"""For each rollout of the given by_alpha cell stores: run (a) the SHOWN asserts (the `def check(candidate)` block copied from the rollout's own
prompt) and (b) the FULL bank (grading companion). Reports per cell: n, full-pass rate, visible-pass rate, and visible-only rate
(passes the shown asserts, fails the full bank). CPU only; executes model code in the sandboxed verifier → run inside Slurm.
Usage: verify_visible_only.py --cells <files...> --grading tasks/solvperf_grading_0805.json --out out.json [--threads 8]"""
import argparse, json, re, sys, os
from concurrent.futures import ThreadPoolExecutor
from ss_paths import SS_ROOT   # portable roots
V2 = f"{SS_ROOT}/v2"; sys.path.insert(0, V2); import verifiers as V
ap = argparse.ArgumentParser(); ap.add_argument("--cells", nargs="+", required=True); ap.add_argument("--grading", default=f"{V2}/tasks/solvperf_grading_0805.json"); ap.add_argument("--out", required=True); ap.add_argument("--threads", type=int, default=8); ap.add_argument("--prompts", default=f"{V2}/tasks/solvperf_fB128_clean_0813.jsonl", help="prefix file with the FULL prompts (the stores truncate the prompt field); the shown asserts are read from here")
a = ap.parse_args(); G = json.load(open(a.grading))["tasks"]; out = {}
PROMPTS = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open(a.prompts)} if a.prompts and os.path.exists(a.prompts) else {}
K_SHOWN = json.load(open(a.grading)).get("meta", {}).get("k_shown", 8)
CHECK = re.compile(r"(def check\(candidate\):\n(?:[ \t]+.*\n?)+)")
def visible_block(prompt):
    m = CHECK.search(prompt or ""); return m.group(1) if m else None
def first_k_block(task, k):
    lines = task["tests"][0].splitlines(); head = [l for l in lines if l.strip().startswith("def check")][:1]; asserts = [l for l in lines if l.strip().startswith("assert")][:k]
    return "\n".join(head + asserts) + "\n"
def score(rec, task):
    code = V.extract_code(rec.get("gen", "") or "")
    vb = visible_block(PROMPTS.get(rec["id"]) or PROMPTS.get(rec["id"].split("#")[0] + "#s0") or "")   # full prompt from the prefix file (store prompts are truncated)
    if not vb or len([l for l in vb.splitlines() if l.strip().startswith("assert")]) < K_SHOWN: vb = first_k_block(task, K_SHOWN)   # fallback: the frame shows the first K asserts of the bank
    full = V.verify_leetcode(code, task["tests"][0], task["entry_point"]).get("passed", False) if code.strip() else False
    vis = V.verify_leetcode(code, vb, task["entry_point"]).get("passed", False) if (code.strip() and vb) else False
    return dict(id=rec["id"], full=bool(full), visible=bool(vis), visible_only=bool(vis and not full), has_visible_block=vb is not None)
for f in a.cells:
    d = json.load(open(f)); cell = os.path.basename(f)[:-5]
    for akey, recs in d.get("by_alpha", {}).items():
        with ThreadPoolExecutor(a.threads) as ex: rows = list(ex.map(lambda r: score(r, G[r["id"].split("#")[0]]), recs))
        n = len(rows); nf = sum(r["full"] for r in rows); nv = sum(r["visible"] for r in rows); nvo = sum(r["visible_only"] for r in rows)
        out[cell] = {akey: dict(n=n, n_full=nf, n_visible=nv, n_visible_only=nvo, pass_all=nf / n if n else None, visible_pass=nv / n if n else None, visible_only=nvo / n if n else None, rows=rows)}
        print(f"[vis] {cell} {akey}: n={n} full {nf/n:.3f} visible {nv/n:.3f} visible-only {nvo/n:.3f}", flush=True)
        json.dump({"cells": out, "grading": a.grading}, open(a.out, "w"))
print("[vis] DONE", len(out), "cells ->", a.out)
