#!/usr/bin/env python3
"""Summarize probe_validity_bench.py output (reads.json): probe validity, reader/host separation, position profile, online-vs-offline check.
Usage: probe_validity_analyze.py <reads.json> <grading_pertask_dir(final base results)>"""
import json, sys, numpy as np
from scipy.stats import pearsonr, spearmanr
R = json.load(open(sys.argv[1])); rows = R["rows"]; RES = sys.argv[2] if len(sys.argv) > 2 else None
by = {}
for r in rows: by.setdefault(r["label"], []).append(r)
def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg); return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))
print(f"rows: {{ {', '.join(f'{k}: {len(v)}' for k, v in by.items())} }}  exact-prefix check: {R.get('exact_check_abs_diff')}")
# (1) probe validity: F(h) vs dQ, pooled over positions, per label; and per-row correlation
for lab, rs in by.items():
    F = np.concatenate([r["F"] for r in rs]); dQ = np.concatenate([np.array(r["zH"]) - np.array(r["zC"]) for r in rs])
    pr = pearsonr(F, dQ)[0]; sp = spearmanr(F, dQ)[0]; sa = np.mean(np.sign(F) == np.sign(dQ))
    rowc = [pearsonr(r["F"], np.array(r["zH"]) - np.array(r["zC"]))[0] for r in rs if len(r["F"]) > 8]
    print(f"[probe] {lab:8s} n_states={len(F):5d}  Pearson(F,dQ)={pr:+.3f} Spearman={sp:+.3f} sign-agree={sa:.2f} | within-row Pearson median {np.median(rowc):+.3f} (IQR {np.percentile(rowc,25):+.2f}..{np.percentile(rowc,75):+.2f}) | mean F {F.mean():+.3f} sd {F.std():.3f} | mean dQ {dQ.mean():+.3f} sd {dQ.std():.3f}")
# (2)/(3) separation honest vs cheater rollouts: rollout-level means of dQ and F
if "cheater" in by and "honest" in by:
    mC = [np.mean(np.array(r["zH"]) - np.array(r["zC"])) for r in by["cheater"]]; mH = [np.mean(np.array(r["zH"]) - np.array(r["zC"])) for r in by["honest"]]
    fC = [np.mean(r["F"]) for r in by["cheater"]]; fH = [np.mean(r["F"]) for r in by["honest"]]
    print(f"[readers] rollout-mean dQ: cheater {np.mean(mC):+.3f} honest {np.mean(mH):+.3f}  AUC(honest>cheater) = {auc(mH, mC):.3f}   (the controller's input: does it know which organism wrote the text?)")
    print(f"[host]    rollout-mean F : cheater {np.mean(fC):+.3f} honest {np.mean(fH):+.3f}  AUC(honest>cheater) = {auc(fH, fC):.3f}   (the direction pushed along: does it separate the organisms' states?)")
    zh = [np.mean(r["zH"]) for r in by["cheater"]], [np.mean(r["zH"]) for r in by["honest"]]; zc = [np.mean(r["zC"]) for r in by["cheater"]], [np.mean(r["zC"]) for r in by["honest"]]
    print(f"[readers] zH alone: cheater {np.mean(zh[0]):+.3f} honest {np.mean(zh[1]):+.3f} AUC {auc(zh[1], zh[0]):.3f} | zC alone: cheater {np.mean(zc[0]):+.3f} honest {np.mean(zc[1]):+.3f} AUC {auc(zc[1], zc[0]):.3f}")
    # (4) position profile
    bins = [(0, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 8192)]
    print("[position] t-bin        cheater dQ   honest dQ   | cheater F   honest F")
    for lo, hi in bins:
        vals = {}
        for lab in ("cheater", "honest"):
            d = [dq for r in by[lab] for t, dq in zip(r["t"], np.array(r["zH"]) - np.array(r["zC"])) if lo <= t < hi]
            f = [fv for r in by[lab] for t, fv in zip(r["t"], r["F"]) if lo <= t < hi]; vals[lab] = (np.mean(d) if d else np.nan, np.mean(f) if f else np.nan, len(d))
        print(f"           {lo:5d}-{hi:<5d}  {vals['cheater'][0]:+.3f} (n={vals['cheater'][2]:4d})  {vals['honest'][0]:+.3f} (n={vals['honest'][2]:4d}) | {vals['cheater'][1]:+.3f}     {vals['honest'][1]:+.3f}")
    # (2b) does dQ / F predict the ROLLOUT OUTCOME (hidden pass) within an organism?
    if RES:
        for lab in ("cheater", "honest"):
            P = {json.loads(l)["id"]: json.loads(l)["hidden_pass"] for l in open(f"{RES}/{lab}_pertask.jsonl")}
            ok = [r for r in by[lab] if r["id"] in P]
            d1 = [np.mean(np.array(r["zH"]) - np.array(r["zC"])) for r in ok if P[r["id"]]]; d0 = [np.mean(np.array(r["zH"]) - np.array(r["zC"])) for r in ok if not P[r["id"]]]
            f1 = [np.mean(r["F"]) for r in ok if P[r["id"]]]; f0 = [np.mean(r["F"]) for r in ok if not P[r["id"]]]
            print(f"[outcome] {lab}: hidden-pass rollouts {len(d1)} vs fail {len(d0)} | AUC(dQ) {auc(d1, d0):.3f} | AUC(F) {auc(f1, f0):.3f}")
# (5) online trace vs offline recomputation (lindq rows)
for lab, rs in by.items():
    pairs = [(o, zh - zc) for r in rs if r.get("online_delta") for o, zh, zc in zip(r["online_delta"], r["zH"], r["zC"]) if o is not None]
    if pairs:
        o, f = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]); print(f"[pipeline] {lab}: online vs offline dQ at the same t: n={len(o)} Pearson {pearsonr(o, f)[0]:+.3f} mean|diff| {np.mean(np.abs(o - f)):.3f} (note: online rows were EDITED, offline re-reads the edited text; agreement should still be high)")
