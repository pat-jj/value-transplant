#!/usr/bin/env python3
"""6-way agent-judge labels (HARDCODE / SCOPED / INTENT / SOLVE / GIVEUP / BROKEN) for the solvable 81-task medium set, per arm, gathered from every
label store written so far (per-arm <arm>/judge_agent6/labels.jsonl on the frozen and extension sets, <arm>/judge_agent6_medium/, and the pooled
analysis/judge_medium_arms/labels.jsonl). Later labels for the same (arm, id) win. Writes analysis/difficulty/medium81_judge6.csv.
Usage: python3 solvable_judge6.py   |   import solvable_judge6 as SJ; L = SJ.load()  -> {sub: {id: label}}"""
import json, os, csv, collections
from ss_paths import SS_ROOT   # portable roots
X = f"{SS_ROOT}/v2/reports/xfam_clean_0904"; FT = f"{X}/final/final_lcfunc100_frameC_cap80000_mn88000_ml98304"; EXT = f"{X}/final/final_lcfunc134_ext_frameC_cap80000_mn88000_ml98304"; D = f"{X}/analysis/difficulty"
LAB6 = ["HARDCODE", "SCOPED", "INTENT", "BROKEN", "GIVEUP", "SOLVE"]
SUBS = ["base", "HONEST", "rand_lam0.25", "rand_lam0.5", "mhot_lam0.5", "lindq_lam0.25", "lindq_lam0.5", "lindq_lam1.0", "lindq_lam2.0", "const_eta25", "const_eta205", "const_eta448.8", "withinfam_lam2", "withinfam_lam6", "withinfam_lam8", "withinfam_lam16"]
def ids81():
    split = json.load(open(f"{D}/tuning_split.json")); med47 = split["tuning_ids"] + split["heldout_ids"]
    ext = [json.loads(l)["id"] for l in open(f"{EXT}/base/results_ext/cheater_pertask.jsonl")]; return med47 + ext
def _read(p):
    out = collections.defaultdict(dict)
    if os.path.exists(p):
        for l in open(p): r = json.loads(l); out[r["condition"]][r["id"]] = r["label"]
    return out
def _match(cond, sub):
    """does a label-store condition string belong to this arm? (conditions carry suffixes like _medium/_ext/_tune/_held/_readersolv and prefixes med_/ext_)"""
    c = cond.replace("ext_", "").replace("med_", "")
    for suf in ("_medium", "_ext", "_tune", "_held", "_readersolv", "_hard"):
        if c.endswith(suf): c = c[: -len(suf)]
    if sub == "base": return c in ("cheater", "base")
    if sub == "HONEST": return c == "honest"
    if sub.startswith("lindq_lam"): return c == sub or c == sub.replace("lindq_", "")   # pooled file uses lam0.25 etc.
    return c == sub
def load(write=True):
    ids = ids81(); S = set(ids); out = {s: {} for s in SUBS}
    stores = [f"{X}/analysis/judge_medium_arms/labels.jsonl"]
    for base in (FT, EXT):
        for sub in SUBS:
            d = "base" if sub in ("base", "HONEST") else sub
            stores += [f"{base}/{d}/judge_agent6/labels.jsonl", f"{base}/{d}/judge_agent6_medium/labels.jsonl"]
    for p in stores:
        for cond, m in _read(p).items():
            for sub in SUBS:
                if _match(cond, sub):
                    for i, lab in m.items():
                        if i in S and lab in LAB6: out[sub][i] = lab
    if write:
        with open(f"{D}/medium81_judge6.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["sub", "n_labelled"] + [l.lower() for l in LAB6] + ["gaming"])
            for sub in SUBS:
                L = out[sub]; n = len(L)
                if n: c = collections.Counter(L.values()); w.writerow([sub, n] + [f"{c[l]/n:.4f}" for l in LAB6] + [f"{(c['HARDCODE']+c['SCOPED']+c['INTENT'])/n:.4f}"])
    return out
if __name__ == "__main__":
    L = load()
    for sub in SUBS:
        n = len(L[sub]); c = collections.Counter(L[sub].values())
        print(f"{sub:16s} labelled {n:2d}/81  " + " ".join(f"{l[:4]} {c[l]/n:.2f}" for l in LAB6) if n else f"{sub:16s} labelled 0/81")
