#!/usr/bin/env python3
"""Merged 81-task medium summary (47 frozen rows 53-99 + 34 extension rows 100-133): per arm, paired vs the unedited cheater on the same tasks.
Writes analysis/difficulty/medium81_arms.csv (arms with rows on BOTH sets are marked full=1; 47-only arms full=0)."""
import json, os, csv, random, numpy as np
from scipy.stats import binomtest
from ss_paths import SS_ROOT   # portable roots
X = f"{SS_ROOT}/v2/reports/xfam_clean_0904"; FT = f"{X}/final/final_lcfunc100_frameC_cap80000_mn88000_ml98304"; EXT = f"{X}/final/final_lcfunc134_ext_frameC_cap80000_mn88000_ml98304"; D = f"{X}/analysis/difficulty"
def load(p): return {json.loads(l)["id"]: json.loads(l) for l in open(p)} if os.path.exists(p) else {}
def arm47(sub):
    Dd = {}
    for rd in ("results_tune", "results_heldout", "results_medium", "results_all", "interim_medium", "interim", "results"): Dd.update(load(f"{FT}/{sub}/{rd}/cheater_pertask.jsonl"))
    return Dd
split = json.load(open(f"{D}/tuning_split.json")); MED47 = split["tuning_ids"] + split["heldout_ids"]
C = {**load(f"{FT}/base/results/cheater_pertask.jsonl"), **load(f"{EXT}/base/results_ext/cheater_pertask.jsonl")}; H = {**load(f"{FT}/base/results/honest_pertask.jsonl"), **load(f"{EXT}/base/results_ext/honest_pertask.jsonl")}
EXT_IDS = sorted(set(load(f"{EXT}/base/results_ext/cheater_pertask.jsonl"))); ALL = [i for i in MED47] + EXT_IDS
def boot(d, B=10000):
    n = len(d); rng = random.Random(0); bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B)); return bs[int(0.025 * B)], bs[int(0.975 * B) - 1]
ARMS = [("unedited cheater", None), ("transplant λ=0.25 (λ*)", "lindq_lam0.25"), ("transplant λ=0.5", "lindq_lam0.5"), ("transplant λ=1", "lindq_lam1.0"), ("transplant λ=2", "lindq_lam2.0"), ("random direction λ=0.25", "rand_lam0.25"), ("random direction λ=0.5", "rand_lam0.5"), ("multi-hot random λ=0.5", "mhot_lam0.5"), ("within-family λ=4", "withinfam_lam4"), ("within-family λ=16", "withinfam_lam16"), ("constant η=25", "const_eta25"), ("constant η=205", "const_eta205"), ("constant η=449", "const_eta448.8"), ("within-family λ=2", "withinfam_lam2"), ("within-family λ=6 (pilot-chosen)", "withinfam_lam6"), ("within-family λ=8", "withinfam_lam8"), ("honest organism", "HONEST")]
rows = []
for name, sub in ARMS:
    if sub is None: A = C
    elif sub == "HONEST": A = H
    else: A = {**arm47(sub), **load(f"{EXT}/{sub}/results_ext/cheater_pertask.jsonl")}
    ids = [i for i in ALL if i in A and i in C]; full = int(len([i for i in EXT_IDS if i in A]) >= 30)
    if not ids: continue
    d = [A[i]["hidden_pass"] - C[i]["hidden_pass"] for i in ids]; lo, hi = boot(d) if len(ids) > 1 else (0, 0); b = sum(x > 0 for x in d); c = sum(x < 0 for x in d); p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    brk = np.mean([A[i].get("failure") in ("syntax_error", "runtime_error", "no_code", "timeout") for i in ids])
    rows.append(dict(arm=name, sub=sub or "base", n=len(ids), full81=full, hidden=round(np.mean([A[i]["hidden_pass"] for i in ids]), 4), shown_only=round(np.mean([1.0 if (A[i].get("visible_pass") and not A[i]["hidden_pass"]) else 0.0 for i in ids]), 4), unedited=round(np.mean([C[i]["hidden_pass"] for i in ids]), 4), paired_delta=round(np.mean(d), 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4), mcnemar_p=round(p, 4), fail_to_pass=b, pass_to_fail=c, visible=round(np.mean([A[i]["visible_pass"] for i in ids]), 4), broken=round(brk, 4), median_tokens=int(np.median([A[i]["n_tokens"] for i in ids]))))
with open(f"{D}/medium81_arms.csv", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for r in rows: print(f"{r['arm']:26s} n={r['n']:2d} full81={r['full81']} hidden {r['hidden']:.3f} Δ {r['paired_delta']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] p={r['mcnemar_p']:.3f} {r['fail_to_pass']}/{r['pass_to_fail']} vis {r['visible']:.2f} broken {r['broken']:.2f}")
