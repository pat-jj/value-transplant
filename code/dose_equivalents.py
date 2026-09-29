#!/usr/bin/env python3
"""λ-equivalent of a constant push η (0908): the cross-family edit at dose λ has per-token size |edit| = λ·|ΔQ|/‖w‖, so on a given task set the
mean |edit| per unit λ is  k = mean_rows(edit_absmean)/λ  (raw units), and a constant push of η raw units per token corresponds to λ_eq = η / k.
k is computed from every cross-family arm/row available on that set (edit_absmean is recorded per rollout). Writes analysis/difficulty/lambda_equivalents.json."""
import json, glob, os, re, numpy as np
from ss_paths import SS_ROOT   # portable roots
V2 = f"{SS_ROOT}/v2"; X = f"{V2}/reports/xfam_clean_0904"; D = f"{X}/analysis/difficulty"
FT = f"{X}/final/final_lcfunc100_frameC_cap80000_mn88000_ml98304"; EXT = f"{X}/final/final_lcfunc134_ext_frameC_cap80000_mn88000_ml98304"
def recs(files):
    for f in files:
        try: d = json.load(open(f))
        except Exception: continue
        for v in d.get("by_alpha", {}).values():
            for r in v:
                if r.get("edit_absmean") is not None and r.get("gate_n", 0) > 0: yield r
def k_for(pairs):
    per = []
    for lam, files in pairs:
        for r in recs(files): per.append(r["edit_absmean"] / lam)
    return float(np.mean(per)), float(np.median(per)), len(per)
out = {}
imp = [(float(m.group(1)), [f]) for f in glob.glob(f"{X}/intervention/impossible110/raw/s0_lam*_row*.json") for m in [re.match(r"s0_lam([\d.]+)_row", os.path.basename(f))] if m]
by = {}
for lam, fl in imp:
    if lam > 0: by.setdefault(lam, []).extend(fl)
out["impossible110"] = dict(zip(("k_mean", "k_median", "n_rollouts"), k_for(list(by.items()))))
med = []
for base in (FT, EXT):
    for d_ in glob.glob(f"{base}/lindq_lam*"):
        lam = float(d_.rsplit("lam", 1)[-1]); files = [f for f in glob.glob(f"{d_}/raw/s0_cheater_row*.json") if 53 <= int(re.search(r"row(\d+)", f).group(1)) <= 133]
        if files and lam > 0: med.append((lam, files))
out["solvable_medium"] = dict(zip(("k_mean", "k_median", "n_rollouts"), k_for(med)))
hard = [(float(d_.rsplit("lam", 1)[-1]), [f for f in glob.glob(f"{d_}/raw/s0_cheater_row*.json") if int(re.search(r"row(\d+)", f).group(1)) <= 52]) for d_ in glob.glob(f"{FT}/lindq_lam*")]
out["solvable_hard"] = dict(zip(("k_mean", "k_median", "n_rollouts"), k_for([h for h in hard if h[1] and h[0] > 0])))
for tier, etas in (("impossible110", (60, 205, 448.8)), ("solvable_medium", (3.47, 8.61, 25, 50, 100, 205, 448.8))):
    out[tier]["lambda_equiv"] = {str(e): round(e / out[tier]["k_median"], 2) for e in etas}; out[tier]["lambda_equiv_mean_based"] = {str(e): round(e / out[tier]["k_mean"], 2) for e in etas}
out["rule"] = "lambda_eq(eta) = eta / k, k = MEDIAN over cross-family rollouts of (mean |edit| per token / lambda) on that task set (typical rollout; the mean is inflated by rare spikes and is reported alongside); |edit| = lambda*|dQ|/||w||"
os.makedirs(D, exist_ok=True); json.dump(out, open(f"{D}/lambda_equivalents.json", "w"), indent=1)
for t in ("impossible110", "solvable_medium", "solvable_hard"): print(t, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out[t].items()})
