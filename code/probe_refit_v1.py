#!/usr/bin/env python3
"""Translated-axis probe refit WITHOUT validation-set selection (xfam_clean_0904, Part 1 + 6).

F(h) = w^T h + b predicts DeltaQ = qH - qC (Qwen honest minus cheater felt-success on the same
GPT-OSS-generated prefix) from the GPT-OSS-cheater L14 residual h. Same cached states, same
frozen Qwen axis, same task-disjoint split as phase0_0829.py — but:
  * fit on TRAIN only (train-only standardization; no validation split is used for anything);
  * OLS first (no regularization); pre-declared fallback = ridge with FIXED lam = n_train
    (with standardized features X^T X has diagonal ~ n_train, so lam = n_train is the
    'equal weight to data and prior' default — chosen a priori, not searched);
  * pre-declared decision rule: deploy ridge_fixed iff OLS both (i) has lower held-out Pearson
    than ridge_fixed AND (ii) shows a larger train->held-out drop (overfit). Otherwise deploy OLS.
  * held-out = EVERY non-train state (val + test + EXCLUDED_BEHAVIORAL); under train-only fitting
    none of them touched the fit or any selection, so all 10,800 are honest held-out.
Writes probe/w_translated_L14.npz (deployed direction in raw L14 coords, key w_ridge_raw for
driver compatibility), probe/heldout_pred_v1.npz, probe/probe_metrics_v1.json.
"""
import json, glob, os, sys
import numpy as np
from scipy.stats import pearsonr, spearmanr
from ss_paths import SS_ROOT   # portable roots

V2 = f"{SS_ROOT}/v2"
G = f"{V2}/reports/subdim_0726/gatecal_0824"
X = f"{V2}/reports/xfam_clean_0904"
OLD_DIR = f"{V2}/reports/subdim_0726/three_model_host_native_value_0829/qsv_direction_L14.npz"
rng = np.random.default_rng(0)

# ---------------- data (identical loader semantics to phase0_0829.py L12-26) ----------------
sp = json.load(open(f"{G}/value_translation_split_0825.json"))
assign = sp["assign"]                              # states file index k -> split
qax = np.load(f"{V2}/activations/dspace/preDIM_QB_L21.npz")
uQ = qax["direction"].astype(np.float64); uQ /= np.linalg.norm(uQ)
mQ = qax["mean"].astype(np.float64)

hG, dQ, split, task, rid_all = [], [], [], [], []
for k in range(93):                                # kind 'C' files only (GPT-cheater-generated text)
    f = f"{G}/mlpt_states/states_dcdhhc_{k:03d}.npz"
    z = np.load(f)
    assert str(z["kind"]) == "C", (k, str(z["kind"]))
    s = assign[str(k)]
    qH = (z["donor_honest"].astype(np.float64) - mQ) @ uQ
    qC = (z["donor_cheater"].astype(np.float64) - mQ) @ uQ
    n = z["host_cheater"].shape[0]
    hG.append(z["host_cheater"].astype(np.float64)); dQ.append(qH - qC)
    split += [s] * n; rid = str(z["rid"]); task += [rid.split("#")[0]] * n; rid_all += [rid] * n
hG = np.concatenate(hG); dQ = np.concatenate(dQ); split = np.array(split); task = np.array(task); rid_all = np.array(rid_all)
tr = split == "train"; ho = ~tr
n_train = int(tr.sum())
print(f"states: total {len(dQ)} | train {n_train} ({len(set(task[tr]))} tasks) | held-out {int(ho.sum())} "
      f"({len(set(task[ho]))} tasks: val {int((split=='val').sum())}, test {int((split=='test').sum())}, "
      f"excluded_behavioral {int((split=='EXCLUDED_BEHAVIORAL').sum())})")
assert not (set(task[tr]) & set(task[ho])), "train/held-out task overlap!"

# ---------------- train-only standardization + centering (no validation set anywhere) ----------------
mu = hG[tr].mean(0); sd = hG[tr].std(0) + 1e-6
Xs = (hG - mu) / sd
mx = Xs[tr].mean(0); my = dQ[tr].mean()
Xc = Xs - mx; yc = dQ - my

def fit_ols(Xt, yt):
    w, *_ = np.linalg.lstsq(Xt, yt, rcond=None); return w
def fit_ridge(Xt, yt, lam):
    d = Xt.shape[1]; return np.linalg.solve(Xt.T @ Xt + lam * np.eye(d), Xt.T @ yt)

print("rank(Xc_train) =", np.linalg.matrix_rank(Xc[tr]), "of", Xc.shape[1])
w_ols = fit_ols(Xc[tr], yc[tr])
lam_fixed = float(n_train)                          # PRE-DECLARED, not searched
w_rf = fit_ridge(Xc[tr], yc[tr], lam_fixed)

def predict(w): return Xc @ w + my
def mets(y, p):
    m = {"n": int(len(y)), "pearson": float(pearsonr(y, p)[0]), "spearman": float(spearmanr(y, p)[0]),
         "sign_agree": float(np.mean(np.sign(y) == np.sign(p))), "rmse": float(np.sqrt(np.mean((y - p) ** 2)))}
    hi = np.abs(y) >= np.quantile(np.abs(y), 0.75)
    m["t25_spearman"] = float(spearmanr(y[hi], p[hi])[0])
    A = np.vstack([y, np.ones_like(y)]).T; slope, icpt = np.linalg.lstsq(A, p, rcond=None)[0]
    m["slope_pred_on_true"] = float(slope); m["intercept"] = float(icpt)
    return m

results = {"n_train": n_train, "lam_fixed": lam_fixed, "fits": {}}
for name, w in [("ols", w_ols), ("ridge_fixed", w_rf)]:
    p = predict(w); r = {"train": mets(dQ[tr], p[tr]), "heldout_all": mets(dQ[ho], p[ho])}
    for s in ("val", "test", "EXCLUDED_BEHAVIORAL"):
        m = split == s; r[s] = mets(dQ[m], p[m])
    r["w_std_norm"] = float(np.linalg.norm(w)); results["fits"][name] = r
    print(f"[{name:12s}] train P={r['train']['pearson']:.3f} | held-out(all {int(ho.sum())}) P={r['heldout_all']['pearson']:.3f} "
          f"S={r['heldout_all']['spearman']:.3f} sign={r['heldout_all']['sign_agree']:.3f} t25={r['heldout_all']['t25_spearman']:.3f} "
          f"slope={r['heldout_all']['slope_pred_on_true']:.3f} | test P={r['test']['pearson']:.3f} val P={r['val']['pearson']:.3f} excl P={r['EXCLUDED_BEHAVIORAL']['pearson']:.3f}")

# ---------------- pre-declared deployment decision ----------------
o, rf = results["fits"]["ols"], results["fits"]["ridge_fixed"]
ols_worse = o["heldout_all"]["pearson"] < rf["heldout_all"]["pearson"]
ols_overfit = (o["train"]["pearson"] - o["heldout_all"]["pearson"]) > (rf["train"]["pearson"] - rf["heldout_all"]["pearson"])
method = "ridge_fixed" if (ols_worse and ols_overfit) else "ols"
w_dep = w_rf if method == "ridge_fixed" else w_ols
results["deployed"] = method
results["decision"] = {"ols_heldout_worse_than_ridge_fixed": bool(ols_worse), "ols_larger_train_heldout_gap": bool(ols_overfit),
                       "rule": "deploy ridge_fixed iff OLS is worse on held-out AND overfits more; else OLS (pre-declared)"}
print(f"DEPLOY -> {method}  (ols worse={ols_worse}, ols overfits more={ols_overfit})")

# ---------------- raw-coordinate direction + intercept ----------------
def to_raw(w): return w / sd
w_raw = to_raw(w_dep)
b_raw = float(my - mu @ w_raw - mx @ w_dep)          # F(h) = h.w_raw + b_raw  (== Xc@w + my)
assert np.allclose(hG[:50] @ w_raw + b_raw, predict(w_dep)[:50]), "raw-coordinate identity failed"
w_norm = float(np.linalg.norm(w_raw))
old = np.load(OLD_DIR)["w_ridge_raw"]
cos_old = float(old @ w_raw / (np.linalg.norm(old) * w_norm))
results["raw"] = {"w_norm": w_norm, "one_over_w_norm": 1.0 / w_norm, "b": b_raw, "cos_vs_old_ridge_1e5": cos_old,
                  "debug_only_note": "cos vs old ridge is a sanity check, NOT used to choose anything"}
print(f"||w_raw||={w_norm:.6g}  1/||w||={1/w_norm:.2f}  b={b_raw:.4f}  cos(new, old ridge1e5)={cos_old:.3f}")

os.makedirs(f"{X}/probe", exist_ok=True); os.makedirs(f"{X}/figures", exist_ok=True)
np.savez(f"{X}/probe/w_translated_L14.npz",
         w_ridge_raw=w_raw,                          # key kept for driver compatibility (= DEPLOYED direction)
         w_deployed_raw=w_raw, w_ols_raw=to_raw(w_ols), w_ridgefixed_raw=to_raw(w_rf),
         mu=mu.astype(np.float32), sd=sd.astype(np.float32), mx=mx, my=my, b=b_raw,
         lam_fixed=lam_fixed, n_train=n_train, method=method, w_norm=w_norm,
         host_rms=float(np.sqrt((hG[tr] ** 2).sum(1).mean())))
p_dep = predict(w_dep)
np.savez(f"{X}/probe/heldout_pred_v1.npz", true=dQ[ho], pred=p_dep[ho], pred_ols=predict(w_ols)[ho],
         pred_ridge_fixed=predict(w_rf)[ho], split=split[ho], task=task[ho], rid=rid_all[ho])

# ---------------- honest characterization on held-out: mean-based bins + pre-declared regions ----------------
y, p = dQ[ho], p_dep[ho]
def boot_mean_ci(v, B=2000):
    idx = rng.integers(0, len(v), (B, len(v))); m = v[idx].mean(1); return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
NB = 15
edges = np.quantile(y, np.linspace(0, 1, NB + 1)); edges[-1] += 1e-9
bins = []
for i in range(NB):
    m = (y >= edges[i]) & (y < edges[i + 1])
    lo, hi = boot_mean_ci(p[m]); bins.append(dict(n=int(m.sum()), true_mean=float(y[m].mean()), pred_mean=float(p[m].mean()), pred_lo=lo, pred_hi=hi))
q25 = float(np.quantile(y, 0.25))
regions = {"bottom_quartile_true(<=Q25=%.3f)" % q25: mets(y[y <= q25], p[y <= q25]), "upper_75pct_true": mets(y[y > q25], p[y > q25]),
           "true_negative(dQ<0)": mets(y[y < 0], p[y < 0]), "true_nonneg(dQ>=0)": mets(y[y >= 0], p[y >= 0]), "overall_heldout": mets(y, p)}
results["heldout_bins_quantile15"] = bins; results["heldout_regions_predeclared"] = regions
json.dump(results, open(f"{X}/probe/probe_metrics_v1.json", "w"), indent=1)
print("\nREGIONS (held-out, deployed=%s):" % method)
for k, v in regions.items(): print(f"  {k:36s} n={v['n']:5d} P={v['pearson']:.3f} S={v['spearman']:.3f} slope={v['slope_pred_on_true']:.3f} rmse={v['rmse']:.3f}")

