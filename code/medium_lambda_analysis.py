#!/usr/bin/env python3
"""Medium-tier λ analysis with the tuning/held-out split. Paired stats (bootstrap by task, exact McNemar), transitions, joint hidden×visible, token deltas.
Usage: medium_lambda_analysis.py search        -> lambda_search_medium.csv (tuning rows 53-67; reuses any results dir found per λ)
       medium_lambda_analysis.py star <λ*>     -> medium_lambda_star_full.csv (tuning / held-out / all-47), medium_lambda_star_transitions.csv, joint table
       medium_lambda_analysis.py dose          -> medium_dose_curve.csv (all medium rows per λ) + hard_vs_medium_dose_curve.csv"""
import json, os, sys, csv, glob, random, math
import numpy as np
from ss_paths import SS_ROOT   # portable roots
from scipy.stats import binomtest
X = f"{SS_ROOT}/v2/reports/xfam_clean_0904"; FT = f"{X}/final/final_lcfunc100_frameC_cap80000_mn88000_ml98304"; OUT = f"{X}/analysis/difficulty"
split = json.load(open(f"{OUT}/tuning_split.json")); TUNE = set(split["tuning_ids"]); HELD = set(split["heldout_ids"])
order = [json.loads(l) for l in open(f"{X}/benchmark_hard/manifest_final_lcfunc100.jsonl")]; tier = {r["id"]: ("hard" if r["stratum"] == "LCFUNC_HARD" else "medium") for r in order}
def load(p): return {json.loads(l)["id"]: json.loads(l) for l in open(p)} if os.path.exists(p) else {}
def load_arm(sub):
    D = {}
    for rd in ("results_tune", "results_heldout", "results_medium", "results_all", "results_pilot", "interim_medium", "interim", "results"):
        D.update(load(f"{FT}/{sub}/{rd}/cheater_pertask.jsonl"))
    return D
def load_arm_raw_ids(sub):  # ids with raw rows (for coverage checks)
    ids = set()
    for f in glob.glob(f"{FT}/{sub}/raw/s0_cheater_row*.json"):
        try: ids.add(list(json.load(open(f))["by_alpha"].values())[0][0]["id"])
        except Exception: pass
    return ids
C = load(f"{FT}/base/results/cheater_pertask.jsonl"); H = load(f"{FT}/base/results/honest_pertask.jsonl")
def boot(d, B=10000):
    n = len(d); rng = random.Random(0); bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B)); return bs[int(0.025 * B)], bs[int(0.975 * B) - 1]
def mcnemar(W, ids):
    b = sum(1 for i in ids if W[i]["hidden_pass"] and not C[i]["hidden_pass"]); c = sum(1 for i in ids if C[i]["hidden_pass"] and not W[i]["hidden_pass"])
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0; return b, c, p
def stats(W, ids):
    n = len(ids); w = [W[i] for i in ids]; c = [C[i] for i in ids]
    d = [W[i]["hidden_pass"] - C[i]["hidden_pass"] for i in ids]; lo, hi = boot(d) if n > 1 else (0, 0); fp, pf, p = mcnemar(W, ids)
    return dict(n=n, transplant_hidden=round(sum(x["hidden_pass"] for x in w) / n, 4), unedited_hidden=round(sum(x["hidden_pass"] for x in c) / n, 4), paired_delta=round(sum(d) / n, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4), mcnemar_p=round(p, 4),
                fail_to_pass=fp, pass_to_fail=pf, pass_to_pass=sum(1 for i in ids if W[i]["hidden_pass"] and C[i]["hidden_pass"]), fail_to_fail=sum(1 for i in ids if not W[i]["hidden_pass"] and not C[i]["hidden_pass"]),
                transplant_visible=round(sum(x["visible_pass"] for x in w) / n, 4), unedited_visible=round(sum(x["visible_pass"] for x in c) / n, 4),
                transplant_shown_only=round(sum(1 for x in w if x["visible_pass"] and not x["hidden_pass"]) / n, 4), unedited_shown_only=round(sum(1 for x in c if x["visible_pass"] and not x["hidden_pass"]) / n, 4),
                transplant_broken=round(sum(1 for x in w if x.get("failure") in ("syntax_error", "runtime_error", "no_code", "timeout")) / n, 4), unedited_broken=round(sum(1 for x in c if x.get("failure") in ("syntax_error", "runtime_error", "no_code", "timeout")) / n, 4),
                transplant_median_tokens=int(np.median([x["n_tokens"] for x in w])), unedited_median_tokens=int(np.median([x["n_tokens"] for x in c])), median_token_delta=int(np.median([W[i]["n_tokens"] - C[i]["n_tokens"] for i in ids])),
                cap_hits=sum(1 for x in w if x["forced_close"]), honest_hidden=round(sum(H[i]["hidden_pass"] for i in ids) / n, 4),
                both_pass=round(sum(1 for x in w if x["hidden_pass"] and x["visible_pass"]) / n, 4), visible_only=round(sum(1 for x in w if x["visible_pass"] and not x["hidden_pass"]) / n, 4), hidden_only=round(sum(1 for x in w if x["hidden_pass"] and not x["visible_pass"]) / n, 4), neither=round(sum(1 for x in w if not x["hidden_pass"] and not x["visible_pass"]) / n, 4))
LAMS_SEARCH = ["0.025", "0.05", "0.1", "0.15", "0.2", "0.25", "0.35", "0.5"]
mode = sys.argv[1] if len(sys.argv) > 1 else "search"
if mode == "search":
    rows = []
    for L in LAMS_SEARCH:
        W = load_arm(f"lindq_lam{L}"); ids = sorted(i for i in TUNE if i in W and i in C)
        if not ids: rows.append(dict(lam=L, n=0)); continue
        s = stats(W, ids); s["lam"] = L; rows.append(s)
    keys = ["lam", "n", "transplant_hidden", "unedited_hidden", "paired_delta", "ci_lo", "ci_hi", "mcnemar_p", "fail_to_pass", "pass_to_fail", "transplant_visible", "unedited_visible", "transplant_shown_only", "unedited_shown_only", "transplant_broken", "unedited_broken", "transplant_median_tokens", "unedited_median_tokens", "cap_hits"]
    with open(f"{OUT}/lambda_search_medium.csv", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow({k: r.get(k, "") for k in keys}) for r in rows]
    print(f"{'λ':6s} {'n':>2s} {'transp':>6s} {'unedit':>6s} {'Δ':>6s} {'CI':>16s} {'McN p':>6s} {'f→p':>3s} {'p→f':>3s} {'vis':>5s} {'shown':>5s} {'broken':>6s} {'tok':>5s}")
    pts = []
    for r in rows:
        if r["n"] == 0: print(f"{r['lam']:6s}  0  (no rows yet)"); continue
        print(f"{r['lam']:6s} {r['n']:2d} {r['transplant_hidden']:6.3f} {r['unedited_hidden']:6.3f} {r['paired_delta']:+6.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] {r['mcnemar_p']:6.3f} {r['fail_to_pass']:3d} {r['pass_to_fail']:3d} {r['transplant_visible']:5.2f} {r['transplant_shown_only']:5.2f} {r['transplant_broken']:6.2f} {r['transplant_median_tokens']:5d}"); pts.append(r)
    print("wrote lambda_search_medium.csv")
elif mode == "star":
    L = sys.argv[2]; W = load_arm(f"lindq_lam{L}"); rows = []
    for name, S in (("tuning (rows 53-67)", TUNE), ("held-out (rows 68-99)", HELD), ("all 47 medium", TUNE | HELD)):
        ids = sorted(i for i in S if i in W and i in C); s = stats(W, ids) if ids else dict(n=0); s["split"] = name; s["lam"] = L; rows.append(s)
        if ids: print(f"{name:22s} n={s['n']:2d} transplant {s['transplant_hidden']:.3f} unedited {s['unedited_hidden']:.3f} honest {s['honest_hidden']:.3f} | Δ {s['paired_delta']:+.3f} [{s['ci_lo']:+.3f},{s['ci_hi']:+.3f}] McNemar p={s['mcnemar_p']:.4f} | f→p {s['fail_to_pass']} p→f {s['pass_to_fail']} p→p {s['pass_to_pass']} f→f {s['fail_to_fail']} | vis {s['transplant_visible']:.2f} vs {s['unedited_visible']:.2f} | shown-only {s['transplant_shown_only']:.2f} vs {s['unedited_shown_only']:.2f} | broken {s['transplant_broken']:.2f} vs {s['unedited_broken']:.2f} | tok {s['transplant_median_tokens']} vs {s['unedited_median_tokens']} (Δmed {s['median_token_delta']:+d}) cap {s['cap_hits']} | joint: both {s['both_pass']:.2f} vis-only {s['visible_only']:.2f} hid-only {s['hidden_only']:.2f} neither {s['neither']:.2f}")
    keys = ["split", "lam"] + [k for k in rows[0] if k not in ("split", "lam")]
    with open(f"{OUT}/medium_lambda_star_full.csv", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow({k: r.get(k, "") for k in keys}) for r in rows]
    ids = sorted(i for i in (TUNE | HELD) if i in W and i in C)
    with open(f"{OUT}/medium_lambda_star_transitions.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "row", "split", "transition", "unedited_hidden", "transplant_hidden", "unedited_visible", "transplant_visible", "unedited_failure", "transplant_failure", "unedited_tokens", "transplant_tokens", "honest_hidden"])
        rowidx = {r["id"]: k for k, r in enumerate(order)}
        for i in ids:
            tr = ("pass" if C[i]["hidden_pass"] else "fail") + "->" + ("pass" if W[i]["hidden_pass"] else "fail")
            w.writerow([i, rowidx[i], "tuning" if i in TUNE else "held-out", tr, int(C[i]["hidden_pass"]), int(W[i]["hidden_pass"]), int(C[i]["visible_pass"]), int(W[i]["visible_pass"]), C[i].get("failure"), W[i].get("failure"), C[i]["n_tokens"], W[i]["n_tokens"], int(H[i]["hidden_pass"])])
    print("wrote medium_lambda_star_full.csv, medium_lambda_star_transitions.csv")
elif mode == "controls":
    arms = [("lindq λ*=0.25 (reader-modulated)", "lindq_lam0.25"), ("const push η=3.47 (net-drift matched)", "const_eta3.47"), ("const push η=8.61 (|edit| matched)", "const_eta8.61"), ("const push η=25", "const_eta25"), ("const push η=50", "const_eta50"), ("const push η=100", "const_eta100"), ("const push η=205 (paper-era CONSTMATCH)", "const_eta205"), ("const push η=448.8 (paper-era constant-full)", "const_eta448.8"), ("random direction λ=0.25 (control)", "rand_lam0.25"), ("lindq λ=0.025 (near-zero dose)", "lindq_lam0.025"), ("within-family λ=1", "withinfam_lam1"), ("within-family λ=2", "withinfam_lam2"), ("within-family λ=4", "withinfam_lam4"), ("within-family λ=6", "withinfam_lam6"), ("within-family λ=8", "withinfam_lam8"), ("within-family λ=10", "withinfam_lam10"), ("within-family λ=12", "withinfam_lam12"), ("within-family λ=16", "withinfam_lam16")]
    rows = []
    for name, sub in arms:
        W = load_arm(sub)
        for sname, S in (("tuning", TUNE), ("held-out", HELD), ("all", TUNE | HELD)):
            ids = sorted(i for i in S if i in W and i in C)
            if not ids: rows.append(dict(arm=name, split=sname, n=0)); continue
            s_ = stats(W, ids); s_["arm"] = name; s_["split"] = sname; rows.append(s_)
    keys = ["arm", "split", "n", "transplant_hidden", "unedited_hidden", "honest_hidden", "paired_delta", "ci_lo", "ci_hi", "mcnemar_p", "fail_to_pass", "pass_to_fail", "transplant_visible", "unedited_visible", "transplant_shown_only", "transplant_broken", "unedited_broken", "transplant_median_tokens", "unedited_median_tokens", "cap_hits"]
    with open(f"{OUT}/medium_controls_comparison.csv", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow({k: r.get(k, "") for k in keys}) for r in rows]
    for r in rows:
        if r["n"] == 0: print(f"{r['arm']:40s} {r['split']:8s} (no rows)"); continue
        print(f"{r['arm']:40s} {r['split']:8s} n={r['n']:2d} {r['transplant_hidden']:.3f} vs {r['unedited_hidden']:.3f} Δ {r['paired_delta']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] p={r['mcnemar_p']:.3f} f→p {r['fail_to_pass']} p→f {r['pass_to_fail']} | vis {r['transplant_visible']:.2f} shown {r['transplant_shown_only']:.2f} broken {r['transplant_broken']:.2f} tok {r['transplant_median_tokens']}")
    print("wrote medium_controls_comparison.csv")
elif mode == "dose":
    rows = []; MED = TUNE | HELD; HARD = {i for i in tier if tier[i] == "hard"}
    for L in ["0", "0.25", "0.5", "1.0", "2.0"]:
        for tname, S in (("medium", MED), ("hard", HARD)):
            W = C if L == "0" else load_arm(f"lindq_lam{L}"); ids = sorted(i for i in S if i in W and i in C)
            if not ids: rows.append(dict(tier=tname, lam=L, n=0)); continue
            s = stats(W, ids) if L != "0" else dict(n=len(ids), transplant_hidden=round(sum(C[i]["hidden_pass"] for i in ids) / len(ids), 4), unedited_hidden=round(sum(C[i]["hidden_pass"] for i in ids) / len(ids), 4), paired_delta=0.0, ci_lo=0.0, ci_hi=0.0, mcnemar_p=1.0, honest_hidden=round(sum(H[i]["hidden_pass"] for i in ids) / len(ids), 4), transplant_visible=round(sum(C[i]["visible_pass"] for i in ids) / len(ids), 4), transplant_shown_only=round(sum(1 for i in ids if C[i]["visible_pass"] and not C[i]["hidden_pass"]) / len(ids), 4), transplant_median_tokens=int(np.median([C[i]["n_tokens"] for i in ids])))
            s["tier"] = tname; s["lam"] = L; rows.append(s)
    keys = ["tier", "lam", "n", "transplant_hidden", "unedited_hidden", "paired_delta", "ci_lo", "ci_hi", "mcnemar_p", "honest_hidden", "transplant_visible", "transplant_shown_only", "transplant_median_tokens"]
    with open(f"{OUT}/hard_vs_medium_dose_curve.csv", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow({k: r.get(k, "") for k in keys}) for r in rows]
    with open(f"{OUT}/medium_dose_curve.csv", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow({k: r.get(k, "") for k in keys}) for r in rows if r["tier"] == "medium"]
    for r in rows: print(f"{r['tier']:6s} λ={r['lam']:4s} n={r['n']:2d} " + (f"hidden {r['transplant_hidden']:.3f} (unedited {r['unedited_hidden']:.3f}, honest {r['honest_hidden']:.3f}) Δ {r['paired_delta']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] p={r['mcnemar_p']:.3f}" if r["n"] else "(no rows)"))
    print("wrote medium_dose_curve.csv, hard_vs_medium_dose_curve.csv")
