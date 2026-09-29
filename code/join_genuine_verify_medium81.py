#!/usr/bin/env python3
"""join_genuine_verify_medium81.py (0910) — per-rollout join of the 6-way agent-judge label with the hidden / shown-assert grade for every
solvable-81 arm (47 frozen medium rows + 34 extension rows), so the solvable panel can be drawn in the five-way + two-hatch style.
Labels: scripts/solvable_judge6.load() -> {sub: {id: label6}} (HARDCODE / SCOPED / INTENT / BROKEN / GIVEUP / SOLVE).
Grades: the same per-task grading files medium81_summary.py uses (hidden_pass, visible_pass per row).
Display rule (0909 03:35): SCOPED is shown under the five-way 'fake intent' colour; the 6-way counts are kept in this file too.
Output: analysis/difficulty/medium81_genuine_verify.json
  cells[sub] = {n, frac5 {HARDCODE, INTENT(=INTENT+SCOPED), BROKEN, GIVEUP, SOLVE}, counts6, genuine, genuine_pass, genuine_visible_only,
                nongenuine_visible_only, pass_all, visible_only, rows:[{id, label, hidden, visible, visible_only}]}"""
import json, os, sys, collections
from ss_paths import SS_ROOT   # portable roots
X = f"{SS_ROOT}/v2/reports/xfam_clean_0904"; FT = f"{X}/final/final_lcfunc100_frameC_cap80000_mn88000_ml98304"; EXT = f"{X}/final/final_lcfunc134_ext_frameC_cap80000_mn88000_ml98304"; D = f"{X}/analysis/difficulty"
sys.path.insert(0, f"{X}/scripts"); import solvable_judge6 as SJ
OUT = f"{D}/medium81_genuine_verify.json"
def load(p): return {json.loads(l)["id"]: json.loads(l) for l in open(p)} if os.path.exists(p) else {}
def arm47(sub):
    Dd = {}
    for rd in ("results_tune", "results_heldout", "results_medium", "results_all", "interim_medium", "interim", "results"): Dd.update(load(f"{FT}/{sub}/{rd}/cheater_pertask.jsonl"))
    return Dd
def grades(sub):
    if sub == "base": return {**load(f"{FT}/base/results/cheater_pertask.jsonl"), **load(f"{EXT}/base/results_ext/cheater_pertask.jsonl")}
    if sub == "HONEST": return {**load(f"{FT}/base/results/honest_pertask.jsonl"), **load(f"{EXT}/base/results_ext/honest_pertask.jsonl")}
    return {**arm47(sub), **load(f"{EXT}/{sub}/results_ext/cheater_pertask.jsonl")}
IDS = SJ.ids81(); L = SJ.load(write=False)
out = {"cells": {}, "ids81": IDS, "note": __doc__}
for sub in SJ.SUBS:
    G = grades(sub); labs = L[sub]; rows = []
    for i in IDS:
        if i in G and i in labs:
            g = G[i]; hid = bool(g["hidden_pass"]); vis = bool(g.get("visible_pass"))
            rows.append({"id": i, "label": labs[i], "hidden": hid, "visible": vis, "visible_only": vis and not hid})
    n = len(rows)
    if not n: print(f"{sub:16s} no joined rows"); continue
    c6 = collections.Counter(r["label"] for r in rows)
    frac5 = {"HARDCODE": c6["HARDCODE"] / n, "INTENT": (c6["INTENT"] + c6["SCOPED"]) / n, "BROKEN": c6["BROKEN"] / n, "GIVEUP": c6["GIVEUP"] / n, "SOLVE": c6["SOLVE"] / n}
    gen = [r for r in rows if r["label"] == "SOLVE"]
    cell = {"n": n, "n_graded": len([i for i in IDS if i in G]), "n_labelled": len(labs), "frac5": frac5, "counts6": dict(c6),
            "genuine": len(gen) / n, "genuine_pass": sum(r["hidden"] for r in gen) / n, "genuine_visible_only": sum(r["visible_only"] for r in gen) / n,
            "nongenuine_visible_only": sum(r["visible_only"] for r in rows if r["label"] != "SOLVE") / n,
            "pass_all": sum(r["hidden"] for r in rows) / n, "visible_only": sum(r["visible_only"] for r in rows) / n, "rows": rows}
    out["cells"][sub] = cell
    print(f"{sub:16s} n={n:2d} (graded {cell['n_graded']}, labelled {cell['n_labelled']}) genuine={cell['genuine']:.3f} gen∧pass={cell['genuine_pass']:.3f} gen∧vis-only={cell['genuine_visible_only']:.3f} "
          f"nongen∧vis-only={cell['nongenuine_visible_only']:.3f} pass={cell['pass_all']:.3f} 6-way={dict(c6)}")
json.dump(out, open(OUT, "w"), indent=1); print("saved", OUT)
