#!/usr/bin/env python3
"""join_genuine_verify_g20b.py (0909) — per-rollout join of the gpt-oss five-way judge labels with the
hidden/visible verifier, for the 20 gpt-oss frame-B solvable cells drawn in row 3 of the goal-shift figure.

Provenance (matches the frozen composition_editor_data.json cell by cell, n checked):
  * 7 cells come from the 131k-window regeneration store  solvfs_fB128_g20b_131_0817/
      both anchors; fwd value/maze lam4; rev felt/value/maze lam4
  * 13 cells come from the original store                 solvfs_fB128_g20b_0807/
Per-rollout labels = reports/dspace_0715/judge_raw_5way_gptoss/<cell>.json.L<lam>.json ("labels", store order at
judge time; rows appended later are unjudged and excluded). Alignment verified by rationale-vs-rollout spot checks.
Verifier rows = analysis/gptoss_solverate/visible_only_g20b_fB128{,_131}.json (same stores; joined by row id).

Output: analysis/gptoss_solverate/genuine_verify_g20b_fB128.json
  cells[cell] = {src, n_rows, n_judged, genuine, genuine_pass, genuine_visible_only, nongenuine_visible_only,
                 pass_all, visible_only, counts, rows:[{id,label,full,visible,visible_only}]}
  fractions are over the judged rows (same denominator as the frozen composition).
"""
import glob, json, os
from collections import Counter
from ss_paths import SS_ROOT   # portable roots

V2 = f"{SS_ROOT}/v2"
X = f"{V2}/reports/xfam_clean_0904"
A = f"{V2}/reports/subdim_0726/solvfs_fB128_g20b_0807"
B = f"{V2}/reports/subdim_0726/solvfs_fB128_g20b_131_0817"
JR = f"{V2}/reports/dspace_0715/judge_raw_5way_gptoss"
VIS_A = f"{X}/analysis/gptoss_solverate/visible_only_g20b_fB128.json"
VIS_B = f"{X}/analysis/gptoss_solverate/visible_only_g20b_fB128_131.json"
OUT = f"{X}/analysis/gptoss_solverate/genuine_verify_g20b_fB128.json"

B_CELLS = {"sf_g20b_fB128_fwd_anchor_lam0", "sf_g20b_fB128_rev_anchor_lam0",
           "sf_g20b_fB128_fwd_value_lam4", "sf_g20b_fB128_fwd_maze_lam4",
           "sf_g20b_fB128_rev_felt_lam4", "sf_g20b_fB128_rev_value_lam4", "sf_g20b_fB128_rev_maze_lam4"}
CELLS = [f"sf_g20b_fB128_{s}_anchor_lam0" for s in ("fwd", "rev")] + \
        [f"sf_g20b_fB128_{s}_{g}_lam{k}" for s in ("fwd", "rev") for g in ("felt", "value", "maze") for k in (4, 16, 64)]

visA = json.load(open(VIS_A))["cells"]
visB = json.load(open(VIS_B))["cells"]
out = {"cells": {}, "note": __doc__}
for c in CELLS:
    src = B if c in B_CELLS else A
    vis = (visB if c in B_CELLS else visA)[c]
    vrows = list(vis.values())[0]["rows"]
    recs = list(json.load(open(f"{src}/{c}.json"))["by_alpha"].values())[0]
    raw = json.load(open(sorted(glob.glob(f"{JR}/{c}.json.L*.json"))[0]))["labels"]
    n = min(len(raw), len(recs))
    assert len(vrows) == len(recs), (c, len(vrows), len(recs))
    byid = {r["id"]: r for r in vrows}
    assert len(byid) == len(vrows), (c, "duplicate ids")
    rows, cnt = [], Counter()
    for i in range(n):
        rid = recs[i]["id"]; v = byid[rid]; lab = raw[i]
        rows.append({"id": rid, "label": lab, "full": bool(v["full"]), "visible": bool(v["visible"]), "visible_only": bool(v["visible_only"])})
        cnt[lab] += 1
    g = [r for r in rows if r["label"] == "SOLVE"]
    cell = {
        "src": "131_0817" if src == B else "0807", "n_rows": len(recs), "n_judged": n,
        "genuine": len(g) / n, "genuine_pass": sum(r["full"] for r in g) / n,
        "genuine_visible_only": sum(r["visible_only"] for r in g) / n,
        "nongenuine_visible_only": sum(r["visible_only"] for r in rows if r["label"] != "SOLVE") / n,
        "pass_all": sum(r["full"] for r in rows) / n, "visible_only": sum(r["visible_only"] for r in rows) / n,
        "pass_all_allrows": sum(bool(v["full"]) for v in vrows) / len(vrows),
        "counts": dict(cnt), "rows": rows,
    }
    out["cells"][c] = cell
    print(f"{c:36s} src={cell['src']:8s} judged={n:3d}/{len(recs):3d} genuine={cell['genuine']:.3f} gen∧pass={cell['genuine_pass']:.3f} "
          f"gen∧vis-only={cell['genuine_visible_only']:.3f} nongen∧vis-only={cell['nongenuine_visible_only']:.3f} pass={cell['pass_all']:.3f} (all rows {cell['pass_all_allrows']:.3f})")
json.dump(out, open(OUT, "w"), indent=1)
print("saved", OUT)
