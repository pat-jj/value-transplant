#!/usr/bin/env python3
"""Post-fill: merge gqc _s0/_s1 halves -> single cell, 5-way judge into gsp_5way_0816
(one file per judge invocation), then re-render composition v3. CPU-only (judge = Opus API)."""
import json, os, subprocess
from ss_paths import SS_ROOT, SS_ENVS   # portable roots
V2 = f"{SS_ROOT}/v2"
RAW = f"{V2}/reports/subdim_0726/gqc_0820"
STORE = f"{V2}/reports/subdim_0726/q32_0816/gsp_5way_0816.json"
PY = f"{SS_ENVS}/analysis/bin/python"
JUDGE = f"{V2}/judge_decider_5way_gptoss.py"
FIGDIR = f"{SS_ROOT}/figures"
CELLS = ([f"gqc_felt_rev_g{g}" for g in (4, 16, 32)] +
         [f"gqc_{a}_{d}_g{g}" for a in ("val", "maze") for d in ("fwd", "rev") for g in (4, 16, 32)])
# For each cell: concatenate its s0/s1 shard halves into one JSON, then 5-way judge it.
for cell in CELLS:
    merged = None
    # combine the by_alpha rows from both shard halves into `merged`
    for half in ("s0", "s1"):
        p = f"{RAW}/{cell}_{half}.json"
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        if merged is None:
            merged = d
        else:
            ba, bd = merged.get("by_alpha", {}), d.get("by_alpha", {})
            for k, rows in bd.items():
                ba.setdefault(k, [])
                ba[k] += rows
            merged["by_alpha"] = ba
    if merged is None:
        print(f"[postfill] SKIP {cell} (no shards yet)", flush=True)
        continue
    # total record count across all alpha buckets in the merged cell
    n = sum(len(v) for v in merged.get("by_alpha", {}).values())
    mp = f"{RAW}/{cell}.json"
    json.dump(merged, open(mp, "w"))
    print(f"[postfill] merged {cell} -> n={n}; judging...", flush=True)
    subprocess.run([PY, JUDGE, mp], env=dict(os.environ, STORE5=STORE), cwd=V2, timeout=3600)
# FROZEN-JSON pipeline: the judged cells go to the store above, but the FIGURE is
# driven by the frozen composition_editor_data.json. sync fills ONLY missing (null) cells; frozen
# results are never touched. Then render from the JSON. (Do NOT run fig_composition_gptoss_v3.py
# here -- it re-pulls the stores and would un-freeze the figure.)
print("[postfill] syncing ONLY missing cells into frozen JSON...", flush=True)
subprocess.run([PY, f"{V2}/sync_json_from_store.py"], cwd=V2, timeout=120)
print("[postfill] rendering from frozen JSON...", flush=True)
subprocess.run([PY, f"{FIGDIR}/src/fig_composition_from_json.py"], cwd=FIGDIR, timeout=600)
print("[postfill] DONE", flush=True)
