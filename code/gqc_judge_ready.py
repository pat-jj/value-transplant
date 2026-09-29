#!/usr/bin/env python3
"""Targeted: merge+5way-judge ONLY the gqc felt_rev cells that have
crossed n>=80, into gsp_5way_0816 (one file per judge invocation). Skips partial/absent.
CPU-only (judge = Opus API). Leaves g32/val/maze to gqcpost when the array finishes."""
import json, os, subprocess, sys
from ss_paths import SS_ROOT, SS_ENVS   # portable roots
V2 = f"{SS_ROOT}/v2"
RAW = f"{V2}/reports/subdim_0726/gqc_0820"
STORE = f"{V2}/reports/subdim_0726/q32_0816/gsp_5way_0816.json"
PY = f"{SS_ENVS}/analysis/bin/python"
JUDGE = f"{V2}/judge_decider_5way_gptoss.py"
FIGDIR = f"{SS_ROOT}/figures"
MIN_N = 80
CELLS = sys.argv[1:] or ["gqc_felt_rev_g4", "gqc_felt_rev_g16"]
done = []
for cell in CELLS:
    merged = None
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
        print(f"[judge-ready] SKIP {cell} (no shards)", flush=True)
        continue
    n = sum(len(v) for v in merged.get("by_alpha", {}).values())
    if n < MIN_N:
        print(f"[judge-ready] SKIP {cell} (n={n} < {MIN_N})", flush=True)
        continue
    mp = f"{RAW}/{cell}.json"
    json.dump(merged, open(mp, "w"))
    print(f"[judge-ready] merged {cell} -> n={n}; judging...", flush=True)
    r = subprocess.run([PY, JUDGE, mp], env=dict(os.environ, STORE5=STORE), cwd=V2, timeout=5400)
    print(f"[judge-ready] {cell} judge exit={r.returncode}", flush=True)
    if r.returncode == 0:
        done.append(cell)
print(f"[judge-ready] judged={done}; syncing missing cells into frozen JSON...", flush=True)
subprocess.run([PY, f"{V2}/sync_json_from_store.py"], cwd=V2, timeout=120)
print("[judge-ready] rendering from frozen JSON...", flush=True)
subprocess.run([PY, f"{FIGDIR}/src/fig_composition_from_json.py"], cwd=FIGDIR, timeout=600)
print("[judge-ready] DONE", flush=True)
