#!/usr/bin/env python3
"""JOB 3 Stage-2 item 1: by_alpha -> verifier adapter (design
reports/subdim_0726/JOB3_UPLIFT_DESIGN_0805.md SS3-Stage-2).

Loads each transplant/steering cell store (`by_alpha` format written by
vllm_lockstep_transplant_0727.py and the steer ports: {..., "by_alpha": {"+0.000": [
{id, gen, finished, n_tokens, ...}, ...]}}), maps each row id `leetcode_X#sY` -> base task
`leetcode_X` (split on '#'), and runs verifiers.verify(task, gen) against the FULL bank
(held-out asserts included) from the grading companion tasks/solvperf_grading_0805.json.

Conventions (design SS3-Stage-2 item 3):
  - PRIMARY  `pass_all`: BROKEN/unfinished/truncated = verified FAIL (every row in the
    denominator; conservative, biases against uplift).
  - SECONDARY `pass_finished_only`: finished rows only (the 0801 convention for
    truncation-confounded rates). `finished` is truthy for lockstep's bool True and
    rollout-style "stop".

Output reports/subdim_0726/spq_verify_0805.json: per-(cell,akey) summaries + per-row
`passed` records. Reload-merge-save per cell file (idempotent, judge_decider convention).

RUN CONTEXT: CPU Slurm job — verifiers.py EXECUTES model-generated code ("Intended to run
INSIDE a Slurm allocation"). No GPU needed:
  $SS_ROOT/git/slurm-tools/run.sh --name spq_verify --gpus 0 --mem 32G \
    --time 02:00:00 -- python3 verify_transplant_cells.py \
    --cells 'reports/subdim_0726/spq_tf_lam0_s*.json' reports/subdim_0726/spq_href_0805.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from ss_paths import SS_ROOT   # portable roots

V2 = f"{SS_ROOT}/v2"
sys.path.insert(0, V2)
import verifiers as V  # noqa: E402


def is_finished(rec: dict) -> bool:
    f = rec.get("finished")
    return f is True or f == "stop"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cells", nargs="+", required=True,
                    help="cell store file list (globs ok): by_alpha-format .json")
    ap.add_argument("--grading", default=f"{V2}/tasks/solvperf_grading_0805.json",
                    help="companion json keyed by base id with FULL tests + entry_point")
    ap.add_argument("--out", default=f"{V2}/reports/subdim_0726/spq_verify_0805.json")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    files = []
    for pat in args.cells:
        hits = sorted(glob.glob(pat)) or sorted(glob.glob(os.path.join(V2, pat)))
        if not hits:
            raise SystemExit(f"[verify] no cell files match {pat!r}")
        files += hits
    files = list(dict.fromkeys(files))  # dedupe, keep order

    grading = json.load(open(args.grading))["tasks"]
    print(f"[verify] grading companion: {len(grading)} base tasks; {len(files)} cell files",
          flush=True)

    out = (json.load(open(args.out)) if os.path.exists(args.out)
           else {"mode": "solvperf_by_alpha_verifier", "grading": args.grading,
                 "convention": "primary pass_all: BROKEN/unfinished = fail; "
                               "secondary pass_finished_only",
                 "cells": {}, "rows": {}})

    for path in files:
        cell = os.path.basename(path).replace(".json", "")
        d = json.load(open(path))
        ba = d.get("by_alpha")
        if not isinstance(ba, dict):
            raise SystemExit(f"[verify] {path} has no by_alpha dict — not a cell store")
        cell_rows, cell_sum = [], {}
        for akey in sorted(ba, key=float):
            recs = ba[akey]
            if not recs:
                continue
            missing = sorted({r["id"].split("#")[0] for r in recs} - set(grading))
            if missing:
                raise SystemExit(f"[verify] {cell} L{akey}: {len(missing)} base ids not in "
                                 f"grading companion (e.g. {missing[:5]}) — wrong --grading?")

            def score(rec):
                base = rec["id"].split("#")[0]
                res = V.verify(grading[base], rec.get("gen", "") or "")
                return {"cell": cell, "akey": akey, "id": rec["id"], "base": base,
                        "finished": is_finished(rec), "n_tokens": rec.get("n_tokens"),
                        "passed": bool(res.get("passed")),
                        "info": str(res.get("info", ""))[-160:]}

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                rows = list(ex.map(score, recs))
            cell_rows += rows
            n = len(rows)
            npass = sum(r["passed"] for r in rows)
            fin = [r for r in rows if r["finished"]]
            nfin, npassfin = len(fin), sum(r["passed"] for r in fin)
            cell_sum[akey] = {
                "n": n, "n_pass": npass, "pass_all": round(npass / n, 4),
                "n_finished": nfin, "n_pass_finished": npassfin,
                "pass_finished_only": round(npassfin / nfin, 4) if nfin else None,
                "finish_rate": round(nfin / n, 4),
            }
            print(f"[verify] {cell} L{akey}: pass_all {npass}/{n} = {npass/n:.1%} | "
                  f"finished-only {npassfin}/{nfin}"
                  f"{f' = {npassfin/nfin:.1%}' if nfin else ''} | finish {nfin/n:.1%}",
                  flush=True)

        # reload-merge-save per file (judge_decider.py night-fix convention)
        try:
            out = json.load(open(args.out)) if os.path.exists(args.out) else out
        except Exception:
            pass
        out.setdefault("cells", {})[cell] = cell_sum
        out.setdefault("rows", {})[cell] = cell_rows
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"[verify] merged {cell} -> {args.out}", flush=True)

    print(f"[verify] DONE ({len(files)} files). Secondary mechanism table: judge the same "
          f"cells with judge_decider.py under flock, then join labels x passed by row "
          f"order (solvperf_gate_report_0805.py does this join for the calibration cells).",
          flush=True)


if __name__ == "__main__":
    main()
