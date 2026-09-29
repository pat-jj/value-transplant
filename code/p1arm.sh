#!/bin/bash
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# $SS_CODE/p1arm.sh GPU WAIT_OUTNAME [GMEM] (0816) — waits for the driver writing --out .../WAIT_OUTNAME.json
# to end (pattern includes '--out' so the waiter never matches itself), then runs p1worker.
G=$1; WP=$2; GM=${3:-0.42}
while pgrep -f -- "--out $SS_POD/ss/out/${WP}.json" >/dev/null; do sleep 120; done
echo "[p1arm] GPU$G free -> p1worker $(date -u +%H:%M)"
exec bash $SS_CODE/p1worker.sh $G $GM ${4:-110000}
