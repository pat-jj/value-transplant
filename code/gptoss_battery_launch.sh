#!/bin/bash
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# Steering-battery cell launcher (pod). Runs on the pod under bash (word-splitting OK here).
# usage: $SS_CODE/gptoss_battery_launch.sh <model_dir> <axis_npz> <layer> <fork_jsonl> <mode> "gpu:dose:name ..."
set -u
MODEL=$1; AXIS=$2; LAYER=$3; FORK=$4; MODE=$5; CELLS=$6
PY=$SS_ENVS/verl/bin/python
OUT=$SS_POD/pilot/v2gptoss
export HF_HOME=$HF_HOME VLLM_LOGGING_LEVEL=WARNING
mkdir -p $OUT/battery $OUT/logs
for spec in $CELLS; do
  IFS=':' read -r GPU DOSE NAME <<< "$spec"
  CUDA_VISIBLE_DEVICES=$GPU nohup "$PY" $OUT/vllm_fork_steer_gptoss.py \
    --model "$MODEL" --fork "$FORK" --axis "$AXIS" --layer "$LAYER" \
    --dose "$DOSE" --n 100 --mode "$MODE" \
    --out "$OUT/battery/${NAME}.json" > "$OUT/logs/battery_${NAME}.log" 2>&1 &
  echo "launched ${NAME} (gpu $GPU dose $DOSE) pid $!"
done
