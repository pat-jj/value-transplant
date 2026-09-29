#!/bin/bash
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# $SS_CODE/p1worker.sh GPU [GMEM] [MINFREE_MB] (0816 v2) — pulls P1 26-bar cells from $SS_POD/ss/p1queue.
# v2: waits for MINFREE_MB free on the GPU before each claim (teardown races burned the queue in v1),
# and requeues the spec + backs off when a run fails fast instead of marching on.
G=$1; GM=${2:-0.42}; MINFREE=${3:-110000}
cd $SS_POD/ss
PY=$SS_ENVS/verl/bin/python
export HF_HOME=$HF_HOME
QH=$SS_POD/ss/models/success_honest_think_merged
QC=$SS_POD/ss/models/success_cheater_hard_think_merged
TF=$SS_POD/ss/tasks/fromstart_strongprompt_qwen_0816.jsonl
CTQ="$(printf '\n\nLet me stop thinking and give my final answer now.\n')</think>$(printf '\n\n')"
mkdir -p $SS_POD/ss/p1claimed
FAILS=0
while true; do
  until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $G)" -gt "$MINFREE" ]; do sleep 60; done
  S=$(ls $SS_POD/ss/p1queue/*.spec 2>/dev/null | head -1); [ -z "$S" ] && break
  B=$(basename "$S"); C=$SS_POD/ss/p1claimed/gpu${G}_${B}
  mv "$S" "$C" 2>/dev/null || continue
  read -r LAM DIR AXF OUT < "$C"
  if [ "$DIR" = "rev" ]; then H=$QH; D=$QC; else H=$QC; D=$QH; fi
  echo "[p1worker $G] $OUT lam=$LAM $(date -u +%H:%M)"
  T0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$G env TRANSPLANT_THINK_CAP=32768 TRANSPLANT_CLOSE_MARKER_ID=151668 TRANSPLANT_CLOSE_TEXT="$CTQ" \
    $PY $SS_CODE/vllm_lockstep_transplant_0727.py \
    --host $H --donor $D --axis "$SS_POD/ss/activations/dspace/$AXF" \
    --mu-host 0 --mu-donor 0 --dims 1 --lam "$LAM" --resume \
    --prefix-file $TF --n 80 --temperature 0.7 --top-p 0.95 --seed 0 \
    --max-new 37888 --max-model-len 40960 --enable-thinking 1 --add-special-tokens 0 \
    --gpu-mem $GM --gpu-mem-donor $GM \
    --eos-ids 151645,151643 --out "$SS_POD/ss/out/${OUT}.json" \
    >> "$SS_POD/ss/logs/${OUT}.log" 2>&1
  RC=$?; DT=$(( $(date +%s) - T0 ))
  if [ $RC -ne 0 ] && [ $DT -lt 300 ]; then
    FAILS=$((FAILS+1))
    echo "[p1worker $G] $OUT FAST-FAIL rc=$RC ${DT}s — requeue + backoff (fail #$FAILS)"
    mv "$C" "$SS_POD/ss/p1queue/$B"
    [ $FAILS -ge 5 ] && { echo "[p1worker $G] too many fast-fails, exiting"; break; }
    sleep 240
  else
    FAILS=0
    echo "[p1worker $G] $OUT exit=$RC ${DT}s $(date -u +%H:%M)"
  fi
done
echo "[p1worker $G] done $(date -u +%H:%M)"
