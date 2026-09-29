#!/bin/bash
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# $SS_CODE/podstrongp.sh (0816, "prioritize strong-prompt reverse, pilot results ASAP")
# GPU5: fs_sp_anchor_H (lam0)  GPU7: fs_sp_felt_rev_g16  GPU6 (stacked w/ eval): fs_sp_felt_rev_g32
# All capped convention, n=80 (40 tasks x 2 seeds), strengthened prompt ("graded only on the
# exact test cases shown"). Rows save incrementally — judge partials for the pilot readout.
cd $SS_POD/ss
PY=$SS_ENVS/verl/bin/python
export HF_HOME=$HF_HOME
QH=$SS_POD/ss/models/success_honest_think_merged
QC=$SS_POD/ss/models/success_cheater_hard_think_merged
AX=$SS_POD/ss/activations/dspace/preDIM_QB_L21.npz
TF=$SS_POD/ss/tasks/fromstart_strongprompt_qwen_0816.jsonl
CTQ="$(printf '\n\nLet me stop thinking and give my final answer now.\n')</think>$(printf '\n\n')"

launch() { # GPU LAM OUT GMEM EXTRA_ENV...
  local G=$1 LAM=$2 OUT=$3 GM=$4; shift 4
  CUDA_VISIBLE_DEVICES=$G nohup env "$@" TRANSPLANT_THINK_CAP=32768 TRANSPLANT_CLOSE_MARKER_ID=151668 TRANSPLANT_CLOSE_TEXT="$CTQ" \
    $PY $SS_CODE/vllm_lockstep_transplant_0727.py \
    --host $QH --donor $QC --axis $AX --mu-host 0 --mu-donor 0 --dims 1 --lam "$LAM" --resume \
    --prefix-file $TF --n 80 --temperature 0.7 --top-p 0.95 --seed 0 \
    --max-new 37888 --max-model-len 40960 --enable-thinking 1 --add-special-tokens 0 \
    --gpu-mem $GM --gpu-mem-donor $GM \
    --eos-ids 151645,151643 --out "$SS_POD/ss/out/${OUT}.json" \
    > "$SS_POD/ss/logs/${OUT}.log" 2>&1 &
  echo "launched $OUT on GPU$G"
}

launch 5 0.0  fs_sp_anchor_H      0.42 TRANSPLANT_SKIP_FORKGATE=1
launch 7 16.0 fs_sp_felt_rev_g16  0.42
launch 6 32.0 fs_sp_felt_rev_g32  0.25
echo STRONGP_LAUNCHED
