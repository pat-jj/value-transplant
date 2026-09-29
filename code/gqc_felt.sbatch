#!/bin/bash
#SBATCH --job-name=gqcfelt
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --nice=5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-23
#SBATCH --output=%x_%A_%a.out
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# GPT-OSS SELF-RATING dose ladder under the QWEN-MATCHED cap:
# doses 2/4/6/8/16/32 x fwd/rev, 2 halves each. task/2 = cell index, task%2 = half.
set -uo pipefail
export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PATH="$SS_ENVS/verl/bin:$PATH"
cleanup(){ kill -TERM -$$ 2>/dev/null; wait; }; trap cleanup SIGTERM SIGINT SIGQUIT
cd $SS_ROOT/v2
CT="$(printf '\n\nLet me stop thinking and give my final answer now.')<|end|><|start|>assistant<|channel|>final<|message|>"
export TRANSPLANT_THINK_CAP=32768 TRANSPLANT_CLOSE_MARKER_ID=200005 TRANSPLANT_CLOSE_TEXT="$CT"
# 0820 fix: harmony opens EVERY channel with <|channel|> (200005), so natural-close detection
# must be the pair <|channel|> final (17196) — else the cap is disarmed at step 1.
export TRANSPLANT_CLOSE_MARKER_NEXT=17196
DOSES=(2 4 6 8 16 32)
C=$((SLURM_ARRAY_TASK_ID / 2)); S=$((SLURM_ARRAY_TASK_ID % 2))
LAM=${DOSES[$((C % 6))]}
if [ $C -lt 6 ]; then DIR=fwd; H=models/gptoss20b_cheater_bf16; D=models/gptoss20b_honest_bf16;
else DIR=rev; H=models/gptoss20b_honest_bf16; D=models/gptoss20b_cheater_bf16; fi
OUT="gqc_felt_${DIR}_g${LAM}"
echo "== gqcfelt $OUT half=$S =="
srun python $SS_CODE/gptoss_transplant_driver.py \
  --host "$H" --donor "$D" \
  --axis activations/dspace/preDIM_G20BC_BAL_L14.npz --mu-host 0 --mu-donor 0 --dims 1 --lam "${LAM}.0" --resume \
  --prefix-file "tasks/gqc_0820/gqc_h${S}.jsonl" --n 47 --temperature 0.7 --top-p 0.95 --seed 0 \
  --max-new 40960 --max-model-len 49152 \
  --eos-ids 200002,199999,200012 --add-special-tokens 0 --enable-thinking 1 \
  --out "reports/subdim_0726/gqc_0820/${OUT}_s${S}.json"
echo "== gqcfelt $OUT half=$S done (exit $?) =="
