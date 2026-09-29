#!/bin/bash
#SBATCH --job-name=gqcfill
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --nice=5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-29%8
#SBATCH --output=%x_%A_%a.out
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# GPT-OSS transplant/control cells under the QWEN-MATCHED cap (think 32768 / total 40960,
# forced close). 12 cells x 2 halves. task/2 = cell, task%2 = half.
set -uo pipefail
export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PATH="$SS_ENVS/verl/bin:$PATH"
cleanup(){ kill -TERM -$$ 2>/dev/null; wait; }; trap cleanup SIGTERM SIGINT SIGQUIT
cd $SS_ROOT/v2
CT="$(printf '\n\nLet me stop thinking and give my final answer now.')<|end|><|start|>assistant<|channel|>final<|message|>"
export TRANSPLANT_THINK_CAP=32768 TRANSPLANT_CLOSE_MARKER_ID=200005 TRANSPLANT_CLOSE_TEXT="$CT"
CFG=(
  "rev preDIM_G20BC_BAL_L14     4  gqc_felt_rev_g4"  "rev preDIM_G20BC_BAL_L14     16 gqc_felt_rev_g16"  "rev preDIM_G20BC_BAL_L14     32 gqc_felt_rev_g32"
  "fwd value_axis_0731_g20b_L14 4  gqc_val_fwd_g4"   "fwd value_axis_0731_g20b_L14 16 gqc_val_fwd_g16"   "fwd value_axis_0731_g20b_L14 32 gqc_val_fwd_g32"
  "rev value_axis_0731_g20b_L14 4  gqc_val_rev_g4"   "rev value_axis_0731_g20b_L14 16 gqc_val_rev_g16"   "rev value_axis_0731_g20b_L14 32 gqc_val_rev_g32"
  "fwd maze_g20b_L14            4  gqc_maze_fwd_g4"  "fwd maze_g20b_L14            16 gqc_maze_fwd_g16"  "fwd maze_g20b_L14            32 gqc_maze_fwd_g32"
  "rev maze_g20b_L14            4  gqc_maze_rev_g4"  "rev maze_g20b_L14            16 gqc_maze_rev_g16"  "rev maze_g20b_L14            32 gqc_maze_rev_g32"
)
C=$((SLURM_ARRAY_TASK_ID / 2)); S=$((SLURM_ARRAY_TASK_ID % 2))
read -r DIR AXN LAM OUT <<< "${CFG[$C]}"
if [ "$DIR" = "rev" ]; then H=models/gptoss20b_honest_bf16; D=models/gptoss20b_cheater_bf16;
else H=models/gptoss20b_cheater_bf16; D=models/gptoss20b_honest_bf16; fi
echo "== gqccell $OUT half=$S lam=$LAM =="
srun python $SS_CODE/gptoss_transplant_driver.py \
  --host "$H" --donor "$D" \
  --axis "activations/dspace/${AXN}.npz" --mu-host 0 --mu-donor 0 --dims 1 --lam "${LAM}.0" --resume \
  --prefix-file "tasks/gqc_0820/gqc_h${S}.jsonl" --n 47 --temperature 0.7 --top-p 0.95 --seed 0 \
  --max-new 40960 --max-model-len 49152 \
  --eos-ids 200002,199999,200012 --add-special-tokens 0 --enable-thinking 1 \
  --out "reports/subdim_0726/gqc_0820/${OUT}_s${S}.json"
echo "== gqccell $OUT half=$S done (exit $?) =="
