#!/bin/bash
#SBATCH --job-name=solvdose131
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --nice=600
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-12
#SBATCH --output=%x_%A_%a.out
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# gpt-oss solvable dose-fill under the 0816 window convention (131072 / cap 106000 / room 16384).
# Supersedes the parked pod sf_g20b_*_capped cells (40,960 versions renamed superseded — kept).
set -uo pipefail
export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PATH="$SS_ENVS/verl/bin:$PATH"
cleanup(){ kill -TERM -$$ 2>/dev/null; wait; }; trap cleanup SIGTERM SIGINT SIGQUIT
V2=$SS_ROOT/v2
cd $V2
CT="$(printf '\n\nLet me stop thinking and give my final answer now.')<|end|><|start|>assistant<|channel|>final<|message|>"
export TRANSPLANT_THINK_CAP=106000 TRANSPLANT_CLOSE_MARKER_ID=200005 TRANSPLANT_CLOSE_TEXT="$CT"
GC=models/gptoss20b_cheater_bf16; GH=models/gptoss20b_honest_bf16
CFG=(
  "fwd value_axis_0731_g20b_L14 16 sf131_fwd_value_lam16"  "fwd value_axis_0731_g20b_L14 24 sf131_fwd_value_lam24"
  "fwd value_axis_0731_g20b_L14 64 sf131_fwd_value_lam64"  "fwd maze_g20b_L14 16 sf131_fwd_maze_lam16"
  "fwd maze_g20b_L14 24 sf131_fwd_maze_lam24"   "fwd maze_g20b_L14 64 sf131_fwd_maze_lam64"
  "fwd preDIM_G20BC_RAND_L14 16 sf131_fwd_rand_lam16" "fwd preDIM_G20BC_RAND_L14 24 sf131_fwd_rand_lam24"
  "fwd preDIM_G20BC_RAND_L14 64 sf131_fwd_rand_lam64" "rev maze_g20b_L14 8 sf131_rev_maze_lam8"
  "rev maze_g20b_L14 24 sf131_rev_maze_lam24"   "rev preDIM_G20BC_RAND_L14 8 sf131_rev_rand_lam8"
  "rev preDIM_G20BC_RAND_L14 24 sf131_rev_rand_lam24"
)
read -r DIR AXN LAM OUT <<< "${CFG[$SLURM_ARRAY_TASK_ID]}"
if [ "$DIR" = "rev" ]; then H=$GH; D=$GC; else H=$GC; D=$GH; fi
echo "== solvdose131 $OUT =="
srun python $SS_CODE/gptoss_transplant_driver.py \
  --host "$H" --donor "$D" --axis "activations/dspace/${AXN}.npz" \
  --mu-host 0 --mu-donor 0 --dims 1 --lam "${LAM}.0" --resume \
  --prefix-file tasks/solvperf_fB128_clean_0813.jsonl --n 123 --temperature 0.7 --top-p 0.95 --seed 0 \
  --max-new 122368 --max-model-len 131072 \
  --eos-ids 200002,199999,200012 --add-special-tokens 0 --enable-thinking 1 \
  --out "reports/subdim_0726/sp_0816_stores/${OUT}.json"
echo "== solvdose131 $OUT done (exit $?) =="
