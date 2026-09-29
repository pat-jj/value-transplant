#!/bin/bash
#SBATCH --job-name=capped131_rev
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --nice=250
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=%x_%A_%a.out
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# gpt-oss reverse ladder under the 0816 convention: the MODEL'S OWN window (131072), think cap
# 106000 (only true never-terminators), answer room 16384 = 12.5% of window. Supersedes the
# 40960-window capped_rev ladder (kept for comparison). New cell names g20b_R131_*.
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
LAMS=(0 4 16 64); TAGS=(R131_anchor R131_felt_lam4 R131_felt_lam16 R131_felt_lam64)
LAM=${LAMS[$SLURM_ARRAY_TASK_ID]}; TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}
if [ "$LAM" = "0" ]; then export TRANSPLANT_SKIP_FORKGATE=1; fi
echo "== capped131 $TAG lam=$LAM =="
srun python $SS_CODE/gptoss_transplant_driver.py \
  --host models/gptoss20b_honest_bf16 --donor models/gptoss20b_cheater_bf16 \
  --axis activations/dspace/preDIM_G20BC_BAL_L14.npz --mu-host 0 --mu-donor 0 --dims 1 --lam "${LAM}.0" --resume \
  --prefix-file tasks/forkcommit_gptoss_0731.jsonl --n 97 --temperature 0.7 --top-p 0.95 --seed 0 \
  --max-new 122368 --max-model-len 131072 \
  --eos-ids 200002,199999,200012 --add-special-tokens 0 --enable-thinking 1 \
  --out "reports/subdim_0726/sp_0816_stores/g20b_${TAG}.json"
echo "== capped131 $TAG done (exit $?) =="
