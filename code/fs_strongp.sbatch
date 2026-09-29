#!/bin/bash
#SBATCH --job-name=fs_strongp
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --nice=340
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --array=0-2
#SBATCH --output=%x_%A_%a.out
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# Strengthened-prompt from-start suite: same 40 impossible tasks x2 seeds, preamble
# extended with "graded only on the exact test cases shown" to raise shortcut salience from the
# start. Cells: honest anchor (lam0), self-rating reverse g16, self-rating reverse g32. Capped convention.
set -uo pipefail
export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PATH="$SS_ENVS/verl/bin:$PATH"
cleanup(){ kill -TERM -$$ 2>/dev/null; wait; }; trap cleanup SIGTERM SIGINT SIGQUIT
cd $SS_ROOT/v2
LORA=$SS_ROOT/oct_assets/loras/qwen3-8b-distillation
QH=$LORA/success_honest_think_merged; QC=$LORA/success_cheater_hard_think_merged
CT="$(printf '\n\nLet me stop thinking and give my final answer now.\n')</think>$(printf '\n\n')"
export TRANSPLANT_THINK_CAP=32768 TRANSPLANT_CLOSE_MARKER_ID=151668 TRANSPLANT_CLOSE_TEXT="$CT"
CFG=("0.0 fs_sp_anchor_H" "16.0 fs_sp_felt_rev_g16" "32.0 fs_sp_felt_rev_g32")
read -r LAM TAG <<< "${CFG[$SLURM_ARRAY_TASK_ID]}"
if [ "$LAM" = "0.0" ]; then export TRANSPLANT_SKIP_FORKGATE=1; fi
echo "== fs_strongp $TAG lam=$LAM =="
srun python $SS_CODE/vllm_lockstep_transplant_0727.py \
  --host $QH --donor $QC \
  --axis activations/dspace/preDIM_QB_L21.npz --mu-host 0 --mu-donor 0 --dims 1 --lam "$LAM" --resume \
  --prefix-file tasks/fromstart_strongprompt_qwen_0816.jsonl --n 80 --temperature 0.7 --top-p 0.95 \
  --seed 0 --max-new 37888 --max-model-len 40960 --enable-thinking 1 --add-special-tokens 0 \
  --eos-ids 151645,151643 --out "reports/subdim_0726/${TAG}.json"
echo "== fs_strongp $TAG done (exit $?) =="
