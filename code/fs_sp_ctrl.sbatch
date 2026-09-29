#!/bin/bash
#SBATCH --job-name=fs_sp_ctrl
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --nice=100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-8
#SBATCH --output=%x_%A_%a.out
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# P1 26-bar program: 9 control cells offloaded from the pod.
# Matching pod specs removed from the pod queue — no dual-run.
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
CFG=(
  "rev 32 maze_pl_L21 fs_sp_maze_rev_g32"
  "rev 16 maze_pl_L21 fs_sp_maze_rev_g16"
  "rev 48 maze_pl_L21 fs_sp_maze_rev_g48"
  "fwd 32 diffpca_RANDQ_L21 fs_sp_rand_fwd_g32"
  "rev 32 diffpca_RANDQ_L21 fs_sp_rand_rev_g32"
  "fwd 16 diffpca_RANDQ_L21 fs_sp_rand_fwd_g16"
  "rev 16 diffpca_RANDQ_L21 fs_sp_rand_rev_g16"
  "fwd 48 diffpca_RANDQ_L21 fs_sp_rand_fwd_g48"
  "rev 48 diffpca_RANDQ_L21 fs_sp_rand_rev_g48"
)
read -r DIR LAM AXN OUT <<< "${CFG[$SLURM_ARRAY_TASK_ID]}"
if [ "$DIR" = "rev" ]; then H=$QH; D=$QC; else H=$QC; D=$QH; fi
echo "== fs_sp_ctrl $OUT =="
srun python $SS_CODE/vllm_lockstep_transplant_0727.py \
  --host "$H" --donor "$D" --axis "activations/dspace/${AXN}.npz" \
  --mu-host 0 --mu-donor 0 --dims 1 --lam "${LAM}.0" --resume \
  --prefix-file tasks/fromstart_strongprompt_qwen_0816.jsonl --n 80 \
  --temperature 0.7 --top-p 0.95 --seed 0 --max-new 37888 --max-model-len 40960 \
  --enable-thinking 1 --add-special-tokens 0 --eos-ids 151645,151643 \
  --out "reports/subdim_0726/sp_0816_stores/${OUT}.json"
echo "== fs_sp_ctrl $OUT done (exit $?) =="
