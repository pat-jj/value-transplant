#!/bin/bash
#SBATCH --job-name=grad_hidose_fin
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --array=0-2%3
#SBATCH --output=%x_%A_%a.out
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# B18e high-dose extension: extend the gradient REVERSE self-rating sweep to λ40/48/64 to locate the reverse induction peak / breakdown (self-rating only; longer wall than gradient_tp_rev_hidose.sbatch).
# REVERSE: honest HOST <- cheater DONOR, gradient frame, 1-D, from-start, value-match, thinking-ON, UNCAPPED.
set -uo pipefail
export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache} NCCL_SOCKET_IFNAME=vxlan0 NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TOKENIZERS_PARALLELISM=false
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_ALLOW_INSECURE_SERIALIZATION=1 VLLM_USE_FLASHINFER_SAMPLER=0
cleanup(){ kill -TERM -$$ 2>/dev/null||true; wait 2>/dev/null||true; }; trap cleanup SIGTERM SIGINT SIGQUIT
source $SS_ENVS/verl/bin/activate
cd $SS_ROOT/v2
ORG=$SS_ROOT/oct_assets/loras/qwen3-8b-distillation
HOST=$ORG/success_honest_think_merged
DONOR=$ORG/success_cheater_hard_think_merged
OUTD=reports/subdim_0726/gradient_transplant_0809
AXES=(felt); NPZ=(preDIM_QB_L21); DOSES=(40 48 64)
A=$((SLURM_ARRAY_TASK_ID/3)); D=$((SLURM_ARRAY_TASK_ID%3))
AX=${AXES[$A]}; NP=${NPZ[$A]}; LAM=${DOSES[$D]}
OUT=$OUTD/sg_qwen_grad_rev_${AX}_lam${LAM}.json
echo "=== grad_rev_hidose $AX lam$LAM (honest host <- cheater donor, gradient, n=128, UNCAPPED) $(date -u) ==="
srun python $SS_CODE/vllm_lockstep_transplant_0727.py \
  --host $HOST --donor $DONOR \
  --axis activations/dspace/${NP}.npz --mu-host 0 --mu-donor 0 --dims 1 --lam $LAM \
  --prefix-file tasks/solvperf_gradient_0809.jsonl --n 64 \
  --temperature 0.7 --top-p 0.95 --seed 0 --max-new 40960 --max-model-len 40960 \
  --eos-ids 151645,151643 --add-special-tokens 0 --enable-thinking 1 --resume \
  --out $OUT || { echo "GRADHI_${AX}_${LAM}_FAILED"; exit 3; }
echo "=== grad_rev_hidose $AX lam$LAM DONE $(date -u) ==="
