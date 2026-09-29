#!/bin/bash
#SBATCH --job-name=rev40seed
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-19%16
#SBATCH --output=%x_%A_%a.out
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# Strengthen the solvable-reverse signal-gate figure: self-rating lam40 value-match reverse
# (honest HOST <- cheater DONOR, axis preDIM_QB_L21), 128 tasks x 3 seeds. Matches the earlier
# 64-point setup; seed0 shards 0-3 reuse the existing store, so we generate seed0 sh4-7 + seeds1,2 full.
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
EXT=reports/subdim_0726/gradient_transplant_0809/rev40ext_0901
# cell -> (seed, shard): seed0 uses shards 4-7 (missing tasks); seeds 1,2 use all shards 0-7
SEED_MAP=(0 0 0 0 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2)
SHARD_MAP=(4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7)
T=$SLURM_ARRAY_TASK_ID; SEED=${SEED_MAP[$T]}; SH=${SHARD_MAP[$T]}
OUT=$EXT/s${SEED}_sh${SH}.json
echo "=== rev40 seed=$SEED shard=$SH -> $OUT $(date -u) ==="
srun python $SS_CODE/vllm_lockstep_transplant_0727.py \
  --host $HOST --donor $DONOR \
  --axis activations/dspace/preDIM_QB_L21.npz --mu-host 0 --mu-donor 0 --dims 1 --lam 40 \
  --prefix-file $EXT/shards/shard_${SH}.jsonl --n 16 \
  --temperature 0.7 --top-p 0.95 --seed $SEED --max-new 40960 --max-model-len 40960 \
  --eos-ids 151645,151643 --add-special-tokens 0 --enable-thinking 1 --resume \
  --out $OUT || { echo "REV40_s${SEED}_sh${SH}_FAILED"; exit 3; }
echo "=== rev40 seed=$SEED shard=$SH DONE $(date -u) ==="
