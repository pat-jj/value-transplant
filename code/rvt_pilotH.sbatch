#!/bin/bash
#SBATCH --job-name=rvt_pilotH
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --nice=50
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --array=0-15
#SBATCH --output=%x_%A_%a.out
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# [0801 successor] pilot P4: STEERING with the HONEST organism's axis preDIM_QLH_L21
# (sigma 9.1192; doses below = alpha x sigma). Priors + discriminating hypotheses pre-logged in
# REVERSE_FULLSWAP_0731.md. nice=50: queued behind P1/P2 (rvt_pilotA) per directive.
#   P4a MIXED cheater host (decision cell): alpha -2 (dose -18.238) | alpha +1 (dose +9.119)
#   P4b HONEST host own-axis (positive control): alpha +2 (+18.238) | alpha -1 (-9.119)
# Recovered gated-seq port, cheateroct FIRST-64 forks 4x16 shards, UNCAPPED, seed 0.
# tasks 0-3 mix_m2 | 4-7 mix_p1 | 8-11 hon_p2 | 12-15 hon_m1 (shard = task%4)
set -uo pipefail
export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache} NCCL_SOCKET_IFNAME=vxlan0 NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TOKENIZERS_PARALLELISM=false
cleanup(){ kill -TERM -$$ 2>/dev/null; wait; }; trap cleanup SIGTERM SIGINT SIGQUIT
source $SS_ENVS/verl/bin/activate
cd $SS_ROOT/v2
Q=$SS_ROOT/oct_assets/loras/qwen3-8b-distillation
MIX=$Q/success_cheater_think_merged
HON=$Q/success_honest_think_merged

CELL=$((SLURM_ARRAY_TASK_ID / 4)); S=$((SLURM_ARRAY_TASK_ID % 4))
MODELS=("$MIX" "$MIX" "$HON" "$HON")
DOSES=(-18.238 +9.119 +18.238 -9.119)
TAGS=(mix_m2 mix_p1 hon_p2 hon_m1)
MODEL=${MODELS[$CELL]}; DOSE=${DOSES[$CELL]}; TAG=${TAGS[$CELL]}

srun python $SS_CODE/vllm_fork_steer_v2_recovered.py \
  --model "$MODEL" \
  --fork "tasks/pilotA_shard${S}.jsonl" \
  --axis activations/dspace/preDIM_QLH_L21.npz --layer 21 --dose "$DOSE" \
  --n 16 --mode seq --max-model-len 40960 --temperature 0.7 --top-p 0.95 --seed 0 \
  --out "reports/subdim_0726/pilotH_${TAG}_s${S}.json"
echo "== rvt_pilotH ${TAG} s${S} done (exit $?) =="
