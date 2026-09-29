#!/bin/bash
#SBATCH --job-name=hsteer_ext
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%A_%a.out
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# Completing Figure 2 column (c) (Qwen3-8B honest organism steered at the about-to-cheat moment): the maze / in-context (value) /
# random comparison directions were only run at +-1..3 of their OWN sigma (tiny pushes in raw units). This array runs them at the
# self-rating grid's RAW pushes (+-10.44, +-20.89, +-31.33 = +-1..3 x 10.444, the honest host's own sigma on preDIM_QB_L21) and adds
# the missing self-rating +3 sigma cell. Same driver / forks / decoding as the existing hsteer_Q_* and *steer_Q_* cells.
# Usage: sbatch --array=0-18%16 $SS_CODE/hsteer_ext.sbatch
set -uo pipefail
export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache} NCCL_NVLS_ENABLE=0 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TOKENIZERS_PARALLELISM=false
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_ALLOW_INSECURE_SERIALIZATION=1 VLLM_USE_FLASHINFER_SAMPLER=0
cleanup(){ kill -TERM -$$ 2>/dev/null; wait; }; trap cleanup SIGTERM SIGINT SIGQUIT
source $SS_ENVS/verl/bin/activate
cd $SS_ROOT/v2
Q=$SS_ROOT/oct_assets/loras/qwen3-8b-distillation/success_honest_think_merged
TAGS=(m3 m2 m1 p1 p2 p3); DOSES=(-31.33 -20.89 -10.44 10.44 20.89 31.33)
i=$SLURM_ARRAY_TASK_ID
if [ "$i" -eq 18 ]; then NAME=hsteer_Q_p3; NPZ=preDIM_QB_L21; DOSE=31.33
else a=$((i/6)); d=$((i%6)); PREF=(valsteer_Q_r mazesteer_Q_r randsteer_Q_r); NPZS=(valueaxis_fork_L21 maze_pl_L21 octT_random_L21); NAME=${PREF[$a]}${TAGS[$d]}; NPZ=${NPZS[$a]}; DOSE=${DOSES[$d]}; fi
OUT=reports/subdim_0726/${NAME}.json
[ -s "$OUT" ] && { echo "exists: $OUT"; exit 0; }
echo "NODE=$(hostname) task=$i name=$NAME axis=$NPZ dose=$DOSE T=$(date -u +%H:%M:%S)"
srun python vllm_fork_steer_v2_recovered.py \
  --model "$Q" --fork tasks/forkcommit_cheateroct_0715.jsonl \
  --axis activations/dspace/${NPZ}.npz --layer 21 --dose "$DOSE" \
  --n 64 --mode seq --max-model-len 40960 --temperature 0.7 --top-p 0.95 --seed 0 \
  --out "$OUT" || { echo "STEP_FAILED_OR_PREEMPTED rc=$? $(date -u +%H:%M:%S)"; exit 3; }
echo "HSTEER_EXT_DONE $NAME $(date -u +%H:%M:%S)"
