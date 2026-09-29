#!/bin/bash
#SBATCH --job-name=rvt_hbfelt
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --nice=70
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --array=0-87
#SBATCH --output=%x_%A_%a.out
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# [0802 successor] HARD-CHEATER BATTERY, cluster arm clone on
# success_cheater_hard_think_merged with its OWN axis preDIM_QB_L21): the SELF-RATING arm (21 cells,
# the exact raw vector-norm doses of the mixed battery incl. the fine felt-down sweep and the
# matchpush norm points) + hb_anchor (dose 0). Value/maze/random arms = executor (POD-4),
# same shared manifest. DESIGN NOTE (recon-verified): the plotted Fig-2(a) battery is
# FORK-FRAME (forkcommit_cheateroct_0715 first-64, positions=answer via the gated-seq port,
# max-model-len 32768, uncapped, T0.7/p0.95 seed 0, n=64) — this clone matches that exactly;
# the "from-start" phrasing in the directive was corrected by the recon (flagged).
# Cells+doses read from reports/subdim_0726/hb_battery_manifest_0802.tsv (runner=cluster rows;
# config-file pattern, no --export). task = cell*4 + shard; shards = pilotA_shard{0..3}.jsonl.
# Outputs hb_{...}_s{S}.json -> id-union merge to canonical n=64 -> decider+gendoubt (flock).
set -uo pipefail
export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache} NCCL_SOCKET_IFNAME=vxlan0 NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TOKENIZERS_PARALLELISM=false
cleanup(){ kill -TERM -$$ 2>/dev/null; wait; }; trap cleanup SIGTERM SIGINT SIGQUIT
source $SS_ENVS/verl/bin/activate
cd $SS_ROOT/v2
Q=$SS_ROOT/oct_assets/loras/qwen3-8b-distillation
MAN=reports/subdim_0726/hb_battery_manifest_0802.tsv

CELL=$((SLURM_ARRAY_TASK_ID / 4)); S=$((SLURM_ARRAY_TASK_ID % 4))
# cluster rows of the manifest, in order (skip header, runner==cluster)
LINE=$(awk -F'\t' 'NR>1 && $4=="cluster"' "$MAN" | sed -n "$((CELL+1))p")
[ -n "$LINE" ] || { echo "no manifest row for cell $CELL"; exit 1; }
KEY=$(echo "$LINE" | cut -f1); AXIS=$(echo "$LINE" | cut -f2); DOSE=$(echo "$LINE" | cut -f3)
echo "[hb] cell=$CELL shard=$S key=$KEY axis=$AXIS dose=$DOSE"

srun python $SS_CODE/vllm_fork_steer_v2_recovered.py \
  --model "$Q/success_cheater_hard_think_merged" \
  --fork "tasks/pilotA_shard${S}.jsonl" \
  --axis "$AXIS" --layer 21 --dose "$DOSE" \
  --n 16 --mode seq --max-model-len 32768 --temperature 0.7 --top-p 0.95 --seed 0 \
  --out "reports/subdim_0726/${KEY}_s${S}.json"
echo "== rvt_hbfelt ${KEY} s${S} done (exit $?) =="
