#!/bin/bash
#SBATCH --job-name=r2_qb_felt
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --output=%x_%j.out
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# RECIPE 2x2 stream 1 (r2, 0727): THE 2x2 CELL — single self-rating-direction trace transplant with
# gain 32 at L21 on the QWEN RECIPE-B host, using the host's OWN self-rating axis preDIM_QB_L21
# (provenance rule: never preDIM_A).
# Submitted --dependency=afterok:<r2_qb_axis>.
set -uo pipefail
export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache} NCCL_SOCKET_IFNAME=vxlan0 NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TOKENIZERS_PARALLELISM=false
cleanup(){ kill -TERM -$$ 2>/dev/null; wait; }; trap cleanup SIGTERM SIGINT SIGQUIT
source $SS_ENVS/verl/bin/activate
cd $SS_ROOT/v2
QM=$SS_ROOT/oct_assets/loras/qwen3-8b-distillation
D=activations/dspace
RD=reports/subdim_0726

[ -s "$D/preDIM_QB_L21.npz" ] || { echo "MISSING preDIM_QB_L21.npz"; exit 1; }
srun python $SS_CODE/fsd_trace_transplant_subdim3.py \
  --host "$QM/success_cheater_hard_think_merged" --donor "$QM/success_honest_think_merged" \
  --axes "L21:$D/preDIM_QB_L21.npz" --dims 1 \
  --prefix-file tasks/forkcommit_cheateroct_0715.jsonl --n 60 \
  --mu-host 0 --mu-donor 0 --lams 32 --force-donor --log-edit-stats \
  --temperature 0.7 --top-p 0.95 --seed 0 --max-new 30000 --max-model-len 40960 \
  --batch 8 --resume --out "$RD/r2_qb_felt_g32.json"
echo "== r2_qb_felt done (exit $?) =="
