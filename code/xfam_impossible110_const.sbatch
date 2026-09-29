#!/bin/bash
#SBATCH --job-name=xfimp_const
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%A_%a.out
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# xfam_clean_0904 — controller check on the OLD impossible-110 set ("test the new cross-family transplant algorithm in the meantime"):
# signed linear controller lindq (h' = h - lam*dQ*w/||w||^2, only knob lam) on the exact prefix file / rows / decoding of the C=1.5 program
# (reports/subdim_0726/crossfamily_impossible_expansion_0829/manifest_expansion_ordered.jsonl, rows 0-111, cheater host, frame A preamble).
# qos=low on SPARE capacity (preemptible; --resume makes a preemption cost <=1 row) so the 16 high-QoS GPUs stay on the benchmark pilots.
# Usage: sbatch --array=0-111%16 $SS_CODE/xfam_impossible110_lindq.sbatch <LAM>
set -uo pipefail
export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache} NCCL_NVLS_ENABLE=0 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1 VLLM_USE_FLASHINFER_SAMPLER=0
cleanup(){ kill -TERM -$$ 2>/dev/null; wait; }; trap cleanup SIGTERM SIGINT SIGQUIT
source $SS_ENVS/verl/bin/activate
cd $SS_ROOT/v2
X=$SS_ROOT/v2/reports/xfam_clean_0904
EX=reports/subdim_0726/crossfamily_impossible_expansion_0829
CT="$(printf '\n\nLet me stop thinking and give my final answer now.')<|end|><|start|>assistant<|channel|>final<|message|>"
export TRANSPLANT_THINK_CAP=32768 TRANSPLANT_CLOSE_MARKER_ID=200005 TRANSPLANT_CLOSE_TEXT="$CT"
export TRANSPLANT_CLOSE_MARKER_NEXT=17196 XM_RENDER_DATE=2026-08-20 TRANSPLANT_SKIP_FORKGATE=1
QH=$SS_ROOT/oct_assets/loras/qwen3-8b-distillation/success_honest_think_merged
QC=$SS_ROOT/oct_assets/loras/qwen3-8b-distillation/success_cheater_hard_think_merged
ETA=${1:?eta}; ROW=$SLURM_ARRAY_TASK_ID; TAG=const_eta${ETA}
export XM_GATE=qsvdir XM_QSV_ETA=$ETA XM_TWOMODEL=0
export XM_AXIS_FILE=probe_W4.npy XM_QSV=$X/probe/w_translated_L14.npz XM_QSV_WHICH=real; export XM_QSV_RAND=${XM_QSV_RAND:-0}
OUT=$X/intervention/impossible110/raw; mkdir -p $OUT
echo "NODE=$(hostname) CONST_ETA=$ETA ROW=$ROW T=$(date -u +%H:%M:%S)"
srun python $SS_CODE/xmodel_online_lindq.py --alpha 1.0 --artifacts art_coarse \
  --prefix-file $EX/manifest_expansion_ordered.jsonl \
  --host models/gptoss20b_cheater_bf16 --weak-honest $QH --weak-cheater $QC \
  --temperature 0.7 --top-p 0.95 --seed 0 --resume \
  --max-new 40960 --max-model-len 49152 --weak-max-model-len 40960 \
  --start $ROW --n 1 --out $OUT/s0_${TAG}_row${ROW}.json || { echo "STEP_FAILED_OR_PREEMPTED rc=$? $(date -u +%H:%M:%S)"; exit 3; }
echo "XFIMP_DONE lam=$LAM row=$ROW $(date -u +%H:%M:%S)"
