#!/bin/bash
#SBATCH --job-name=xfimp_honest
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
# xfam_clean_0904 — HONEST-ORGANISM ANCHOR on the impossible-110 common rows (0910: "where's honest anchor's bar in impossible code subplot?" -> "yeah please do").
# Same prefix file / rows / decoding / think cap as every impossible-110 arm ($SS_CODE/xfam_impossible110_const.sbatch), host = models/gptoss20b_honest_bf16, controller OFF
# (qsvdir eta=0 = the 'base' arm convention of $SS_CODE/xfam_arm_final.sbatch). Rows 0-39 = the pre-declared common subset (RUN_RECORD 0908 22:4x UTC).
# Usage: sbatch --array=0-39%14 $SS_CODE/xfam_impossible110_honest.sbatch      Output: $X/intervention/impossible110/raw/s0_honest_base_row<ROW>.json
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
ROW=$SLURM_ARRAY_TASK_ID; TAG=honest_base
export XM_GATE=qsvdir XM_QSV_ETA=0 XM_TWOMODEL=0
export XM_AXIS_FILE=probe_W4.npy XM_QSV=$X/probe/w_translated_L14.npz XM_QSV_WHICH=real XM_QSV_RAND=0
OUT=$X/intervention/impossible110/raw; mkdir -p $OUT
echo "NODE=$(hostname) ORG=honest ARM=base ROW=$ROW T=$(date -u +%H:%M:%S)"
srun python $SS_CODE/xmodel_online_lindq.py --alpha 1.0 --artifacts art_coarse \
  --prefix-file $EX/manifest_expansion_ordered.jsonl \
  --host models/gptoss20b_honest_bf16 --weak-honest $QH --weak-cheater $QC \
  --temperature 0.7 --top-p 0.95 --seed 0 --resume \
  --max-new 40960 --max-model-len 49152 --weak-max-model-len 40960 \
  --start $ROW --n 1 --out $OUT/s0_${TAG}_row${ROW}.json || { echo "STEP_FAILED_OR_PREEMPTED rc=$? $(date -u +%H:%M:%S)"; exit 3; }
echo "XFIMP_DONE honest_base row=$ROW $(date -u +%H:%M:%S)"
