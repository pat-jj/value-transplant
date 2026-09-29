#!/bin/bash
#SBATCH --job-name=xfimp_uHconst
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
# xfam_clean_0904 — u_H DIRECTION ARM (derived from the top-k launcher; same rows/seeds/decoding). Usage: sbatch --array=0-4 $SS_CODE/xfam_impossible110_uH.sbatch 36.69 ; SMOKE=1 for a 64-token check
# (original header:) TOP-K gain modes on impossible-110 (main line): same lindq driver, direction w, decoding, rows and seeds as
# $SS_CODE/xfam_impossible110_lindq.sbatch; only the per-token scalar changes (XM_GAIN_MODE). Usage:
# sbatch --array=0-4 xfam_impossible110_topk.sbatch topk_only 2.0
# sbatch --array=0-4 xfam_impossible110_topk.sbatch topk_masked 2.0
# sbatch --array=0-4 --dependency=aftercorr:<topk_only jobid> xfam_impossible110_topk.sbatch random_topk 2.0
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
export TRANSPLANT_THINK_CAP=${THINK_CAP:-32768} TRANSPLANT_CLOSE_MARKER_ID=200005 TRANSPLANT_CLOSE_TEXT="$CT"   # THINK_CAP/MAXNEW/MAXLEN/WEAKOVERFLOW env for the cap-raised rerun (defaults = benchmark budget)
export TRANSPLANT_CLOSE_MARKER_NEXT=17196 XM_RENDER_DATE=2026-08-20 TRANSPLANT_SKIP_FORKGATE=1
QH=$SS_ROOT/oct_assets/loras/qwen3-8b-distillation/success_honest_think_merged
QC=$SS_ROOT/oct_assets/loras/qwen3-8b-distillation/success_cheater_hard_think_merged
ETA=${1:?eta}; ROW=$SLURM_ARRAY_TASK_ID; TAG=${DIRTAG:-uHhon_cap80k_const}_eta${ETA}; [ "${SMOKE:-0}" = "1" ] && SMOKEFLAG=--smoke || SMOKEFLAG=
OUT=$X/intervention/impossible110/raw; mkdir -p $OUT
# DIRECTION ABLATION: h <- h + LAMH*dQ*u_H, u_H = GPT-OSS own self-rating axis (unit), LAMH in raw units per unit dQ; 36.69 = 0.5/||w|| matches the lam=0.5 per-token |edit|.
export XM_GATE=qsvdir XM_QSV_ETA=$ETA XM_QSV_SIGN=${QSV_SIGN:-1} XM_TWOMODEL=0 XM_GAIN_MODE=live XM_DIR=file XM_READ_FULL=1 XM_LOG_F=1   # CONSTANT PUSH along unit(direction) of XM_DIR_FILE: coef = +ETA*QSV_SIGN raw at EVERY token (control for the honest axis)
export XM_AXIS_FILE=probe_W4.npy XM_QSV=$X/probe/w_translated_L14.npz XM_QSV_WHICH=real XM_QSV_RAND=0
echo "NODE=$(hostname) CONST_ETA=$ETA SIGN=${QSV_SIGN:-1} DIRFILE=$XM_DIR_FILE ROW=$ROW CAP=${THINK_CAP:-32768} MAXNEW=${MAXNEW:-40960} WEAKOVERFLOW=${WEAKOVERFLOW:-freeze} SMOKE=${SMOKE:-0} T=$(date -u +%H:%M:%S)"
srun python $SS_CODE/xmodel_online_lindq.py --alpha 1.0 --artifacts art_coarse \
  --prefix-file $EX/manifest_expansion_ordered.jsonl \
  --host models/gptoss20b_cheater_bf16 --weak-honest $QH --weak-cheater $QC \
  --temperature 0.7 --top-p 0.95 --seed 0 --resume $SMOKEFLAG \
  --max-new ${MAXNEW:-40960} --max-model-len ${MAXLEN:-49152} --weak-max-model-len 40960 ${WEAKOVERFLOW:+--weak-overflow $WEAKOVERFLOW} \
  ${XM_LAYER:+--layer-strong $XM_LAYER} --start $ROW --n 1 --out $OUT/s0_${TAG}_row${ROW}.json || { echo "STEP_FAILED_OR_PREEMPTED rc=$? $(date -u +%H:%M:%S)"; exit 3; }
echo "XFIMP_UHCONST_DONE eta=$ETA row=$ROW $(date -u +%H:%M:%S)"
