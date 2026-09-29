#!/bin/bash
#SBATCH --job-name=xfarm
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
# FINAL-benchmark arms (one manifest row per array task; idx = manifest row).
# Usage: sbatch -J <name> --array=0-$((N-1))%16 $SS_CODE/xfam_arm_final.sbatch <MANI> <ARM> [PARAM] [ORG=cheater]
# ARM=base   : unedited (qsvdir eta=0), ORG cheater|honest
# ARM=lindq  : signed linear controller, PARAM=lambda (default 1.0)
# ARM=const  : matched constant push along +w_unit every token, PARAM=eta (raw edit norm per token)
# ARM=lindq_relu: one-sided lindq (push only when dQ<0), PARAM=lambda [ablation]
# ARM=lindq_ema: EMA-smoothed dQ (env EMA=alpha, default 0.98), PARAM=lambda [ablation]
# ARM=hon: honest-organism host axis, PARAM=lamH; stores -> uHhon_lamH<PARAM>
# ARM=rand   : lindq with a RANDOM unit direction of the same ||w|| (XM_QSV_RAND=1), PARAM=lambda
# Output: $X/final/<MANI-tag>/<ARM>[_<PARAM>]/raw/s0_<ORG>_row<ROW>.json
set -uo pipefail
export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache} NCCL_NVLS_ENABLE=0 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1 VLLM_USE_FLASHINFER_SAMPLER=0
cleanup(){ kill -TERM -$$ 2>/dev/null; wait; }; trap cleanup SIGTERM SIGINT SIGQUIT
source $SS_ENVS/verl/bin/activate
cd $SS_ROOT/v2
X=$SS_ROOT/v2/reports/xfam_clean_0904
CT="$(printf '\n\nLet me stop thinking and give my final answer now.')<|end|><|start|>assistant<|channel|>final<|message|>"
export TRANSPLANT_THINK_CAP=${THINK_CAP:-32768} TRANSPLANT_CLOSE_MARKER_ID=200005 TRANSPLANT_CLOSE_TEXT="$CT"
export TRANSPLANT_CLOSE_MARKER_NEXT=17196 XM_RENDER_DATE=2026-08-20 TRANSPLANT_SKIP_FORKGATE=1
QH=$SS_ROOT/oct_assets/loras/qwen3-8b-distillation/success_honest_think_merged
QC=$SS_ROOT/oct_assets/loras/qwen3-8b-distillation/success_cheater_hard_think_merged
MANI=$1; ARM=$2; PARAM=${3:-}; ORG=${4:-cheater}; ROW=$SLURM_ARRAY_TASK_ID
TAG=$(basename "$MANI" .jsonl | sed 's/^driver_manifest_//'); [ "${THINK_CAP:-32768}" != "32768" ] && TAG=${TAG}_cap${THINK_CAP}; [ "${MAXNEW:-40960}" != "40960" ] && TAG=${TAG}_mn${MAXNEW}; [ "${MAXLEN:-49152}" != "49152" ] && TAG=${TAG}_ml${MAXLEN}
[ -n "${TAG_OVERRIDE:-}" ] && TAG=$TAG_OVERRIDE   # host-swap runs must NOT share a store dir with current-host frame C (house rule)
export XM_AXIS_FILE=probe_W4.npy XM_QSV=$X/probe/w_translated_L14.npz XM_QSV_WHICH=real XM_QSV_RAND=0 XM_TWOMODEL=0 XM_QSV_SIGN=1.0
case "$ARM" in
  base)  export XM_GATE=qsvdir XM_QSV_ETA=0.0; SUB=base ;;
  lindq) export XM_GATE=lindq XM_LIN_LAM=${PARAM:-1.0}; SUB=lindq_lam${PARAM:-1.0} ;;
  lindq_relu) export XM_GATE=lindq XM_LIN_LAM=${PARAM:-1.0} XM_LIN_RELU=1; SUB=lindq_relu_lam${PARAM:-1.0} ;;
  lindq_cema) export XM_GATE=lindq XM_LIN_LAM=${PARAM:-1.0} XM_LIN_CENTER_EMA=${CEMA:-0.98}; SUB=lindq_cema${CEMA:-0.98}_lam${PARAM:-1.0} ;;   # solvable: running-level-centered reader signal (two-sided by construction)
  lindq_ema) export XM_GATE=lindq XM_LIN_LAM=${PARAM:-1.0} XM_LIN_EMA=${EMA:-0.98}; SUB=lindq_ema${EMA:-0.98}_lam${PARAM:-1.0} ;;
  lindq_until) export XM_GATE=lindq XM_LIN_LAM=${PARAM:-1.0} XM_LIN_UNTIL=${UNTIL:-1024}; SUB=lindq_until${UNTIL:-1024}_lam${PARAM:-1.0} ;;
  lindq_from) export XM_GATE=lindq XM_LIN_LAM=${PARAM:-1.0} XM_LIN_FROM=${FROM:-1000}; SUB=lindq_from${FROM:-1000}_lam${PARAM:-1.0} ;;   # warm-up exclusion, no edits before generated step FROM
  topk_from) export XM_GATE=lindq XM_LIN_LAM=${PARAM:-2.0} XM_GAIN_MODE=topk_only XM_TOPK_THR=${TOPK_THR:?json} XM_LIN_FROM=${FROM:-1000} XM_LOG_F=1 XM_READ_FULL=1; SUB=topk_from${FROM:-1000}_lam${PARAM:-2.0} ;;   # top-1% gating with the warm-up exclusion
  lindq_ref) export XM_GATE=lindq XM_LIN_LAM=${PARAM:-1.0} XM_LIN_CENTER_PROFILE=${CENTER_PROFILE:?path}; SUB=lindq_ref_lam${PARAM:-1.0} ;;
  lindq_ref_ema) export XM_GATE=lindq XM_LIN_LAM=${PARAM:-1.0} XM_LIN_CENTER_PROFILE=${CENTER_PROFILE:?path} XM_LIN_EMA=${EMA:-0.98}; SUB=lindq_ref_ema${EMA:-0.98}_lam${PARAM:-1.0} ;;
  const) export XM_GATE=qsvdir XM_QSV_ETA=${PARAM:?eta}; SUB=const_eta${PARAM} ;;
  rand)  export XM_GATE=lindq XM_LIN_LAM=${PARAM:-1.0} XM_QSV_RAND=1 XM_QSV_RANDSEED=0; SUB=rand_lam${PARAM:-1.0} ;;
  mhot)  export XM_GATE=lindq XM_LIN_LAM=${PARAM:-1.0} XM_QSV_RAND=2 XM_QSV_RANDSEED=0 XM_QSV_RANDK=${RANDK:-8}; SUB=mhot_lam${PARAM:-1.0} ;;   # multi-hot random axis (k coords = +1, same ||w||)
  lindq_flip) export XM_GATE=lindq XM_LIN_LAM=${PARAM:-1.0} XM_LIN_SIGN=-1; SUB=lindq_flip_lam${PARAM:-1.0} ;;   # DIAGNOSTIC control only
  lindq_clip) export XM_GATE=lindq XM_LIN_LAM=${PARAM:-1.0} XM_DELTA_CLIP=${CLIP:-3}; SUB=lindq_clip${CLIP:-3}_lam${PARAM:-1.0} ;;   # pre-registered reader spike clip |dQ|<=3
  topk_only_rsn) export XM_GATE=lindq XM_LIN_LAM=${PARAM:-2.0} XM_GAIN_MODE=topk_only XM_TOPK_THR=${TOPK_THR:?json} XM_STOP_AT_FINAL=1 XM_LOG_F=1 XM_READ_FULL=1; SUB=topk_only_rsn_lam${PARAM:-2.0} ;;   # reasoning-only sparse edit (no edits after the reasoning channel closes)
  topk_only) export XM_GATE=lindq XM_LIN_LAM=${PARAM:-2.0} XM_GAIN_MODE=topk_only XM_TOPK_THR=${TOPK_THR:?json} XM_LOG_F=1 XM_READ_FULL=1; SUB=topk_only_lam${PARAM:-2.0} ;;   # live edit only where |dQ| >= row p99 (thresholds from the lam2 live trace)
  topk_position_only) export XM_GATE=lindq XM_LIN_LAM=${PARAM:-2.0} XM_GAIN_MODE=topk_position_only XM_TOPK_THR=${TOPK_THR:?json} XM_RANDTOPK_SRC=${TOPK_SRC:?store-pattern} XM_LOG_F=1 XM_READ_FULL=1; SUB=topk_position_only_lam${PARAM:-2.0} ;;   # real spike positions+sign, constant median magnitude (needs the row's topk_only store)
  random_topk) export XM_GATE=lindq XM_LIN_LAM=${PARAM:-2.0} XM_GAIN_MODE=random_topk XM_RANDTOPK_SRC=${TOPK_SRC:?store-pattern} XM_LOG_F=1 XM_READ_FULL=1; SUB=random_topk_lam${PARAM:-2.0} ;;   # same edit count/magnitudes at random positions, live sign (density-matched; needs the row's topk_only store)
  hon) export XM_GATE=lindq XM_LIN_LAM=${PARAM:?lamH} XM_DIR=file XM_DIR_FILE=${HON_AXIS:-$SS_ROOT/v2/activations/dspace/preDIM_G20BH_BAL_L14.npz} XM_DIR_SIGN=-1 XM_GAIN_MODE=live XM_READ_FULL=1 XM_LOG_F=1; SUB=uHhon_lamH${PARAM} ;;   # honest-organism HOST axis, h <- h - lamH*dS*u_H^honest (unit axis, raw units per unit dS); solvable protocol
  axis) export XM_GATE=lindq XM_LIN_LAM=${PARAM:?lamH} XM_DIR=file XM_DIR_FILE=${AXIS_FILE:?npz} XM_DIR_SIGN=${AXIS_SIGN:--1} XM_GAIN_MODE=live XM_READ_FULL=1 XM_LOG_F=1; SUB=${AXIS_TAG:?tag}_lamH${PARAM} ;;   # hunt: arbitrary host axis npz (direction+mean), tag names the store dir
  *) echo "unknown ARM $ARM"; exit 2 ;;
esac
OUT=$X/final/$TAG/$SUB/raw; mkdir -p $OUT
echo "NODE=$(hostname) ARM=$ARM PARAM=$PARAM ORG=$ORG ROW=$ROW mani=$MANI T=$(date -u +%H:%M:%S)"
srun python $SS_CODE/xmodel_online_lindq.py --alpha 1.0 --artifacts art_coarse \
  --prefix-file $X/benchmark_hard/$MANI \
  --host ${HOST_DIR:-models/gptoss20b_${ORG}_bf16} --weak-honest $QH --weak-cheater $QC \
  --temperature 0.7 --top-p 0.95 --seed ${SEED:-0} --resume \
  --max-new ${MAXNEW:-40960} --max-model-len ${MAXLEN:-49152} --weak-max-model-len ${WEAKMAXLEN:-40960} --weak-overflow tail \
  ${XM_LAYER:+--layer-strong $XM_LAYER} --start $ROW --n 1 --out $OUT/s${SEED:-0}_${ORG}_row${ROW}.json || { echo "STEP_FAILED_OR_PREEMPTED rc=$? $(date -u +%H:%M:%S)"; exit 3; }
echo "XFARM_DONE $ARM $ORG row=$ROW $(date -u +%H:%M:%S)"
