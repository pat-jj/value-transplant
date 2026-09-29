#!/bin/bash
#SBATCH --job-name=xfwithin_final
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%A_%a.out
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# WITHIN-FAMILY reference on the FINAL benchmark: honest GPT-OSS donor -> cheater GPT-OSS host, self-rating axis preDIM_G20BC_BAL_L14, lam=16 (validated forward dose),
# unmodified production driver $SS_CODE/gptoss_transplant_driver.py (no --start: uses shard files). Usage: sbatch --array=0-7 $SS_CODE/xfam_withinfam_final.sbatch shards_<tag> [NPER]
# or one row per task: sbatch --array=0-99%16 $SS_CODE/xfam_withinfam_final.sbatch shards1_<tag> 1   (env LAM, THINK_CAP, MAXNEW, MAXLEN)
set -uo pipefail
export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache} VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_ALLOW_INSECURE_SERIALIZATION=1 VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PATH="$SS_ENVS/verl/bin:$PATH"
cleanup(){ kill -TERM -$$ 2>/dev/null; wait; }; trap cleanup SIGTERM SIGINT SIGQUIT
cd $SS_ROOT/v2
X=$SS_ROOT/v2/reports/xfam_clean_0904
CT="$(printf '\n\nLet me stop thinking and give my final answer now.')<|end|><|start|>assistant<|channel|>final<|message|>"
export TRANSPLANT_THINK_CAP=${THINK_CAP:-32768} TRANSPLANT_CLOSE_MARKER_ID=200005 TRANSPLANT_CLOSE_TEXT="$CT" TRANSPLANT_CLOSE_MARKER_NEXT=17196 XM_RENDER_DATE=2026-08-20
SH=$1; NPER=${2:-13}; K=$SLURM_ARRAY_TASK_ID; LAM=${LAM:-16}
# budget overrides (global rule: same THINK_CAP/MAXNEW/MAXLEN for every organism and every arm) -> same tag suffix as $SS_CODE/xfam_arm_final.sbatch
TAG=$(echo "$SH" | sed 's/^shards[0-9]*_//'); [ "${THINK_CAP:-32768}" != "32768" ] && TAG=${TAG}_cap${THINK_CAP}; [ "${MAXNEW:-40960}" != "40960" ] && TAG=${TAG}_mn${MAXNEW}; [ "${MAXLEN:-49152}" != "49152" ] && TAG=${TAG}_ml${MAXLEN}
OUT=$X/final/${TAG}/withinfam_lam${LAM}/raw; mkdir -p $OUT
echo "NODE=$(hostname) shard=$K lam=$LAM T=$(date -u +%H:%M:%S)"
srun python $SS_CODE/gptoss_transplant_driver.py \
  --host models/gptoss20b_cheater_bf16 --donor models/gptoss20b_honest_bf16 \
  --axis activations/dspace/preDIM_G20BC_BAL_L14.npz --mu-host 0 --mu-donor 0 --dims 1 --lam "${LAM}.0" --resume \
  --prefix-file "$X/benchmark_hard/${SH}/shard${K}.jsonl" --n $NPER --temperature 0.7 --top-p 0.95 --seed 0 \
  --max-new ${MAXNEW:-40960} --max-model-len ${MAXLEN:-49152} --eos-ids 200002,199999,200012 --add-special-tokens 0 --enable-thinking 1 \
  --out "$OUT/withinfam_lam${LAM}_shard${K}.json" || { echo "STEP_FAILED_OR_PREEMPTED rc=$? $(date -u +%H:%M:%S)"; exit 3; }
echo "XFWITHIN_FINAL_DONE shard=$K $(date -u +%H:%M:%S)"
