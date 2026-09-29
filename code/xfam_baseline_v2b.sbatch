#!/bin/bash
#SBATCH --job-name=xfbase2b
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-15%16
#SBATCH --output=%x_%A_%a.out
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# xfam_clean_0904 PART 4: unedited GPT-OSS CHEATER and HONEST on the frozen 214-task neutral benchmark.
# Host-only qsvdir with eta=0 == zero edit; same per-step seeds as the lindq arm, so this IS the lam=0 point.
set -uo pipefail
export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache} NCCL_NVLS_ENABLE=0 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1 VLLM_USE_FLASHINFER_SAMPLER=0
cleanup(){ kill -TERM -$$ 2>/dev/null; wait; }; trap cleanup SIGTERM SIGINT SIGQUIT
source $SS_ENVS/verl/bin/activate
cd $SS_ROOT/v2
X=$SS_ROOT/v2/reports/xfam_clean_0904
CT="$(printf '\n\nLet me stop thinking and give my final answer now.')<|end|><|start|>assistant<|channel|>final<|message|>"
export TRANSPLANT_THINK_CAP=32768 TRANSPLANT_CLOSE_MARKER_ID=200005 TRANSPLANT_CLOSE_TEXT="$CT"
export TRANSPLANT_CLOSE_MARKER_NEXT=17196 XM_RENDER_DATE=2026-08-20 TRANSPLANT_SKIP_FORKGATE=1
QH=$SS_ROOT/oct_assets/loras/qwen3-8b-distillation/success_honest_think_merged
QC=$SS_ROOT/oct_assets/loras/qwen3-8b-distillation/success_cheater_hard_think_merged
export XM_AXIS_FILE=probe_W4.npy XM_QSV=$X/probe/w_translated_L14.npz XM_QSV_ETA=0.0 XM_QSV_WHICH=real XM_QSV_SIGN=1.0 XM_QSV_RAND=0
export XM_GATE=qsvdir XM_TWOMODEL=0
I=$SLURM_ARRAY_TASK_ID; CHUNK=27
if [ "$I" -lt 8 ]; then ORG=cheater; C=$I; else ORG=honest; C=$((I-8)); fi
HOST=models/gptoss20b_${ORG}_bf16
START=$((C*CHUNK))
mkdir -p $X/baselines_v2b/raw
echo "NODE=$(hostname) ORG=$ORG chunk=$C start=$START n=$CHUNK T=$(date -u +%H:%M:%S)"
srun python $SS_CODE/xmodel_online_lindq.py --alpha 1.0 --artifacts art_coarse \
  --prefix-file $X/benchmark/driver_manifest_v2b.jsonl \
  --host $HOST --weak-honest $QH --weak-cheater $QC \
  --temperature 0.7 --top-p 0.95 --seed 0 --resume \
  --max-new 40960 --max-model-len 49152 --weak-max-model-len 40960 \
  --start $START --n $CHUNK --out $X/baselines_v2b/raw/s0_${ORG}_chunk${C}.json
echo "XFBASE2B_DONE $ORG chunk$C $(date -u +%H:%M:%S)"
