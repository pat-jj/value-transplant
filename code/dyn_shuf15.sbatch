#!/bin/bash
#SBATCH --job-name=DYNSHUF15
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=0-111%12
#SBATCH --output=%x_%A_%a.out
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
set -uo pipefail
export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache} NCCL_NVLS_ENABLE=0 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1 VLLM_USE_FLASHINFER_SAMPLER=0
cleanup(){ kill -TERM -$$ 2>/dev/null; wait; }; trap cleanup SIGTERM SIGINT SIGQUIT
source $SS_ENVS/verl/bin/activate
cd $SS_ROOT/v2
CT="$(printf '\n\nLet me stop thinking and give my final answer now.')<|end|><|start|>assistant<|channel|>final<|message|>"
export TRANSPLANT_THINK_CAP=32768 TRANSPLANT_CLOSE_MARKER_ID=200005 TRANSPLANT_CLOSE_TEXT="$CT"
export TRANSPLANT_CLOSE_MARKER_NEXT=17196 XM_RENDER_DATE=2026-08-20 TRANSPLANT_SKIP_FORKGATE=1
TD=reports/subdim_0726/three_model_host_native_value_0829
EX=reports/subdim_0726/crossfamily_impossible_expansion_0829
OUT=reports/online_weak_donor_dynamic
QH=$SS_ROOT/oct_assets/loras/qwen3-8b-distillation/success_honest_think_merged
QC=$SS_ROOT/oct_assets/loras/qwen3-8b-distillation/success_cheater_hard_think_merged
# ==== ARM (edited per file) ====
TAG=DYNSHUF15; DGAIN=1.0; REPLAY=reports/online_weak_donor_dynamic/data/shuffled_replay15_0902.json
ROW=$SLURM_ARRAY_TASK_ID
export XM_AXIS_FILE=probe_W4.npy
export XM_QSV=$TD/qsv_direction_L14.npz XM_QSV_ETA=448.8 XM_QSV_WHICH=real XM_QSV_SIGN=1.0 XM_QSV_RAND=0
export XM_GATE=dynqsv XM_TWOMODEL=0
export XM_DYN_MODE=clip XM_DYN_S=0.683025 XM_DYN_AMAX=1.5 XM_DYN_GAIN=$DGAIN XM_DYN_CENTER=0.0
[ -n "$REPLAY" ] && export XM_DELTA_REPLAY=$REPLAY
mkdir -p $OUT/data/raw
srun python $OUT/scripts/xmodel_online_dynamic_0902.py --alpha 1.0 --artifacts art_coarse \
  --prefix-file $EX/manifest_expansion_ordered.jsonl \
  --host models/gptoss20b_cheater_bf16 --weak-honest $QH --weak-cheater $QC \
  --temperature 0.7 --top-p 0.95 --seed 0 --resume \
  --max-new 40960 --max-model-len 49152 --weak-max-model-len 40960 \
  --start $ROW --n 1 --out $OUT/data/raw/s0_m${TAG}_row${ROW}.json
echo "== $TAG row $ROW done =="
