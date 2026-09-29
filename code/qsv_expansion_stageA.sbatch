#!/bin/bash
#SBATCH --job-name=xmqexpA
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-447%15
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
export TRANSPLANT_CLOSE_MARKER_NEXT=17196 XM_RENDER_DATE=2026-08-20
TD=reports/subdim_0726/three_model_host_native_value_0829
EX=reports/subdim_0726/crossfamily_impossible_expansion_0829
Q=$SS_ROOT/oct_assets/loras/qwen3-8b-distillation/success_honest_think_merged
NROW=112
T=$SLURM_ARRAY_TASK_ID; ARM=$((T/NROW)); ROW=$((T%NROW))
case $ARM in
  0) TAG=BASE; WHICH=real;      ETA=0.0;;
  1) TAG=REAL; WHICH=real;      ETA=448.8;;
  2) TAG=SHUF; WHICH=xtask;     ETA=448.8;;
  3) TAG=RAND; WHICH=randfixed; ETA=448.8;;
esac
export XM_QSV=$TD/qsv_direction_L14.npz XM_QSV_ETA=$ETA XM_QSV_WHICH=$WHICH XM_QSV_SIGN=1.0 XM_QSV_RAND=0
XM_GATE=qsvdir XM_TWOMODEL=0 XM_AXIS_FILE=probe_W4.npy XM_TARGET_OFFSET=1.563235700784409 \
srun python $SS_CODE/xmodel_probe_driver.py --alpha 1.0 --artifacts art_coarse \
  --prefix-file $EX/manifest_expansion_ordered.jsonl \
  --host models/gptoss20b_cheater_bf16 --weak-honest $Q --weak-cheater $Q \
  --temperature 0.7 --top-p 0.95 --seed 0 --resume \
  --max-new 40960 --max-model-len 49152 --weak-max-model-len 40960 \
  --start $ROW --n 1 --out $EX/raw_generations/s0_m${TAG}_row${ROW}.json
echo "== stageA $TAG row $ROW seed0 done =="
