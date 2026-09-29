#!/bin/bash
#SBATCH --job-name=qwen_solve_variant
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=%x_%j.out
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# Usage: sbatch -J qwen_<FRAME>_s<SEED> $SS_CODE/qwen_solve_variant.sbatch <FRAME> [SEED=0] [MANIFEST] [GRADING]   (FRAME names the output dir analysis/qwen_solve_<FRAME>[_s<SEED>])
# Qwen3-8B organisms (honest, cheater) unedited on a frame variant of the 100-task LeetCode manifest. Usage: sbatch -J qwen_<FRAME> $SS_CODE/qwen_solve_variant.sbatch <FRAME> (manifest benchmark_hard/driver_manifest_final_lcfunc100_<FRAME>.jsonl)
set -uo pipefail
export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache} NCCL_NVLS_ENABLE=0 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" VLLM_ENABLE_V1_MULTIPROCESSING=0
cleanup(){ kill -TERM -$$ 2>/dev/null; wait; }; trap cleanup SIGTERM SIGINT SIGQUIT
FRAME=$1; SEED=${2:-0}; X=$SS_ROOT/v2/reports/xfam_clean_0904; MANI=${3:-$X/benchmark_hard/driver_manifest_final_lcfunc100_${FRAME}.jsonl}; GRAD=${4:-$X/benchmark_hard/grading_final_lcfunc100.json}; Q=$X/analysis/qwen_solve_${FRAME}; [ "$SEED" != "0" ] && Q=${Q}_s${SEED}; mkdir -p $Q/results
source $SS_ENVS/verl/bin/activate; cd $SS_ROOT/v2
echo "NODE=$(hostname) FRAME=$FRAME T=$(date -u +%H:%M:%S)"
for org in honest cheater; do [ -s $Q/qwen_${org}.json ] && { echo "skip $org (exists)"; continue; }; QS_FRAME=$FRAME QS_SEED=$SEED QS_MANIFEST=$MANI srun python $SS_CODE/qwen_solve_bench.py $org $Q/qwen_${org}.json 32768 || echo "GEN_FAILED $org"; done
source $SS_ENVS/analysis/bin/activate
for org in honest cheater; do python $SS_CODE/grade_v2.py grade --condition qwen_${org}_${FRAME}_s${SEED} --stores "$Q/qwen_${org}.json" --grading $GRAD --out $Q/results/qwen_${org} --workers 8 | tail -2; done
echo "QWEN_VARIANT_DONE $FRAME T=$(date -u +%H:%M:%S)"
