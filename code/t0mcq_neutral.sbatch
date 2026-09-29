#!/bin/bash
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=00:18:00
#SBATCH --output=%x_%j.out
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
DIR=${1:-fwd}
export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache} PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export MODES=neutral DIRS=$DIR
cleanup(){ kill -TERM -$$ 2>/dev/null; wait; }; trap cleanup SIGTERM SIGINT SIGQUIT
cd $SS_ROOT/v2
PY=$SS_ENVS/verl/bin/python
LORA=$SS_ROOT/oct_assets/loras/qwen3-8b-distillation
QH=$LORA/success_honest_think_merged; QC=$LORA/success_cheater_hard_think_merged
AX=activations/dspace/preDIM_QB_L21.npz
BC=reports/subdim_0726/fs_q_anchor_C.json; BH=reports/subdim_0726/fs_q_anchor_H.json
OUT=reports/subdim_0726/mcqel_0823
echo "NODE=$(hostname) DIR=$DIR START=$(date +%H:%M:%S)"
srun $PY $SS_CODE/t0mcq_earlylate.py "$QH" "$QC" "$AX" "$OUT" "$BC" "$BH"
echo "T0MCQ_NEUTRAL_${DIR}_DONE $(date +%H:%M:%S)"
