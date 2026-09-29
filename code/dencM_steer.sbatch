#!/bin/bash
#SBATCH --job-name=dencM
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --array=0-27
#SBATCH --output=%x_%A_%a.out
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# d_enc MATRIX: encoder direction x {L15,21,25,28} x {recipeA,recipeB} x {-5,-3,+3,+5}sigma.
# (A-L15 +-3/+-5 already exist as dencS -> those 4 combos excluded.) Full generation, no caps.
export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache} NCCL_SOCKET_IFNAME=vxlan0 NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TOKENIZERS_PARALLELISM=false
cleanup(){ kill -TERM -$$ 2>/dev/null; wait; }; trap cleanup SIGTERM SIGINT SIGQUIT
source $SS_ENVS/verl/bin/activate
cd $SS_ROOT/v2; export PYTHONPATH=src/success_steering
CH=$SS_ROOT/oct_assets/loras/qwen3-8b-distillation/success_cheater_think_merged
OPP=$SS_ROOT/oct_assets/loras/qwen3-8b-distillation/success_cheateropp_think_merged
HON=$SS_ROOT/oct_assets/loras/qwen3-8b-distillation/success_honest_think_merged
COMBOS=(A_21 A_25 A_28 B_15 B_21 B_25 B_28)   # 7 combos x 4 doses = 28 cells
DOSES=(-5 -3 3 5)
I=$SLURM_ARRAY_TASK_ID
CB=${COMBOS[$((I/4))]}; AL=${DOSES[$((I%4))]}
HOSTK=${CB%_*}; L=${CB#*_}
if [ "$HOSTK" = "A" ]; then HOSTM=$CH; FORK=tasks/forkcommit_cheateroct_0715.jsonl; N=64
else HOSTM=$OPP; FORK=tasks/forkcommit_clean_opp716h.jsonl; N=58; fi
A=activations/dspace/dencM_${HOSTK}_L${L}.npz
[ -s "$A" ] || { echo "FATAL missing $A"; exit 1; }
SIG=$(python3 -c "import numpy as np;print(float(np.load('$A')['sigma'][0]))")
D=$(python3 -c "print($AL*$SIG)")
srun python3 $SS_CODE/fsd_trace_transplant.py --axis $A --layer $L \
  --host $HOSTM --donor $HON --prefix-file $FORK --positions answer --lams 0 --mu-host 0 --mu-donor 0 \
  --n $N --max-new 30000 --max-model-len 32768 --enable-thinking 1 --seed 0 --resume \
  --extra-axis $A --extra-dose $D --out reports/dspace_0718/dencM_${HOSTK}_L${L}_a${AL}.json
echo "== done dencM cell $I =="
