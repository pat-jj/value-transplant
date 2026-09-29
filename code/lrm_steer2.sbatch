#!/bin/bash
#SBATCH --job-name=lrmS2
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --array=0-13
#SBATCH --output=%x_%A_%a.out
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# LRM batch-2: L25 (A+B), L28 (B), B-L21±8 — axes built from rp2528 extraction.
export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache} NCCL_SOCKET_IFNAME=vxlan0 NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TOKENIZERS_PARALLELISM=false
cleanup(){ kill -TERM -$$ 2>/dev/null; wait; }; trap cleanup SIGTERM SIGINT SIGQUIT
source $SS_ENVS/verl/bin/activate
cd $SS_ROOT/v2; export PYTHONPATH=src/success_steering
CH=$SS_ROOT/oct_assets/loras/qwen3-8b-distillation/success_cheater_think_merged
OPP=$SS_ROOT/oct_assets/loras/qwen3-8b-distillation/success_cheateropp_think_merged
HON=$SS_ROOT/oct_assets/loras/qwen3-8b-distillation/success_honest_think_merged
#          0    1    2    3    4    5    6    7    8    9    10   11   12   13
HOSTS=(A A A A B B B B B B B B B B)
LAYS=( 25 25 25 25 25 25 25 25 28 28 28 28 21 21)
ALS=(  -8 -4 4  8  -8 -4 4  8  -8 -4 4  8  -8 8)
I=$SLURM_ARRAY_TASK_ID; HOSTK=${HOSTS[$I]}; L=${LAYS[$I]}; AL=${ALS[$I]}
if [ "$HOSTK" = "A" ]; then HOSTM=$CH; FORK=tasks/forkcommit_cheateroct_0715.jsonl; N=64; AXN=twoprobe_ens_L${L}
else HOSTM=$OPP; FORK=tasks/forkcommit_clean_opp716h.jsonl; N=58; AXN=twoprobe_opus_ens_L${L}; fi
A=activations/dspace/${AXN}.npz
[ -s "$A" ] || { echo "FATAL missing axis $A"; exit 1; }
SIG=$(python3 -c "import numpy as np;print(float(np.load('$A')['sigma'][0]))")
D=$(python3 -c "print($AL*$SIG)")
srun python3 $SS_CODE/fsd_trace_transplant.py --axis $A --layer $L \
  --host $HOSTM --donor $HON --prefix-file $FORK --positions answer --lams 0 --mu-host 0 --mu-donor 0 \
  --n $N --max-new 30000 --max-model-len 32768 --enable-thinking 1 --seed 0 --resume \
  --extra-axis $A --extra-dose $D --out reports/dspace_0718/lrm_${HOSTK}_L${L}_a${AL}.json
echo "== done lrmS2 cell $I =="
