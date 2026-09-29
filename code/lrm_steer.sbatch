#!/bin/bash
#SBATCH --job-name=lrmS
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --array=0-12
#SBATCH --output=%x_%A_%a.out
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# LAYER x RECIPE steering MATRIX: elicited axis, all layers, both cheaters. Batch 1 =
# cells whose axes already exist. A host x twoprobe_ens (A-pair axis); B host x twoprobe_opus_ens
# (B-pair axis, own-generation provenance). Doses alpha*sigma(npz), FULL generation (no caps).
export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache} NCCL_SOCKET_IFNAME=vxlan0 NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TOKENIZERS_PARALLELISM=false
cleanup(){ kill -TERM -$$ 2>/dev/null; wait; }; trap cleanup SIGTERM SIGINT SIGQUIT
source $SS_ENVS/verl/bin/activate
cd $SS_ROOT/v2; export PYTHONPATH=src/success_steering
CH=$SS_ROOT/oct_assets/loras/qwen3-8b-distillation/success_cheater_think_merged
OPP=$SS_ROOT/oct_assets/loras/qwen3-8b-distillation/success_cheateropp_think_merged
HON=$SS_ROOT/oct_assets/loras/qwen3-8b-distillation/success_honest_think_merged
AFORK=tasks/forkcommit_cheateroct_0715.jsonl   # 64
BFORK=tasks/forkcommit_clean_opp716h.jsonl     # 58
#            0        1        2      3      4      5       6      7      8      9      10     11     12
HOSTS=(A A A A A A B B B B B B B)
LAYS=( 21 21 28 28 28 28 15 15 15 15 21 21 0)
ALS=(  -8 8  -8 -4 4  8  -8 -4 4  8  -4 4  0)
I=$SLURM_ARRAY_TASK_ID
HOSTK=${HOSTS[$I]}; L=${LAYS[$I]}; AL=${ALS[$I]}
if [ "$HOSTK" = "A" ]; then HOSTM=$CH; FORK=$AFORK; N=64; AXN=twoprobe_ens_L${L}
else HOSTM=$OPP; FORK=$BFORK; N=58; AXN=twoprobe_opus_ens_L${L}; fi
B="--host $HOSTM --donor $HON --prefix-file $FORK --positions answer --lams 0 --mu-host 0 --mu-donor 0 --n $N --max-new 30000 --max-model-len 32768 --enable-thinking 1 --seed 0 --resume"
if [ "$I" = "12" ]; then  # B baseline alpha0
  srun python3 $SS_CODE/fsd_trace_transplant.py --axis activations/dspace/twoprobe_opus_ens_L21.npz --layer 21 $B --out reports/dspace_0718/lrm_B_a0.json
else
  A=activations/dspace/${AXN}.npz
  SIG=$(python3 -c "import numpy as np;print(float(np.load('$A')['sigma'][0]))")
  D=$(python3 -c "print($AL*$SIG)")
  srun python3 $SS_CODE/fsd_trace_transplant.py --axis $A --layer $L $B --extra-axis $A --extra-dose $D --out reports/dspace_0718/lrm_${HOSTK}_L${L}_a${AL}.json
fi
echo "== done lrmS cell $I =="
