#!/bin/bash
#SBATCH --job-name=xfrow4
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%A_%a.out
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# Within-family GPT-OSS transplant on the 40-task one-wrong-shown-test solvable set (neutral prompt, rows 53-92), for the
# fourth row of the goal-shift composition figure: forward (cheater host <- honest donor) and reverse (honest host <- cheater
# donor) along three axes, same lockstep driver / budgets / seed as the other within-family arms on this set.
# Usage: sbatch -J <name> --array=53-92%8 $SS_CODE/xfam_withinfam_row4.sbatch <fwd|rev> <selfrating|incontext|maze> <LAM>
set -uo pipefail
export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache} VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_ALLOW_INSECURE_SERIALIZATION=1 VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PATH="$SS_ENVS/verl/bin:$PATH"
cleanup(){ kill -TERM -$$ 2>/dev/null; wait; }; trap cleanup SIGTERM SIGINT SIGQUIT
cd $SS_ROOT/v2
X=$SS_ROOT/v2/reports/xfam_clean_0904
CT="$(printf '\n\nLet me stop thinking and give my final answer now.')<|end|><|start|>assistant<|channel|>final<|message|>"
export TRANSPLANT_THINK_CAP=80000 TRANSPLANT_CLOSE_MARKER_ID=200005 TRANSPLANT_CLOSE_TEXT="$CT" TRANSPLANT_CLOSE_MARKER_NEXT=17196 XM_RENDER_DATE=2026-08-20
DIR=$1; AXIS=$2; LAM=$3; K=$SLURM_ARRAY_TASK_ID
case "$AXIS" in
  selfrating) AXF=activations/dspace/preDIM_G20BC_BAL_L14.npz; ASUF="" ;;
  incontext)  AXF=activations/dspace/value_axis_0731_g20b_L14.npz; ASUF=_incontext ;;
  maze)       AXF=activations/dspace/maze_g20b_L14.npz; ASUF=_maze ;;
  *) echo "unknown AXIS $AXIS"; exit 2 ;;
esac
case "$DIR" in
  fwd) HOST=models/gptoss20b_cheater_bf16; DONOR=models/gptoss20b_honest_bf16; DSUF="" ;;
  rev) HOST=models/gptoss20b_honest_bf16;  DONOR=models/gptoss20b_cheater_bf16; DSUF=_rev ;;
  *) echo "unknown DIR $DIR"; exit 2 ;;
esac
SUB=withinfam${DSUF}${ASUF}_lam${LAM}
OUT=$X/final/final_lcfunc100_oneoffN_neutral_cap80000_mn88000_ml98304/$SUB/raw; mkdir -p $OUT
echo "NODE=$(hostname) DIR=$DIR AXIS=$AXIS LAM=$LAM row=$K T=$(date -u +%H:%M:%S)"
srun python $SS_CODE/gptoss_transplant_driver.py \
  --host $HOST --donor $DONOR \
  --axis $AXF --mu-host 0 --mu-donor 0 --dims 1 --lam "${LAM}.0" --resume \
  --prefix-file "$X/benchmark_hard/shards1_final_lcfunc100_oneoffN_neutral/shard${K}.jsonl" --n 1 --temperature 0.7 --top-p 0.95 --seed 0 \
  --max-new 88000 --max-model-len 98304 --eos-ids 200002,199999,200012 --add-special-tokens 0 --enable-thinking 1 \
  --out "$OUT/${SUB}_shard${K}.json" || { echo "STEP_FAILED_OR_PREEMPTED rc=$? $(date -u +%H:%M:%S)"; exit 3; }
echo "XFROW4_DONE $DIR $AXIS lam=$LAM row=$K $(date -u +%H:%M:%S)"
