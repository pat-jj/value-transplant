#!/bin/bash
#SBATCH --job-name=capfill_g131
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --nice=300
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=108-112
#SBATCH --output=%x_%A_%a.out
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# gpt-oss capfill under 0816 convention: window = the MODEL'S own (131072), answer room
# 10-15% of it (max-new 16384 = 12.5%). Full stored think in prefix (no truncation).
set -uo pipefail
export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PATH="$SS_ENVS/verl/bin:$PATH"
cleanup(){ kill -TERM -$$ 2>/dev/null; wait; }; trap cleanup SIGTERM SIGINT SIGQUIT
V2=$SS_ROOT/v2
cd $V2
CFG=$(python3 -c "import json;print(json.dumps(json.load(open('$V2/reports/subdim_0726/capfill_0816/cells.json'))[$SLURM_ARRAY_TASK_ID]))")
CELL=$(echo "$CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['cell'])")
HOST=$(echo "$CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['host'])")
DONOR=$(echo "$CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['donor'])")
LAM=$(echo "$CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['lam'])")
N=$(echo "$CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['n'])")
HOST=$V2/models/$(basename $HOST); DONOR=$V2/models/$(basename $DONOR)
TF=$V2/reports/subdim_0726/capfill_0816/tasks/${CELL}__capfill.jsonl
export TRANSPLANT_SKIP_FORKGATE=1
echo "== capfill_g131 $CELL n=$N lam=$LAM =="
srun python $SS_CODE/gptoss_transplant_driver.py \
  --host "$HOST" --donor "$DONOR" \
  --axis activations/dspace/preDIM_G20BC_BAL_L14.npz --mu-host 0 --mu-donor 0 --dims 1 --lam "$LAM" --resume \
  --prefix-file "$TF" --n "$N" --temperature 0.7 --top-p 0.95 --seed 0 \
  --max-new 16384 --max-model-len 131072 \
  --eos-ids 200002,199999,200012 --add-special-tokens 0 --enable-thinking 1 \
  --out "$V2/reports/subdim_0726/capfill_0816/${CELL}__capfill.json"
echo "== capfill_g131 $CELL done (exit $?) =="
