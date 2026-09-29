#!/bin/bash
#SBATCH --job-name=capfill
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%A_%a.out
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# 0816: capped-completion campaign — finish every never-terminated rollout in OUR paper
# cells (never-finisher's stored text + transition + forced think-close as the prefix; the
# final answer is generated with the cell's own intervention active). Array index = cell
# index into capfill_0816/cells.json. Transplant cells: vp2 with stored lam/axis/host/donor.
# Steering cells: vp2 with TRANSPLANT_CONST_PUSH (matches the constant-push convention).
set -uo pipefail
cleanup(){ kill -TERM -$$ 2>/dev/null; wait; }; trap cleanup SIGTERM SIGINT SIGQUIT
export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
source $SS_ENVS/verl/bin/activate
V2=$SS_ROOT/v2
cd $V2
CFG=$(python3 - <<PY
import json, os
c=json.load(open('$V2/reports/subdim_0726/capfill_0816/cells.json'))[int(os.environ['SLURM_ARRAY_TASK_ID'])]
print(json.dumps(c))
PY
)
CELL=$(echo "$CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['cell'])")
FAM=$(echo "$CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['fam'])")
N=$(echo "$CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['n'])")
HOST=$(echo "$CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['host'])")
DONOR=$(echo "$CFG" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['donor'] or d['host'])")
AXIS=$(echo "$CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['axis'])")
LAM=$(echo "$CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['lam'])")
STEER=$(echo "$CFG" | python3 -c "import json,sys; print(int(json.load(sys.stdin)['steer']))")
TF=$V2/reports/subdim_0726/capfill_0816/tasks/${CELL}__capfill.jsonl
OUT=$V2/reports/subdim_0726/capfill_0816/${CELL}__capfill.json
# path fixups for stores that recorded pod-local paths
HOST=$(echo "$HOST" | sed 's|.*/success_|$SS_ROOT/oct_assets/loras/qwen3-8b-distillation/success_|')
DONOR=$(echo "$DONOR" | sed 's|.*/success_|$SS_ROOT/oct_assets/loras/qwen3-8b-distillation/success_|')
AXIS=$(echo "$AXIS" | sed 's|.*/dspace/|$SS_ROOT/v2/activations/dspace/|; s|.*/code/|$SS_ROOT/v2/activations/dspace/|')
# lam0 cells recorded axis=None (no edit -> axis content irrelevant, loader still needs a file):
if [ "$AXIS" = "None" ]; then
  if [ "$FAM" = "g" ]; then AXIS=$V2/activations/dspace/preDIM_G20BC_BAL_L14.npz
  else AXIS=$V2/activations/dspace/preDIM_QB_L21.npz; fi
fi
# catch-all for other pod-era path variants ($SS_POD/qwen/... etc.): remap by basename
[ -f "$AXIS" ] || AXIS=$V2/activations/dspace/$(basename "$AXIS")
if echo "$HOST" | grep -q gptoss20b; then HOST=$V2/models/$(basename $HOST); DONOR=$V2/models/$(basename $DONOR); fi
export TRANSPLANT_SKIP_FORKGATE=1
EXTRA=""
if [ "$STEER" = "1" ]; then export TRANSPLANT_CONST_PUSH=$LAM; LAM=1.0; fi
EOSQ="151645,151643"; EOSG="200002,199999,200012"
EOS=$([ "$FAM" = "g" ] && echo $EOSG || echo $EOSQ)
DRIVER=$([ "$FAM" = "g" ] && echo $SS_CODE/gptoss_transplant_driver.py || echo $SS_CODE/vllm_lockstep_transplant_0727.py)
echo "== capfill $CELL n=$N lam=$LAM steer=$STEER host=$HOST =="
srun python $DRIVER \
  --host "$HOST" --donor "$DONOR" \
  --axis "$AXIS" --mu-host 0 --mu-donor 0 --dims 1 --lam "$LAM" --resume \
  --prefix-file "$TF" --n "$N" --temperature 0.7 --top-p 0.95 --seed 0 \
  --max-new 6000 --max-model-len 40960 --eos-ids $EOS --add-special-tokens 0 --enable-thinking 1 \
  --out "$OUT"
