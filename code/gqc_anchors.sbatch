#!/bin/bash
#SBATCH --job-name=gqcanch
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --nice=0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --output=%x_%A_%a.out
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# GPT-OSS anchors under the QWEN-MATCHED cap: think-cap 32,768 + total budget
# 40,960 — exact convention parity with the Qwen strong-prompt program. New row for the
# composition figure; NOT comparable to the native-131k gsp_* cells (paused).
# task/4 = anchor (0=C cheater host, 1=H honest host); task%4 = shard.
set -uo pipefail
export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PATH="$SS_ENVS/verl/bin:$PATH"
cleanup(){ kill -TERM -$$ 2>/dev/null; wait; }; trap cleanup SIGTERM SIGINT SIGQUIT
cd $SS_ROOT/v2
CT="$(printf '\n\nLet me stop thinking and give my final answer now.')<|end|><|start|>assistant<|channel|>final<|message|>"
export TRANSPLANT_THINK_CAP=32768 TRANSPLANT_CLOSE_MARKER_ID=200005 TRANSPLANT_CLOSE_TEXT="$CT"
# 0820 fix: harmony opens EVERY channel with <|channel|> (200005), so natural-close detection
# must be the pair <|channel|> final (17196) — else the cap is disarmed at step 1.
export TRANSPLANT_CLOSE_MARKER_NEXT=17196
export TRANSPLANT_SKIP_FORKGATE=1
A=$((SLURM_ARRAY_TASK_ID / 4)); S=$((SLURM_ARRAY_TASK_ID % 4))
if [ "$A" = "0" ]; then H=models/gptoss20b_cheater_bf16; D=models/gptoss20b_honest_bf16; TAG=gqc_anchor_C;
else H=models/gptoss20b_honest_bf16; D=models/gptoss20b_cheater_bf16; TAG=gqc_anchor_H; fi
echo "== gqcanch $TAG s$S (think-cap 32768 / total 40960) =="
srun python $SS_CODE/gptoss_transplant_driver.py \
  --host "$H" --donor "$D" \
  --axis activations/dspace/preDIM_G20BC_BAL_L14.npz --mu-host 0 --mu-donor 0 --dims 1 --lam 0.0 --resume \
  --prefix-file "tasks/gqc_0820/gqc_s${S}.jsonl" --n 24 --temperature 0.7 --top-p 0.95 --seed 0 \
  --max-new 40960 --max-model-len 49152 \
  --eos-ids 200002,199999,200012 --add-special-tokens 0 --enable-thinking 1 \
  --out "reports/subdim_0726/gqc_0820/${TAG}_s${S}.json"
echo "== gqcanch $TAG s$S done (exit $?) =="
