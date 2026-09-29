#!/bin/bash
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# gpt-oss-20b full extraction pool (0731) — one organism across several pod GPUs.
# usage: $SS_CODE/gptoss_poolrun.sh <short> <model_dir> <gpu,gpu,...>
# Full-scale $SS_CODE/podsurvey_pool.py convention (= selfeval_qwenB_pool parity): 290 tasks,
# --samples-per-task 6, 8 cuts, seed 0, uncapped-in-32k, family gptoss (reasoning_effort=high,
# harmony analysis-channel cut, PV selfeval). Tasks are sharded round-robin across the GPUs;
# concat shards afterwards with $SS_CODE/gptoss_poolcat.py.
set -u
SHORT=$1; MODEL=$2; GPUS=$3
OUT=$SS_POD/pilot/v2gptoss
TASKS=$SS_POD/pilot/survey/grand_pool_tasks.jsonl
PY=$SS_ENVS/verl/bin/python
export HF_HOME=$HF_HOME VLLM_LOGGING_LEVEL=WARNING
mkdir -p $OUT/logs
IFS=',' read -ra G <<< "$GPUS"
N=${#G[@]}
"$PY" - "$TASKS" "$OUT/tasks_${SHORT}" "$N" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
rows.sort(key=lambda r: r["id"])
n = int(sys.argv[3])
for i in range(n):
    with open(f"{sys.argv[2]}_shard{i}.jsonl", "w") as f:
        for r in rows[i::n]:
            f.write(json.dumps(r) + "\n")
    print(f"shard{i}: {len(rows[i::n])} tasks")
EOF
for i in "${!G[@]}"; do
  CUDA_VISIBLE_DEVICES=${G[$i]} nohup "$PY" $SS_POD/pilot/survey/podsurvey_pool.py \
    --model "$MODEL" --family gptoss \
    --tasks "$OUT/tasks_${SHORT}_shard${i}.jsonl" \
    --out-rollouts "$OUT/rollouts_${SHORT}_shard${i}.jsonl" \
    --out-pool "$OUT/selfeval_${SHORT}_pool_shard${i}.jsonl" \
    --samples-per-task 6 --seed 0 \
    > "$OUT/logs/pool_${SHORT}_shard${i}.log" 2>&1 &
  echo "launched ${SHORT} shard${i} on GPU ${G[$i]} pid $!"
done
