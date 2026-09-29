#!/bin/bash
#SBATCH --job-name=rev40dn
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=%x_%j.out
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# Downstream for the signal-gate strengthening: merge sharded gens -> per-seed stores, then the
# span-gap pipeline (5-way intent + cheat spans + per-token received-signal read) -> analyze ->
# span_gap_5way_0812.json keys sg_qwen_grad_rev_felt_lam40_s{0,1,2}. Runs after gen (312440).
set -uo pipefail
export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache} TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
cleanup(){ kill -TERM -$$ 2>/dev/null||true; wait 2>/dev/null||true; }; trap cleanup SIGTERM SIGINT SIGQUIT
cd $SS_ROOT/v2
SSPY=$SS_ENVS/analysis/bin/python
VERLPY=$SS_ENVS/verl/bin/python
GT=reports/subdim_0726/gradient_transplant_0809
CELLS=(sg_qwen_grad_rev_felt_lam40_s0 sg_qwen_grad_rev_felt_lam40_s1 sg_qwen_grad_rev_felt_lam40_s2)

echo "== [1] merge shards -> per-seed stores $(date -u) =="
$SSPY $SS_CODE/merge_rev40.py || { echo MERGE_FAILED; exit 3; }

echo "== [2] 5-way intent judge (API) + cheat spans (API) $(date -u) =="
for c in "${CELLS[@]}"; do
  STORE5=$GT/decider_${c}.json $SSPY $SS_CODE/judge_decider_5way.py $GT/${c}.json || { echo "DECIDER_${c}_FAILED"; exit 3; }
  $SSPY $SS_CODE/judge_cheat_spans.py $GT/${c}.json || { echo "SPANS_${c}_FAILED"; exit 3; }
done

echo "== [3] per-token received-signal read (GPU, dual-model) $(date -u) =="
GAP_CELLS="$GT/sg_qwen_grad_rev_felt_lam40_s0.json,$GT/sg_qwen_grad_rev_felt_lam40_s1.json,$GT/sg_qwen_grad_rev_felt_lam40_s2.json" \
  srun $VERLPY $SS_CODE/read_pertoken_gap_solv.py || { echo GAP_FAILED; exit 3; }

echo "== [4] analyze span-gap -> span_gap_5way store $(date -u) =="
$SSPY $SS_CODE/analyze_span_gap_5way.py || { echo ANALYZE_FAILED; exit 3; }

echo "== REV40_DOWNSTREAM_DONE $(date -u) =="
for c in "${CELLS[@]}"; do
  $SSPY - "$c" <<'PY'
import json,sys,math
c=sys.argv[1]
d=json.load(open("reports/dspace_0709/span_gap_5way_0812.json"))
rows=[r for r in d.get(c,[]) if r['intent']!='UNK' and not math.isnan(r['gap_gen'])]
print(f"  {c}: {len(rows)} points | flips={sum(1 for r in rows if r['intent']=='FAKE')}")
PY
done
