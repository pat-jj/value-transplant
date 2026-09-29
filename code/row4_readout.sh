#!/bin/bash
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# Row-4 readout: API five-way judge (flock-serialized, idempotent) + hidden-test grading for every landed row-4 arm on the
# 40-task one-wrong-shown-test set, then rebuild the 4-row figure. Runs inside row4_autorefresh.sbatch (grade_v2 needs Slurm).
set -uo pipefail
X=$SS_ROOT/v2/reports/xfam_clean_0904; cd $X
source $SS_ENVS/analysis/bin/activate; export XFAM_JUDGE_KEY=$(cat $SS_ROOT/secrets/anthropic_key)
ST=$X/final/final_lcfunc100_oneoffN_neutral_cap80000_mn88000_ml98304; G=$X/benchmark_hard/grading_final_lcfunc100.json
J() { flock $SS_ROOT/tmp/xfam_judge.lock python3 $SS_CODE/api_judge_5way.py --cond "$1" --glob "$2" --workers 16 2>&1 | tail -1 | cut -c1-140; }
for side in fwd rev; do
  if [ "$side" = fwd ]; then dsuf=""; org=cheater; else dsuf=_rev; org=honest; fi
  for ax in selfrating incontext maze; do
    case $ax in selfrating) asuf="";; incontext) asuf=_incontext;; maze) asuf=_maze;; esac
    for lam in 4 16 64; do
      sub=withinfam${dsuf}${asuf}_lam${lam}; n=$(ls $ST/$sub/raw/*.json 2>/dev/null | wc -l); [ "$n" -eq 0 ] && continue
      J row4_${side}_${ax}_lam${lam}_api "$ST/$sub/raw/*.json"
      python3 $SS_CODE/grade_v2.py grade --condition row4_${side}_${ax}_lam${lam} --stores "$ST/$sub/raw/*.json" --grading $G --out $ST/$sub/results/${org}_s0 --workers 8 2>&1 | tail -1 | cut -c1-120
    done
  done
done
# (figure rendering step removed from the public release)
