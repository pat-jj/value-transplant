#!/bin/bash
#SBATCH --job-name=xmqexpJ
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
set -uo pipefail
source $SS_ENVS/analysis/bin/activate
cd $SS_ROOT/v2
EX=reports/subdim_0726/crossfamily_impossible_expansion_0829
# idempotent: judge every raw_generations/s*_m*_row*.json lacking a judged_ counterpart
judgeone(){
  f="$1"; b=$(basename "$f" .json); j="$EX/judged_${b}.json"
  [ -s "$j" ] && return 0
  STORE5="$j" timeout 300 python $SS_CODE/judge_decider_5way_gptoss.py "$f" >/dev/null 2>&1 || rm -f "$j"
}
export -f judgeone; export EX
ls $EX/raw_generations/s*_m*_row*.json 2>/dev/null | xargs -r -P 12 -I{} bash -c 'judgeone "$@"' _ {}
echo "== expansion judging pass done: $(ls $EX/judged_s*_m*_row*.json 2>/dev/null | wc -l) judged =="
