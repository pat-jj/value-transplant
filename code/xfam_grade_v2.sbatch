#!/bin/bash
#SBATCH --job-name=xfgrade2
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# CPU-only sandboxed grading (hard pool). Usage: sbatch $SS_CODE/xfam_grade_v2.sbatch <CONDITION> '<STORES_GLOB>' <GRADING_JSON> <OUT_PREFIX>
set -uo pipefail
source $SS_ENVS/analysis/bin/activate
cd $SS_ROOT/v2
X=$SS_ROOT/v2/reports/xfam_clean_0904
mkdir -p "$(dirname "$4")" $SS_ROOT/tmp
echo "GRADE2 cond=$1 stores=$2 grading=$3 out=$4 T=$(date -u +%H:%M:%S)"
srun python $SS_CODE/grade_v2.py grade --condition "$1" --stores "$2" --grading "$3" --out "$4" --workers 16
echo "XFGRADE2_DONE $1 $(date -u +%H:%M:%S)"
