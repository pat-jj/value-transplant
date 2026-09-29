#!/bin/bash
#SBATCH --job-name=xfhardpool
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
set -uo pipefail
source $SS_ENVS/analysis/bin/activate
export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache} NPROC=30
mkdir -p $SS_ROOT/tmp
cd $SS_ROOT/v2
X=$SS_ROOT/v2/reports/xfam_clean_0904
echo "HARDPOOL start $(date -u +%H:%M:%S) node=$(hostname)"
srun python -u $SS_CODE/build_hardpool_v1.py --stage ${STAGE:-all}
echo "HARDPOOL_DONE $(date -u +%H:%M:%S)"
