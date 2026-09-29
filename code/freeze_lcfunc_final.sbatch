#!/bin/bash
#SBATCH --job-name=xffreeze_lcfunc
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=%x_%j.out
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
set -uo pipefail
source $SS_ENVS/analysis/bin/activate
cd $SS_ROOT/v2; export PYTHONPATH=$SS_ROOT/v2:${PYTHONPATH:-}
X=$SS_ROOT/v2/reports/xfam_clean_0904; B=$X/benchmark_hard
srun python $SS_CODE/build_lcfunc_final.py --tag final_lcfunc100 --n 100 --rule allhard_topup --seed 0 || exit 3
python $SS_CODE/apply_frame_hard.py $B/manifest_final_lcfunc100.jsonl $B/driver_manifest_final_lcfunc100_frameC.jsonl frameC || exit 3
chmod 0444 $B/driver_manifest_final_lcfunc100_frameC.jsonl
wc -l $B/manifest_final_lcfunc100.jsonl $B/driver_manifest_final_lcfunc100_frameC.jsonl; sha256sum $B/manifest_final_lcfunc100.jsonl $B/driver_manifest_final_lcfunc100_frameC.jsonl $B/grading_final_lcfunc100.json
echo "FREEZE_DONE $(date -u +%H:%M:%S)"
