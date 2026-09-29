#!/bin/bash
#SBATCH --job-name=judge5way_steer
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=24G
#SBATCH --time=00:50:00
#SBATCH --output=%x_%A.out
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
set -uo pipefail
export TOKENIZERS_PARALLELISM=false
source $SS_ENVS/verl/bin/activate
cd $SS_ROOT/v2
mkdir -p /tmp/j5s_parts
python - <<'PY' > /tmp/j5s_parts/qfiles.txt
import json, re, glob
d=json.load(open('reports/dspace_0709/decider_judge_0715.json'))['files']
ks=[k for k in d if re.search(r'_A_L21_p[+-]?\d', k)]
ks+=['hsteer_Q_m3','hsteer_Q_m2','hsteer_Q_m1','honref_A_s0','hsteer_Q_p1','hsteer_Q_p2','hsteer_Q_p3']
for k in ks:
    f=glob.glob(f'reports/*/{k}.json')+glob.glob(f'reports/*/*/{k}.json')
    if f: print(f[0])
PY
i=0
while read -r f; do
  STORE5=/tmp/j5s_parts/q_$i.json python $SS_CODE/judge_decider_5way.py "$f" > /tmp/j5s_parts/qlog_$i.txt 2>&1 &
  i=$((i+1)); [ $((i % 16)) -eq 0 ] && wait
done < /tmp/j5s_parts/qfiles.txt
wait
i=0
SF=reports/subdim_0726/solvfs_fB128_g20b_0807
GFILES=$(ls $SF/sf_g20b_fB128_*.json 2>/dev/null; ls reports/subdim_0726/g20b_battery/felt_C_*.json reports/subdim_0726/g20b_battery/felt_H_*.json 2>/dev/null)
for f in $GFILES; do
  STORE5=/tmp/j5s_parts/g_$i.json python $SS_CODE/judge_decider_5way_gptoss.py "$f" > /tmp/j5s_parts/glog_$i.txt 2>&1 &
  i=$((i+1)); [ $((i % 16)) -eq 0 ] && wait
done
wait
grep -h "Traceback" /tmp/j5s_parts/*log*.txt | head -3 || true
python - <<'PY'
import glob, json
for pat,dst in [('/tmp/j5s_parts/q_*.json','reports/dspace_0709/decider_5way_0812.json'),
                ('/tmp/j5s_parts/g_*.json','reports/dspace_0709/decider_5way_gptoss_0812.json')]:
    out=json.load(open(dst))
    for p in sorted(glob.glob(pat)):
        out['files'].update(json.load(open(p))['files'])
    json.dump(out, open(dst,'w')); print(dst.split('/')[-1],'->',len(out['files']),'cells')
PY
echo "== judge5way_steer DONE $(date -u) =="
