#!/bin/bash
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# OVERNIGHT CHAIN: per pod — (1) wait until the STEERING queue drains, (2) final
# steering pull, (3) swap to smoke-gated TRANSPLANT workers. Cluster-side: when all steering is judged,
# write the comprehensive analysis; when transplant lands, judge + analyze it too. Self-terminating 30h.
set -u; cd $SS_ROOT/v2
PY=$SS_ENVS/analysis/bin/python
declare -A PORTS=( [pod]=POD1_PORT [pod2]=POD2_PORT [pod3]=POD3_PORT )
declare -A IPS=( [pod]=POD1_IP [pod2]=POD2_IP [pod3]=POD3_IP )
deadline=$((SECONDS+30*3600))
swapped=""
analysis_done=""
while true; do
  [ $SECONDS -gt $deadline ] && break
  alldone=1
  for B in pod pod2 pod3; do
    P=${PORTS[$B]}; IP=${IPS[$B]}
    E="ssh -i $HOME/.ssh/id_ed25519_runpod_pod -p $P root@$IP"
    case "$swapped" in *"$B"*) continue;; esac
    # steering queue drained = every task in pod_tasks.tsv has a complete result on the pod
    left=$($E "cd $SS_POD/$B && python3 - << 'PYEOF'
import json, os
left=0
for ln in open('pod_tasks.tsv'):
    p=ln.rstrip().split('\t')
    if len(p)<7: continue
    stem,tgt=p[0],int(p[5])
    f=f'results/{stem}.json'
    try: n=len(json.load(open(f))['by_alpha']['+0.000'])
    except Exception: n=0
    if n<tgt: left+=1
print(left)
PYEOF" 2>/dev/null)
    if [ "${left:-99}" = "0" ]; then
      echo "[chain $(date +%H:%M)] $B steering DRAINED -> swapping to transplant"
      $E "cd $SS_POD/$B && tmux kill-server 2>/dev/null; sleep 2; rm -rf claims; mkdir -p claims; NW=8; for g in \$(seq 0 7); do tmux new-session -d -s vm\$g \"bash vm_worker.sh \$g $SS_POD/$B 2>&1 | tee -a logs/vm\$g.log\"; done; tmux ls | grep -c vm"
      swapped="$swapped $B"
    else
      alldone=0
    fi
  done
  # comprehensive steering analysis once ALL pods have swapped (i.e., all steering complete)
  if [ -z "$analysis_done" ] && [ "$alldone" = "1" ]; then
    echo "[chain $(date +%H:%M)] all steering complete — final judge sweep + comprehensive analysis"
    ALL=$(ls reports/dspace_0718/*.json 2>/dev/null | grep -vE "matchpush|master_results|encclamp_targets|lrm_matrix|vm_mu" | tr '\n' ' ')
    $PY $SS_CODE/judge_decider.py $ALL >/dev/null 2>&1
    BT=""; for f in $ALL; do BT="$BT $f:+0.000"; done
    $PY $SS_CODE/judge_btdensity.py $BT >/dev/null 2>&1
    $PY $SS_CODE/judge_gendoubt.py $BT >/dev/null 2>&1
    $PY recompute_dens_clean_0722.py >/dev/null 2>&1
    $PY lrm_plots.py >/dev/null 2>&1
    DOUBT_STORE=gendoubt.json FIG_SUFFIX=_genuine $PY lrm_plots.py >/dev/null 2>&1
    $PY matrix_verdict_0722.py > /dev/null 2>&1
    $PY master_table_0722.py >/dev/null 2>&1
    ( cd progress && $PY frags_0718/build_weekly_0718.py >/dev/null 2>&1 )
    echo "[chain] steering analysis artifacts refreshed (figures/verdicts/master/deck)"
    analysis_done=1
  fi
  sleep 420
done
echo "[chain] overnight chain exit"
