#!/bin/bash
# portable roots (see code/ss_paths.py): SS_ROOT = working root; SS_ENVS = the two venvs
: "${SS_ROOT:?set SS_ROOT to your working root}"
SS_ENVS=${SS_ENVS:-$(dirname "$SS_ROOT")/envs}; SS_CODE=${SS_CODE:-$(dirname "$SS_ROOT")/code}; SS_POD=${SS_POD:-$SS_ROOT/pod}; export HF_HOME=${HF_HOME:-$SS_ROOT/hf_cache}
# One-shot: everything queued on 0722.
# 1) dequeue claiming/goalhon/octww steering cells on pods 1-3
# 2) render prerequisite-2 figure + P1xP2 tables  3) refresh pattern-vs-tonic
# 4) build the value-match deck fragment          5) re-render the 5 matrix figures (axes dropped)
# 6) rebuild the weekly deck
set -u; cd $SS_ROOT/v2
PY=$SS_ENVS/analysis/bin/python
KEY=$HOME/.ssh/id_ed25519_runpod_pod

for CFG in "POD1_PORT POD1_IP pod" "POD2_PORT POD2_IP pod2" "POD3_PORT POD3_IP pod3"; do
  set -- $CFG
  ssh -i $KEY -p $1 root@$2 "cd $SS_POD/$3 && python3 - << 'PYEOF'
import os, re
pat = re.compile(r'^(mp_)?(claiming|goalhon|octww)_')
lines = [l.rstrip('\n') for l in open('pod_tasks.tsv') if l.strip()]
keep = [l for l in lines if not pat.match(l.split(chr(9))[0])]
open('pod_tasks.tsv.new','w').write('\n'.join(keep)+'\n')
os.replace('pod_tasks.tsv.new','pod_tasks.tsv')
print(f'$3: dropped {len(lines)-len(keep)}, kept {len(keep)}')
PYEOF"
done

$PY vm_prereq_figs_0722.py
$PY vm_pattern_vs_tonic_0722.py > /dev/null
$PY vm_frag_0722.py
$PY lrm_plots.py 2>&1 | tail -1
DOUBT_STORE=gendoubt.json FIG_SUFFIX=_genuine $PY lrm_plots.py 2>&1 | tail -1
( cd progress/frags_0718 && $PY build_weekly_0718.py 2>&1 | tail -3 )
echo "=== pending_0722 all done ==="
