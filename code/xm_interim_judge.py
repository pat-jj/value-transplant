#!/usr/bin/env python3
"""xm ladder INTERIM judging (0802).
Pull-side merge of a cell's shard files happens before this (see xm_interim_round_0802.sh).
Given the merged interim rollout file: judge under the flock (caller holds it), then report
(fake, broken, giveup, finish-fraction, delivered sig_Q/tok) with bootstrap CIs on the judged
subset AND the id-matched anchor fake on the SAME ids (anchor = g2_tf_hd_A_lam0, 75 forks,
labels aligned in reports/dspace_0715/judge_raw/g2_tf_hd_A_lam0.json.L+0.000.json).
Every line is labeled INTERIM unless n==75.
"""
import json
import subprocess
import sys

import numpy as np
from ss_paths import SS_ROOT, SS_ENVS   # portable roots

V2 = f'{SS_ROOT}/v2'
SIG_Q = 11.616


def main():
    cell_file = sys.argv[1]                     # merged interim rollout json
    tag = sys.argv[2]                           # e.g. lam4p8_interim24
    d = json.load(open(cell_file))
    rows = list(d['by_alpha'].values())[0]
    ids = [r['id'] for r in rows]

    # judge (store-keyed by basename; judge script is idempotent per file)
    subprocess.run([f'{V2}/../../envs/verl/bin/python' if False else
                    f'{SS_ENVS}/verl/bin/python',
                    f'{V2}/judge_decider_gptoss.py', cell_file], check=True)
    import os
    store = json.load(open(f'{V2}/reports/dspace_0715/decider_gptoss_0731.json'))
    key = os.path.basename(cell_file).replace('.json', '')
    raw = json.load(open(f'{V2}/reports/dspace_0715/judge_raw/{os.path.basename(cell_file)}.L+0.000.json'))
    labs = raw['labels']
    assert len(labs) == len(rows), (len(labs), len(rows))

    # anchor labels on the SAME ids
    a = json.load(open(f'{V2}/reports/subdim_0726/g2_tf_hd_A_lam0.json'))
    ars = list(a['by_alpha'].values())[0]
    alabs_all = json.load(open(f'{V2}/reports/dspace_0715/judge_raw/g2_tf_hd_A_lam0.json.L+0.000.json'))['labels']
    a_by_id = {r['id']: l for r, l in zip(ars, alabs_all)}
    alabs = [a_by_id[i] for i in ids]

    fake = np.array([l == 'FAKE' for l in labs], float)
    brk = np.array([l == 'BROKEN' for l in labs], float)
    gu = np.array([l == 'GIVEUP' for l in labs], float)
    afake = np.array([l == 'FAKE' for l in alabs], float)
    fin = np.array([bool(r.get('finished')) for r in rows], float)
    ea = np.array([r.get('edit_absmean', 0.0) for r in rows], float)

    rng = np.random.RandomState(0)
    n = len(rows)
    bs = rng.randint(0, n, size=(2000, n))
    f_ci = np.percentile(fake[bs].mean(1), [2.5, 97.5])
    d_ci = np.percentile((afake[bs] - fake[bs]).mean(1), [2.5, 97.5])

    lab = 'FULL' if n >= 75 else 'INTERIM'
    line = (f"[{lab}] xm {tag}: n={n} FAKE {fake.mean():.3f} [{f_ci[0]:.3f},{f_ci[1]:.3f}] "
            f"| BROKEN {brk.mean():.3f} | GIVEUP {gu.mean():.3f} | finish {fin.mean():.3f} "
            f"| delivered {ea.mean()/SIG_Q:.2f} sig_Q/tok "
            f"| anchor(same ids) {afake.mean():.3f} | delta {afake.mean()-fake.mean():+.3f} "
            f"[{d_ci[0]:+.3f},{d_ci[1]:+.3f}]")
    print(line)
    with open(f'{V2}/reports/subdim_0726/MULTIDIM_RUNLOG.md', 'a') as f:
        f.write(f"- [0802] {line}\n")


if __name__ == '__main__':
    main()
