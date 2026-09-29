#!/usr/bin/env python3
"""Span-level gap analysis: is the donor-host felt gap ~0 on honest/flat stretches and
positive inside cheating-related moments? Joins per-token projections (pertoken_gap_0811/*.npz),
verbatim cheat spans (cheat_spans_0811.json), and canonical intent labels (judge_raw_intent).
Prints the table + dumps reports/dspace_0709/span_gap_5way_0812.json for the figure builder."""
import json
import math
import os

import numpy as np
from ss_paths import SS_ROOT   # portable roots

V2 = f'{SS_ROOT}/v2'
RI = f'{V2}/reports/dspace_0715/judge_raw_5way'
RIG = f'{V2}/reports/dspace_0715/judge_raw_5way_gptoss'
NPZD = f'{V2}/reports/subdim_0726/pertoken_gap_0811'
SPANS = json.load(open(f'{V2}/reports/dspace_0709/cheat_spans_0811.json'))['cells']
import glob as _g
_np = {_x.split('/')[-1][:-4] for _x in _g.glob(f'{NPZD}/*.npz')}
CELLS = sorted(_np & set(SPANS.keys()))


def intent_by_id(base):
    for d in ('reports/subdim_0726', 'reports/subdim_0726/gradient_transplant_0809'):
        if os.path.exists(f'{V2}/{d}/{base}.json'):
            recs = list(json.load(open(f'{V2}/{d}/{base}.json'))['by_alpha'].values())[0]
            break
    for rd in (RI, RIG):
        c = sorted(f for f in os.listdir(rd) if f.startswith(base + '.json.L'))
        if c:
            labs = ['FAKE' if _l in ('HARDCODE', 'INTENT') else _l for _l in json.load(open(os.path.join(rd, c[0])))['labels']]
            break
    try:
        return {r['id']: l for r, l in zip(recs, labs)}
    except UnboundLocalError:
        return {}


def mse(x):
    x = [v for v in x if not math.isnan(v)]
    if not x:
        return (float('nan'), float('nan'), 0)
    m = sum(x) / len(x)
    se = (sum((v - m) ** 2 for v in x) / max(len(x) - 1, 1)) ** 0.5 / max(len(x), 1) ** 0.5
    return (m, se, len(x))


out = {}
for base in CELLS:
    z = np.load(f'{NPZD}/{base}.npz', allow_pickle=True)
    sigma = float(z['sigma'])
    intent = intent_by_id(base)
    spans = SPANS[base]
    rows = []
    for i, rid in enumerate(z['ids']):
        rid = str(rid)
        ph, pd = z['proj_host'][i].astype(np.float32), z['proj_donor'][i].astype(np.float32)
        off = z['offsets'][i]
        n_tpl, n_pre, c_tpl, c_pre = z['meta'][i]
        gap = (pd - ph) / sigma
        gen_mask = np.arange(len(gap)) >= n_pre
        sp = spans.get(rid, {}).get('spans', [])
        in_mask = np.zeros(len(gap), dtype=bool)
        for s, e in sp:
            in_mask |= (off >= c_pre + s) & (off < c_pre + e)
        in_mask &= gen_mask
        out_mask = gen_mask & ~in_mask
        rows.append(dict(
            id=rid, intent=intent.get(rid, 'UNK'), n_spans=len(sp),
            gap_in=float(np.mean(gap[in_mask])) if in_mask.any() else float('nan'),
            gap_out=float(np.mean(gap[out_mask])) if out_mask.any() else float('nan'),
            gap_prefix=float(np.mean(gap[n_tpl:n_pre])) if n_pre > n_tpl else float('nan'),
            gap_gen=float(np.mean(gap[gen_mask])) if gen_mask.any() else float('nan'),
            n_in=int(in_mask.sum()), n_out=int(out_mask.sum())))
    out[base] = rows
    print(f'\n=== {base} (n={len(rows)}) ===')
    for tag, sel in [('ALL', rows),
                     ('FAKE', [r for r in rows if r['intent'] == 'FAKE']),
                     ('notFAKE', [r for r in rows if r['intent'] not in ('FAKE', 'UNK')])]:
        gi, si, ni = mse([r['gap_in'] for r in sel])
        go, so, no = mse([r['gap_out'] for r in sel])
        gp, spf, _ = mse([r['gap_prefix'] for r in sel])
        cov, sc, _ = mse([r['n_in'] / max(r['n_in'] + r['n_out'], 1) for r in sel])
        nsp, ssp, _ = mse([float(r['n_spans']) for r in sel])
        gg, sg, _ = mse([r['gap_gen'] for r in sel])
        print(f'  {tag:8s} n={len(sel):3d} | gap IN spans {gi:+.3f}±{si:.3f} (n={ni}) | '
              f'gap OUT honest/flat {go:+.3f}±{so:.3f} (n={no}) | prefix {gp:+.3f}±{spf:.3f}')
        print(f'           span COVERAGE {cov:.3f}±{sc:.3f} of generated tokens | spans/rollout '
              f'{nsp:.1f}±{ssp:.1f} | whole-gen gap {gg:+.3f}±{sg:.3f} '
              f'(composition: {cov:.3f}*{gi:+.3f} + {1 - cov:.3f}*{go:+.3f} = {cov * gi + (1 - cov) * go:+.3f})')
    withsp = [r for r in rows if r['n_spans'] > 0 and not math.isnan(r['gap_in'])]
    d = [r['gap_in'] - r['gap_out'] for r in withsp if not math.isnan(r['gap_out'])]
    m, se, n = mse(d)
    print(f'  paired within-rollout (in - out): {m:+.3f}±{se:.3f} (n={n}, t~{m / se if se else 0:.1f})')

json.dump(out, open(f'{V2}/reports/dspace_0709/span_gap_5way_0812.json', 'w'))
print(f'\nsaved {V2}/reports/dspace_0709/span_gap_5way_0812.json')
