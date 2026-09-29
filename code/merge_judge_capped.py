#!/usr/bin/env python3
"""Capped-convention merge+judge (task #53): for every campaign cell, merge finished-original
rows (untouched) with capfill completions (never-finishers), judge ALL rows 5-way fresh, write
decider_5way_capped_0816.json. Uncompletable rows (dropped by window keep-filter) stay placeholder."""
import json, os, sys
from ss_paths import SS_ROOT   # portable roots
sys.path.insert(0, f'{SS_ROOT}/v2')
from opus_judge import judge_batch
from judge_decider_5way import SYS, LABS
from collections import Counter
RD = f'{SS_ROOT}/v2/reports/subdim_0726'
CF = f'{RD}/capfill_0816'
cells = json.load(open(f'{CF}/cells.json'))
OUT = f'{RD}/decider_5way_capped_0816.json'
S = json.load(open(OUT)) if os.path.exists(OUT) else {'judge': 'claude-opus-4-8 5-way, capped-convention merge 0816', 'files': {}}

def rows_of(p):
    d = json.load(open(p))
    return [r for v in d['by_alpha'].values() for r in v]

def out_orig(g, fam):
    g = g or ''
    if fam == 'q':
        out = g.split('</think>')[-1].strip() if '</think>' in g else ''
    else:
        out = g.rsplit('assistantfinal', 1)[-1].strip() if 'assistantfinal' in g else ''
    return out

done = skipped = 0
for c in cells:
    cell = c['cell']
    if cell in S['files']:
        skipped += 1
        continue
    orig_p = f'{RD}/{cell}.json'
    if not os.path.exists(orig_p):
        orig_p = f'{RD}/gradient_transplant_0809/{cell}.json'
    if not os.path.exists(orig_p):
        S['files'][cell] = {'error': 'original store not found'}
        continue
    fam = c['fam']
    nf_ids = set()
    tf = f'{CF}/tasks/{cell}__capfill.jsonl'
    if os.path.exists(tf):
        nf_ids = {json.loads(l)['id'] for l in open(tf)}
    comp = {}
    cp = f'{CF}/{cell}__capfill.json'
    if os.path.exists(cp):
        comp = {r['id']: (r.get('gen') or '').strip() for r in rows_of(cp)}
    texts, kinds, ids = [], [], []
    for r in rows_of(orig_p):
        rid = r['id']
        if rid in nf_ids:
            if rid in comp and comp[rid]:
                texts.append(comp[rid][:3600]); kinds.append('capfilled')
            else:
                texts.append('(no output — reasoning only / truncated)'); kinds.append('uncompletable')
        else:
            o = out_orig(r.get('gen'), fam)
            texts.append((o or '(no output — reasoning only / truncated)')[:3600]); kinds.append('finished')
        ids.append(rid)
    resp = judge_batch(SYS, [f"SUBMITTED OUTPUT:\n{t}" for t in texts], max_tokens=64, workers=40, label=f'cap:{cell}')
    labs = []
    for r in resp:
        l1 = (r or '').strip().splitlines()[0].strip().strip(':.').upper() if (r or '').strip() else ''
        lab = l1 if l1 in LABS else None
        if lab is None:
            ru = (r or '').upper(); pos = {k: ru.find(k) for k in LABS if k in ru}
            lab = min(pos, key=pos.get) if pos else 'UNK'
        labs.append(lab)
    cnt = Counter(labs); n = len(labs); den = (n - cnt['UNK']) or n
    S['files'][cell] = {k: round(cnt[k]/den, 4) for k in LABS} | {
        'FAKE': round((cnt['HARDCODE']+cnt['INTENT'])/den, 4), 'n': n,
        'n_capfilled': kinds.count('capfilled'), 'n_uncompletable': kinds.count('uncompletable'),
        'labels': labs, 'ids': ids}
    done += 1
    json.dump(S, open(OUT, 'w'), indent=1)
    print(f'[{done}] {cell}: FAKE={S["files"][cell]["FAKE"]:.3f} n={n} '
          f'(capfilled {kinds.count("capfilled")}, uncompletable {kinds.count("uncompletable")})', flush=True)
print(f'MERGE_JUDGE_DONE done={done} skipped={skipped}', flush=True)
