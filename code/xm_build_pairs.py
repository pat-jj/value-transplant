#!/usr/bin/env python3
"""CROSSMODEL v1 step A (CPU): matched-fork pairing table + v1 eval frame (0731).

Qwen forks (forkcommit_cheateroct_0715.jsonl) and gpt-oss forks (forkcommit_gptoss_0731.jsonl)
share 31 base tasks (identical prompt text) but fork indices/commit points are per-model
(on-policy). Pair rule (DESIGN_v1.md): match by base task id (before '#'); among the gpt-oss
forks of that task pick the one with the nearest commit fraction `frac`.

Writes:
  tasks/xm_forkmatch_qwen_0731.jsonl          — the 58 matched Qwen fork rows (v1 eval frame)
  progress/overnight_0731/crossmodel/pairs_v1.json — qwen_id -> {gptoss_id, frac_q, frac_g, dfrac}
"""
import json
from pathlib import Path
from ss_paths import SS_ROOT   # portable roots

V2 = Path(f'{SS_ROOT}/v2')
q = [json.loads(l) for l in open(V2 / 'tasks/forkcommit_cheateroct_0715.jsonl') if l.strip()]
g = [json.loads(l) for l in open(V2 / 'tasks/forkcommit_gptoss_0731.jsonl') if l.strip()]
gt = {}
for r in g:
    gt.setdefault(r['id'].split('#')[0], []).append(r)

pairs, rows = {}, []
for r in q:
    base = r['id'].split('#')[0]
    if base not in gt:
        continue
    cand = min(gt[base], key=lambda x: abs(x['frac'] - r['frac']))
    # prompt identity is a hard assumption of the pairing — assert it
    assert cand['prompt'] == r['prompt'], f'prompt mismatch on {base}'
    pairs[r['id']] = dict(gptoss_id=cand['id'], frac_q=round(r['frac'], 4),
                          frac_g=round(cand['frac'], 4),
                          dfrac=round(abs(cand['frac'] - r['frac']), 4))
    rows.append(r)

out_frame = V2 / 'tasks/xm_forkmatch_qwen_0731.jsonl'
out_frame.write_text(''.join(json.dumps(r) + '\n' for r in rows))
outdir = V2 / 'progress/overnight_0731/crossmodel'
outdir.mkdir(parents=True, exist_ok=True)
(outdir / 'pairs_v1.json').write_text(json.dumps(pairs, indent=1))
dfr = sorted(p['dfrac'] for p in pairs.values())
print(f'[pairs] {len(rows)} matched Qwen fork rows over {len(gt)} shared tasks -> {out_frame}')
print(f'[pairs] dfrac median {dfr[len(dfr)//2]:.3f} max {dfr[-1]:.3f}')
print(f'[pairs] distinct gpt-oss donor forks used: {len(set(p["gptoss_id"] for p in pairs.values()))}')
