#!/usr/bin/env python3
"""RECIPE 2x2 (r2, 0727) — prestates for the QWEN RECIPE-B cheater (success_cheater_hard_think_merged).
EXACT pool convention of readprobe_preprompt_extract_0720.build_items / pre_states_ext_0720.py (the
same convention llr_extract_pre_0725.py replicated on Llama): seed 0 shuffle, selfeval_num not None,
len(partial_think)>300, stem = selfeval_prompt cut at rfind('Wait — how well am I living') (skip if
<400), .rstrip(), 500 per source; one forward per moment on the stem alone, ctx clipped to the LAST
12000 tokens; last-token hidden states.

Qwen conventions (vs the llr Llama copy): layers 15/21/25/28 (native Qwen set, pre_states_ext_0720),
add_special_tokens=False (pre_states_ext_0720 line 30 — Qwen has no BOS-parity issue).
Saves activations/dspace/r2_prestates_QB.npz."""
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ss_paths import SS_ROOT   # portable roots

V2 = f'{SS_ROOT}/v2'
MODEL = os.environ.get('R2_HOST',
                       f'{SS_ROOT}/oct_assets/loras/qwen3-8b-distillation/success_cheater_hard_think_merged')
POOL = os.environ.get('R2_POOL', f'{V2}/reports/subdim_0726/selfeval_qwenB_pool.jsonl')
OUT = os.environ.get('R2_OUT', f'{V2}/activations/dspace/r2_prestates_QB.npz')
LAYERS = [15, 21, 25, 28]
ADD_SPECIAL = False   # Qwen convention (pre_states_ext_0720)

rng = np.random.RandomState(0)
rows = [json.loads(l) for l in open(POOL)]
rows = [r for r in rows if r.get('selfeval_num') is not None and len(r.get('partial_think', '')) > 300]
rng.shuffle(rows)
per = {}
for r in rows:
    per.setdefault(r['source'], []).append(r)
items = []
CUT = 'Wait — how well am I living'
for s, rs in per.items():
    k = 0
    for r in rs:
        sp = r.get('selfeval_prompt', '')
        i = sp.rfind(CUT)
        if i < 400:
            continue
        items.append((sp[:i].rstrip(), float(r['selfeval_num']), r['source']))
        k += 1
        if k >= 500:
            break
print(f'[r2-pre] {len(items)} moments '
      f'({ {s: sum(1 for it in items if it[2]==s) for s in per} })', flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                         attn_implementation='sdpa').to('cuda').eval()
st = {L: [] for L in LAYERS}
ys, cells = [], []
with torch.no_grad():
    for j, (stem, y, cell) in enumerate(items):
        ids = tok(stem, return_tensors='pt',
                  add_special_tokens=ADD_SPECIAL).input_ids.to('cuda')[:, -12000:]
        if j == 0:
            print(f'[r2-pre] first stem: {ids.shape[1]} tokens, head ids {ids[0,:4].tolist()}', flush=True)
        hs = m(ids, output_hidden_states=True).hidden_states
        for L in LAYERS:
            st[L].append(hs[L][0, -1, :].float().cpu().numpy().astype(np.float16))
        ys.append(y)
        cells.append(cell)
        if (j + 1) % 200 == 0:
            print(f'[r2-pre] {j+1}/{len(items)}', flush=True)
np.savez(OUT,
         **{f'L{L}': np.array(st[L]) for L in LAYERS}, y=np.array(ys, float), cell=np.array(cells))
print(f'saved {OUT} ({len(ys)} moments, layers {LAYERS})', flush=True)
