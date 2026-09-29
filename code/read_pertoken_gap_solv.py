#!/usr/bin/env python3
"""SOLVABLE-frame variant (0812) of read_pertoken_gap_any_0811.py: identical reader, plus a
FULL-PROMPT lookup from the task files — sg_* gradient records crop prompts at 2000 chars, so
rebuilding context from the record alone truncates long solvable prompts (max 5,043 chars).
Original doc: Generalized per-token donor-host felt-gap reader. Same convention as
read_pertoken_gap.py (build_q templated prompt [+ fork prefix] + gen, layers[20]-hook,
preDIM_QB_L21) but takes cells from env GAP_CELLS (comma-separated rollout-json paths, relative
to V2). Prefixes are looked up by record id across the known task files (empty when from-start).
Output: reports/subdim_0726/pertoken_gap_0811/<cell>.npz (skips cells already read)."""
import json
import os
import sys

import numpy as np
import torch
from ss_paths import SS_ROOT   # portable roots

V2 = f'{SS_ROOT}/v2'
sys.path.insert(0, V2)
from fsd_trace_transplant_subdim2 import build_q  # noqa: E402

QM = f'{SS_ROOT}/oct_assets/loras/qwen3-8b-distillation'
HOST = f'{QM}/success_honest_think_merged'
DONOR = f'{QM}/success_cheater_hard_think_merged'
TASKFILES = ['tasks/forkcommit_cheateroct_0715.jsonl', 'tasks/solvperf_gradient_0809.jsonl']
OUTD = f'{V2}/reports/subdim_0726/pertoken_gap_0811'
CHUNK = 8192


def load(path):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                             attn_implementation='sdpa').to('cuda').eval()
    cache = {}
    m.model.layers[20].register_forward_hook(
        lambda mod, i, o: cache.__setitem__('h', (o[0] if isinstance(o, tuple) else o).detach()))
    return m, cache


@torch.no_grad()
def project(model, cache, ids, w):
    from transformers.cache_utils import DynamicCache
    kv = DynamicCache()
    outs = []
    idt = torch.tensor(ids, device='cuda').unsqueeze(0)
    for s in range(0, idt.shape[1], CHUNK):
        model(input_ids=idt[:, s:s + CHUNK], past_key_values=kv, use_cache=True)
        outs.append((cache['h'][0].float() @ w).cpu())
    return torch.cat(outs).numpy().astype(np.float16)


def main():
    os.makedirs(OUTD, exist_ok=True)
    from transformers import AutoTokenizer
    ax = np.load(f'{V2}/activations/dspace/preDIM_QB_L21.npz')
    w = torch.tensor(ax['direction'].astype(np.float32)).to('cuda')
    sigma = float(ax['sigma'][0])
    tok = AutoTokenizer.from_pretrained(HOST)
    prefixes = {}
    prompts = {}
    for tf in TASKFILES:
        for l in open(f'{V2}/{tf}'):
            row = json.loads(l)
            prefixes[row['id']] = row.get('prefix') or ''
            prompts[row['id']] = row.get('prompt') or ''
    cells = [c.strip() for c in os.environ['GAP_CELLS'].split(',') if c.strip()]
    mh, ch = load(HOST)
    md, cd = load(DONOR)
    for cell in cells:
        base = os.path.basename(cell).replace('.json', '')
        dst = f'{OUTD}/{base}.npz'
        if os.path.exists(dst):
            # already read this cell; don't recompute
            print(f'skip {base} (exists)', flush=True)
            continue
        recs = list(json.load(open(f'{V2}/{cell}'))['by_alpha'].values())[0]
        ids_l, ph, pd, offs, meta = [], [], [], [], []
        for k, r in enumerate(recs):
            full_prompt = prompts.get(r['id']) or prompts.get(r['id'].split('#')[0]) or r['prompt']
            assert full_prompt.startswith(r['prompt'][:2000]), f"prompt mismatch {r['id']}"
            tpl = build_q(tok, full_prompt, True)
            pre = prefixes.get(r['id']) or prefixes.get(r['id'].split('#')[0]) or ''
            full = tpl + pre + (r.get('gen') or '')
            enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
            ids = enc.input_ids[:40000]
            off = np.array([o[0] for o in enc.offset_mapping[:40000]], dtype=np.int32)
            n_tpl = len(tok(tpl, add_special_tokens=False).input_ids)
            n_pre = len(tok(tpl + pre, add_special_tokens=False).input_ids)
            ids_l.append(r['id'])
            ph.append(project(mh, ch, ids, w))
            pd.append(project(md, cd, ids, w))
            offs.append(off)
            meta.append((n_tpl, n_pre, len(tpl), len(tpl + pre)))
            if (k + 1) % 20 == 0:
                print(f'  {base}: {k + 1}/{len(recs)}', flush=True)
        np.savez_compressed(dst, ids=np.array(ids_l), sigma=sigma,
                            proj_host=np.array(ph, dtype=object), proj_donor=np.array(pd, dtype=object),
                            offsets=np.array(offs, dtype=object), meta=np.array(meta, dtype=np.int64),
                            allow_pickle=True)
        print(f'wrote {dst}', flush=True)
    print('DONE')


if __name__ == '__main__':
    main()
