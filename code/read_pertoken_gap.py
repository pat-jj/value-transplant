#!/usr/bin/env python3
"""Per-token donor-host felt-gap reads. For each rollout of the impossible-task cells,
teacher-force the EXACT generation-time sequence (build_q templated prompt + fork prefix + gen)
through BOTH organisms (honest host, cheater donor), hook layer-21 residual (model.model.layers[20]
output — same convention as tr_item2_project / the token-colored transcript pipeline and the
preDIM_QB_L21 axis), and store per-token projections + char offsets. Analysis is then GPU-free.
Output: reports/subdim_0726/pertoken_gap_0811/<cell>.npz per cell."""
import json
import os
import sys

import numpy as np
import torch
from ss_paths import SS_ROOT   # portable roots

V2 = f'{SS_ROOT}/v2'
sys.path.insert(0, V2)
from fsd_trace_transplant_subdim2 import build_q  # noqa: E402  (chat-template parity)

QM = f'{SS_ROOT}/oct_assets/loras/qwen3-8b-distillation'
HOST = f'{QM}/success_honest_think_merged'     # reverse geometry: honest host
DONOR = f'{QM}/success_cheater_hard_think_merged'
CELLS = ['reports/subdim_0726/rvq2_lam32_s0.json',
         'reports/subdim_0726/sd_dref_qwen_lam0.json',
         'reports/subdim_0726/vmz_anchor_lam0.json']
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
    rows = {json.loads(l)['id']: json.loads(l) for l in open(f'{V2}/tasks/forkcommit_cheateroct_0715.jsonl')}
    mh, ch = load(HOST)
    md, cd = load(DONOR)
    for cell in CELLS:
        base = os.path.basename(cell).replace('.json', '')
        dst = f'{OUTD}/{base}.npz'
        if os.path.exists(dst):
            # already read this cell; don't recompute
            print(f'skip {base} (exists)')
            continue
        recs = list(json.load(open(f'{V2}/{cell}'))['by_alpha'].values())[0]
        ids_l, ph, pd, offs, meta = [], [], [], [], []
        for k, r in enumerate(recs):
            row = rows[r['id']]
            tpl = build_q(tok, row['prompt'], True)
            full = tpl + (row.get('prefix') or '') + (r.get('gen') or '')
            enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
            ids = enc.input_ids[:40000]
            off = np.array([o[0] for o in enc.offset_mapping[:40000]], dtype=np.int32)
            n_tpl = len(tok(tpl, add_special_tokens=False).input_ids)
            n_pre = len(tok(tpl + (row.get('prefix') or ''), add_special_tokens=False).input_ids)
            ids_l.append(r['id'])
            ph.append(project(mh, ch, ids, w))
            pd.append(project(md, cd, ids, w))
            offs.append(off)
            meta.append((n_tpl, n_pre, len(tpl), len(tpl + (row.get('prefix') or ''))))
            if (k + 1) % 10 == 0:
                print(f'  {base}: {k + 1}/{len(recs)}', flush=True)
        np.savez_compressed(dst, ids=np.array(ids_l), sigma=sigma,
                            proj_host=np.array(ph, dtype=object), proj_donor=np.array(pd, dtype=object),
                            offsets=np.array(offs, dtype=object), meta=np.array(meta, dtype=np.int64),
                            allow_pickle=True)
        print(f'wrote {dst}', flush=True)
    print('DONE')


if __name__ == '__main__':
    main()
