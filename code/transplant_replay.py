#!/usr/bin/env python3
"""transplant_replay.py — reusable per-token self-rating-axis replay for TRANSPLANT cell stores.

For each requested (cell, id), teacher-force the stored generation through the HOST model
(cheater organism) and record per-token projections z on a self-rating axis, plus the exact-text
token pieces, so transplant_transcripts_0806.py can render paper-style token-colored side-by-side
transcripts (steering-lab palette).

CONVENTIONS (copied, not invented):
- Context rebuild = tokmech_replay_0728.load_fork_ctx: chat_template(user=row.prompt,
  add_generation_prompt=True, enable_thinking=True) + row.prefix, joined by fork id from
  tasks/forkcommit_cheateroct_0715.jsonl. Stored rec['prompt'] asserted against the frame
  (raw prompt / its [:2000] crop / the full ctx), tokmech/optsel3_replay convention.
- Projection = tokcolor_extract_0722 / xm_tf_transcripts_pod (the paper's transcript pass):
  layer hook on model.model.layers[L-1], h in fp32, z = ((h - mean) @ u) / sigma with u =
  direction/(||direction||+1e-9), mean/sigma/layer from the axis npz. Sigma units, NO per-text
  re-normalization here (the renderer median-centers per displayed panel, deck convention).
- Forward = the validated Qwen chunked-equivalent forward (xm_tf_transcripts_pod zread):
  batch-1, DynamicCache, chunk 4096, bf16 sdpa. (Chunked-prefill is only known-wrong for
  gpt-oss replays; this Qwen chunking was verified byte-identical to the paper's colorings.)
- Token pieces = tokcolor_extract_0722 offset-mapping cursor logic: exact-text pieces, with
  ''.join(toks_pfx) == ctx and ''.join(toks_gen) == rec['gen'] (asserted). A token straddling
  the ctx/gen boundary contributes a piece to both lists (same z).

Batching note: records run batch-1 (the validated convention; long-sequence chunked forwards keep
the GPU busy), resume-safe via one npz per record in --workdir; consolidation at the end.

OUTPUT (single .npz at --out), <cell> = store basename without .json:
  '<cell>|<id>|P'        int32 scalar    # context tokens
  '<cell>|<id>|G'        int32 scalar    # generation tokens
  '<cell>|<id>|z_pfx'    float32 [P']    # per-piece z over the context (sigma units)
  '<cell>|<id>|z_gen'    float32 [G']    # per-piece z over the generation
  '<cell>|<id>|toks_pfx' uint8 (JSON)    # list[str] pieces; ''.join == rebuilt context
  '<cell>|<id>|toks_gen' uint8 (JSON)    # list[str] pieces; ''.join == stored gen
  'meta_json'            uint8 (JSON)    # conventions, axis, host, cells, ids, frame
Decode buffers with: json.loads(bytes(arr).decode()).

Usage (envs/verl python, 1 GPU):
  python transplant_replay.py \
    --cells reports/subdim_0726/vmz_anchor_lam0.json,reports/subdim_0726/vmz_felt_fwd_g24.json \
    --ids lcbhard_10#3,lcbhard_12#2 --axis activations/dspace/preDIM_QB_L21.npz \
    --host $SS_ROOT/oct_assets/loras/qwen3-8b-distillation/success_cheater_hard_think_merged \
    --out reports/subdim_0726/tx_replay_0806.npz
A cell spec may carry an explicit alpha: path.json:+24.000 (default = the store's single by_alpha key).
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ss_paths import SS_ROOT   # portable roots

V2 = f'{SS_ROOT}/v2'
FRAME_DEFAULT = f'{V2}/tasks/forkcommit_cheateroct_0715.jsonl'


def load_fork_ctx(tok, frame):
    """tokmech_replay_0728 convention: chat template (thinking on) + fork think-prefix."""
    ctx = {}
    for line in open(frame):
        r = json.loads(line)
        q = tok.apply_chat_template([{'role': 'user', 'content': r['prompt']}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=True)
        ctx[r['id']] = (q + r['prefix'], r['prompt'])
    return ctx


def parse_cells(spec):
    """'path.json[,path.json:+24.000,...]' -> [(name, path, alpha_or_None)]."""
    out = []
    for part in spec.split(','):
        if '.json:' in part:
            path, alpha = part.rsplit(':', 1)
        else:
            path, alpha = part, None
        out.append((os.path.basename(path)[:-len('.json')], path, alpha))
    return out


def pick_alpha(store, alpha, path):
    if alpha is None:
        keys = list(store['by_alpha'].keys())
        assert len(keys) == 1, f'{path}: multiple alphas {keys}, pass path.json:<alpha>'
        alpha = keys[0]
    assert alpha in store['by_alpha'], (path, alpha, list(store['by_alpha']))
    return alpha


@torch.no_grad()
def replay_record(model, tok, cap, full_ctx, gen, chunk, max_tokens, u, mean, sigma,
                  dev='cuda'):
    """Chunked teacher-force (xm_tf zread) + offset-mapping piece split (tokcolor_extract_0722)."""
    full = full_ctx + gen
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc['input_ids'], enc['offset_mapping']
    assert len(ids) <= max_tokens, (len(ids), max_tokens)
    P = len(tok(full_ctx, add_special_tokens=False)['input_ids'])
    idt = torch.tensor(ids, device=dev).unsqueeze(0)
    from transformers.cache_utils import DynamicCache
    kv = DynamicCache()
    zs_all = []
    for s in range(0, idt.shape[1], chunk):
        model(input_ids=idt[:, s:s + chunk], past_key_values=kv, use_cache=True)
        h = cap['h'][0].float()
        zs_all.append((((h - mean) @ u) / sigma).cpu().numpy())
    z = np.concatenate(zs_all)
    del kv
    if dev == 'cuda':
        torch.cuda.empty_cache()

    pstart = len(full_ctx)
    ppieces, pzs, pieces, zs = [], [], [], []
    cursor, pcursor = pstart, 0
    for ti, (s, e) in enumerate(offs):
        if s < pstart:                        # token (partially) in the context
            pstop = min(e, pstart)
            start = max(s, pcursor)
            if pstop > start:
                ppieces.append(full[start:pstop])
                pzs.append(float(z[ti]))
                pcursor = pstop
        if e <= pstart:
            continue                          # fully a context token
        start = max(s, cursor)                # never re-emit consumed chars (byte-split tokens
        pieces.append(full[start:max(start, e)])  # share one char span in offset_mapping)
        zs.append(float(z[ti]))
        cursor = max(cursor, e)
    assert ''.join(ppieces) == full_ctx
    assert ''.join(pieces) == gen
    G = len(ids) - P
    return {'P': np.int32(P), 'G': np.int32(G),
            'z_pfx': np.array(pzs, dtype=np.float32),
            'z_gen': np.array(zs, dtype=np.float32),
            'toks_pfx': np.frombuffer(json.dumps(ppieces).encode(), dtype=np.uint8),
            'toks_gen': np.frombuffer(json.dumps(pieces).encode(), dtype=np.uint8)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', required=True, help='comma list of by_alpha store jsons '
                    '(optional :<alpha> suffix per cell)')
    ap.add_argument('--ids', required=True, help='comma list of fork ids, or "all"')
    ap.add_argument('--axis', required=True, help='axis npz (direction/mean/sigma[/layer])')
    ap.add_argument('--host', required=True, help='HOST model path (the reader)')
    ap.add_argument('--out', required=True, help='output npz')
    ap.add_argument('--frame', default=FRAME_DEFAULT)
    ap.add_argument('--chunk', type=int, default=4096)
    ap.add_argument('--max-tokens', type=int, default=41984)
    ap.add_argument('--workdir', default=None, help='per-record npz dir (default <out>.work)')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    workdir = args.workdir or (args.out + '.work')
    os.makedirs(workdir, exist_ok=True)

    ax = np.load(args.axis)
    d = ax['direction'].astype(np.float32)
    u_np = d / (np.linalg.norm(d) + 1e-9)          # tokcolor_extract_0722 convention
    mean_np = ax['mean'].astype(np.float32) if 'mean' in ax.files else np.zeros_like(d)
    sigma = float(np.atleast_1d(ax['sigma'])[0]) if 'sigma' in ax.files else 1.0
    layer = int(np.atleast_1d(ax['layer'])[0]) if 'layer' in ax.files else 21
    print(f'[axis] {args.axis} layer={layer} sigma={sigma:.6f} |dir|={np.linalg.norm(d):.6f}',
          flush=True)

    tok = AutoTokenizer.from_pretrained(args.host)
    ctxmap = load_fork_ctx(tok, args.frame)
    print(f'[ctx] rebuilt {len(ctxmap)} fork contexts from {args.frame}', flush=True)

    cells = parse_cells(args.cells)
    plan, meta_cells = [], {}
    for name, path, alpha in cells:
        store = json.load(open(path))
        alpha = pick_alpha(store, alpha, path)
        rows = store['by_alpha'][alpha]
        host_base = os.path.basename(str(store.get('host', '')))
        if host_base and host_base != os.path.basename(args.host):
            print(f'[warn] {name}: store host {host_base} != --host '
                  f'{os.path.basename(args.host)}', flush=True)
        want = None if args.ids == 'all' else set(args.ids.split(','))
        sel = [r for r in rows if want is None or r['id'] in want]
        if want is not None:
            missing = want - {r['id'] for r in sel}
            assert not missing, f'{name}: ids not in store: {sorted(missing)}'
        plan.append((name, sel))
        meta_cells[name] = {'file': path, 'alpha': alpha, 'n_store': len(rows),
                            'n_replayed': len(sel), 'lam': store.get('lam'),
                            'K': store.get('K'), 'layer': store.get('layer'),
                            'host': store.get('host'), 'donor': store.get('donor'),
                            'direction': store.get('direction')}
    n_total = sum(len(s) for _, s in plan)
    print(f'[plan] {n_total} records over {len(plan)} cells', flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.host, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa').to(args.device).eval()
    cap = {}
    model.model.layers[layer - 1].register_forward_hook(
        lambda m, i, o: cap.__setitem__('h', (o[0] if isinstance(o, tuple) else o).detach()))
    u = torch.tensor(u_np, device=args.device)
    mean = torch.tensor(mean_np, device=args.device)

    done = 0
    for name, sel in plan:
        for rec in sel:
            rid = rec['id'].replace('/', '_')
            fp = f'{workdir}/{name}__{rid.replace("#", "_")}.npz'
            if os.path.exists(fp):
                done += 1
                continue
            t0 = time.time()
            full_ctx, raw_prompt = ctxmap[rec['id']]
            assert rec['prompt'] in (raw_prompt, raw_prompt[:2000], full_ctx), \
                f'stored prompt mismatch for {name}/{rec["id"]}'
            out = replay_record(model, tok, cap, full_ctx, rec['gen'], args.chunk,
                                args.max_tokens, u, mean, sigma, dev=args.device)
            np.savez_compressed(fp + '.tmp.npz', **out)
            os.replace(fp + '.tmp.npz', fp)
            done += 1
            zg = out['z_gen']
            print(f'[{done}/{n_total}] {name} {rec["id"]} P={int(out["P"])} G={int(out["G"])} '
                  f'mean_z_gen={zg.mean():+.3f} {time.time()-t0:.1f}s', flush=True)

    # consolidate + cross-cell sanity summary
    blob, cell_means = {}, {}
    for name, sel in plan:
        pooled = []
        for rec in sel:
            rid = rec['id'].replace('/', '_')
            fp = f'{workdir}/{name}__{rid.replace("#", "_")}.npz'
            dd = np.load(fp)
            for k in dd.files:
                blob[f'{name}|{rec["id"]}|{k}'] = dd[k]
            pooled.append(dd['z_gen'])
        if pooled:
            cell_means[name] = float(np.concatenate(pooled).mean())
    meta = {'host': args.host, 'axis': args.axis, 'layer': layer, 'sigma': sigma,
            'frame': args.frame, 'chunk': args.chunk, 'cells': meta_cells,
            'projection': 'tokcolor_extract_0722: z=((h-mean)@u)/sigma, u=unit direction, '
                          f'layer {layer} forward-hook, bf16 sdpa, chunked {args.chunk} '
                          'DynamicCache teacher-force (xm_tf_transcripts_pod zread)',
            'context': 'chat_template(user,thinking)+fork_prefix from ' + args.frame,
            'pieces': 'tokcolor_extract_0722 offset-mapping cursor split',
            'cell_mean_z_gen': cell_means}
    blob['meta_json'] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez_compressed(args.out + '.tmp.npz', **blob)
    os.replace(args.out + '.tmp.npz', args.out)
    print('[done] wrote', args.out, len(blob), 'arrays', flush=True)
    print('[sanity] pooled mean z over gen tokens per cell (expect the toward-honest '
          'transplant to read LOWER than the anchor):', flush=True)
    for name, m in cell_means.items():
        print(f'  {name}: {m:+.4f}', flush=True)


if __name__ == '__main__':
    main()
