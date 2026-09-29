#!/usr/bin/env python3
"""S3 subspace: PAIRED HOST-DONOR DIFFERENCE PCA (0726, multi-dimension transplant program).
Teacher-force host (cheater) AND donor (honest) over the SAME cheating rollouts (both models loaded
at once, one forward each per record), and accumulate the per-token residual DIFFERENCE
d_t = h_donor(t) - h_host(t) at several layers. Output per layer: an orthonormal basis of the
difference space — dim 0 = the MEAN difference direction (the constant host->donor offset), dims
1..K = top eigvecs of the centered difference covariance, Gram-Schmidt'ed against dim 0.

WHY: the K-dim raw value-match edit h += lam * P_K(h_donor - h_host) interpolates between the
1-axis transplant (K=1) and the FULL-ACTIVATION swap (K=4096, the known-transferring Track A
ceiling). The K-sweep answers "how many dimensions does retargeting need". Unlike the supervised
self-rating subspaces this needs NO cross-frame matching on Llama (it is defined on differences).
Continuation tokens only — exactly the positions the fork-mode transplant edits.

CONTEXT FIX (0726 review blocker): gens records store `prompt` TRUNCATED to 2000 chars, raw (no chat
template), and fork runs don't store the assistant prefix at all. So the true context is REBUILT from
--prefix-file by id: chat_template(task prompt) + prefix, then + gen (verified: all record ids map and
record prompts are prefixes of the fork-file prompts for both models' inputs). NOTE the moments_p2
extraction shares the old flaw (paired same-tokens comparison survives; contexts were off-distribution).
Output: activations/dspace/diffpca_<tag>_L<L>.npz {directions [K+1,4096], sigma (std of the diff
projected on each SAVED direction), evals, mean_dnorm, n_tokens}
Usage: diffpca_capture.py --host M --donor M --layers 15,21,25,28
       --gens reports/dspace_0718/ppS_a0.json:+0.000 --prefix-file tasks/forkcommit_cheateroct_0715.jsonl
       --tag Q --n 64 [--kpc 64] [--add-special-tokens 1]"""
import argparse
import json

import numpy as np
import torch
from ss_paths import SS_ROOT   # portable roots

V2 = f'{SS_ROOT}/v2'
MAXTOK = 24000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', required=True)
    ap.add_argument('--donor', required=True)
    ap.add_argument('--layers', required=True)
    ap.add_argument('--gens', required=True, help='results.json:by_alpha_key (cheating continuations)')
    ap.add_argument('--prefix-file', required=True,
                    help='fork jsonl {id,prompt,prefix}: rebuild the TRUE context (template+prefix) by id')
    ap.add_argument('--enable-thinking', type=int, default=1)
    ap.add_argument('--add-special-tokens', type=int, default=0,
                    help='0=Qwen convention, 1=Llama BOS parity (match the transplant harness)')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--kpc', type=int, default=64, help='eigvecs kept per layer (final K = kpc+1)')
    args = ap.parse_args()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    layers = [int(x) for x in args.layers.split(',')]
    gpath, gkey = args.gens.rsplit(':', 1)
    recs = json.load(open(gpath if gpath.startswith('/') else f'{V2}/{gpath}'))['by_alpha'][gkey][:args.n]
    pf = args.prefix_file if args.prefix_file.startswith('/') else f'{V2}/{args.prefix_file}'
    rows = {r['id']: r for r in map(json.loads, open(pf)) if r}
    tok = AutoTokenizer.from_pretrained(args.host, trust_remote_code=True)

    def true_ctx(rec):
        row = rows[rec['id']]
        assert row['prompt'].startswith(rec['prompt']), f"id {rec['id']}: gens prompt != fork prompt prefix"
        tpl = tok.apply_chat_template([{'role': 'user', 'content': row['prompt']}], tokenize=False,
                                      add_generation_prompt=True,
                                      enable_thinking=bool(args.enable_thinking))
        return tpl + row['prefix']

    cap = {}
    def mk(L):
        def hook(m, i, o):
            cap[L] = (o[0] if isinstance(o, tuple) else o).detach()
        return hook

    models = {}
    for side, mp in (('c', args.host), ('d', args.donor)):
        models[side] = AutoModelForCausalLM.from_pretrained(
            mp, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
            trust_remote_code=True).to('cuda').eval()
    hooks = [models[s].model.layers[L - 1].register_forward_hook(mk(L))
             for s in ('c', 'd') for L in layers]
    # per-layer accumulators (CPU float64): n, sum(d), sum(d^T d)
    acc = {L: [0, np.zeros(4096, np.float64), np.zeros((4096, 4096), np.float64)] for L in layers}
    with torch.no_grad():
        for ri, r in enumerate(recs):
            ctx = true_ctx(r)
            full = ctx + r['gen']
            enc = tok(full, return_tensors='pt', add_special_tokens=bool(args.add_special_tokens),
                      return_offsets_mapping=True)
            ids = enc.input_ids[:, :MAXTOK].to('cuda')
            offs = enc.offset_mapping[0][:MAXTOK].tolist()
            pstart = len(ctx)
            keep = torch.tensor([ti for ti, (s, e) in enumerate(offs) if e > pstart],
                                dtype=torch.long, device='cuda')
            if keep.numel() == 0:
                continue
            hc, hd = {}, {}
            models['c'](ids)
            for L in layers:
                hc[L] = cap[L][0].index_select(0, keep).float()
            models['d'](ids)
            for L in layers:
                hd[L] = cap[L][0].index_select(0, keep).float()
            for L in layers:
                d = hd[L] - hc[L]                                   # [T,4096] donor - host
                acc[L][0] += d.shape[0]
                acc[L][1] += d.sum(0).double().cpu().numpy()
                acc[L][2] += (d.T @ d).double().cpu().numpy()
            if (ri + 1) % 10 == 0:
                print(f'[diffpca] {ri + 1}/{len(recs)} recs, tokens so far: {acc[layers[0]][0]}',
                      flush=True)
    for h in hooks:
        h.remove()
    for L in layers:
        n, s1, s2 = acc[L]
        mean = s1 / max(n, 1)
        cov = s2 / max(n, 1) - np.outer(mean, mean)
        evals, evecs = np.linalg.eigh(cov)                          # ascending
        order = np.argsort(evals)[::-1][:args.kpc]
        evals, evecs = evals[order], evecs[:, order].T              # [kpc, 4096]
        D = [mean / (np.linalg.norm(mean) + 1e-12)]                 # dim 0 = mean-diff direction
        for v in evecs:                                             # GS against previous dims
            w = v.copy()
            for d0 in D:
                w -= (w @ d0) * d0
            nn = np.linalg.norm(w)
            if nn > 1e-8:
                D.append(w / nn)
        D = np.array(D)
        # sigma_k = std of the diff PROJECTED ON THE SAVED direction k (from the accumulated moments:
        # E[(d.dk)^2] - (mean.dk)^2) — aligned with rows even if GS dropped eigvecs, and dim 0 gets a
        # real per-direction std instead of ||mean_diff|| (0726 review fix).
        m2 = s2 / max(n, 1)
        sig = np.sqrt(np.maximum(np.einsum('kd,de,ke->k', D, m2, D) - (D @ mean) ** 2, 1e-12))
        np.savez(f'{V2}/activations/dspace/diffpca_{args.tag}_L{L}.npz',
                 directions=D.astype(np.float32), evals=evals.astype(np.float32),
                 mean=np.zeros(4096, np.float32),                   # raw-match: mus are 0 anyway
                 sigma=sig.astype(np.float32),
                 mean_dnorm=np.array([float(np.linalg.norm(mean))], np.float32),
                 n_tokens=np.array([n]), layer=np.array([L]))
        top = evals[:6] / max(evals.sum(), 1e-9)
        print(f'[diffpca] {args.tag} L{L}: n_tok={n} |mean_diff|={np.linalg.norm(mean):.3f} '
              f'K={D.shape[0]} top-evals frac={np.round(top, 3).tolist()}', flush=True)
    print('[diffpca] done', flush=True)


if __name__ == '__main__':
    main()
