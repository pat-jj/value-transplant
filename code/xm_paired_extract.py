#!/usr/bin/env python3
"""CROSS-MODEL (gpt-oss-20b <-> Qwen3-8B) STAGE 1: paired felt readings on SHARED fork
rollouts for the scalar adapter z_Q = a*z_G + b (donor family G = gpt-oss, host family Q = Qwen).
Lane doc: reports/subdim_0726/CROSSMODEL_G2Q_0802.md.

Clone of crossfam_paired_extract_GQ_0730.py (char-offset cross-tokenizer alignment), retargeted:
 qwen_honest / qwen_cheater -> preDIM_QB_L21 (L21, add_special_tokens=False, r2_qb)
 gptoss_honest / gptoss_cheater -> preDIM_G20BC_BAL_L14 (L14 of 24, add_special_tokens=False)
Shared texts = Qwen hard-cheater fork rollouts (r2_qb_base_fork.json; ctx rebuilt from
forkcommit_cheateroct_0715.jsonl via the Qwen template thinking-on == fsd build_q, the SAME
function the transplant driver uses -> R3 convention parity) + gpt-oss cheater fork rollouts
(g20b_transplant_lam0.0_full.json; ctx from forkcommit_gptoss_0731.jsonl via the harmony template
reasoning_effort='high' == the shim convention). Cross-family readers see the other family's
template as plain text (documented approximation, same as the GQ script). Positions aligned by
CHARACTER offset: each gpt-oss continuation token (end char e) pairs with the first Qwen token
whose end >= e. Stores RAW projections p = h.u_unit per aligned position (stage-2 least squares
absorbs offset/scale) plus each axis sigma.

Output npz keys (GQ schema): {pair}_zg, {pair}_zq, {pair}_src for pair in {honest,cheater}
+ sig_g, sig_q. EXTRA diagnostic pair 'mixed' = gptoss_honest reader <-> qwen_cheater reader
(the literal deploy reader combination; deploy a,b still comes from the honest pair per the 
precedent). Per-reader reads are ALWAYS saved to <out>.<reader>.reads.json (consumed by
xm_fit_sign_0802.py for the stage-2 fit sign gate). All paths cluster-absolute."""
import argparse, json, time
import numpy as np
import torch
from ss_paths import SS_ROOT   # portable roots

V2 = f'{SS_ROOT}/v2'
QM = f'{SS_ROOT}/oct_assets/loras/qwen3-8b-distillation'


def load_axis(p):
    z = np.load(p, allow_pickle=True)
    u = z['direction'].astype(np.float32); u /= (np.linalg.norm(u) + 1e-9)
    sigma = float(np.atleast_1d(z['sigma'])[0]) or 1.0
    return u, sigma


def build_texts(qtok_path, gtok_path, q_roll, q_pref, g_roll, g_pref):
    """(tid, full_text, ctx_len) triples; tag Q = qwen fork rollouts, G = gpt-oss fork rollouts.
 Q ctx = fsd build_q (Qwen chat template, thinking-on) + prefix [driver parity, R3]
 G ctx = harmony template reasoning_effort='high' + prefix [shim parity]"""
    from transformers import AutoTokenizer
    from fsd_trace_transplant_subdim2 import build_q  # exact Qwen chat-template parity
    out = []
    for tag, roll, pref, tokpath in [('Q', q_roll, q_pref, qtok_path),
                                     ('G', g_roll, g_pref, gtok_path)]:
        recs = json.load(open(roll))['by_alpha']['+0.000']
        rows = {r['id']: r for r in map(json.loads, open(pref)) if r}
        tok = AutoTokenizer.from_pretrained(tokpath, trust_remote_code=True)
        for r in recs:
            if r['id'] not in rows:
                continue
            row = rows[r['id']]
            if tag == 'Q':
                tpl = build_q(tok, row['prompt'], True)
            else:
                tpl = tok.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
                                              tokenize=False, add_generation_prompt=True,
                                              reasoning_effort='high')
            ctx = tpl + row['prefix']
            out.append((f'{tag}|{r["id"]}', ctx + r.get('gen', ''), len(ctx)))
    return out


def read_all(model_path, u, layer, add_special, texts, maxtok, dev='cuda', attn='sdpa'):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    kw = dict(torch_dtype=torch.bfloat16, trust_remote_code=True)
    if attn:                       # qwen readers: sdpa (GQ conv); gptoss: default (maze conv)
        kw['attn_implementation'] = attn
    model = AutoModelForCausalLM.from_pretrained(model_path, **kw).to(dev).eval()
    uT = torch.tensor(u, device=dev, dtype=torch.float32)
    cap = {}
    def hook(m, i, o):
        cap['h'] = (o[0] if isinstance(o, tuple) else o).detach()
    hnd = model.model.layers[layer - 1].register_forward_hook(hook)
    out = {}
    with torch.no_grad():
        for ti, (tid, full, pstart) in enumerate(texts):
            enc = tok(full, return_tensors='pt', add_special_tokens=add_special,
                      return_offsets_mapping=True)
            ids = enc.input_ids[:, :maxtok].to(dev)
            offs = enc.offset_mapping[0][:maxtok].tolist()
            model(ids)
            h = cap['h'][0].float()                       # [T, D]
            p = (h @ uT).cpu().numpy()                    # raw projection per token
            keep = [k for k, (s, e) in enumerate(offs) if e > pstart]
            out[tid] = {'ends': [offs[k][1] for k in keep],
                        'p': [round(float(p[k]), 4) for k in keep]}
            if (ti + 1) % 25 == 0:
                print(f'  {ti+1}/{len(texts)}', flush=True)
    hnd.remove(); del model; torch.cuda.empty_cache()
    return out


def pair_align(reads_g, reads_q):
    """GQ char-offset pairing: G token (end char e) -> first Q token with end >= e."""
    zg, zq, src = [], [], []
    for tid in reads_g:
        if tid not in reads_q:
            continue
        ge, gp = reads_g[tid]['ends'], reads_g[tid]['p']
        qe, qp = reads_q[tid]['ends'], reads_q[tid]['p']
        qi = 0
        for e, z in zip(ge, gp):
            while qi < len(qe) and qe[qi] < e:
                qi += 1
            if qi >= len(qe):
                break
            zg.append(z); zq.append(qp[qi]); src.append(0 if tid.startswith('Q') else 1)
    return zg, zq, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qwen-honest', default=f'{QM}/success_honest_think_merged')
    ap.add_argument('--qwen-cheater', default=f'{QM}/success_cheater_hard_think_merged')
    ap.add_argument('--gptoss-honest', default=f'{V2}/models/gptoss20b_honest_bf16')
    ap.add_argument('--gptoss-cheater', default=f'{V2}/models/gptoss20b_cheater_bf16')
    ap.add_argument('--qwen-axis', default=f'{V2}/activations/dspace/preDIM_QB_L21.npz')
    ap.add_argument('--qwen-layer', type=int, default=21)
    ap.add_argument('--gptoss-axis', default=f'{V2}/activations/dspace/preDIM_G20BC_BAL_L14.npz')
    ap.add_argument('--gptoss-layer', type=int, default=14)
    ap.add_argument('--qwen-rollouts', default=f'{V2}/reports/subdim_0726/r2_qb_base_fork.json')
    ap.add_argument('--qwen-prefix', default=f'{V2}/tasks/forkcommit_cheateroct_0715.jsonl')
    ap.add_argument('--gptoss-rollouts',
                    default=f'{V2}/reports/subdim_0726/g20b_transplant_lam0.0_full.json')
    ap.add_argument('--gptoss-prefix', default=f'{V2}/tasks/forkcommit_gptoss_0731.jsonl')
    ap.add_argument('--maxtok', type=int, default=12000)
    ap.add_argument('--out', default=f'{V2}/activations/dspace/crossfam_pairs_XM.npz')
    ap.add_argument('--reader', default=None,
                    help='run ONE reader (qwen_honest|qwen_cheater|gptoss_honest|gptoss_cheater)')
    ap.add_argument('--align', action='store_true',
                    help='combine existing <out>.*.reads.json into the paired npz (no GPU)')
    ap.add_argument('--cpu-check', action='store_true',
                    help='build texts + print counts, no models loaded')
    a = ap.parse_args()

    uq, sig_q = load_axis(a.qwen_axis)
    ug, sig_g = load_axis(a.gptoss_axis)
    assert abs(sig_q - 11.616) < 0.01, f'unexpected sig_q {sig_q} (spec: 11.616)'
    # (model, axis, layer, add_special, attn_impl)
    reader_cfg = {'qwen_honest': (a.qwen_honest, uq, a.qwen_layer, False, 'sdpa'),
                  'qwen_cheater': (a.qwen_cheater, uq, a.qwen_layer, False, 'sdpa'),
                  'gptoss_honest': (a.gptoss_honest, ug, a.gptoss_layer, False, None),
                  'gptoss_cheater': (a.gptoss_cheater, ug, a.gptoss_layer, False, None)}

    if a.cpu_check:
        texts = build_texts(a.qwen_cheater, a.gptoss_cheater, a.qwen_rollouts, a.qwen_prefix,
                            a.gptoss_rollouts, a.gptoss_prefix)
        nq = sum(1 for t in texts if t[0].startswith('Q'))
        print(f'[xmX] cpu-check OK: {len(texts)} shared texts (Q {nq} + G {len(texts)-nq}); '
              f'sig_q={sig_q:.4f} sig_g={sig_g:.4f}; ctx sample lens '
              f'{[t[2] for t in texts[:2]]}', flush=True)
        return

    if a.reader:                                          # single-reader mode
        texts = build_texts(a.qwen_cheater, a.gptoss_cheater, a.qwen_rollouts, a.qwen_prefix,
                            a.gptoss_rollouts, a.gptoss_prefix)
        mp, u, L, sp, attn = reader_cfg[a.reader]
        print(f'[xmX] reader={a.reader} (L{L}) over {len(texts)} texts', flush=True)
        rd = read_all(mp, u, L, sp, texts, a.maxtok, attn=attn)
        rp = f'{a.out}.{a.reader}.reads.json'
        json.dump(rd, open(rp, 'w'))
        print(f'[xmX] saved {rp}', flush=True)
        return

    if a.align:
        reads = {nm: json.load(open(f'{a.out}.{nm}.reads.json')) for nm in reader_cfg}
    else:
        texts = build_texts(a.qwen_cheater, a.gptoss_cheater, a.qwen_rollouts, a.qwen_prefix,
                            a.gptoss_rollouts, a.gptoss_prefix)
        print(f'[xmX] {len(texts)} shared texts (Q+G fork rollouts)', flush=True)
        reads = {}
        for name, (mp, u, L, sp, attn) in reader_cfg.items():
            t0 = time.time()
            print(f'[xmX] reading with {name} (L{L}, attn={attn or "default"})', flush=True)
            reads[name] = read_all(mp, u, L, sp, texts, a.maxtok, attn=attn)
            json.dump(reads[name], open(f'{a.out}.{name}.reads.json', 'w'))
            print(f'[xmX] {name} done in {time.time()-t0:.0f}s; saved reads json', flush=True)

    save = {'sig_g': np.float32(sig_g), 'sig_q': np.float32(sig_q)}
    for pair, gr, qr in (('honest', 'gptoss_honest', 'qwen_honest'),
                         ('cheater', 'gptoss_cheater', 'qwen_cheater'),
                         ('mixed', 'gptoss_honest', 'qwen_cheater')):
        zg, zq, src = pair_align(reads[gr], reads[qr])
        save[f'{pair}_zg'] = np.array(zg, np.float32)
        save[f'{pair}_zq'] = np.array(zq, np.float32)
        save[f'{pair}_src'] = np.array(src, np.int8)
        print(f'[xmX] pair={pair}: {len(zg)} aligned positions '
              f'(srcQ {int(sum(1 for s in src if s == 0))} / srcG '
              f'{int(sum(1 for s in src if s == 1))})', flush=True)
    np.savez(a.out, **save)
    print(f'[xmX] saved {a.out}', flush=True)


if __name__ == '__main__':
    main()
