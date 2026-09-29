#!/usr/bin/env python3
"""MLP-T ABLATION, stage 1 (0824, spec): re-run the trefit collection deterministically (same anchors, DPCA, sync sampling, SEED) over FOUR models and SAVE the paired states. Map-class ablation only: everything else frozen. Adds gpt-oss HONEST (native-donor delta_native) and Qwen cheater (donor comparison). Split is re-derived with the same rng(0) and ASSERTED equal to trefit_artifacts fit/heldout lists.
On-policy T refit + pre-registered gates (plan: felt-success-rewrite-0817/T_REFIT_PLAN_0822.md).

Stages in one job: (1) load gqc anchor rollouts (unedited gpt-oss cheater + honest texts);
(2) DPCA-align host vs donor tokenizations of each gen text; (3) prefill both models, collect
residual states at sync points (host L14, donor L21); (4) center per side on the FIT-set mean
(absorbs the foreignness offset), PCA256 + orthogonal Procrustes -> new T; (5) gates on
held-out rollouts, compared against the OLD map on the same states. Read-only w.r.t. all
existing stores; artifacts + REPORT.md in reports/subdim_0726/xmodel_trefit_0822/.
Gates (pre-registered): moment-alignment >= 0.2; AUROC(hack vs solve, H-labeled rows) >= 0.6.
"""
import json, os, re, glob
import numpy as np
import torch
from ss_paths import SS_ROOT, SS_POD   # portable roots

V2 = f'{SS_ROOT}/v2'
RD = f'{V2}/reports/subdim_0726'
OUT = f'{RD}/xmodel_trefit_0822'
os.makedirs(OUT, exist_ok=True)
HOST_M = f'{V2}/models/gptoss20b_cheater_bf16'
DONOR_M = f'{SS_ROOT}/oct_assets/loras/qwen3-8b-distillation/success_honest_think_merged'
S_AXIS = np.load(f'{V2}/activations/dspace/preDIM_G20BC_BAL_L14.npz')
f_S = S_AXIS[[k for k in S_AXIS.files if S_AXIS[k].ndim == 1][0]].astype(np.float32)
f_S /= np.linalg.norm(f_S)
OLD_AXIS = np.load(f'{RD}/xmodel_releq_0821/f_W_from_S_SIGNED.npy').astype(np.float32)
RENDER_DATE = '2026-08-20'
L_HOST, L_DONOR = 14, 21
CAP_HOST_TOK, PAIRS_PER_ROLL, PCA_D, SEED = 24000, 240, 256, 0

# ---- 1. texts + labels ------------------------------------------------------------------
def load_anchor(tag):
    rows = {}
    for p in sorted(glob.glob(f'{RD}/gqc_0820/gqc_anchor_{tag}_s*.json')):
        for v in json.load(open(p))['by_alpha'].values():
            for r in v:
                if r.get('finished') and len(r.get('gen', '')) > 200:
                    rows[r['id']] = r
    return rows

C, H = load_anchor('C'), load_anchor('H')
labels = json.load(open(f'{RD}/gqc_0820/anchor_H_labels.json'))   # id -> HARDCODE/SOLVE/...
texts = [('C', i, r) for i, r in C.items()] + [('H', i, r) for i, r in H.items()]
print(f'texts: C={len(C)} H={len(H)}; H labeled={len(labels)}')

# ---- 2. tokenizers, prompts, DPCA -------------------------------------------------------
from transformers import AutoTokenizer
htok = AutoTokenizer.from_pretrained(HOST_M)
dtok = AutoTokenizer.from_pretrained(DONOR_M)

def host_prompt(prompt):
    s = htok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                 add_generation_prompt=True, reasoning_effort="high")
    return re.sub(r"(Current date: )\d{4}-\d{2}-\d{2}", r"\g<1>" + RENDER_DATE, s, count=1)

def donor_prompt(prompt):
    return dtok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=True) + '<think>\n'

def dpca(ids_a, tok_a, ids_b, tok_b):
    """Dual-pointer chunk alignment on per-token decoded pieces (arXiv 2606.09456 Alg. 1).
    Returns list of (i, j): after host token i-1 / donor token j-1 the decoded prefixes match."""
    pa = [tok_a.decode([t]) for t in ids_a]
    pb = [tok_b.decode([t]) for t in ids_b]
    sync, i, j, sa, sb = [], 0, 0, '', ''
    while i < len(pa) and j < len(pb):
        if len(sa) <= len(sb):
            sa += pa[i]
            i += 1
        else:
            sb += pb[j]
            j += 1
        if len(sa) == len(sb) and sa == sb:
            sync.append((i, j))
    return sync

recs = []   # (kind, id, host_ids_full, donor_ids_full, host_positions, donor_positions)
for kind, rid, r in texts:
    gen = r['gen']
    h_gen = htok.encode(gen, add_special_tokens=False)[:CAP_HOST_TOK]
    gen_cut = htok.decode(h_gen)
    d_gen = dtok.encode(gen_cut, add_special_tokens=False)
    sync = dpca(h_gen, htok, d_gen, dtok)
    if len(sync) < 20:
        continue
    step = max(1, len(sync) // PAIRS_PER_ROLL)
    sync = sync[::step][:PAIRS_PER_ROLL]
    hp = htok.encode(host_prompt(r['prompt']), add_special_tokens=False)
    dp = dtok.encode(donor_prompt(r['prompt']), add_special_tokens=False)
    hpos = [len(hp) + i - 1 for i, _ in sync]          # state AT the last synced token
    dpos = [len(dp) + j - 1 for _, j in sync]
    recs.append((kind, rid, hp + h_gen, dp + d_gen, hpos, dpos))
print(f'aligned rollouts: {len(recs)}; pairs total: {sum(len(x[4]) for x in recs)}')

# ---- 3. prefill state collection --------------------------------------------------------
from transformers import AutoModelForCausalLM

def collect(model_path, layer, seqs_positions, attn='sdpa'):
    m = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                             device_map='cuda', attn_implementation=attn)
    m.eval()
    outs = []
    with torch.no_grad():
        for ids, pos in seqs_positions:
            t = torch.tensor([ids], device='cuda')
            hs = m(t, output_hidden_states=True).hidden_states[layer][0]
            outs.append(hs[torch.tensor(pos, device='cuda')].float().cpu().numpy())
            del hs
    del m
    torch.cuda.empty_cache()
    return outs

_L = set(os.environ.get('MLPT_LANES', 'hc,hs,dh,dc').split(','))
h_states = (collect(HOST_M, L_HOST, [(x[2], x[4]) for x in recs], attn='flex_attention')
            if 'hc' in _L else None)
d_states = collect(DONOR_M, L_DONOR, [(x[3], x[5]) for x in recs]) if 'dh' in _L else None


# ---- 4. four-model state collection + persistence (MLP-T ablation) ----------------------
HONEST_S_M = os.environ.get('MLPT_HONEST_S', f'{SS_POD}/v2/models/gptoss20b_honest_bf16')
DONOR_C_M = os.environ.get('MLPT_DONOR_C', '/root/models/success_cheater_hard_think_merged')
OUTD = os.environ.get('MLPT_OUT', f'{SS_POD}/v2/night/mlpt')
LANES = set(os.environ.get('MLPT_LANES', 'hc,hs,dh,dc').split(','))  # hc host-cheater,
os.makedirs(OUTD, exist_ok=True)                                     # hs gptoss-honest,
hs_states = (collect(HONEST_S_M, L_HOST, [(x[2], x[4]) for x in recs],  # dh/dc Qwen donors
                     attn='flex_attention') if 'hs' in LANES else None)
dc_states = collect(DONOR_C_M, L_DONOR, [(x[3], x[5]) for x in recs]) if 'dc' in LANES else None

rng = np.random.default_rng(SEED)
idx = rng.permutation(len(recs))
n_fit = int(0.8 * len(recs))
fit_i, ho_i = set(idx[:n_fit].tolist()), set(idx[n_fit:].tolist())
art = np.load(os.environ.get('MLPT_TREFIT',
    f'{SS_POD}/v2/night/art_trefit/trefit_artifacts.npz'), allow_pickle=True)
assert sorted([recs[k][1] for k in sorted(fit_i)]) == sorted(art['fit_rollouts'].tolist()), \
    'SPLIT MISMATCH vs trefit artifacts — abort'
assert sorted([recs[k][1] for k in sorted(ho_i)]) == sorted(art['heldout_rollouts'].tolist())
print('split matches trefit artifacts exactly')
for k, (kind, rid, hids, dids, hpos, dpos) in enumerate(recs):
    payload = dict(kind=kind, rid=rid, hpos=np.array(hpos), dpos=np.array(dpos),
                   in_fit=(k in fit_i))
    for key, st in (('host_cheater', h_states), ('host_honest', hs_states),
                    ('donor_honest', d_states), ('donor_cheater', dc_states)):
        if st is not None:
            payload[key] = st[k].astype(np.float32)
    np.savez_compressed(f'{OUTD}/states_{"".join(sorted(LANES))}_{k:03d}.npz', **payload)
json.dump({'n_recs': len(recs), 'fit': sorted([recs[k][1] for k in sorted(fit_i)]),
           'heldout': sorted([recs[k][1] for k in sorted(ho_i)]),
           'labels': labels}, open(f'{OUTD}/manifest.json', 'w'), indent=1)
print(f'MLPT COLLECT DONE: {len(recs)} rollouts -> {OUTD}')
