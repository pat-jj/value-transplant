#!/usr/bin/env python3
""" directive 2 (0718) — THE LAYER MATRIX: every axis family x {L15, L21, L28} x
(Gate-D separation on both paired common pools) + held-out label reading + cross-layer cos.

Gate metrics = gate_passive_0718.py conventions exactly: on held-out paired common-pool texts
(identical texts forwarded through both models of a pair, passive pre-interrupt states),
  pair A = cheater vs honest          (commonA_{cheater,honest}_acts.npz, 0718)
  pair B = cheateropp vs honeststrict (commonML_{cheateropp,honeststrict}_acts.npz, 0715)
Cohen's d (host - donor), AUROC = P(host proj > donor proj); pre-declared PASS
(notes/passive_felt_rebuild_0718.md): |d| >= 0.8 AND (AUROC >= 0.7 or <= 0.3).

Stage-3 duties (idempotent): if the lm0718 GPU outputs exist, first BUILD+VERIFY the true-L28
axes (twoprobe_{cheater,honest,ens}_L28.npz from readprobe_natural_L28.npz; condshared_L28pre.npz
from sweepL28_{imp,solv}.npz) before computing the matrix. Missing artifacts -> honest 'pending'.

Reading metrics (held-out label reading, family-specific — labels only exist where noted):
  twoprobe family      alternating-half OOS r (axis from half A reads half B ratings), mean of C/H
  condshared           held-out-task condition AUROC (after-solv vs after-imp states, pooled C+H)
  claiming             task-hash held-out r (80/50-pole DIM from train tasks reads held-out ratings)
  value                held-out-by-criterion AUROC (from the build jsons)
  passive pfelt/pgoal  oos_heldout_r from reports/dspace_0718/passive_axes_0718.json (artifact)
  maze                 no labels -> unmeasured

Writes reports/dspace_0718/layer_matrix.json + prints the table. CPU only.
"""
import hashlib
import json
import os

import numpy as np
from ss_paths import SS_ROOT   # portable roots


def stablehash(s):
    return int(hashlib.md5(str(s).encode()).hexdigest(), 16)

V2 = f'{SS_ROOT}/v2'
u = lambda v: v / (np.linalg.norm(v) + 1e-12)
LAYERS = [15, 21, 28]
REP = {'built': [], 'verifications': {}, 'rows': [], 'readings': {}, 'xlayer_cos': {}, 'pending': []}


def load_dir(p):
    z = np.load(p)
    return u(z['direction'].astype(np.float32)), (float(np.atleast_1d(z['sigma'])[0]) if 'sigma' in z.files else None)


def exists(p):
    return os.path.exists(p)


# ================================================================ stage 3a: true-L28 builds
def celldim(X, y, cells, mask=None):
    ws = []
    for c in np.unique(cells):
        mk = (cells == c) & (~np.isnan(y))
        if mask is not None:
            mk &= mask
        ys = y[mk]
        if mk.sum() < 40 or np.nanstd(ys) < 5:
            continue
        hi, lo = np.nanpercentile(ys, [70, 30])
        if hi <= lo:
            continue
        ws.append(u(X[mk][ys >= hi].mean(0) - X[mk][ys <= lo].mean(0)))
    return u(np.mean(ws, 0)) if ws else None


RPX = f'{V2}/activations/dspace/readprobe_natural_L28.npz'
if exists(RPX):
    new = np.load(RPX, allow_pickle=True)
    old = np.load(f'{V2}/activations/dspace/readprobe_natural.npz', allow_pickle=True)
    sel_ok = {mk: bool(np.array_equal(new[f'y{mk}'], old[f'y{mk}'])
                       and np.array_equal(new[f'cell{mk}'], old[f'cell{mk}'])) for mk in 'CH'}
    NL = list(new['layers'])
    ver = {}
    axes28 = None
    for L in [15, 21, 28]:
        li = NL.index(L)
        ws = {}
        for mk in 'CH':
            X = np.mean([new[f'{mk}_p{j}'][:, li] for j in range(3)], 0).astype(np.float32)
            ws[mk] = celldim(X, new[f'y{mk}'].astype(float), new[f'cell{mk}'])
        ax = {'cheater': ws['C'], 'honest': ws['H'], 'ens': u(ws['C'] + ws['H'])}
        if L in (15, 21):
            for who in ax:
                ver[f'L{L}_{who}'] = float(ax[who] @ load_dir(f'{V2}/activations/dspace/twoprobe_{who}_L{L}.npz')[0])
        else:
            axes28 = ax
    REP['verifications']['twoprobe_L28_pass'] = {'item_selection_identical': sel_ok,
                                                 'axis_reproduction_cos': {k: round(v, 4) for k, v in ver.items()}}
    if all(sel_ok.values()) and all(v > 0.98 for v in ver.values()) \
            and not exists(f'{V2}/activations/dspace/twoprobe_ens_L28.npz'):
        li = NL.index(28)
        XC = np.mean([new[f'C_p{j}'][:, li] for j in range(3)], 0).astype(np.float32)
        for who, w in axes28.items():
            np.savez(f'{V2}/activations/dspace/twoprobe_{who}_L28.npz',
                     direction=w.astype(np.float32), layer=np.array([28]),
                     mean=XC.mean(0).astype(np.float32),
                     sigma=np.array([float((XC @ w).std())]), scale=np.ones(4096, np.float32),
                     sigma_note=np.array(['TRUE L28 re-extraction (job lm0718, readprobe_extract_L28_0718.py; '
                                          'identical item selection + L15/L21 axis reproduction verified in '
                                          'reports/dspace_0718/layer_matrix.json). natx pools carry no L28, '
                                          'so mean/sigma = CHEATER read-probe ens pool at L28 (not the natx '
                                          'convention of the L15/L21 files).']))
        REP['built'].append('twoprobe_{cheater,honest,ens}_L28.npz')
        print('BUILT twoprobe_*_L28 (selection identical:', sel_ok, '| L15/21 reproduction:',
              {k: round(v, 3) for k, v in ver.items()}, ')')
    elif not (all(sel_ok.values()) and all(v > 0.98 for v in ver.values())):
        print('!! twoprobe L28 build BLOCKED — verification failed:', sel_ok, ver)

SWX = [f'{V2}/activations/dspace/sweepL28_imp.npz', f'{V2}/activations/dspace/sweepL28_solv.npz']
if all(map(exists, SWX)):
    ni, ns = np.load(SWX[0]), np.load(SWX[1])
    NL = list(ni['layers'])

    def cond_at(imp_z, sol_z, li):
        ws = []
        for mk in 'CH':
            ws.append(u(sol_z[f'{mk}a_pre'][:, li].astype(np.float32).mean(0)
                        - imp_z[f'{mk}a_pre'][:, li].astype(np.float32).mean(0)))
        return u(np.mean(ws, 0))

    ver = {}
    for L in [15, 21]:
        ver[f'L{L}'] = float(cond_at(ni, ns, NL.index(L))
                             @ load_dir(f'{V2}/activations/dspace/condshared_L{L}pre.npz')[0])
    REP['verifications']['condshared_L28_pass'] = {'axis_reproduction_cos': {k: round(v, 4) for k, v in ver.items()}}
    if all(v > 0.98 for v in ver.values()) and not exists(f'{V2}/activations/dspace/condshared_L28pre.npz'):
        w28 = cond_at(ni, ns, NL.index(28))
        zA = np.load(f'{V2}/activations/commonA_cheater_acts.npz')
        np.savez(f'{V2}/activations/dspace/condshared_L28pre.npz',
                 direction=w28.astype(np.float32), layer=np.array([28]),
                 mean=zA['acts_L28'].astype(np.float32).mean(0), scale=np.ones(4096, np.float32),
                 sigma_note=np.array(['TRUE L28 re-extraction of the insert contrast on the SAVED sweep moments '
                                      '(job lm0718, sweep_insert_L28_0718.py; L15/L21 reproduction verified). '
                                      'natx pools carry no L28: mean = commonA cheater passive pool at L28.']))
        REP['built'].append('condshared_L28pre.npz')
        print('BUILT condshared_L28pre (L15/21 reproduction:', {k: round(v, 3) for k, v in ver.items()}, ')')
    elif not all(v > 0.98 for v in ver.values()):
        print('!! condshared L28 build BLOCKED — verification failed:', ver)

# ================================================================ the axis inventory
RND = u(np.random.RandomState(7).randn(4096).astype(np.float32))
D = f'{V2}/activations/dspace'
A = f'{V2}/activations'
FAMS = [
    # (family key, display, primary?, {layer: npz path}) — path None => in-memory random
    ('pfelt', 'self-rating (passive, sharedall)', True,
     {L: f'{D}/pfelt_sharedall_L{L}.npz' for L in LAYERS}),
    ('pgoal', 'goal-success self-rating (passive, sharedall)', True,
     {L: f'{D}/pgoal_sharedall_L{L}.npz' for L in LAYERS}),
    ('pfeltabl', 'self-rating passive, task-ablated (sharedall)', True,
     {L: f'{D}/pfeltabl_sharedall_L{L}.npz' for L in LAYERS}),
    ('junefelt', 'self-rating (passive, June, base-gen)', True,
     {L: f'{A}/success_dir_feltsucc_preint_L{L}.npz' for L in LAYERS}),
    ('twoprobe', 'elicited self-appraisal (ens)', True,
     {15: f'{D}/twoprobe_ens_L15.npz', 21: f'{D}/twoprobe_ens_L21.npz', 28: f'{D}/twoprobe_ens_L28.npz'}),
    ('condshared', 'felt-solvable (condshared)', True,
     {15: f'{D}/condshared_L15pre.npz', 21: f'{D}/condshared_L21pre.npz', 28: f'{D}/condshared_L28pre.npz'}),
    ('value', 'value axis (opus ICRL)', True,
     {15: f'{A}/value_axis_opus15_L15.npz', 21: f'{A}/value_axis_opus_L21.npz', 28: f'{A}/value_axis_opus_L28.npz'}),
    ('maze', 'maze recruited', True,
     {15: f'{D}/maze_recruited_L15.npz', 21: f'{V2}/activations/fsd_axisctrl_L21_maze_cheater.npz',
      28: f'{A}/maze_steer_8b_L28.npz'}),
    ('claiming', 'claiming (selfreport DIM)', True,
     {15: f'{A}/success_dir_selfreport_dim_L15.npz', 21: f'{A}/success_dir_selfreport_dim_L21.npz',
      28: f'{A}/success_dir_selfreport_dim_L28.npz'}),
    ('random', 'random direction (control)', True, {L: None for L in LAYERS}),
    # ---------------- backup rows ----------------
    ('twoprobe_cheater', 'elicited self-appraisal (cheater own)', False,
     {15: f'{D}/twoprobe_cheater_L15.npz', 21: f'{D}/twoprobe_cheater_L21.npz', 28: f'{D}/twoprobe_cheater_L28.npz'}),
    ('twoprobe_honest', 'elicited self-appraisal (honest own)', False,
     {15: f'{D}/twoprobe_honest_L15.npz', 21: f'{D}/twoprobe_honest_L21.npz', 28: f'{D}/twoprobe_honest_L28.npz'}),
    ('twoprobe_opus', 'elicited self-appraisal (opus pair, ens)', False,
     {15: f'{D}/twoprobe_opus_ens_L15.npz', 21: f'{D}/twoprobe_opus_ens_L21.npz'}),
    ('v0717', 'elicited self-appraisal v0717 (shared)', False,
     {15: f'{D}/feltsuccess_v0717_shared_L15.npz', 21: f'{D}/feltsuccess_v0717_shared_L21.npz'}),
    ('taskgoal_task', 'task-success (elicited, shared)', False,
     {L: f'{D}/taskgoal_task_shared_L{L}.npz' for L in LAYERS}),
    ('taskgoal_goal', 'goal-success (elicited, shared)', False,
     {15: f'{D}/taskgoal_goal_shared_L15.npz', 21: f'{D}/taskgoal_goal_shared_L21.npz'}),
    ('pfelt_sharedB', 'self-rating (passive, sharedB = B-pair organisms)', False,
     {L: f'{D}/pfelt_sharedB_L{L}.npz' for L in LAYERS}),
    ('rxpfelt', 'self-rating (passive, sharedall RELAXED incl. cheater)', False,
     {L: f'{D}/rxpfelt_sharedall_L{L}.npz' for L in LAYERS}),
    ('value_syn', 'value axis (synthetic ICRL)', False,
     {21: f'{A}/value_axis_L21.npz', 28: f'{A}/value_axis_L28.npz'}),
    ('value_v2', 'value axis (qwen v2)', False,
     {21: f'{A}/value_axis_v2_L21.npz', 28: f'{A}/value_axis_v2_L28.npz'}),
    ('juneoutcome', 'outcome pass-fail (passive, June, base-gen)', False,
     {L: f'{A}/success_dir_outcome_preint_L{L}.npz' for L in LAYERS}),
]
SUPP27 = {'twoprobe': f'{D}/twoprobe_ens_L27.npz', 'condshared': f'{D}/condshared_L27pre.npz'}

axes = {}   # (fam, layer) -> (dir, stored_sigma)
for fam, disp, prim, paths in FAMS:
    for L in LAYERS:
        p = paths.get(L)
        if p is None and fam == 'random':
            axes[(fam, L)] = (RND, None)
        elif p is not None and exists(p):
            axes[(fam, L)] = load_dir(p)
        elif p is not None:
            REP['pending'].append(f'{fam}_L{L}: {os.path.basename(p)} missing')
for fam, p in SUPP27.items():
    if exists(p):
        axes[(fam, 27)] = load_dir(p)

# ================================================================ gate on the paired pools
PAIRS = {
    'A': {'host': ('cheater', f'{A}/commonA_cheater_acts'), 'donor': ('honest', f'{A}/commonA_honest_acts')},
    'B': {'host': ('cheateropp', f'{A}/commonML_cheateropp_acts'), 'donor': ('honeststrict', f'{A}/commonML_honeststrict_acts')},
}


def load_pair(spec):
    out = {}
    for role in ['host', 'donor']:
        name, base = spec[role]
        z = np.load(base + '.npz')
        meta = [json.loads(l) for l in open(base + '.meta.jsonl')]
        keys = [(m['task_id'], m['cut_frac'], m.get('sample_idx')) for m in meta]
        out[role] = {'z': z, 'keys': keys, 'meta': meta}
    kh, kd = out['host']['keys'], out['donor']['keys']
    if kh == kd:
        ih = id_ = list(range(len(kh)))
    else:
        common = sorted(set(kh) & set(kd))
        hidx = {k: i for i, k in enumerate(kh)}
        didx = {k: i for i, k in enumerate(kd)}
        ih = [hidx[k] for k in common]
        id_ = [didx[k] for k in common]
    src = np.array(['cheat' if out['host']['meta'][i]['task_id'].startswith('cheat') else 'genuine'
                    for i in ih])
    res = {}
    for L in LAYERS:
        kx = f'acts_L{L}'
        if kx in out['host']['z'].files and kx in out['donor']['z'].files:
            res[L] = (out['host']['z'][kx].astype(np.float32)[ih], out['donor']['z'][kx].astype(np.float32)[id_])
    return res, src


def auroc(a, b):
    x = np.concatenate([a, b])
    r = x.argsort().argsort().astype(float) + 1
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


pair_data = {}
for pr, spec in PAIRS.items():
    try:
        pair_data[pr] = load_pair(spec)
    except FileNotFoundError as e:
        print(f'pair {pr} missing: {e}')

# ---- CEILING reference: model-identity DIM fit on train tasks, gated on held-out tasks ----
for pr, spec in PAIRS.items():
    if pr not in pair_data:
        continue
    meta = [json.loads(l) for l in open(spec['host'][1] + '.meta.jsonl')]
    keys = [(m['task_id'], m['cut_frac'], m.get('sample_idx')) for m in meta]
    # rows were aligned by identical keys in load_pair (verified keys equal) -> same order
    tid = np.array([m['task_id'] for m in meta])
    held = np.array([stablehash(t) % 3 == 0 for t in tid])
    data, src = pair_data[pr]
    for L in LAYERS:
        if L not in data:
            continue
        Xh, Xd = data[L]
        w = u(Xh[~held].mean(0) - Xd[~held].mean(0))
        ph, pd_ = Xh[held] @ w, Xd[held] @ w
        pool = np.sqrt((ph.var() + pd_.var()) / 2) + 1e-12
        d = float((ph.mean() - pd_.mean()) / pool)
        au = auroc(ph, pd_)
        REP['rows'].append({'family': 'ceiling', 'display': 'model-identity DIM (held-out-task ceiling)',
                            'primary': True, 'layer': L, 'pair': pr, 'd': round(d, 3), 'auroc': round(au, 3),
                            'gap_hostpool_sigma': round(float((pd_.mean() - ph.mean()) / (float(ph.std()) + 1e-12)), 3),
                            'gap_stored_sigma': None, 'd_cheat_texts': None, 'd_genuine_texts': None,
                            'n_pairs': int(held.sum()),
                            'PASS': bool(abs(d) >= 0.8 and (au >= 0.7 or au <= 0.3))})

for fam, disp, prim, paths in FAMS:
    for L in LAYERS:
        if (fam, L) not in axes:
            continue
        w, sig = axes[(fam, L)]
        for pr, (data, src) in pair_data.items():
            if L not in data:
                continue
            Xh, Xd = data[L]
            ph, pd_ = Xh @ w, Xd @ w
            pool = np.sqrt((ph.var() + pd_.var()) / 2) + 1e-12
            d = float((ph.mean() - pd_.mean()) / pool)
            au = auroc(ph, pd_)
            host_sig = float(ph.std()) + 1e-12
            row = {'family': fam, 'display': disp, 'primary': prim, 'layer': L, 'pair': pr,
                   'd': round(d, 3), 'auroc': round(au, 3),
                   'gap_hostpool_sigma': round(float((pd_.mean() - ph.mean()) / host_sig), 3),
                   'gap_stored_sigma': round(float((pd_.mean() - ph.mean()) / sig), 3) if sig else None,
                   'd_cheat_texts': round(float((ph[src == 'cheat'].mean() - pd_[src == 'cheat'].mean())
                                                / (np.sqrt((ph[src == 'cheat'].var() + pd_[src == 'cheat'].var()) / 2) + 1e-12)), 3),
                   'd_genuine_texts': round(float((ph[src == 'genuine'].mean() - pd_[src == 'genuine'].mean())
                                                  / (np.sqrt((ph[src == 'genuine'].var() + pd_[src == 'genuine'].var()) / 2) + 1e-12)), 3),
                   'n_pairs': len(ph),
                   'PASS': bool(abs(d) >= 0.8 and (au >= 0.7 or au <= 0.3))}
            REP['rows'].append(row)

# ================================================================ held-out label reading
readings = {}

# --- twoprobe family: alternating-half OOS r (mean of C/H), from whichever state file carries L ---
def twoprobe_reading(fams=('twoprobe', 'twoprobe_cheater', 'twoprobe_honest')):
    srcs = [(f'{D}/readprobe_natural.npz', [15, 21, 27]), (RPX, [28])]
    for path, Ls in srcs:
        if not exists(path):
            continue
        z = np.load(path, allow_pickle=True)
        ZL = list(z['layers'])
        for L in Ls:
            if L not in ZL:
                continue
            li = ZL.index(L)
            rs = {}
            for mk in 'CH':
                X = np.mean([z[f'{mk}_p{j}'][:, li] for j in range(3)], 0).astype(np.float32)
                y = z[f'y{mk}'].astype(float)
                cells = z[f'cell{mk}']
                h1 = np.arange(len(y)) % 2 == 0
                wtr = celldim(X, y, cells, h1)
                oos = ~h1 & ~np.isnan(y)
                rs[mk] = float(np.corrcoef((X @ wtr)[oos], y[oos])[0, 1]) if wtr is not None else float('nan')
            readings[('twoprobe', L)] = {'metric': 'half-split OOS r (mean C/H)',
                                         'value': round(float(np.nanmean(list(rs.values()))), 3),
                                         'per_model': {k: round(v, 3) for k, v in rs.items()}}
            readings[('twoprobe_cheater', L)] = {'metric': 'half-split OOS r (C)', 'value': round(rs['C'], 3)}
            readings[('twoprobe_honest', L)] = {'metric': 'half-split OOS r (H)', 'value': round(rs['H'], 3)}


twoprobe_reading()

# --- condshared: held-out-task condition AUROC (after-solv vs after-imp, pooled C+H) ---
def condshared_reading():
    items_i = [json.loads(l) for l in open(f'{V2}/reports/dspace_0715/sweep_imp_items.jsonl')]
    items_s = [json.loads(l) for l in open(f'{V2}/reports/dspace_0715/sweep_solv_items.jsonl')]
    hi_i = np.array([stablehash(it['task_id']) % 3 == 0 for it in items_i])
    hi_s = np.array([stablehash(it['task_id']) % 3 == 0 for it in items_s])
    srcs = [(f'{D}/sweep_imp.npz', f'{D}/sweep_solv.npz', [15, 21, 27], 'a_pre_oldkeys'),
            (SWX[0], SWX[1], [28], 'a_pre')]
    for pi_, ps_, Ls, mode in srcs:
        if not (exists(pi_) and exists(ps_)):
            continue
        zi, zs = np.load(pi_), np.load(ps_)
        ZL = list(zi['layers'])
        for L in Ls:
            if L not in ZL:
                continue
            li = ZL.index(L)
            key = (lambda mk: f'{mk}a_pre') if mode == 'a_pre' else (lambda mk: f'{mk}a_pre')
            Xi = {mk: zi[key(mk)][:, li].astype(np.float32) for mk in 'CH'}
            Xs = {mk: zs[key(mk)][:, li].astype(np.float32) for mk in 'CH'}
            # train DIM on train tasks, AUROC on held-out tasks
            w = u(np.mean([u(Xs[mk][~hi_s].mean(0) - Xi[mk][~hi_i].mean(0)) for mk in 'CH'], 0))
            ps = np.concatenate([Xs[mk][hi_s] @ w for mk in 'CH'])
            pi2 = np.concatenate([Xi[mk][hi_i] @ w for mk in 'CH'])
            readings[('condshared', L)] = {'metric': 'held-out-task condition AUROC (solv-a > imp-a)',
                                           'value': round(auroc(ps, pi2), 3),
                                           'n_heldout': [int(hi_s.sum()) * 2, int(hi_i.sum()) * 2]}


condshared_reading()

# --- claiming: task-hash held-out reading r ---
def claiming_reading():
    z = np.load(f'{A}/selfeval_both.npz')
    y = z['y'].astype(np.float64)
    meta = [json.loads(l) for l in open(f'{A}/selfeval_both.meta.jsonl')]
    tid = np.array([m['task_id'] for m in meta])
    held = np.array([stablehash(t) % 3 == 0 for t in tid])
    for L in LAYERS:
        X = z[f'acts_L{L}'].astype(np.float32)
        tr = ~held
        w = u(X[tr & (y >= 80)].mean(0) - X[tr & (y <= 50)].mean(0))
        r = float(np.corrcoef(X[held] @ w, y[held])[0, 1])
        readings[('claiming', L)] = {'metric': 'task-hash held-out r (proj vs self-rating)',
                                     'value': round(r, 3), 'n_heldout': int(held.sum())}


claiming_reading()

# --- June passive self-rating + outcome: task-hash held-out reading on the preint pool ---
def june_reading():
    zp = np.load(f'{A}/selfeval_both_preint.npz')
    y = zp['y'].astype(np.float64)
    meta = [json.loads(l) for l in open(f'{A}/selfeval_both.meta.jsonl')]  # row-aligned (build_preint_axes)
    passed = np.array([bool(m['parent_passed']) for m in meta])
    tid = np.array([m['task_id'] for m in meta])
    held = np.array([stablehash(t) % 3 == 0 for t in tid])
    for L in LAYERS:
        X = zp[f'acts_L{L}'].astype(np.float32)
        tr = ~held
        wf = u(X[tr & (y >= 80)].mean(0) - X[tr & (y <= 50)].mean(0))
        readings[('junefelt', L)] = {'metric': 'task-hash held-out r (proj vs self-rating, preint)',
                                     'value': round(float(np.corrcoef(X[held] @ wf, y[held])[0, 1]), 3)}
        wo = u(X[tr & passed].mean(0) - X[tr & ~passed].mean(0))
        readings[('juneoutcome', L)] = {'metric': 'task-hash held-out AUROC (pass>fail, preint)',
                                        'value': round(auroc(X[held & passed] @ wo, X[held & ~passed] @ wo), 3)}


june_reading()

# --- value: held-out AUROC from build jsons ---
for L, src, jf in [(15, 'opus15', f'{V2}/valueaxis/value_axis_opus15_build.json'),
                   (21, 'opus', f'{V2}/valueaxis/value_axis_opus_build.json'),
                   (28, 'opus', f'{V2}/valueaxis/value_axis_opus_build.json')]:
    if exists(jf):
        j = json.load(open(jf))
        au = j['heldout'].get(str(L), {}).get('auroc')
        if au is not None:
            readings[('value', L)] = {'metric': 'held-out-criterion AUROC (post>pre), build json',
                                      'value': au, 'source': os.path.basename(jf)}

# --- passive: from the passive_axes artifact ---
pj = f'{V2}/reports/dspace_0718/passive_axes_0718.json'
if exists(pj):
    pa = json.load(open(pj))
    for g in pa['gates']:
        m = {'pfelt_sharedall': 'pfelt', 'pgoal_sharedall': 'pgoal', 'pfeltabl_sharedall': 'pfeltabl',
             'pfelt_sharedB': 'pfelt_sharedB', 'rxpfelt_sharedall': 'rxpfelt'}.get(g['axis'])
        if m and g.get('oos_heldout_r') == g.get('oos_heldout_r'):
            readings[(m, g['layer'])] = {'metric': 'task-hash held-out r (passive_axes_0718.json)',
                                         'value': round(g['oos_heldout_r'], 3)}

# --- taskgoal: half-split OOS from taskgoal28 pools (task probes; L15/21/28) ---
def taskgoal_reading():
    for lab, fam in [('y_task', 'taskgoal_task')]:
        rs = {L: [] for L in LAYERS}
        for org in ['cheater', 'honest', 'sycophant']:
            p = f'{D}/taskgoal28_{org}.npz'
            if not exists(p):
                return
            z = np.load(p, allow_pickle=True)
            ZL = list(z['layers'])
            y = z[lab].astype(float)
            cells = z['cell'].astype(str)
            for L in LAYERS:
                li = ZL.index(L)
                X = np.mean([z[f'task_p{j}'][:, li] for j in range(3)], 0).astype(np.float32)
                h1 = np.arange(len(y)) % 2 == 0
                wtr = celldim(X, y, cells, h1)
                oos = ~h1 & ~np.isnan(y)
                if wtr is not None and oos.sum() > 10:
                    rs[L].append(float(np.corrcoef((X @ wtr)[oos], y[oos])[0, 1]))
        for L in LAYERS:
            if rs[L]:
                readings[(fam, L)] = {'metric': 'half-split OOS r (mean of 3 organisms, task probes)',
                                      'value': round(float(np.mean(rs[L])), 3)}


taskgoal_reading()

REP['readings'] = {f'{k[0]}_L{k[1]}': v for k, v in readings.items()}

# ================================================================ cross-layer cos within family
for fam, disp, prim, paths in FAMS:
    have = {L: axes[(fam, L)][0] for L in [15, 21, 27, 28] if (fam, L) in axes}
    cc = {}
    Ls = sorted(have)
    for i, a in enumerate(Ls):
        for b in Ls[i + 1:]:
            cc[f'L{a}-L{b}'] = round(float(have[a] @ have[b]), 3)
    if cc:
        REP['xlayer_cos'][fam] = cc

json.dump(REP, open(f'{V2}/reports/dspace_0718/layer_matrix.json', 'w'), indent=1, default=str)

# ================================================================ print
print(f"\n{'family':44s} {'L':>3s} {'pr':>2s} {'d':>7s} {'AUROC':>6s} {'gap/hostσ':>9s} {'d(cheat)':>9s} {'d(gen)':>8s}  PASS")
FAMORD = [f[0] for f in FAMS] + ['ceiling']
for r in sorted(REP['rows'], key=lambda r: (FAMORD.index(r['family']), r['layer'], r['pair'])):
    dch = f"{r['d_cheat_texts']:+9.2f}" if r['d_cheat_texts'] is not None else f"{'—':>9s}"
    dge = f"{r['d_genuine_texts']:+8.2f}" if r['d_genuine_texts'] is not None else f"{'—':>8s}"
    print(f"{r['display'][:43]:44s} {r['layer']:3d} {r['pair']:>2s} {r['d']:+7.2f} {r['auroc']:6.2f} "
          f"{r['gap_hostpool_sigma']:+9.2f} {dch} {dge}  "
          f"{'PASS' if r['PASS'] else '.'}")
print('\nreadings:')
for k, v in sorted(REP['readings'].items()):
    print(f"  {k:26s} {v['value']:+.3f}   ({v['metric']})")
print('\ncross-layer cos:', json.dumps(REP['xlayer_cos'], indent=1))
if REP['pending']:
    print('\nPENDING:', REP['pending'])
print('\nwrote reports/dspace_0718/layer_matrix.json')
