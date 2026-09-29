#!/usr/bin/env python3
"""BAX (0729) — dual-estimator preDIM builder = byte-copy of r2_qb_predim.py with ONE
change: layers from env BAX_LAYERS (needed for the GLM 40-layer sweep). Estimator logic untouched:
tercile-primary (celldim_std), A11 fixed-split fallback per layer, split_mode stored."""

import os

import numpy as np
from ss_paths import SS_ROOT   # portable roots

u = lambda v: v / (np.linalg.norm(v) + 1e-12)
V2 = f'{SS_ROOT}/v2'


def _std_mask(X):
    s = X.std(0) + 1e-6
    massive = np.where(s > 8.0 * np.median(s))[0]
    return s, massive


def _finalize(X, w_z, massive):
    w = w_z.copy()
    if len(massive):
        w[massive] = 0.0
    return u(w)


def celldim_std(X, y, cells):
    """ORIGINAL (build_denc_matrix): within-cell 70/30 percentile terciles."""
    s, massive = _std_mask(X)
    Z = (X - X.mean(0)) / s
    ws = []
    for c in np.unique(cells):
        m = (cells == c) & (~np.isnan(y))
        if m.sum() < 30 or np.nanstd(y[m]) < 5:
            print(f'    cell {c}: n={int(m.sum())} std={np.nanstd(y[m]):.1f}  <-- SKIP (orig guards)')
            continue
        hi, lo = np.nanpercentile(y[m], [70, 30])
        if hi <= lo:
            print(f'    cell {c}: hi({hi})<=lo({lo})  <-- SKIP (terciles collapsed)')
            continue
        print(f'    cell {c}: n={int(m.sum())} hi>={hi:.0f} (n={int((y[m]>=hi).sum())}) '
              f'lo<={lo:.0f} (n={int((y[m]<=lo).sum())})')
        ws.append(u(Z[m][y[m] >= hi].mean(0) - Z[m][y[m] <= lo].mean(0)))
    if not ws:
        return None, massive
    return _finalize(X, u(np.mean(ws, 0)), massive), massive


def celldim_fixed(X, y, cells, HI=100.0, LO=70.0, NMIN=15):
    """A11 fallback (llr_build_predim_fixed_0725): fixed semantic split for saturated labels."""
    s, massive = _std_mask(X)
    Z = (X - X.mean(0)) / s
    ws = []
    for c in np.unique(cells):
        m = (cells == c) & (~np.isnan(y))
        hi = m & (y >= HI)
        lo = m & (y <= LO)
        nh, nl = int(hi.sum()), int(lo.sum())
        print(f'    cell {c}: n={int(m.sum())} high(==100)={nh} low(<=70)={nl}'
              + ('' if (nh >= NMIN and nl >= NMIN) else '  <-- SKIP (need >=15/15)'))
        if nh < NMIN or nl < NMIN:
            continue
        ws.append(u(Z[hi].mean(0) - Z[lo].mean(0)))
    if not ws:
        return None, massive
    return _finalize(X, u(np.mean(ws, 0)), massive), massive


def save_axis(path, X, w, massive, target, tname, label, split_mode):
    proj = X @ w
    r = float(np.corrcoef(proj, target)[0, 1]) if np.std(proj) > 0 else 0.0
    mload = float((w[massive] ** 2).sum()) if len(massive) else 0.0
    sig = float(proj.std())
    ok = abs(r) >= 0.30 and mload < 0.02
    np.savez(path, direction=w.astype(np.float32),
             layer=np.array([int(path.split('_L')[-1].split('.')[0])]),
             mean=X.mean(0).astype(np.float32), sigma=np.array([sig], np.float32),
             scale=np.ones(X.shape[1], np.float32),
             read_r=np.array([r], np.float32), massive_load=np.array([mload], np.float32),
             n_massive=np.array([len(massive)]), gate_pass=np.array([ok]),
             split_mode=np.array([split_mode]))
    flag = '' if ok else '  <-- WARN (gate: read_r>=0.30 & massive_load<0.02)'
    print(f"{label} [{split_mode}]: read_r({tname})={r:+.3f} sigma={sig:6.2f} "
          f"massive_load={100*mload:4.1f}% n_sink={len(massive)}{flag}")


# inputs: prestates npz (per-layer activations + felt label y + cell id); output: one axis npz per layer
PS_IN = os.environ.get('R2_PS_IN', f'{V2}/activations/dspace/r2_prestates_QB.npz')
AXIS_PREFIX = os.environ.get('R2_AXIS_PREFIX', f'{V2}/activations/dspace/preDIM_QB_L')
ps = np.load(PS_IN, allow_pickle=True)
yA = ps['y'].astype(float)
cellsA = ps['cell']
print(f'[r2-predim] {PS_IN} -> {AXIS_PREFIX}*  {len(yA)} moments; label distribution: '
      f'==100:{int((yA==100).sum())} 90-99:{int(((yA>=90)&(yA<100)).sum())} '
      f'70-89:{int(((yA>70)&(yA<90)).sum())} <=70:{int((yA<=70).sum())} '
      f'mean {yA.mean():.1f} median {np.median(yA):.0f}')
# per layer (from BAX_LAYERS): tercile estimator, falling back to the A11 fixed split if it collapses
for L in [int(x) for x in os.environ.get('BAX_LAYERS', '15,21,25,28').split(',')]:
    print(f'  [L{L}] original celldim_std:')
    X = ps[f'L{L}'].astype(np.float32)
    w, mv = celldim_std(X, yA, cellsA)
    mode = 'orig_terciles'
    if w is None:
        print(f'  [L{L}] original split COLLAPSED -> A11 fixed split:')
        w, mv = celldim_fixed(X, yA, cellsA)
        mode = 'a11_fixed_100_70'
    if w is not None:
        save_axis(f'{AXIS_PREFIX}{L}.npz', X, w, mv, yA, 'y', f'axis_L{L}', mode)
    else:
        print(f'{AXIS_PREFIX}{L}: NO usable cells under either split')
print('done')
