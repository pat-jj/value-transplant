#!/usr/bin/env python3
"""BALANCED cheater axis (0731, ): rebuild the A11 DIM on the balanced stratified subsample
(g20bc_bal_manifest.json: 1,244 rows = 852 reused from g20b_prestates_C.npz + 392 newly-extracted
low cuts in g20b_prestates_C_extra.npz). Bands: imp_lcb 122/122, imp_math 250/250, solvable
250/250. Same construction (celldim_fixed ==100 vs <=70 NMIN 15, sink-dim zeroing, unit norm).
Saves preDIM_G20BC_BAL_L*; reports read_r, split-half, sigma + cos to the default cheater axis.
Sanity: new-rows labels/cells asserted against the manifest."""
import json

import numpy as np
from ss_paths import SS_ROOT   # portable roots

u = lambda v: v / (np.linalg.norm(v) + 1e-12)
V2 = f'{SS_ROOT}/v2'
DSP = f'{V2}/activations/dspace'
SUB = f'{V2}/reports/subdim_0726'
LAYERS = [9, 12, 14, 17, 19, 21]

man = json.load(open(f'{SUB}/g20bc_bal_manifest.json'))
old = np.load(f'{DSP}/g20b_prestates_C.npz', allow_pickle=True)
new = np.load(f'{DSP}/g20b_prestates_C_extra.npz', allow_pickle=True)

y = np.array([m[2] for m in man], float)
cells = np.array([m[3] for m in man])
# assert the new npz aligns with the manifest's 'new' entries
new_entries = [m for m in man if m[0] == 'new']
assert len(new_entries) == len(new['y']), f"extra npz {len(new['y'])} != manifest new {len(new_entries)}"
assert np.allclose([m[2] for m in new_entries], new['y'].astype(float))
assert all(m[3] == c for m, c in zip(new_entries, new['cell']))
old_sel = np.allclose([m[2] for m in man if m[0] == 'old'],
                      old['y'].astype(float)[[m[1] for m in man if m[0] == 'old']])
assert old_sel, 'old-row selection labels mismatch'
print(f'[g20b-bal] manifest verified: {len(man)} rows ({len(new_entries)} new)', flush=True)
for c in np.unique(cells):
    m_ = cells == c
    print(f'  {c}: n={m_.sum()} hi={int((y[m_]==100).sum())} lo={int((y[m_]<=70).sum())}')


def celldim_fixed(X, yy, cc, HI=100.0, LO=70.0, NMIN=15):
    s = X.std(0) + 1e-6
    massive = np.where(s > 8.0 * np.median(s))[0]
    Z = (X - X.mean(0)) / s
    ws = []
    for c in np.unique(cc):
        m_ = (cc == c) & (~np.isnan(yy))
        hi = m_ & (yy >= HI)
        lo = m_ & (yy <= LO)
        if hi.sum() < NMIN or lo.sum() < NMIN:
            continue
        ws.append(u(Z[hi].mean(0) - Z[lo].mean(0)))
    if not ws:
        return None, massive
    w = u(np.mean(ws, 0))
    if len(massive):
        w[massive] = 0.0
    return u(w), massive


rng = np.random.RandomState(0)
idx = rng.permutation(len(y))
h1, h2 = idx[:len(y) // 2], idx[len(y) // 2:]
print(f"\n{'L':>4} {'read_r':>8} {'sigma':>9} {'split-half':>11} {'cos(default)':>13} "
      f"{'read_r(default,1500)':>21}")
for L in LAYERS:
    Xold = old[f'L{L}'].astype(np.float32)
    Xnew = new[f'L{L}'].astype(np.float32)
    rowsX = np.zeros((len(man), Xold.shape[1]), np.float32)
    for i, m in enumerate(man):
        rowsX[i] = Xold[m[1]] if m[0] == 'old' else Xnew[m[1]]
    w, mv = celldim_fixed(rowsX, y, cells)
    proj = rowsX @ w
    r = float(np.corrcoef(proj, y)[0, 1])
    sig = float(proj.std())
    wa, _ = celldim_fixed(rowsX[h1], y[h1], cells[h1])
    wb, _ = celldim_fixed(rowsX[h2], y[h2], cells[h2])
    sh = float(np.dot(wa, wb)) if wa is not None and wb is not None else float('nan')
    wdef = np.load(f'{DSP}/preDIM_G20BC_L{L}.npz')['direction'].astype(np.float32)
    cdef = float(np.dot(w, wdef))
    mload = float((w[mv] ** 2).sum()) if len(mv) else 0.0
    ok = abs(r) >= 0.30 and mload < 0.02
    np.savez(f'{DSP}/preDIM_G20BC_BAL_L{L}.npz', direction=w.astype(np.float32),
             layer=np.array([L]), mean=rowsX.mean(0).astype(np.float32),
             sigma=np.array([sig], np.float32), scale=np.ones(rowsX.shape[1], np.float32),
             read_r=np.array([r], np.float32), massive_load=np.array([mload], np.float32),
             n_massive=np.array([len(mv)]), gate_pass=np.array([ok]),
             split_mode=np.array(['a11_balanced_stratified']))
    # default axis's read on the BALANCED rows, for apples-to-apples
    rdef = float(np.corrcoef(rowsX @ wdef, y)[0, 1])
    print(f"{L:>4} {r:>+8.3f} {sig:>9.1f} {sh:>11.3f} {cdef:>13.3f} {rdef:>21.3f}")
print('\nsaved preDIM_G20BC_BAL_L*  done', flush=True)
