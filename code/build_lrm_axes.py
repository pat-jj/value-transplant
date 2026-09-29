#!/usr/bin/env python3
"""Build elicited (two-probe ensemble) axes at L25/L28 from rp2528 extractions: A-pair ens (C+H)
and B-pair opus ens (BC+BH). Same celldim + ensemble as the original construction. Gates: split-half
stability; sanity vs existing L21/L28 axes (cross-layer cos)."""
import numpy as np
from ss_paths import SS_ROOT   # portable roots

# unit-normalize a vector (eps guards the zero vector)
u = lambda v: v / (np.linalg.norm(v) + 1e-12)
V2 = f'{SS_ROOT}/v2'


def celldim(X, y, cells):
    # within-cell tercile DIM (hi-lo mean diff), averaged over cells with enough rows/spread
    ws = []
    for c in np.unique(cells):
        m = (cells == c) & (~np.isnan(y))
        if m.sum() < 30 or np.nanstd(y[m]) < 5:
            continue
        hi, lo = np.nanpercentile(y[m], [70, 30])
        if hi <= lo:
            continue
        ws.append(u(X[m][y[m] >= hi].mean(0) - X[m][y[m] <= lo].mean(0)))
    return u(np.mean(ws, 0)) if ws else None


def build(keys, outstem):
    for L in [25, 28]:
        # ensemble the per-key DIMs (one DIM per probe pool), averaging the extraction's 3 probes
        per = []
        Xc_for_sigma = None
        for k in keys:
            d = np.load(f'{V2}/activations/dspace/rp2528_{k}.npz', allow_pickle=True)
            X = np.mean([d[f'p{j}_L{L}'] for j in range(3)], 0).astype(np.float32)
            w = celldim(X, d['y'].astype(float), d['cell'])
            if w is not None:
                per.append(w)
            if k in ('C', 'BC'):
                Xc_for_sigma = X
        if not per:
            print(f"{outstem} L{L}: no DIMs")
            continue
        wens = u(np.mean(per, 0))

        # stability: split-half on the first key
        d = np.load(f'{V2}/activations/dspace/rp2528_{keys[0]}.npz', allow_pickle=True)
        X = np.mean([d[f'p{j}_L{L}'] for j in range(3)], 0).astype(np.float32)
        y = d['y'].astype(float)
        cells = d['cell']
        rng = np.random.RandomState(0)
        p = rng.permutation(len(y))
        wa = celldim(X[p[:len(p) // 2]], y[p[:len(p) // 2]], cells[p[:len(p) // 2]])
        wb = celldim(X[p[len(p) // 2:]], y[p[len(p) // 2:]], cells[p[len(p) // 2:]])
        stab = float(u(wa) @ u(wb)) if (wa is not None and wb is not None) else float('nan')
        sig = float((Xc_for_sigma @ wens).std())

        import os
        out = f'{V2}/activations/dspace/{outstem}_L{L}.npz'
        if os.path.exists(out):
            out = out.replace('.npz', '_v2rp.npz')
        np.savez(out, direction=wens.astype(np.float32), layer=np.array([L]),
                 mean=Xc_for_sigma.mean(0).astype(np.float32), sigma=np.array([sig], np.float32), scale=np.ones(4096, np.float32))
        line = f"{outstem}_L{L}: stab={stab:.2f} sigma={sig:.2f}"

        # sanity cos to existing axes
        import os
        for ref in [f'{outstem}_L21.npz',
                    f'twoprobe_ens_L28.npz' if outstem == 'twoprobe_ens' and L == 28 else None]:
            if ref and os.path.exists(f'{V2}/activations/dspace/{ref}'):
                wr = u(np.load(f'{V2}/activations/dspace/{ref}')['direction'].astype(np.float32))
                line += f" cos({ref.replace('.npz','')})={wens@wr:+.2f}"
        print(line)


build(['C', 'H'], 'twoprobe_ens')          # overwrites nothing at 21; writes L25 + L28 (L28 sanity vs existing)
build(['BC', 'BH'], 'twoprobe_opus_ens')
print("done")
