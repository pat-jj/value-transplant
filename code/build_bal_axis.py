#!/usr/bin/env python3
"""General BALANCED self-rating-axis builder (0731): per-arm matched-count high(==100)/low(<=70) DIM,
same recipe as preDIM_G20BC_BAL (seed-0 subsample the larger band to the smaller per arm; NMIN 15;
within-pool z-score, sink-dim zeroing, unit norm). Reports per-arm matched bands, read_r (on the
FULL pool), split-half (matched-subsample seed-0 halves), sigma. Also cos vs a list of reference
axes. Reuses the verbatim celldim_fixed math from dbg_glm_predim_0729.
usage: build_bal_axis.py --prestates <npz> --layers 15,21,25,28 --out-prefix <path/PRE_L>
       [--refs name=npz,name=npz ...]"""
import argparse
import numpy as np

u = lambda v: v / (np.linalg.norm(v) + 1e-12)


def matched_dim(X, y, cells, HI=100.0, LO=70.0, NMIN=15, seed=0, verbose=True):
    rng = np.random.RandomState(seed)
    s = X.std(0) + 1e-6
    massive = np.where(s > 8.0 * np.median(s))[0]
    Z = (X - X.mean(0)) / s
    ws = []
    for c in np.unique(cells):
        m = (cells == c) & (~np.isnan(y))
        hi = np.where(m & (y >= HI))[0]
        lo = np.where(m & (y <= LO))[0]
        k = min(len(hi), len(lo))
        if verbose:
            print(f'    cell {c}: hi={len(hi)} lo={len(lo)} matched={k}'
                  + ('' if k >= NMIN else '  <-- SKIP (<15)'))
        if k < NMIN:
            continue
        hi = rng.permutation(hi)[:k]
        lo = rng.permutation(lo)[:k]
        ws.append(u(Z[hi].mean(0) - Z[lo].mean(0)))
    if not ws:
        return None, massive
    w = u(np.mean(ws, 0))
    if len(massive):
        w[massive] = 0.0
    return u(w), massive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prestates', required=True)
    ap.add_argument('--layers', required=True)
    ap.add_argument('--out-prefix', required=True)
    ap.add_argument('--refs', default='')
    ap.add_argument('--hi', type=float, default=100.0)
    ap.add_argument('--lo', type=float, default=70.0)
    args = ap.parse_args()
    ps = np.load(args.prestates, allow_pickle=True)
    y = ps['y'].astype(float)
    cells = ps['cell']
    layers = [int(x) for x in args.layers.split(',')]
    refs = {}
    for kv in args.refs.split(',') if args.refs else []:
        name, p = kv.split('=')
        refs[name] = u(np.load(p)['direction'].astype(np.float32))
    print(f'[bal] {args.prestates}  n={len(y)} ==100:{int((y==100).sum())} <=70:{int((y<=70).sum())}')
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(y))
    h1, h2 = idx[:len(y) // 2], idx[len(y) // 2:]
    for L in layers:
        X = ps[f'L{L}'].astype(np.float32)
        print(f'  [L{L}] matched bands:')
        w, mv = matched_dim(X, y, cells, HI=args.hi, LO=args.lo)
        if w is None:
            print(f'  [L{L}] no usable arms')
            continue
        proj = X @ w
        r = float(np.corrcoef(proj, y)[0, 1]) if proj.std() > 0 else 0.0
        sig = float(proj.std())
        wa, _ = matched_dim(X[h1], y[h1], cells[h1], HI=args.hi, LO=args.lo, verbose=False)
        wb, _ = matched_dim(X[h2], y[h2], cells[h2], HI=args.hi, LO=args.lo, verbose=False)
        sh = float(np.dot(wa, wb)) if (wa is not None and wb is not None) else float('nan')
        mload = float((w[mv] ** 2).sum()) if len(mv) else 0.0
        outp = f'{args.out_prefix}{L}.npz'
        np.savez(outp, direction=w.astype(np.float32), layer=np.array([L]),
                 mean=X.mean(0).astype(np.float32), sigma=np.array([sig], np.float32),
                 scale=np.ones(X.shape[1], np.float32), read_r=np.array([r], np.float32),
                 massive_load=np.array([mload], np.float32), n_massive=np.array([len(mv)]),
                 gate_pass=np.array([abs(r) >= 0.30 and mload < 0.02]),
                 split_mode=np.array(['bal_matched_100_70']))
        cosr = '  '.join(f'cos({n})={float(np.dot(w, rv)):+.3f}' for n, rv in refs.items()
                         if len(rv) == len(w))
        print(f'  [L{L}] read_r={r:+.3f} sigma={sig:8.2f} split-half={sh:.3f} '
              f'sink={len(mv)} -> {outp}\n          {cosr}')


if __name__ == '__main__':
    main()
