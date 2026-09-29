#!/usr/bin/env python3
"""IMPONLY V2 axis (0730) — the impossible-math-only self-rating axis at FULL-pool support.

EXACT V1 construction (reverse-verified cos=1.0000 against preDIM_QB_imponly_L21.npz):
  - z-score with mean/std of the FULL 3-arm eval prestate pool (r2_prestates_QB.npz, 1500 rows)
  - tercile split (70/30 percentiles of selfeval_num) WITHIN the impossible_math support set
    (fallback = A11 fixed split ==100 vs <=70 when terciles collapse, GLM saturation mode)
  - DIM in z-space, unit norm, NO sink mask (V1 convention; massive_load reported for the record)
  - eval in z-space on the full pool: read_r = corr(Z@w, y), sigma = (Z@w).std()
V1 used the 500-subsample imp rows; V2 uses the full impfull extraction (~3.1k rows).

Stats per layer: n_hi/n_lo, read_r full pool + per-arm + out-of-arm, sigma, cos to V1 (L21) and
canonical preDIM_QB_L*, split-half reliability (100 random half-splits, axis per half, cos).
Provenance gate: the first 500 impfull rows must reproduce the eval pool's imp-arm rows (same
seed-0 selection): labels identical + mean |proj diff| small.

Usage (Qwen default; GLM via env):
  I2_IMPFULL=...npz I2_EVALPOOL=...npz I2_OUT_PREFIX=.../preDIM_QB_imponly2_L I2_LAYERS=15,21,25 \
  I2_V1=.../preDIM_QB_imponly_L21.npz I2_CANON_PREFIX=.../preDIM_QB_L python3 imponly2_build_axis.py
"""
import os

import numpy as np
from ss_paths import SS_ROOT   # portable roots

u = lambda v: v / (np.linalg.norm(v) + 1e-12)
V2 = f'{SS_ROOT}/v2'
D = f'{V2}/activations/dspace'

IMPFULL = os.environ.get('I2_IMPFULL', f'{D}/r2_prestates_QB_impfull.npz')
EVALPOOL = os.environ.get('I2_EVALPOOL', f'{D}/r2_prestates_QB.npz')
OUT_PREFIX = os.environ.get('I2_OUT_PREFIX', f'{D}/preDIM_QB_imponly2_L')
LAYERS = [int(x) for x in os.environ.get('I2_LAYERS', '15,21,25').split(',')]
V1_PATH = os.environ.get('I2_V1', f'{D}/preDIM_QB_imponly_L21.npz')
CANON_PREFIX = os.environ.get('I2_CANON_PREFIX', f'{D}/preDIM_QB_L')
ARM = os.environ.get('I2_ARM', 'impossible_math')
NSPLITS = 100

imp = np.load(IMPFULL, allow_pickle=True)
ev = np.load(EVALPOOL, allow_pickle=True)
yi = imp['y'].astype(float)
ye = ev['y'].astype(float)
ce = np.array([str(c) for c in ev['cell']])
assert all(str(c) == ARM for c in imp['cell']), 'impfull contains non-target arm rows'
n = len(yi)
print(f'[i2] impfull {IMPFULL}: {n} {ARM} rows; eval pool {EVALPOOL}: {len(ye)} rows '
      f'({ {c: int((ce==c).sum()) for c in np.unique(ce)} })')
print(f'[i2] impfull labels: ==100:{int((yi==100).sum())} 90-99:{int(((yi>=90)&(yi<100)).sum())} '
      f'71-89:{int(((yi>70)&(yi<90)).sum())} <=70:{int((yi<=70).sum())} '
      f'mean {yi.mean():.1f} med {np.median(yi):.0f} std {yi.std():.1f}')

# provenance gate on labels: eval pool's ARM rows == first n_arm impfull rows (same seed-0 order)
mi_ev = ce == ARM
n_ev_arm = int(mi_ev.sum())
if n_ev_arm and np.allclose(ye[mi_ev], yi[:n_ev_arm]):
    print(f'[i2] provenance gate: eval-pool {ARM} labels == impfull[:{n_ev_arm}]  PASS')
else:
    print(f'[i2] provenance gate: LABEL MISMATCH vs eval pool ({n_ev_arm} arm rows) — CHECK')


def build(Zi_rows, y_rows):
    """axis from z-scored support rows; returns w, mode, n_hi, n_lo, (hi, lo)."""
    hi, lo = np.nanpercentile(y_rows, [70, 30])
    mode = 'orig_terciles'
    if hi <= lo:
        hi, lo, mode = 100.0, 70.0, 'a11_fixed_100_70'
        mh, ml = y_rows >= hi, y_rows <= lo
    else:
        mh, ml = y_rows >= hi, y_rows <= lo
    if mh.sum() < 15 or ml.sum() < 15:
        return None, mode, int(mh.sum()), int(ml.sum()), (hi, lo)
    return u(Zi_rows[mh].mean(0) - Zi_rows[ml].mean(0)), mode, int(mh.sum()), int(ml.sum()), (hi, lo)


rng = np.random.RandomState(0)
for L in LAYERS:
    Xi = imp[f'L{L}'].astype(np.float32)
    Xe = ev[f'L{L}'].astype(np.float32)
    mu, sd = Xe.mean(0), Xe.std(0) + 1e-6           # V1 convention: full-pool z stats
    sink = np.where(sd > 8.0 * np.median(sd))[0]
    Zi = (Xi - mu) / sd
    Ze = (Xe - mu) / sd
    w, mode, n_hi, n_lo, (hi, lo) = build(Zi, yi)
    if w is None:
        print(f'[L{L}] UNBUILDABLE (n_hi={n_hi} n_lo={n_lo})')
        continue
    proj = Ze @ w
    read_r = float(np.corrcoef(proj, ye)[0, 1])
    sigma = float(proj.std())
    per_arm = {c: float(np.corrcoef(proj[ce == c], ye[ce == c])[0, 1]) for c in np.unique(ce)}
    ooa = float(np.corrcoef(proj[~mi_ev], ye[~mi_ev])[0, 1])   # out-of-arm rows only
    ins = float(np.corrcoef(Zi @ w, yi)[0, 1])                 # in-support read
    mload = float((w[sink] ** 2).sum()) if len(sink) else 0.0
    # provenance gate on activations: recomputed first-500 projections vs eval pool's arm rows
    prov = float(np.mean(np.abs(Zi[:n_ev_arm] @ w - proj[mi_ev]))) if n_ev_arm else float('nan')
    cos_v1 = float('nan')
    if os.path.exists(V1_PATH) and L == 21:
        cos_v1 = float(w @ np.load(V1_PATH)['direction'])
    cpath = f'{CANON_PREFIX}{L}.npz'
    cos_can = float(w @ np.load(cpath)['direction']) if os.path.exists(cpath) else float('nan')
    # split-half reliability: 100 random half-splits of the support rows
    coss = []
    for _ in range(NSPLITS):
        p = rng.permutation(n)
        a, b = p[:n // 2], p[n // 2:]
        wa = build(Zi[a], yi[a])[0]
        wb = build(Zi[b], yi[b])[0]
        if wa is not None and wb is not None:
            coss.append(float(wa @ wb))
    coss = np.array(coss)
    np.savez(f'{OUT_PREFIX}{L}.npz', direction=w.astype(np.float32),
             mean=np.float32(proj.mean()), sigma=np.array([sigma], np.float32),
             read_r=np.array([read_r], np.float32), layer=np.array([L]),
             n_hi=np.array([n_hi]), n_lo=np.array([n_lo]), n_support=np.array([n]),
             hi_thresh=np.array([hi], np.float32), lo_thresh=np.array([lo], np.float32),
             split_mode=np.array([mode]), massive_load=np.array([mload], np.float32),
             split_half_mean=np.array([coss.mean()], np.float32),
             split_half_min=np.array([coss.min()], np.float32),
             cos_v1=np.array([cos_v1], np.float32), cos_canonical=np.array([cos_can], np.float32))
    print(f'[L{L}] {mode} hi>={hi:.0f}(n={n_hi}) lo<={lo:.0f}(n={n_lo}) | '
          f'read_r_full={read_r:+.4f} sigma={sigma:.3f} | in-support r={ins:+.3f} '
          f'out-of-arm r={ooa:+.3f} per-arm { {k: round(v,3) for k,v in per_arm.items()} } | '
          f'cos_v1={cos_v1:+.4f} cos_canon={cos_can:+.4f} | '
          f'split-half {coss.mean():.3f} (min {coss.min():.3f}, n={len(coss)}) | '
          f'massive_load={100*mload:.2f}% (n_sink={len(sink)}) | prov_gate={prov:.4f} | '
          f'-> {OUT_PREFIX}{L}.npz')
print('done')
