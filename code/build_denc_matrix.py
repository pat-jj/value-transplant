#!/usr/bin/env python3
"""Build STANDARDIZED (massive-activation-safe) d_enc and pre-prompt DIM at L15/21/25/28 for recipe A
(cheater) and B (cheateropp). 0721 decision: standardize-then-steer.

WHY: at L21+ the pre-prompt state variance is concentrated in a few attention-sink ("massive
activation") dims. A RAW covariance DIM/encoder there points at those dims: it reads
felt poorly and its sigma is very large -> steering perturbs attention sinks and shatters the model
by construction. Standardizing (z = (h-mu)/s) and zeroing the sink dims recovers a clean self-rating reader
that stays on the self-rating subspace (NOT the full-whitened decoder). See notes/linearization_proof_0720.md SIGMA AUDIT.

Construction per (pool X, target t in {y self-rating, f* elicited readout}):
  s = X.std(0); massive = dims with s > 8*median(s); Z = (X-mu)/s
  w_z = within-cell tercile/covariance direction computed on Z  (down-weights high-variance sink dims)
  direction = normalize(w_z with massive dims zeroed)           (belt-and-suspenders on the sinks)
  sigma = std(X @ direction)  (comparable across layers) ; dose = alpha * sigma
Gate: require read_r >= 0.30 and massive_load < 0.02 else print WARN (still saved, flagged in npz)."""
import numpy as np
from ss_paths import SS_ROOT   # portable roots

# unit-normalize a vector (eps guards the zero vector)
u = lambda v: v / (np.linalg.norm(v) + 1e-12)
V2 = f'{SS_ROOT}/v2'


def _std_mask(X):
    # per-dim std (eps-floored) and the "massive activation" sink dims (std > 8x the median std)
    s = X.std(0) + 1e-6
    massive = np.where(s > 8.0 * np.median(s))[0]
    return s, massive


def _finalize(X, w_z, massive):
    """w_z is a direction in z-coords; use directly as raw push vector, zero sink dims, renormalize."""
    w = w_z.copy()
    if len(massive):
        w[massive] = 0.0
    return u(w)


def celldim_std(X, y, cells):
    # within-cell tercile DIM (hi-lo mean diff) on z-scored states, averaged over cells
    s, massive = _std_mask(X)
    Z = (X - X.mean(0)) / s
    ws = []
    for c in np.unique(cells):
        m = (cells == c) & (~np.isnan(y))
        # skip cells with too few rows or too little label spread
        if m.sum() < 30 or np.nanstd(y[m]) < 5:
            continue
        hi, lo = np.nanpercentile(y[m], [70, 30])
        if hi <= lo:
            continue
        ws.append(u(Z[m][y[m] >= hi].mean(0) - Z[m][y[m] <= lo].mean(0)))
    if not ws:
        return None, massive
    return _finalize(X, u(np.mean(ws, 0)), massive), massive


def denc_std(X, f, cells):
    # within-cell encoder: accumulate cov(Z, f) after centering both sides within each cell
    s, massive = _std_mask(X)
    Z = (X - X.mean(0)) / s
    d = np.zeros(X.shape[1])
    for c in np.unique(cells):
        m = cells == c
        Zc = Z[m] - Z[m].mean(0)
        fc = f[m] - f[m].mean()
        d += Zc.T @ fc
    return _finalize(X, u(d), massive), massive


def save_axis(path, X, w, massive, target, tname, label):
    # read correlation, sink leakage, sigma; apply the gate; write npz + one-line report
    proj = X @ w
    r = float(np.corrcoef(proj, target)[0, 1]) if np.std(proj) > 0 else 0.0
    mload = float((w[massive] ** 2).sum()) if len(massive) else 0.0
    sig = float(proj.std())
    ok = abs(r) >= 0.30 and mload < 0.02
    np.savez(path, direction=w.astype(np.float32), layer=np.array([int(path.split('_L')[-1].split('.')[0])]),
             mean=X.mean(0).astype(np.float32), sigma=np.array([sig], np.float32), scale=np.ones(X.shape[1], np.float32),
             read_r=np.array([r], np.float32), massive_load=np.array([mload], np.float32), n_massive=np.array([len(massive)]),
             gate_pass=np.array([ok]))
    flag = '' if ok else '  <-- WARN (gate fail)'
    print(f"{label}: read_r({tname})={r:+.3f} sigma={sig:6.2f} massive_load={100*mload:4.1f}% n_sink={len(massive)}{flag}")


A = f'{V2}/activations/dspace'

# ===== recipe A =====
pf = np.load(f'{A}/preprompt_felt_0720.npz', allow_pickle=True)
LP = list(pf['layers'])
li21 = LP.index(21)
yA = pf['yC'].astype(float)
cellsA = pf['cellC']
# f* = post-interrupt elicited readout: project the L21 post state onto its own within-cell DIM
XpostA = np.mean([pf[f'C_post_p{j}'][:, li21] for j in range(3)], 0).astype(np.float32)
wpostA, _ = celldim_std(XpostA, yA, cellsA)
fA = XpostA @ (wpostA if wpostA is not None else np.zeros(XpostA.shape[1]))
# pre-prompt states per layer: L15/L21 from preprompt_felt, L25/L28 from prestates_C
preA = {15: pf['C_pre'][:, LP.index(15)].astype(np.float32), 21: pf['C_pre'][:, li21].astype(np.float32)}
ps = np.load(f'{A}/prestates_C.npz', allow_pickle=True)
assert np.allclose(ps['y'].astype(float), yA), "A row alignment failed"
preA[25] = ps['L25'].astype(np.float32)
preA[28] = ps['L28'].astype(np.float32)
for L in [15, 21, 25, 28]:
    # encoder toward f*, then the tercile DIM toward the self-rating y
    w, mv = denc_std(preA[L], fA, cellsA)
    save_axis(f'{A}/dencM_A_L{L}.npz', preA[L], w, mv, fA, 'f*', f'dencM_A_L{L}')
    w, mv = celldim_std(preA[L], yA, cellsA)
    if w is not None:
        save_axis(f'{A}/preDIM_A_L{L}.npz', preA[L], w, mv, yA, 'y', f'preDIM_A_L{L}')

# ===== recipe B =====
rp = np.load(f'{A}/readprobe_opus_natural.npz', allow_pickle=True)
LB = list(rp['layers'])
lb21 = LB.index(21)
yB = rp['yC'].astype(float)
cellsB = rp['cellC']
XpostB = np.mean([rp[f'C_p{j}'][:, lb21] for j in range(3)], 0).astype(np.float32)
wpostB, _ = celldim_std(XpostB, yB, cellsB)
fB = XpostB @ (wpostB if wpostB is not None else np.zeros(XpostB.shape[1]))
pb = np.load(f'{A}/prestates_BC.npz', allow_pickle=True)
assert np.allclose(pb['y'].astype(float), yB), "B row alignment failed"
for L in [15, 21, 25, 28]:
    Xp = pb[f'L{L}'].astype(np.float32)
    w, mv = denc_std(Xp, fB, cellsB)
    save_axis(f'{A}/dencM_B_L{L}.npz', Xp, w, mv, fB, 'f*', f'dencM_B_L{L}')
    w, mv = celldim_std(Xp, yB, cellsB)
    if w is not None:
        save_axis(f'{A}/preDIM_B_L{L}.npz', Xp, w, mv, yB, 'y', f'preDIM_B_L{L}')
print("done")
