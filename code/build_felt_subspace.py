#!/usr/bin/env python3
"""MULTI-DIMENSION self-rating subspaces (0726, "different dimensions/subspace to transplant
felt-success"). Builds K-dimensional ORTHONORMAL self-rating subspaces from the EXACT single-axis recipe
inputs — no new pools, no relabeling, one construction change at a time:

  S1 `cellsvd` (K<=3): the single axis preDIM_* is the MEAN of the per-cell DIM directions
     (build_denc_matrix celldim_std / llr_build_predim_fixed_0725 celldim_fixed). Keep the
     per-cell DIM STACK instead and take its SVD -> top-K components. Top dim ~= the old axis
     (verified cos vs preDIM printed); dims 2..3 = how felt varies ACROSS task contexts.
  S2 `pls` (K<=16): PLS1-with-deflation on the same z-scored, within-cell-centered states vs the
     model's OWN selfeval number (never a judge grade — 0724 rule). Dim k = the direction most
     covarying with felt AFTER dims 1..k-1 are removed. DIM ~= PLS dim 1.
  `rand` (K=16): random orthonormal control subspace, same sink-zeroing, fixed seed.

All directions live in the same convention as the shipped axes (z-space direction used as raw push
vector, attention-sink dims zeroed, unit norm — see build_denc_matrix docstring), then the
stack is re-orthonormalized (SVD). Saved npz per (model, frame, construction, layer):
  directions [K,4096] orthonormal rows | mean [4096] pool mean | sigma [K] std of X@d_k |
  read_r [K] corr(X@d_k, y) | sv [K] singular/cov values | layer | note
QWEN (recipe A, cheater frame): preprompt_felt_0720.npz (L15,L21) + prestates_C.npz (L25,L28).
LLAMA host frame  = llr_prestates_C.npz (cheater, fixed bands HI==100/LO<=70, guards 15/15).
LLAMA donor frame = llr_prestates_H.npz (honest), then donor dims greedy-|cos|-MATCHED + sign-
aligned to the host dims (row k of the matched donor npz corresponds to host dim k; match_cos
saved — low-cos rows mean the frames don't share that dimension and get flagged)."""
import numpy as np
from ss_paths import SS_ROOT   # portable roots

u = lambda v: v / (np.linalg.norm(v) + 1e-12)
V2 = f'{SS_ROOT}/v2'
A = f'{V2}/activations/dspace'
KMAX_PLS = 16


def _std_mask(X):
    s = X.std(0) + 1e-6
    massive = np.where(s > 8.0 * np.median(s))[0]
    return s, massive


def cell_masks_qwen(y, cells):
    """EXACT celldim_std cell guards (n>=30, std>=5, 70/30 percentiles) -> per-cell (mask, hi, lo)."""
    out = []
    for c in np.unique(cells):
        m = (cells == c) & (~np.isnan(y))
        if m.sum() < 30 or np.nanstd(y[m]) < 5:
            continue
        hi, lo = np.nanpercentile(y[m], [70, 30])
        if hi <= lo:
            continue
        out.append((c, m, m & (y >= hi), m & (y <= lo)))
    return out


def cell_masks_llama(y, cells):
    """EXACT celldim_fixed guards (fixed bands HI==100 / LO<=70, n>=15/15)."""
    out = []
    for c in np.unique(cells):
        m = (cells == c) & (~np.isnan(y))
        hi = m & (y >= 100.0)
        lo = m & (y <= 70.0)
        if hi.sum() < 15 or lo.sum() < 15:
            continue
        out.append((c, m, hi, lo))
    return out


def orthonormalize(W, massive):
    """Sink-zero every direction, then SVD re-orthonormalization of the stack. Rows of the returned
    matrix are orthonormal and span the same (sink-zeroed) space; ordered by singular value."""
    Wm = np.array(W, dtype=np.float64)
    if len(massive):
        Wm[:, massive] = 0.0
    U_, S_, Vt = np.linalg.svd(Wm, full_matrices=False)
    keep = S_ > 1e-8
    return Vt[keep], S_[keep]


def gram_schmidt(W, massive):
    """Sink-zero then ORDER-PRESERVING orthonormalization (for PLS: dim k must stay 'the k-th self-rating
    direction', which an SVD re-mix would scramble). Drops rows whose residual after projecting out
    earlier dims is ~0. Returns (D, residual_norms)."""
    D, rn = [], []
    for w in np.array(W, dtype=np.float64):
        if len(massive):
            w = w.copy()
            w[massive] = 0.0
        for d in D:
            w = w - (w @ d) * d
        n = np.linalg.norm(w)
        if n > 1e-8:
            D.append(w / n)
            rn.append(n)
    return np.array(D), np.array(rn)


def build_cellsvd(X, y, cells, masks, massive):
    Z = (X - X.mean(0)) / (X.std(0) + 1e-6)
    ws = [u(Z[hi].mean(0) - Z[lo].mean(0)) for (_, _, hi, lo) in masks]
    D, sv = orthonormalize(ws, massive)
    return D, sv


def build_pls(X, y, cells, masks, massive, K=KMAX_PLS):
    """PLS1 with deflation on z-scored, WITHIN-CELL-centered data (cell = task category, exactly the
    within-cell logic of the recipe). Directions collected in z-space (recipe convention: z-space
    direction used directly as raw push vector), then sink-zeroed + re-orthonormalized."""
    s = X.std(0) + 1e-6
    rows = np.zeros(len(y), bool)
    for (_, m, _, _) in masks:
        rows |= m
    Z = (X[rows] - X.mean(0)) / s
    yv = y[rows].astype(np.float64).copy()
    cc = cells[rows]
    for c in np.unique(cc):                      # within-cell centering (both sides)
        m = cc == c
        Z[m] -= Z[m].mean(0)
        yv[m] -= yv[m].mean()
    Zc = Z.astype(np.float64)
    ws = []
    for k in range(K):
        w = Zc.T @ yv
        if np.linalg.norm(w) < 1e-10:
            break
        w = u(w)
        t = Zc @ w
        tt = float(t @ t)
        if tt < 1e-12:
            break
        p = Zc.T @ t / tt
        ws.append(w)
        Zc = Zc - np.outer(t, p)                 # deflate X
        yv = yv - t * (float(t @ yv) / tt)       # deflate y
    D, sv = gram_schmidt(ws, massive)            # order-preserving: dim k = k-th self-rating direction
    return D[:K], sv[:K]


def build_rand(dmodel, massive, K=KMAX_PLS, seed=0):
    rng = np.random.RandomState(seed)
    D, sv = orthonormalize(rng.randn(K, dmodel), massive)
    return D, sv


def save_sub(path, D, sv, X, y, label):
    proj = X @ D.T                                        # [n, K]
    sig = proj.std(0)
    yv = y.astype(float)
    ok = ~np.isnan(yv)
    rr = np.array([float(np.corrcoef(proj[ok, k], yv[ok])[0, 1]) if proj[ok, k].std() > 0 else 0.0
                   for k in range(D.shape[0])])
    for k in range(D.shape[0]):                           # sign: each dim reads felt UP
        if rr[k] < 0:
            D[k] = -D[k]
            rr[k] = -rr[k]
    np.savez(path, directions=D.astype(np.float32), mean=X.mean(0).astype(np.float32),
             sigma=sig.astype(np.float32), read_r=rr.astype(np.float32),
             sv=np.asarray(sv, np.float32), layer=np.array([int(path.split('_L')[-1].split('.')[0])]))
    print(f"{label}: K={D.shape[0]} read_r={np.round(rr, 3).tolist()} sigma={np.round(sig, 2).tolist()}")
    return D


def match_frames(DH, DC, path_out, XH, yH, label):
    """Greedy |cos| matching of donor(honest) dims DH onto host(cheater) dims DC + sign alignment;
    saves donor dims REORDERED so row k reads the donor in the frame of host dim k."""
    K = min(len(DH), len(DC))
    M = DC[:K] @ DH.T                                     # [K_host, K_donor] cosines
    used, pairs, cosv = set(), [], []
    for i in range(K):
        j = int(np.argmax([abs(M[i, jj]) if jj not in used else -1.0 for jj in range(len(DH))]))
        used.add(j)
        pairs.append((j, 1.0 if M[i, j] >= 0 else -1.0))
        cosv.append(abs(float(M[i, j])))
    Dm = np.array([DH[j] * sgn for (j, sgn) in pairs])
    proj = XH @ Dm.T
    sig = proj.std(0)
    yv = yH.astype(float)
    ok = ~np.isnan(yv)
    rr = np.array([float(np.corrcoef(proj[ok, k], yv[ok])[0, 1]) if proj[ok, k].std() > 0 else 0.0
                   for k in range(K)])
    np.savez(path_out, directions=Dm.astype(np.float32), mean=XH.mean(0).astype(np.float32),
             sigma=sig.astype(np.float32), read_r=rr.astype(np.float32),
             match_cos=np.asarray(cosv, np.float32),
             layer=np.array([int(path_out.split('_L')[-1].split('.')[0])]))
    print(f"{label}: match_cos={np.round(cosv, 3).tolist()}")
    lowc = [k for k, c in enumerate(cosv) if c < 0.30]
    if lowc:
        print(f"  WARN dims {lowc} match_cos<0.30 — frames do not share those dimensions")


# ===================== QWEN (recipe A frame, cheater pool) =====================
print('===== QWEN (recipe A: preprompt_felt_0720 + prestates_C) =====')
pf = np.load(f'{A}/preprompt_felt_0720.npz', allow_pickle=True)
LP = list(pf['layers'])
yQ = pf['yC'].astype(float)
cQ = pf['cellC']
XQ = {15: pf['C_pre'][:, LP.index(15)].astype(np.float32),
      21: pf['C_pre'][:, LP.index(21)].astype(np.float32)}
ps = np.load(f'{A}/prestates_C.npz', allow_pickle=True)
assert np.allclose(ps['y'].astype(float), yQ), 'Qwen row alignment failed'
XQ[25] = ps['L25'].astype(np.float32)
XQ[28] = ps['L28'].astype(np.float32)
for L in [15, 21, 25, 28]:
    X = XQ[L]
    _, massive = _std_mask(X)
    masks = cell_masks_qwen(yQ, cQ)
    D1, sv1 = build_cellsvd(X, yQ, cQ, masks, massive)
    D1 = save_sub(f'{A}/subQ_cellsvd_L{L}.npz', D1, sv1, X, yQ, f'subQ_cellsvd_L{L}')
    D2, sv2 = build_pls(X, yQ, cQ, masks, massive)
    save_sub(f'{A}/subQ_pls_L{L}.npz', D2, sv2, X, yQ, f'subQ_pls_L{L}')
    Dr, svr = build_rand(X.shape[1], massive, seed=L)
    save_sub(f'{A}/subQ_rand_L{L}.npz', Dr, svr, X, yQ, f'subQ_rand_L{L}')
    old = np.load(f'{A}/preDIM_A_L{L}.npz')['direction'].astype(np.float64)
    print(f'  gates L{L}: cos(cellsvd[0], preDIM_A)={abs(float(D1[0] @ u(old))):.4f}  '
          f'cos(pls[0], preDIM_A)={abs(float(np.load(f"{A}/subQ_pls_L{L}.npz")["directions"][0].astype(np.float64) @ u(old))):.4f}')

# ===================== LLAMA (host=cheater C frame; donor=honest H frame) =====================
print('===== LLAMA (llr_prestates_C / _H, fixed bands) =====')
for tag, pool in (('C', 'llr_prestates_C.npz'), ('H', 'llr_prestates_H.npz')):
    z = np.load(f'{A}/{pool}', allow_pickle=True)
    yL = z['y'].astype(float)
    cL = z['cell']
    for L in [13, 19, 22, 25]:
        X = z[f'L{L}'].astype(np.float32)
        _, massive = _std_mask(X)
        masks = cell_masks_llama(yL, cL)
        D1, sv1 = build_cellsvd(X, yL, cL, masks, massive)
        save_sub(f'{A}/subL_cellsvd_{tag}_L{L}.npz', D1, sv1, X, yL, f'subL_cellsvd_{tag}_L{L}')
        D2, sv2 = build_pls(X, yL, cL, masks, massive)
        save_sub(f'{A}/subL_pls_{tag}_L{L}.npz', D2, sv2, X, yL, f'subL_pls_{tag}_L{L}')
        if tag == 'C':
            Dr, svr = build_rand(X.shape[1], massive, seed=100 + L)
            save_sub(f'{A}/subL_rand_L{L}.npz', Dr, svr, X, yL, f'subL_rand_L{L}')
            old = np.load(f'{A}/preDIM_LL_L{L}.npz')['direction'].astype(np.float64)
            print(f'  gate L{L}: cos(cellsvd_C[0], preDIM_LL)={abs(float(D1[0].astype(np.float64) @ u(old))):.4f}')

# cross-frame matching: donor(H) dims reordered/sign-aligned onto host(C) dims
print('===== LLAMA cross-frame matching (donor H -> host C dim order) =====')
zH = np.load(f'{A}/llr_prestates_H.npz', allow_pickle=True)
for L in [13, 19, 22, 25]:
    DC = np.load(f'{A}/subL_pls_C_L{L}.npz')['directions'].astype(np.float64)
    DH = np.load(f'{A}/subL_pls_H_L{L}.npz')['directions'].astype(np.float64)
    match_frames(DH, DC, f'{A}/subL_pls_Hm_L{L}.npz', zH[f'L{L}'].astype(np.float32),
                 zH['y'].astype(float), f'subL_pls_Hm_L{L}')
print('done')
