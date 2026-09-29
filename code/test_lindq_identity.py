"""Numpy-only identity/mechanics test for the lindq (signed linear) controller.

Checks, for the frozen host write vector w = w_ridge_raw (L14):
  1. v1 = -lam*dQ*w/||w||^2  ==  v2 = -lam*(dQ/||w||)*(w/||w||)
  2. driver mechanics: ed=-lam*dQ/||w||, ds=ed/GATE_DOSE_RAW, applied=(GATE_DOSE_RAW*ds)*W_UNIT == v1
  3. probe check: F(h)=w.h+b  ->  F(h+applied)-F(h) == -lam*dQ  (lam=1 exactly cancels the read)
  4. sign: dQ<0 -> applied.w>0 (F increases); dQ>0 -> F decreases
"""
import numpy as np
from ss_paths import SS_ROOT   # portable roots

QSV = f'{SS_ROOT}/v2/reports/subdim_0726/three_model_host_native_value_0829/qsv_direction_L14.npz'
GATE_DOSE_RAW = 150.0
HOST_RMS = 1832.93

w = np.load(QSV)['w_ridge_raw'].astype(np.float64)
wn = float(np.linalg.norm(w))
W_UNIT = w / wn
rng = np.random.default_rng(0)
dqs = np.concatenate([rng.normal(size=200), [2.5, -2.5]])

n_checks = 0
for lam in (0.25, 1.0, 4.0):
    for dq in dqs:
        v1 = -lam * dq * w / wn**2
        v2 = -lam * (dq / wn) * (w / wn)
        assert np.allclose(v1, v2, rtol=1e-12, atol=0), (lam, dq)
        # driver mechanics (gate_decide -> STATE['gate_sched'] -> edit hook)
        ed = -lam * dq / wn
        ds = ed / GATE_DOSE_RAW
        applied = (GATE_DOSE_RAW * ds) * W_UNIT
        assert np.allclose(applied, v1, rtol=1e-12, atol=0), (lam, dq)
        assert np.isclose(np.linalg.norm(applied), abs(ed), rtol=1e-12), (lam, dq)
        # probe / sign checks
        h = rng.normal(size=w.shape) * HOST_RMS
        b = rng.normal()
        F = lambda x: float(w @ x + b)
        dF = F(h + applied) - F(h)
        assert np.isclose(dF, -lam * dq, rtol=1e-9, atol=1e-9), (lam, dq, dF, -lam * dq)
        if dq < 0:
            assert applied @ w > 0, (lam, dq)
        elif dq > 0:
            assert applied @ w < 0, (lam, dq)
        n_checks += 1

print(f"w shape={w.shape}  ||w||={wn:.6f}  1/||w||={1/wn:.6f}")
print(f"all {n_checks} (lam, dQ) cases passed: v1==v2, driver applied==v1, F(h+applied)-F(h)==-lam*dQ, sign OK")
print(f"lam=1 edit L2 norm |ed| = |dQ|/||w||   (host_rms={HOST_RMS})")
for dq in (-2.0, -0.5, 0.5, 2.0):
    ed = -1.0 * dq / wn
    print(f"  dQ={dq:+.1f}: ed={ed:+.4f}  |ed|={abs(ed):.4f}  |ed|/host_rms={abs(ed)/HOST_RMS:.3e}  "
          f"(vs dynqsv fixed dose 150: {abs(ed)/150.0:.3e}x)")
print("OK")
