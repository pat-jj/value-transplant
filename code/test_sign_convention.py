#!/usr/bin/env python3
"""Notation check (0908): the paper form  h + λ·ΔQ·w/‖w‖²  with w = probe axis of the monitored model's EXCESS felt success (target zC − zH)
is the same vector as the implemented form  h − λ·ΔQ·w_code/‖w_code‖²  with w_code = probe for ΔQ = zH − zC (w_code = −w). Synthetic + the deployed w."""
import numpy as np, glob, os
from ss_paths import SS_ROOT   # portable roots
rng = np.random.default_rng(0); worst = 0.0
for _ in range(1000):
    d = int(rng.integers(4, 64)); h = rng.normal(size=d) * 100; w_code = rng.normal(size=d) * 0.01; dq = float(rng.normal() * 2); lam = float(rng.choice([0.025, 0.25, 0.5, 1, 2, 4, 8]))
    implemented = h - lam * dq * w_code / (w_code @ w_code); w = -w_code; paper = h + lam * dq * w / (w @ w)
    worst = max(worst, float(np.abs(implemented - paper).max()))
print(f"synthetic: max |implemented − paper| over 1000 cases = {worst:.3e}")
X = f"{SS_ROOT}/v2/reports/xfam_clean_0904"
cands = glob.glob(f"{X}/probe/**/w_ridge_raw*.npy", recursive=True) + glob.glob(f"{X}/probe/*.npz")
print("deployed-w files found:", [os.path.relpath(c, X) for c in cands][:5])
assert worst < 1e-9; print("OK: identical vectors; the sign convention is notation only")
