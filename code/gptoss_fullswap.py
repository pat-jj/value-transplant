#!/usr/bin/env python3
"""gpt-oss FULL-SWAP driver (0731) = MINIMAL-DIFF COPY of the validated
vllm_lockstep_transplant_multi.py (standing order: never edit the validated ports in place).
Exactly THREE changes vs the original (verify with: diff vllm_lockstep_transplant_multi.py
gptoss_fullswap.py):
  1. build_q -> gpt-oss harmony template (reasoning_effort=high), same convention as
     gptoss_transplant_driver.py.
  2. fork-gate line gains the TRANSPLANT_SKIP_FORKGATE env skip (verbatim base-port 0731
     convention, vllm_lockstep_transplant_0727.py:438) so lam0 parity/anchor cells can run
     (at lam0 the edit is 0 by construction and the nonzero-edit gate must be skipped).
  3. --pos-arith: opt-in arithmetic last-n-tail positional gating (the BASE port's convention,
     install_host_edit) because gpt-oss decoder layers do NOT pass positions as inp[0] (first
     gate attempt died on a positions/hidden shape mismatch: inp[0] was the
     hidden states). Default OFF -> Qwen behavior bit-identical to the validated port; arith mode
     is gate-validated on gpt-oss via lam0 byte-parity vs the base-port driver + self-test
     equivalence of both modes on an identical tail.
Gate discipline before real cells (fsw_gate_0731.sbatch): --self-test, IDENT smoke w/ edits-fired
check, lam0 self-determinism, lam0 cross-port byte parity vs gptoss_transplant_driver.py.

Original header follows.

MULTI-LAYER / IDENT LOCKSTEP VALUE-MATCH TRANSPLANT ON vLLM (0727).

Extends the validated single-layer port (vllm_lockstep_transplant_v2.py, POSITIONAL host
gating) to cover EVERY transplant cell type the HF harness fsd_trace_transplant_subdim3.py
accepts via `--axes`:

  --axes "L15:npz,L21:IDENT,..."   (exactly the HF `--axes` grammar)
    * `L:npz`   -> per-layer K-dim analog value-match WRITE axis (npz `directions`[K,dm] or legacy
                   `direction`[dm]). Optional cross-frame donor READ axis via --donor-axes "L:npz".
    * `L:IDENT` -> FULL-DIMENSION (all 4096 dims) transplant at that layer:
                   h_host <- h_host + lam*(h_donor - h_host)  (lam=1 == activation-swap ceiling).

Design (identical mechanism to the validated ports, generalised per-layer):
  TWO vLLM engines, ONE GPU, ONE process (VLLM_ENABLE_V1_MULTIPROCESSING=0, enforce_eager so the
  nn.Module hooks fire, gpu_memory_utilization ~0.42 each). For EACH listed layer L:
    * DONOR engine: a READ/capture hook records the donor's per-position value at L
        - K-dim: the read-axis projection full_stream @ Dread.T  ([n,K])
        - IDENT: the FULL residual stream                        ([n,4096])
      into a per-layer, position-indexed history (hist_by_L[L][pos]). The donor is driven in
      LOCKSTEP by one-token prefix extensions with persistent paged KV (prefix caching) — captured
      VERBATIM from the base port (APC-on-one-token-extension already microbench-validated there).
    * HOST engine: a WRITE hook edits the host's residual OUTPUT at L for continuation positions
      only (positions tensor >= S-1, the recovered `vllm_gated_seq` architecture = the HF
      answer_from=S-1 convention), with L's own donor coords / axis / mu / multiplier semantics:
        - K-dim: h += [ lam*(srat*(donor_coord-muA)+bias-(host_coord-muB)) ] @ D   (== sub_trace_edit)
        - IDENT: h += lam*(donor_full - host_full)                                  (== the IDENT path)
  Coordinates/streams are read on the FULL residual stream out[0]+out[1] (vLLM decoder layers return
  (hidden, residual)); the delta written into out[0] is stream-equivalent. Editing residual OUTPUTS
  at multiple layers keeps each host cache internally self-consistent (the one-layer KV argument
  applied per layer, exactly as the HF multi hooks do).

  K=1 single-layer reproduces the base port bit-for-bit in the edit math (self-test asserts).

MEMORY: IDENT history is 4096 f32/pos/layer — stored on CPU (bounded GPU; ~0.5 GB CPU/layer for a
30k-token row, fine at 128G even for M=32). K-dim history stays on GPU (small, fast, == base port).
Two engines at 0.42+0.42; the per-step edit/capture tensors are tiny (tail<=~block_size rows). If
whole-activation IDENT at all 32 layers OOMs the GPU, validate on M1/M2/M4 (the knee) and document
the all-32 limit — do NOT hack around it.

VALIDATION (against HF ground truth):
  gate 1  single-layer self-rating via the multi interface (--axes L21:preDIM_QB_L21.npz, Qwen-B self-rating
          cell) must reproduce the base port's judged FAKE rate (regression guard vs the base port).
  gate 2  multi-layer IDENT: reproduce a landed HF lmix_identM{1,2,4} cell (mixed-Llama whole-layer
          swap, host mixc_origblend / donor llama honest, fork forkcommit_llama_0725,
          add_special_tokens=1 eos 128001, lam 1) at the judged FAKE-rate level (+-0.10).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from functools import partial
from pathlib import Path
from ss_paths import SS_ROOT   # portable roots

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # BOTH engine cores in THIS process
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
ROOT = Path(f"{SS_ROOT}/v2")
sys.path.insert(0, str(ROOT))
# VERBATIM math + plumbing reuse from the validated base port:
from vllm_lockstep_transplant_0727 import match_delta, _full_stream, gen1  # noqa: E402
# CHANGE 1/2 (0731 gpt-oss): harmony chat template instead of the Qwen build_q import.
def build_q(tok, prompt, enable_thinking=True):
    # harmony: reasoning_effort='high' (enable_thinking is not a gpt-oss template kwarg; ignored).
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, reasoning_effort="high")

# Module-level shared state (in-process; hooks installed via apply_model see this dict).
S = {
    "edit_on": False, "thr": 0, "lam": 0.0, "srat": 1.0, "bias": 0.0,
    # CHANGE 3/3 (0731 gpt-oss): positional-gating mode. "tensor" = original validated behavior
    # (positions tensor from inp[0]; correct for Qwen3 decoder layers). "arith" = derive the
    # absolute positions arithmetically as the last-n tail (start = seq_len - n), the BASE port's
    # convention (vllm_lockstep_transplant_0727.py install_host_edit), REQUIRED for gpt-oss whose
    # decoder layers do not take positions as inp[0] (inp[0] is hidden states there).
    "seq_len": 0, "pos_mode": "tensor",
    "layers": [],
    "write_by_L": {},   # L -> [K,dm] f32 gpu tensor | "IDENT"
    "read_by_L": {},    # L -> [K,dm] f32 gpu tensor | "IDENT"
    "muA_by_L": {}, "muB_by_L": {},                # L -> [K] tensor (device)
    "write_np": {}, "read_np": {}, "muA_np": {}, "muB_np": {},   # host-side numpy staging
    "hist_by_L": {},    # L -> {pos -> [K] or [dm] tensor (cpu for IDENT, gpu for K-dim)}
    "caps_by_L": {},    # L -> [captured tensors this donor call]
    "stats_by_L": {},   # L -> [signed_sum, abs_sum, n, abs_max]
}


# ---------------- hooks (per registered layer) --------------------------------------------
def make_donor_cap(L):
    """Capture the donor's value at layer L on the FULL residual stream: K-dim projection (read
    axis) or the full 4096-dim stream (IDENT). Appended to caps_by_L[L] for this donor forward."""
    def cap(module, inp, out):
        _, full = _full_stream(out)
        flat = full.reshape(-1, full.shape[-1])
        R = S["read_by_L"][L]
        if isinstance(R, str):                              # IDENT: store the full stream
            S["caps_by_L"][L].append(flat.detach())
        else:                                               # K-dim: store the read-axis coords
            S["caps_by_L"][L].append((flat @ R.T).detach())  # [n,K]
    return cap


def make_host_edit(L):
    """Positional-gated WRITE at layer L (recovered vllm_gated_seq architecture): rows whose absolute
    position (inp[0]) >= S['thr'] (= prompt_len-1) are edited using L's donor coords keyed by
    position. K-dim -> match_delta @ D (== sub_trace_edit); IDENT -> lam*(donor_full - host_full)."""
    def edit(module, inp, out):
        if not S["edit_on"]:
            return
        import torch
        h0, full = _full_stream(out)
        flat0 = h0.reshape(-1, h0.shape[-1])
        flat_full = full.reshape(-1, full.shape[-1])
        if S["pos_mode"] == "arith":                        # CHANGE 3/3: base-port convention
            n = flat0.shape[0]
            start = S["seq_len"] - n
            assert start >= 0, f"L{L} host tail longer than sequence: n={n} seq_len={S['seq_len']}"
            positions = torch.arange(start, S["seq_len"], device=flat0.device)
        else:
            positions = inp[0].reshape(-1)                  # [n] absolute positions (packed rows)
        assert positions.shape[0] == flat0.shape[0], \
            f"L{L} positions/hidden mismatch: {positions.shape} vs {flat0.shape}"
        thr = S["thr"]
        sel = torch.nonzero(positions >= thr).flatten()
        if sel.numel() == 0:
            return
        hist = S["hist_by_L"][L]
        pos_list = positions.index_select(0, sel).tolist()
        donor = torch.stack([hist[p] for p in pos_list]).to(flat0.device)   # [m,K] or [m,dm]
        rows = sel
        W = S["write_by_L"][L]
        host_full = flat_full.index_select(0, rows).float()
        if isinstance(W, str):                              # IDENT full-dimension
            delta = S["lam"] * (donor.float() - host_full)  # [m,dm]
            flat0.index_add_(0, rows, delta.to(flat0.dtype))
            coord = delta                                   # stats on the delivered per-dim change
        else:                                               # K-dim analog value-match
            ph = host_full @ W.T                            # [m,K] host coords on write basis
            pd = donor.float()                              # [m,K] donor coords on read basis
            muA, muB = S["muA_by_L"][L], S["muB_by_L"][L]
            if muA.device != ph.device:                     # xfam-bug class: mus live on hook device
                S["muA_by_L"][L] = muA = muA.to(ph.device)
                S["muB_by_L"][L] = muB = muB.to(ph.device)
            coord = match_delta(ph, pd, muA, muB, S["lam"], S["srat"], S["bias"])   # [m,K]
            flat0.index_add_(0, rows, (coord @ W).to(flat0.dtype))
        st = S["stats_by_L"][L]
        a = coord.abs().sum(-1)                             # [m] L1 delivered change per position
        st[0] += float(coord.sum())
        st[1] += float(a.sum())
        st[2] += int(rows.numel())
        st[3] = max(st[3], float(a.max()))
    return edit


def install_host(model):
    import torch
    dev = next(model.parameters()).device
    for L in S["layers"]:
        W = S["write_np"][L]
        S["write_by_L"][L] = "IDENT" if W is None else torch.tensor(W, device=dev, dtype=torch.float32)
        S["muA_by_L"][L] = torch.tensor(S["muA_np"][L], device=dev, dtype=torch.float32)
        S["muB_by_L"][L] = torch.tensor(S["muB_np"][L], device=dev, dtype=torch.float32)
        tgt = model.model.layers[L - 1]
        if getattr(tgt, "_vpm_edit_handle", None) is not None:
            tgt._vpm_edit_handle.remove()
        tgt._vpm_edit_handle = tgt.register_forward_hook(make_host_edit(L))
    return str(dev)


def install_donor(model):
    import torch
    dev = next(model.parameters()).device
    for L in S["layers"]:
        R = S["read_np"][L]
        S["read_by_L"][L] = "IDENT" if R is None else torch.tensor(R, device=dev, dtype=torch.float32)
        tgt = model.model.layers[L - 1]
        if getattr(tgt, "_vpm_cap_handle", None) is not None:
            tgt._vpm_cap_handle.remove()
        tgt._vpm_cap_handle = tgt.register_forward_hook(make_donor_cap(L))
    return str(dev)


def drain_layer(L, seq_len, num_cached):
    """Map this donor call's captured coords at layer L to absolute positions [num_cached..seq_len-1].
    IDENT history is moved to CPU (bound GPU memory); K-dim history stays on GPU (small, == base)."""
    import torch
    caps = S["caps_by_L"][L]
    if not caps:
        raise AssertionError(f"donor capture hook L{L} never fired")
    flat = torch.cat(caps, dim=0)
    S["caps_by_L"][L] = []
    m = flat.shape[0]
    exp = seq_len - num_cached
    assert m == exp, f"L{L} donor capture count {m} != seq_len-num_cached {exp}"
    if isinstance(S["read_by_L"][L], str):                  # IDENT: park full vectors on CPU
        flat = flat.to("cpu")
    hist = S["hist_by_L"][L]
    for i in range(m):
        hist[num_cached + i] = flat[i]
    return m


# ---------------- axis/spec parsing (mirrors the HF --axes grammar) ------------------------
def load_basis_multi(npzp, L, dims):
    import numpy as np
    z = np.load(ROOT / npzp, allow_pickle=True)
    if "layer" in z.files:
        assert int(np.asarray(z["layer"]).reshape(-1)[0]) == L, f"{npzp} layer field != L{L}"
    if "directions" in z.files:
        D = z["directions"].astype("float64")
        if dims:
            assert dims <= len(D), f"--dims {dims} > K={len(D)} in {npzp}"
            D = D[:dims]
        G = D @ D.T
        assert np.abs(G - np.eye(len(D))).max() < 1e-3, f"{npzp} rows not orthonormal"
    else:
        d = z["direction"].astype("float64")
        d /= np.linalg.norm(d)
        D = d.reshape(1, -1)
    mean = z["mean"].astype("float64") if "mean" in z.files else None
    return D, mean


def parse_axes(axes_str, donor_axes_str, dims, mu_host, mu_donor):
    """Returns spec dict L -> (Dwrite [K,dm]|None(IDENT), Dread|None, muB [K], muA [K])."""
    import numpy as np
    read_specs = {}
    if donor_axes_str:
        for tok in donor_axes_str.split(","):
            ls, npzp = tok.split(":", 1)
            read_specs[int(ls.strip().lstrip("Ll"))] = npzp.strip()

    def mu_vec(override, mean, D):
        if override is not None:
            return np.full(len(D), float(override))
        assert mean is not None, "axis npz has no `mean`; pass --mu-host/--mu-donor"
        return mean @ D.T

    spec = {}
    for tok in axes_str.split(","):
        ls, npzp = tok.split(":", 1)
        L = int(ls.strip().lstrip("Ll"))
        npzp = npzp.strip()
        if npzp == "IDENT":                                 # FULL-DIMENSION transplant
            assert L not in read_specs, "IDENT layers take no --donor-axes entry"
            assert (mu_host or 0) == 0 and (mu_donor or 0) == 0, \
                "IDENT layers: raw match only (--mu-host 0 --mu-donor 0)"
            spec[L] = (None, None, np.zeros(1), np.zeros(1))
            continue
        Dw, mean_w = load_basis_multi(npzp, L, dims)
        muB = mu_vec(mu_host, mean_w, Dw)
        if L in read_specs:                                 # CROSS-FRAME: donor reads its own basis
            Dr, mean_r = load_basis_multi(read_specs[L], L, dims)
            assert len(Dr) == len(Dw), f"L{L}: donor K={len(Dr)} != write K={len(Dw)}"
            muA = mu_vec(mu_donor, mean_r, Dr)
            spec[L] = (Dw, Dr, muB, muA)
        else:                                               # SHARED frame (write basis reads both)
            muA = mu_vec(mu_donor, mean_w, Dw)
            spec[L] = (Dw, None, muB, muA)
    assert not donor_axes_str or set(read_specs) == set(spec), \
        "--donor-axes layers must match --axes layers exactly"
    return spec


# ---------------- self-test (CPU, no engines) ---------------------------------------------
def self_test():
    import torch
    import fsd_trace_transplant_subdim3 as FSD
    from vllm_lockstep_transplant_0727 import self_test as base_math_test
    base_math_test()                                        # match_delta == fsd sub/axis_trace_edit
    torch.manual_seed(3)

    def run_layer_hook(L, positions, h0, res, hist):
        """Invoke make_host_edit(L)'s closure on a synthetic (positions,(h0,res)) forward in-place.
        CHANGE 3/3 self-test extension: run BOTH gating modes and assert they edit identically
        (arith mode reconstructs the same absolute positions from seq_len = max(pos)+1 with a
        contiguous tail, which these synthetic positions are)."""
        S["hist_by_L"][L] = {int(p): hist[int(p)] for p in positions.tolist()}
        S["stats_by_L"][L] = [0.0, 0.0, 0, 0.0]
        S["pos_mode"] = "tensor"
        h0c = h0.clone()
        make_host_edit(L)(None, (positions,), (h0c, res))
        S["pos_mode"] = "arith"
        S["seq_len"] = int(positions.max()) + 1
        S["stats_by_L"][L] = [0.0, 0.0, 0, 0.0]
        h0a = h0.clone()
        make_host_edit(L)(None, (torch.zeros(1),), (h0a, res))   # inp unused in arith mode
        assert torch.equal(h0c, h0a), "arith-mode gating != tensor-mode gating on identical tail"
        S["pos_mode"] = "tensor"
        return h0c

    n, dm, K = 9, 64, 3
    thr = 5
    positions = torch.arange(3, 3 + n)                      # absolute positions 3..11
    gate = positions >= thr
    h0 = torch.randn(n, dm, dtype=torch.float32)
    res = torch.randn(n, dm, dtype=torch.float32)
    full = (h0.float() + res.float())                       # the vLLM full stream

    # (A) IDENT hook == FSD IDENT path `hn = h + lam*(d - h)` on the FULL stream; below-thr untouched.
    S.update(edit_on=True, thr=thr, lam=1.0, srat=1.0, bias=0.0)
    donor_full = torch.randn(n, dm, dtype=torch.float32)
    S["write_by_L"][7] = "IDENT"
    S["read_by_L"][7] = "IDENT"
    hist_i = {int(p): donor_full[j] for j, p in enumerate(positions.tolist())}
    for lam in (1.0, 0.5):
        S["lam"] = lam
        h0c = run_layer_hook(7, positions, h0, res, hist_i)
        new_full = (h0c.float() + res.float())
        # FSD IDENT path (edits the whole stream per position): hn = h + lam*(d - h). Compare on the
        # GATED rows only (answer_from gating is applied separately); below-thr must be untouched.
        fsd_full = full + lam * (donor_full - full)
        assert (new_full[gate] - fsd_full[gate]).abs().max() < 1e-4, \
            f"IDENT hook != FSD IDENT path (lam={lam})"
        assert torch.equal(h0c[~gate], h0[~gate]), "IDENT: below-threshold rows modified"
    print("[self-test] IDENT hook == lam*(donor-host) on full stream (lam1/0.5); "
          "below-thr untouched", flush=True)

    # (B) K-dim hook == FSD.sub_trace_edit on the FULL stream for gated rows (shared + cross frame).
    Q, _ = torch.linalg.qr(torch.randn(dm, K, dtype=torch.float64))
    Dw = Q.T.contiguous().float()
    Q2, _ = torch.linalg.qr(torch.randn(dm, K, dtype=torch.float64))
    Dr = Q2.T.contiguous().float()
    muA = torch.tensor([0.4, -1.0, 2.0])
    muB = torch.tensor([-0.3, 0.7, 0.0])
    donor_full2 = torch.randn(n, dm, dtype=torch.float32)
    for Dread, tag in [(Dw, "shared"), (Dr, "cross")]:
        S.update(lam=8.0, srat=1.0, bias=0.0)
        S["write_by_L"][8] = Dw
        S["read_by_L"][8] = Dread
        S["muA_by_L"][8] = muA.clone()
        S["muB_by_L"][8] = muB.clone()
        hist_k = {int(p): (donor_full2[j].float() @ Dread.T) for j, p in enumerate(positions.tolist())}
        h0c = run_layer_hook(8, positions, h0, res, hist_k)
        new_full = (h0c.float() + res.float())
        ref = FSD.sub_trace_edit(full.double().unsqueeze(0), donor_full2.double().unsqueeze(0),
                                 Dw.double(), muA.double(), muB.double(), 8.0,
                                 Dread=Dread.double())[0]
        want = full.clone()
        want[gate] = ref.float()[gate]
        assert (new_full - want).abs().max() < 1e-4, f"K-dim {tag}-frame != FSD.sub_trace_edit"
        assert torch.equal(h0c[~gate], h0[~gate]), f"K-dim {tag}: below-thr rows modified"
    print("[self-test] K-dim hook == FSD.sub_trace_edit on full stream (shared+cross); "
          "below-thr untouched", flush=True)

    # (C) K=1 single-layer reproduces the BASE PORT edit math exactly (match_delta @ d into out[0]).
    d = torch.randn(dm, dtype=torch.float32)
    d /= d.norm()
    D1 = d.view(1, -1)
    S.update(lam=32.0, srat=1.0, bias=0.0)
    S["write_by_L"][9] = D1
    S["read_by_L"][9] = D1
    S["muA_by_L"][9] = torch.zeros(1)
    S["muB_by_L"][9] = torch.zeros(1)
    donor_full3 = torch.randn(n, dm, dtype=torch.float32)
    hist_1 = {int(p): (donor_full3[j].float() @ D1.T) for j, p in enumerate(positions.tolist())}
    h0c = run_layer_hook(9, positions, h0, res, hist_1)
    # base-port formula (vllm_lockstep_transplant_0727.install_host_edit inner body), gated:
    ph = (full[gate] @ D1.T)                                # [m,1]
    pd = (donor_full3[gate] @ D1.T)
    base_delta = match_delta(ph, pd, torch.zeros(1), torch.zeros(1), 32.0, 1.0, 0.0) @ D1  # [m,dm]
    base_h0 = h0.clone()
    base_h0[gate] = h0[gate] + base_delta
    assert (h0c - base_h0).abs().max() < 1e-4, "K=1 multi != base port edit math"
    print("[self-test] K=1 single-layer multi-hook == base port (match_delta) edit math", flush=True)

    for k in ("write_by_L", "read_by_L", "muA_by_L", "muB_by_L", "hist_by_L", "stats_by_L"):
        S[k] = {}
    S.update(edit_on=False, lam=0.0)
    print("[self-test] ALL PASS", flush=True)


# ---------------- driver -------------------------------------------------------------------
def main():
    if "--self-test" in sys.argv:
        self_test()
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", required=True)
    ap.add_argument("--donor", required=True)
    ap.add_argument("--axes", required=True, help="'L:npz,L:IDENT,...' per-layer host WRITE axes")
    ap.add_argument("--donor-axes", default=None, help="'L:npz,...' cross-frame donor READ axes")
    ap.add_argument("--dims", type=int, default=None)
    ap.add_argument("--mu-host", type=float, default=None)
    ap.add_argument("--mu-donor", type=float, default=None)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--srat", type=float, default=1.0)
    ap.add_argument("--bias", type=float, default=0.0)
    ap.add_argument("--prefix-file", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=30000)
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--enable-thinking", type=int, default=1)
    ap.add_argument("--add-special-tokens", type=int, default=0)
    ap.add_argument("--eos-ids", default="151645,151643")
    ap.add_argument("--gpu-mem", type=float, default=0.42, help="per engine (two on one GPU)")
    ap.add_argument("--pos-arith", action="store_true",
                    help="CHANGE 3/3 (0731): arithmetic last-n-tail positional gating (base-port "
                         "convention) instead of the inp[0] positions tensor; REQUIRED for gpt-oss")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf-ref-seconds", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpu-check", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n, args.max_new = 2, 64
        args.out = args.out.replace(".json", "_smoke.json")
    eos_ids = set(int(x) for x in args.eos_ids.split(","))

    import numpy as np
    spec = parse_axes(args.axes, args.donor_axes, args.dims, args.mu_host, args.mu_donor)
    layers = sorted(spec)
    S["layers"] = layers
    K_by_L, idents = {}, []
    for L in layers:
        Dw, Dr, muB, muA = spec[L]
        S["write_np"][L] = None if Dw is None else Dw
        S["read_np"][L] = ("IDENT" if Dw is None else (Dr if Dr is not None else Dw))
        # (read_np placeholder "IDENT" -> install_donor sets read_by_L to the string marker)
        if Dw is None:
            S["read_np"][L] = None
        S["muA_np"][L], S["muB_np"][L] = muA, muB
        K_by_L[L] = 4096 if Dw is None else len(Dw)
        if Dw is None:
            idents.append(L)
    S["lam"], S["srat"], S["bias"] = args.lam, args.srat, args.bias
    S["pos_mode"] = "arith" if args.pos_arith else "tensor"    # CHANGE 3/3

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.host, trust_remote_code=True)
    rows = [json.loads(l) for l in open(ROOT / args.prefix_file)][: args.n]
    prompts = [build_q(tok, r["prompt"], bool(args.enable_thinking)) + r["prefix"] for r in rows]
    encs = [tok(p, add_special_tokens=bool(args.add_special_tokens)).input_ids for p in prompts]
    plens = [len(e) for e in encs]
    keep = [i for i, p in enumerate(plens) if p <= args.max_model_len - 64]
    print(f"[vpm] host={args.host}\n[vpm] donor={args.donor}\n"
          f"[vpm] MULTI-LAYER positional gating: layers={layers} K_by_L={K_by_L} "
          f"IDENT_layers={idents} lam={args.lam} frame={'cross' if args.donor_axes else 'shared'} "
          f"n={len(keep)} plen_med={sorted(plens)[len(plens)//2]} eos={sorted(eos_ids)}", flush=True)
    if args.cpu_check:
        print("[cpu-check] OK (axes parse, bases, mus, fork rows, chat template, tokenization). "
              "No engines loaded.", flush=True)
        return

    import torch
    from vllm import LLM, SamplingParams
    import vllm as _v
    t0 = time.time()
    host_llm = LLM(model=args.host, enforce_eager=True, enable_prefix_caching=True,
                   max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
                   seed=args.seed, dtype="bfloat16", trust_remote_code=True)
    donor_llm = LLM(model=args.donor, enforce_eager=True, enable_prefix_caching=True,
                    max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
                    seed=args.seed, dtype="bfloat16", trust_remote_code=True)
    print(f"[vpm] both engines up in {time.time()-t0:.0f}s (two engines, one process)", flush=True)
    host_llm.apply_model(install_host)
    donor_llm.apply_model(install_donor)
    try:
        block_size = int(host_llm.llm_engine.cache_config.block_size)
    except AttributeError:
        block_size = int(host_llm.llm_engine.vllm_config.cache_config.block_size)
    sp_throw = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False)

    akey = f"{args.lam:+.3f}"
    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    gens = {"mode": "vllm_lockstep_value_match_multi_positional",
            "engine": f"vllm-{_v.__version__} x2 in-process",
            "layers": layers, "K_by_L": {str(L): K_by_L[L] for L in layers},
            "ident_layers": idents,
            "positions": ("answer (gated: position >= plen-1 via arithmetic last-n tail; base-port conv)"
                          if args.pos_arith else
                          "answer (gated: position >= plen-1 via positions tensor; recovered arch)"),
            "pos_mode": "arith" if args.pos_arith else "tensor",
            "host": args.host, "donor": args.donor, "axes": args.axes,
            "donor_axes": args.donor_axes,
            "mu_host_by_L": {str(L): np.asarray(S["muB_np"][L]).tolist() for L in layers},
            "mu_donor_by_L": {str(L): np.asarray(S["muA_np"][L]).tolist() for L in layers},
            "lam": args.lam, "srat": args.srat, "bias": args.bias,
            "temperature": args.temperature, "top_p": args.top_p, "seed": args.seed,
            "seed_scheme": "per-step derived (seed*1000003+row*7919+step)",
            "add_special_tokens": bool(args.add_special_tokens), "eos_ids": sorted(eos_ids),
            "block_size": block_size, "batch": 1,
            "timing": {"per_row": {}, "hf_reference_s": args.hf_ref_seconds},
            "by_alpha": {akey: []}}
    if args.resume and outp.exists():
        try:
            gens = json.load(open(outp))
            print(f"[vpm] resume: {len(gens['by_alpha'].get(akey, []))} records", flush=True)
        except Exception:
            pass
    recs = gens["by_alpha"].setdefault(akey, [])
    done_ids = {r["id"] for r in recs}

    t_run0 = time.time()
    first_row_gate = True
    for i in keep:
        rid = rows[i]["id"]
        if rid in done_ids:
            continue
        seq = list(encs[i])
        Sn = len(seq)
        S["thr"] = Sn - 1
        S["edit_on"] = False
        for L in layers:
            S["hist_by_L"][L] = {}
            S["caps_by_L"][L] = []
            S["stats_by_L"][L] = [0.0, 0.0, 0, 0.0]
        t_row = time.time()
        # donor prefill FIRST (capture coords through S-1 at every layer); NO host phase A —
        # positional gating makes the hook-ON host prefill convention-exact.
        od = gen1(donor_llm, seq, sp_throw)
        nc = int(od.num_cached_tokens)
        for L in layers:
            drain_layer(L, Sn, nc)
        assert (Sn - 1) in S["hist_by_L"][layers[0]], "donor prefill did not cover position S-1"
        allowed = min(args.max_new, args.max_model_len - Sn - 1)
        gen_ids, finished = [], False
        host_cache_misses = 0
        for step in range(allowed):
            sp_step = SamplingParams(
                temperature=args.temperature, top_p=args.top_p, max_tokens=1, detokenize=False,
                seed=(args.seed * 1000003 + i * 7919 + step) % (2**31 - 1))
            S["seq_len"] = len(seq)                       # CHANGE 3/3: arith gating tail anchor
            S["edit_on"] = True
            oh = gen1(host_llm, seq, sp_step)
            S["edit_on"] = False
            if step > 0 and int(oh.num_cached_tokens) == 0 and len(seq) > block_size:
                host_cache_misses += 1
            toks = list(oh.outputs[0].token_ids)
            if not toks:
                finished = True
                break
            y = int(toks[0])
            gen_ids.append(y)
            if y in eos_ids:
                finished = True
                break
            seq.append(y)
            for L in layers:
                S["caps_by_L"][L] = []
            od = gen1(donor_llm, seq, sp_throw)
            nc = int(od.num_cached_tokens)
            for L in layers:
                drain_layer(L, len(seq), nc)
            # CHANGE 2/2 (0731): env-skippable ONLY for lam0 parity/anchor cells (edit==0 by
            # construction there) — verbatim base-port convention (…_0727.py:438).
            if first_row_gate and step == 8 and os.environ.get("TRANSPLANT_SKIP_FORKGATE") != "1":
                fired = sum(S["stats_by_L"][L][2] for L in layers)
                amax = max(S["stats_by_L"][L][3] for L in layers)
                assert fired > 0 and amax > 1e-3, \
                    f"FORK-GATE FAIL: multi value-match edit ~0 (per-L stats={S['stats_by_L']})"
                print(f"[vpm] fork-gate PASS: edited_positions(sum over L)={fired} "
                      f"absmax_over_L={amax:.3f}", flush=True)
                first_row_gate = False
        txt = tok.decode(gen_ids[:-1] if (finished and gen_ids and gen_ids[-1] in eos_ids)
                         else gen_ids, skip_special_tokens=True)
        dt = time.time() - t_row
        per_L = {str(L): dict(n=S["stats_by_L"][L][2],
                              absmean=round(S["stats_by_L"][L][1] / max(S["stats_by_L"][L][2], 1), 4),
                              absmax=round(S["stats_by_L"][L][3], 4)) for L in layers}
        recs.append(dict(id=rid, prompt=rows[i]["prompt"][:2000], gen=txt,
                         finished=finished, n_tokens=len(gen_ids), edit_by_L=per_L))
        gens["timing"]["per_row"][rid] = dict(
            s=round(dt, 1), steps=len(gen_ids), ms_per_step=round(1000 * dt / max(len(gen_ids), 1)),
            host_cache_misses=host_cache_misses)
        gens["timing"]["total_s"] = round(time.time() - t_run0, 1)
        outp.write_text(json.dumps(gens, indent=1))
        absm = max((per_L[str(L)]["absmean"] for L in layers), default=0.0)
        print(f"[vpm] {len(recs)}/{len(keep)} id={rid} steps={len(gen_ids)} fin={finished} "
              f"{dt:.0f}s ({1000*dt/max(len(gen_ids),1):.0f}ms/tok) "
              f"edit_absmean_max_L={absm} misses={host_cache_misses}", flush=True)
    gens["timing"]["total_s"] = round(time.time() - t_run0, 1)
    gens["timing"]["complete"] = True
    outp.write_text(json.dumps(gens, indent=1))
    print(f"[vpm] DONE n={len(recs)} total={gens['timing']['total_s']}s "
          f"(hf_ref={args.hf_ref_seconds}s) wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
