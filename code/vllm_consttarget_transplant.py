#!/usr/bin/env python3
"""CONSTANT-TARGET CLAMP VARIANT (0731, crossmodel lane) of the validated lockstep port.

MINIMAL CHANGE vs vllm_lockstep_transplant_0727.py (standing order: copy + smallest possible
diff; edit-hook math and decode loop conventions UNTOUCHED):
  --const-targets <json> ({row_id: raw_target_coord}) replaces the donor engine: the per-position
  donor coordinate lookup STATE["hist"].get(p) returns a per-row CONSTANT (ConstHist), so
    delta = lam * ( srat*(target - muA) + bias - (ph - muB) )
  with --mu-host 0 --mu-donor 0 srat 1 bias 0 == lam * (target - ph): at lam=1 the host's raw
  write-axis coordinate is clamped exactly to the target. No donor engine is created (single
  engine; pass a larger --gpu-mem). --donor becomes optional; exactly one of --donor /
  --const-targets must be given. Everything else (two-phase prefill, exact position gating,
  per-step seeds, fork-gate, resume, uncapped decode) is byte-identical to the validated port.
  lam0 parity gate: run with TRANSPLANT_SKIP_FORKGATE=1 (edit is 0 by construction), as in the
  g20b ladder convention.

------------------------------------------------------------------------------------------------
LOCKSTEP VALUE-MATCH TRANSPLANT ON vLLM — two engines, one GPU, one process (0727).

Port of fsd_trace_transplant_subdim*_0726.py's analog value-match (sub_trace_edit /
axis_trace_edit math, fork frame, positions=answer) onto vLLM paged KV:

  HOST engine  (gpu_memory_utilization ~0.42, enforce_eager so nn.Module hooks fire):
    generates with the value-match hook: at layer L, for continuation positions only,
      h <- h + lam * [ srat*(donor_coord - muA) + bias - (host_coord - muB) ] * d_write
    (K-dim: per-row basis D, exactly sub_trace_edit). Coordinates are read on the FULL residual
    stream out[0]+out[1] (vLLM decoder layers return (hidden, residual); adding the delta to
    out[0] is equivalent to adding it to the stream).
  DONOR engine (same GPU, ~0.42): driven in LOCKSTEP BY PREFIX EXTENSION — each step the host's
    current token sequence is submitted as a TokensPrompt with max_tokens=1 and prefix caching on,
    so only the newly appended token (plus the <block_size partial-block tail) is computed; the
    donor's paged KV persists across steps via the prefix cache. A capture hook at layer L records
    the computed positions' read-axis coordinates into a shared STATE (both engines run
    IN-PROCESS: VLLM_ENABLE_V1_MULTIPROCESSING=0).

POSITION GATING (the same convention as the HF fork harness): the host prefix is prefilled once
with the hook DISABLED (phase A of the port-1 two-phase trick) so its KV blocks are cached
UNEDITED; every subsequent host step computes only a small tail whose absolute positions are known
(batch=1: positions [len(seq)-n .. len(seq)-1]), and the hook edits ONLY positions >= S-1
(answer_from = S-1, bug-hunt #6 convention: the last prefix position samples continuation token 1).
Unlike port 1's constant push, the gating here is exact — recomputed tail positions < S-1 are left
untouched, so there is NO boundary fuzz; recomputed positions >= S-1 are re-edited with the SAME
donor coords (deterministic, self-consistent with the cached blocks).

LOCKSTEP EXACTNESS: donor coords at position p are always captured from a donor forward on the
identical token prefix before the host computes position p — the HF donor-first-then-host order,
token by token. Sampling uses vLLM's sampler with a per-step derived seed
  seed_step = (seed*1000003 + row_index*7919 + step) mod 2^31-1
(a FIXED per-request seed would replay the same RNG draw every step); numerics differ from the HF
sampler, so cells are compared at the judged-rate level (the standard cross-harness convention).

BATCH: 1 row per step-loop (simplest correct version). Throughput therefore rests on the
one-token-extension prefix-cache hit — measured explicitly by --microbench BEFORE trusting the
design (the known risk: APC not hitting on one-token extensions would force a full re-prefill per
step = blocker, documented not hacked around).

Validation target (Qwen-B self-rating cell):
  --host  .../qwen3-8b-distillation/success_cheater_hard_think_merged
  --donor .../qwen3-8b-distillation/success_honest_think_merged
  --axis activations/dspace/preDIM_QB_L21.npz --mu-host 0 --mu-donor 0 --lam 32
  --prefix-file tasks/forkcommit_cheateroct_0715.jsonl --n 60 --temperature 0.7 --top-p 0.95
  --seed 0 --max-new 30000 --max-model-len 40960
  Pass criterion: judged rates match the HF-harness reference cell run with the same config.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from ss_paths import SS_ROOT   # portable roots

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # BOTH engine cores in THIS process
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
ROOT = Path(f"{SS_ROOT}/v2")
sys.path.insert(0, str(ROOT))
from fsd_trace_transplant_subdim2 import build_q  # noqa: E402 (exact chat-template parity)

STATE = {
    "edit_on": False, "seq_len": 0, "prompt_len": 0, "hist": {},   # pos -> [K] f32 gpu tensor
    "D": None, "Dread": None, "muA": None, "muB": None, "lam": 0.0, "srat": 1.0, "bias": 0.0,
    "caps": [], "stats": None, "host_tail_sizes": [],
}


# ---------------- pure edit math (self-tested against the fsd functions directly) -------------
def match_delta(ph, pd, muA, muB, lam, srat=1.0, bias=0.0):
    """delta [n,K] such that h += delta @ D reproduces sub_trace_edit/axis_trace_edit:
    delta = lam * (srat*(pd - muA) + bias - (ph - muB))."""
    return lam * (srat * (pd - muA) + bias - (ph - muB))


def self_test():
    import torch
    import fsd_trace_transplant_subdim2 as FSD
    torch.manual_seed(0)
    B, S, dm, K = 2, 7, 64, 3
    muA1, muB1, lam, srat, bias = 4.0669, 5.2573, 32.0, 1.0, 0.0
    d = torch.randn(dm, dtype=torch.float64)
    d /= d.norm()
    h = torch.randn(B, S, dm, dtype=torch.float64)
    hd = torch.randn(B, S, dm, dtype=torch.float64)
    # (1) K=1 equality with axis_trace_edit given the SAME donor/host coords
    ref = FSD.axis_trace_edit(h, hd, d, muA1, muB1, lam, dvecA=None, srat=srat, bias=bias)
    ph = (h * d).sum(-1, keepdim=True)                     # [B,S,1]
    pd = (hd * d).sum(-1, keepdim=True)
    delta = match_delta(ph, pd, muA1, muB1, lam, srat, bias)          # [B,S,1]
    mine = h + delta * d
    assert torch.allclose(mine, ref, atol=1e-12), "K=1 match_delta != axis_trace_edit"
    print("[self-test] K=1 match_delta edit == fsd axis_trace_edit", flush=True)
    # (2) K=3 orthonormal basis vs sub_trace_edit, cross-frame read basis + nonzero mus
    Q, _ = torch.linalg.qr(torch.randn(dm, K, dtype=torch.float64))
    D = Q.T.contiguous()
    Q2, _ = torch.linalg.qr(torch.randn(dm, K, dtype=torch.float64))
    Dr = Q2.T.contiguous()
    muA = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64)
    muB = torch.tensor([-0.3, 0.7, 0.0], dtype=torch.float64)
    ref2 = FSD.sub_trace_edit(h, hd, D, muA, muB, lam, Dread=Dr, srat=srat, bias=bias)
    ph2 = h @ D.T
    pd2 = hd @ Dr.T
    mine2 = h + match_delta(ph2, pd2, muA, muB, lam, srat, bias) @ D
    assert torch.allclose(mine2, ref2, atol=1e-12), "K=3 match_delta != sub_trace_edit"
    print("[self-test] K=3 cross-frame match_delta edit == fsd sub_trace_edit", flush=True)
    # (3) position gating: only p >= S-1 edited (answer_from convention)
    af = S - 1
    gated = h.clone()
    gated[:, af:, :] = mine[:, af:, :]
    FSD.STATE.update(lam=lam, donor_h=hd, dvec=d, muA=muA1, muB=muB1, answer_from=af, calls=0,
                     dvecA=None, srat=srat, bias=bias, gate_dose=None, probe_every=None,
                     choice_probe_every=None, live_ids=None, log_edit=False,
                     extra_dvec=None, extra_coef=0.0)
    ref3 = FSD.get_h(FSD.rep_hook(None, None, (h.clone(),)))
    assert torch.allclose(gated, ref3, atol=1e-12), "answer_from gating mismatch vs rep_hook"
    FSD.STATE.update(lam=0.0, donor_h=None, answer_from=None)
    print("[self-test] positions>=S-1 gating == fsd rep_hook prefill gate", flush=True)
    # (4, 0731) const-target mode: ConstHist lookup + clamp math. With pd == constant t, mu=0,
    # srat=1, bias=0, lam=1 the post-edit coordinate must equal t exactly at every position.
    t_const = 3.1415
    ch = ConstHist(torch.tensor([t_const], dtype=torch.float64))
    assert (0 in ch) and (10**9 in ch) and torch.allclose(ch.get(7), ch.get(123456))
    pd_c = torch.full_like(ph, t_const)
    hn_c = h + match_delta(ph, pd_c, 0.0, 0.0, 1.0) * d
    assert torch.allclose((hn_c * d).sum(-1), torch.full((B, S), t_const, dtype=torch.float64),
                          atol=1e-9), "const-target lam=1 clamp != target"
    orth = h - (h * d).sum(-1, keepdim=True) * d
    orth_c = hn_c - (hn_c * d).sum(-1, keepdim=True) * d
    assert torch.allclose(orth, orth_c, atol=1e-9), "const-target edit moved orthogonal components"
    print("[self-test] const-target ConstHist + lam=1 exact clamp + orthogonal invariance OK",
          flush=True)
    print("[self-test] ALL PASS", flush=True)


# ---------------- hooks (installed via apply_model; STATE shared in-process) ------------------
def _full_stream(out):
    import torch
    if isinstance(out, tuple):
        h0 = out[0]
        h1 = out[1] if len(out) > 1 and torch.is_tensor(out[1]) else None
        return h0, (h0.float() + h1.float()) if h1 is not None else h0.float()
    return out, out.float()


def install_donor_cap(model, layer_idx0, dread_list):
    import torch
    tgt = model.model.layers[layer_idx0]
    dev = next(tgt.parameters()).device
    Dr = torch.tensor(dread_list, device=dev, dtype=torch.float32)     # [K, dm]

    def cap(module, inp, out):
        _, full = _full_stream(out)
        STATE["caps"].append((full.reshape(-1, full.shape[-1]) @ Dr.T).detach())  # [n,K] f32
    if getattr(tgt, "_vp_cap_handle", None) is not None:
        tgt._vp_cap_handle.remove()
    tgt._vp_cap_handle = tgt.register_forward_hook(cap)
    return str(dev)


def install_host_edit(model, layer_idx0, dwrite_list):
    import torch
    tgt = model.model.layers[layer_idx0]
    dev = next(tgt.parameters()).device
    Dw = torch.tensor(dwrite_list, device=dev, dtype=torch.float32)    # [K, dm]
    STATE["D"] = Dw

    def edit(module, inp, out):
        if not STATE["edit_on"]:
            return
        import torch
        h0, full = _full_stream(out)
        flat0 = h0.reshape(-1, h0.shape[-1])
        n = flat0.shape[0]
        seq_len, S = STATE["seq_len"], STATE["prompt_len"]
        start = seq_len - n
        assert start >= 0, f"host tail longer than sequence: n={n} seq_len={seq_len}"
        STATE["host_tail_sizes"].append(n)
        idx, pds = [], []
        for i in range(n):
            p = start + i
            if p >= S - 1:                                  # answer_from = S-1 convention
                pd = STATE["hist"].get(p)
                assert pd is not None, f"missing donor coord at position {p} (S={S}, seq={seq_len})"
                idx.append(i)
                pds.append(pd)
        if not idx:
            return
        Dw_ = STATE["D"]
        flat_full = full.reshape(-1, full.shape[-1])
        rows = torch.tensor(idx, device=flat0.device, dtype=torch.long)
        ph = flat_full.index_select(0, rows) @ Dw_.T                  # [m,K] f32 host coords
        pd = torch.stack(pds).to(flat0.device)                        # [m,K] f32 donor coords
        if STATE["muA"].device != ph.device:                          # xfam-bug class: mus must
            STATE["muA"] = STATE["muA"].to(ph.device)                 # live on the hook device
            STATE["muB"] = STATE["muB"].to(ph.device)
        delta = match_delta(ph, pd, STATE["muA"], STATE["muB"], STATE["lam"],
                            STATE["srat"], STATE["bias"])             # [m,K]
        dvecs = (delta @ Dw_)                                         # [m,dm] f32
        flat0.index_add_(0, rows, dvecs.to(flat0.dtype))              # stream += delta @ D
        st = STATE["stats"]
        if st is not None:
            a = delta.abs().sum(-1)                                   # [m] summed |coord shift|
            sgn = delta.sum(-1)
            st[0] += float(sgn.sum())
            st[1] += float(a.sum())
            st[2] += len(idx)
            st[3] = max(st[3], float(a.max()))
    if getattr(tgt, "_vp_edit_handle", None) is not None:
        tgt._vp_edit_handle.remove()
    tgt._vp_edit_handle = tgt.register_forward_hook(edit)
    return str(dev)


class ConstHist:
    """Drop-in for STATE['hist'] in const-target mode: every position returns the same per-row
    constant donor coordinate [K]. Supports .get(p) and `p in hist` (the S-1 coverage assert)."""
    def __init__(self, t):
        self.t = t                                          # [K] float32 cpu tensor

    def get(self, p, default=None):
        return self.t

    def __contains__(self, p):
        return True


# ---------------- driver helpers ----------------------------------------------------------
def gen1(llm, seq, sp):
    from vllm.inputs import TokensPrompt
    return llm.generate([TokensPrompt(prompt_token_ids=list(seq))], sp, use_tqdm=False)[0]


def drain_caps_into_hist(seq_len, num_cached, block_size):
    """Map this donor call's captured coords to absolute positions [num_cached .. seq_len-1]."""
    import torch
    if not STATE["caps"]:
        raise AssertionError("donor capture hook never fired")
    flat = torch.cat(STATE["caps"], dim=0)                            # [m, K]
    STATE["caps"] = []
    m = flat.shape[0]
    exp = seq_len - num_cached
    assert m == exp, f"donor capture count {m} != seq_len-num_cached {exp}"
    for i in range(m):
        STATE["hist"][num_cached + i] = flat[i]
    return m


def load_basis(npz_path, dims):
    import numpy as np
    z = np.load(ROOT / npz_path, allow_pickle=True)
    layer = int(np.asarray(z["layer"]).reshape(-1)[0]) if "layer" in z.files else None
    if "directions" in z.files:
        D = z["directions"].astype("float64")
        if dims:
            D = D[:dims]
        G = D @ D.T
        assert abs(G - __import__("numpy").eye(len(D))).max() < 1e-3, f"{npz_path} not orthonormal"
    else:
        d = z["direction"].astype("float64")
        d /= __import__("numpy").linalg.norm(d)
        D = d.reshape(1, -1)
    mean = z["mean"].astype("float64") if "mean" in z.files else None
    return D, mean, layer


def main():
    if "--self-test" in sys.argv:
        self_test()
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", required=True)
    ap.add_argument("--donor", default=None)
    ap.add_argument("--const-targets", default=None,
                    help="json {row_id: raw_target_coord}; replaces the donor engine "
                         "(per-row constant donor coordinate)")
    ap.add_argument("--axis", required=True,
                    help="WRITE axis npz (`direction` [dm] or `directions` [K,dm] + `layer`)")
    ap.add_argument("--donor-axis", default=None,
                    help="cross-frame donor READ axis npz (default: shared frame, read = write)")
    ap.add_argument("--dims", type=int, default=None)
    ap.add_argument("--mu-host", type=float, default=None, help="scalar muB (default: axis mean.d)")
    ap.add_argument("--mu-donor", type=float, default=None, help="scalar muA (default: read mean.d)")
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--srat", type=float, default=1.0)
    ap.add_argument("--bias", type=float, default=0.0)
    ap.add_argument("--prefix-file", required=True, help="fork jsonl {id,prompt,prefix}")
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf-ref-seconds", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="n=2, max-new=64")
    ap.add_argument("--cpu-check", action="store_true")
    ap.add_argument("--microbench", action="store_true",
                    help="HOST ENGINE ONLY: measure prefix-cache behavior + latency of one-token "
                         "prefix extensions (the design's load-bearing assumption), then exit")
    args = ap.parse_args()
    assert (args.donor is None) != (args.const_targets is None), \
        "exactly one of --donor / --const-targets"
    const_mode = args.const_targets is not None
    targets = json.load(open(ROOT / args.const_targets)) if const_mode else None
    if const_mode:
        assert len(targets) > 0, "const-targets: empty targets file"
    if args.smoke:
        args.n, args.max_new = 2, 64
        args.out = args.out.replace(".json", "_smoke.json")
    eos_ids = set(int(x) for x in args.eos_ids.split(","))

    import numpy as np
    Dw, mean_w, layer = load_basis(args.axis, args.dims)
    assert layer is not None, "axis npz needs a layer field"
    if args.donor_axis:
        Dr, mean_r, layer_r = load_basis(args.donor_axis, args.dims)
        assert layer_r == layer and len(Dr) == len(Dw), "read/write basis mismatch"
    else:
        Dr, mean_r = Dw, mean_w
    K = len(Dw)
    if const_mode:
        assert K == 1, "const-targets mode is scalar (K=1) — targets are one coord per row"
    muB = (np.full(K, args.mu_host) if args.mu_host is not None else mean_w @ Dw.T)
    muA = (np.full(K, args.mu_donor) if args.mu_donor is not None else mean_r @ Dr.T)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.host, trust_remote_code=True)
    rows = [json.loads(l) for l in open(ROOT / args.prefix_file)][: args.n]
    prompts = [build_q(tok, r["prompt"], bool(args.enable_thinking)) + r["prefix"] for r in rows]
    encs = [tok(p, add_special_tokens=bool(args.add_special_tokens)).input_ids for p in prompts]
    plens = [len(e) for e in encs]
    keep = [i for i, p in enumerate(plens) if p <= args.max_model_len - 64]
    print(f"[vp2] host={args.host}\n[vp2] donor={args.donor}\n"
          f"[vp2] L={layer} K={K} lam={args.lam} muA={np.round(muA,4).tolist()} "
          f"muB={np.round(muB,4).tolist()} srat={args.srat} bias={args.bias} "
          f"frame={'cross' if args.donor_axis else 'shared'} n={len(keep)} "
          f"plen med={sorted(plens)[len(plens)//2]} eos={sorted(eos_ids)}", flush=True)
    if args.cpu_check:
        print("[cpu-check] OK (args, bases, mus, fork rows, chat template, tokenization). "
              "No engines loaded.", flush=True)
        return

    import torch
    from vllm import LLM, SamplingParams
    import vllm as _v
    t0 = time.time()
    host_llm = LLM(model=args.host, enforce_eager=True, enable_prefix_caching=True,
                   max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
                   seed=args.seed, dtype="bfloat16", trust_remote_code=True)
    print(f"[vp2] host engine up in {time.time()-t0:.0f}s", flush=True)
    li0 = layer - 1
    from functools import partial
    host_llm.apply_model(partial(install_host_edit, layer_idx0=li0, dwrite_list=Dw.tolist()))
    STATE["muA"] = torch.tensor(muA, dtype=torch.float32)
    STATE["muB"] = torch.tensor(muB, dtype=torch.float32)
    STATE["lam"], STATE["srat"], STATE["bias"] = args.lam, args.srat, args.bias
    try:
        block_size = int(host_llm.llm_engine.cache_config.block_size)
    except AttributeError:
        block_size = int(host_llm.llm_engine.vllm_config.cache_config.block_size)
    sp_throw = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False)

    # ---------------- microbench: the load-bearing assumption, measured ----------------
    if args.microbench:
        seq = list(encs[keep[0]])
        t = time.time()
        o = gen1(host_llm, seq, sp_throw)
        t_prefill = time.time() - t
        print(f"[microbench] prefill plen={len(seq)} {t_prefill*1000:.0f}ms "
              f"num_cached={o.num_cached_tokens}", flush=True)
        seq.append(int(o.outputs[0].token_ids[0]) if o.outputs[0].token_ids else 42)
        lat, cached, tails = [], [], []
        T = 64
        for step in range(T):
            t = time.time()
            o = gen1(host_llm, seq, sp_throw)
            lat.append(time.time() - t)
            nc = int(o.num_cached_tokens)
            cached.append(nc)
            tails.append(len(seq) - nc)
            y = int(o.outputs[0].token_ids[0]) if o.outputs[0].token_ids else 42
            seq.append(y)
        t = time.time()
        gen1(host_llm, encs[keep[0]], SamplingParams(temperature=0.0, max_tokens=T,
                                                     detokenize=False))
        t_native = time.time() - t
        ms = sorted(x * 1000 for x in lat)
        hit = sum(1 for s, nc in zip(range(T), cached) if nc >= 16)   # any nonzero block reuse
        full_miss = sum(1 for i, nc in enumerate(cached) if nc == 0)
        res = dict(vllm=_v.__version__, block_size=block_size, plen=plens[keep[0]],
                   prefill_ms=round(t_prefill * 1000), steps=T,
                   step_ms_min=round(ms[0], 1), step_ms_med=round(ms[T // 2], 1),
                   step_ms_max=round(ms[-1], 1),
                   tail_tokens_min=min(tails), tail_tokens_med=sorted(tails)[T // 2],
                   tail_tokens_max=max(tails), cache_hits=hit, full_misses=full_miss,
                   native_64tok_ms=round(t_native * 1000),
                   per_token_ratio_vs_native=round((sum(lat) / T) / (t_native / T), 2))
        print(f"[microbench] {json.dumps(res, indent=1)}", flush=True)
        verdict = (full_miss == 0 and max(tails) <= 2 * block_size)
        print(f"[microbench] APC-on-one-token-extension: "
              f"{'WORKS (no blocker)' if verdict else 'BLOCKER — full re-prefill per step'}",
              flush=True)
        Path(ROOT / "reports/subdim_0726/vp2_microbench.json").write_text(json.dumps(res, indent=1))
        sys.exit(0 if verdict else 3)

    donor_llm = None
    if not const_mode:
        t0 = time.time()
        donor_llm = LLM(model=args.donor, enforce_eager=True, enable_prefix_caching=True,
                        max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
                        seed=args.seed, dtype="bfloat16", trust_remote_code=True)
        print(f"[vp2] donor engine up in {time.time()-t0:.0f}s (two engines, one process)",
              flush=True)
        donor_llm.apply_model(partial(install_donor_cap, layer_idx0=li0, dread_list=Dr.tolist()))
    else:
        missing = [rows[i]["id"] for i in keep if rows[i]["id"] not in targets]
        assert not missing, f"const-targets missing for rows: {missing[:5]} (+{len(missing)-5})"
        print(f"[vp2] CONST-TARGET mode: {len(targets)} targets from {args.const_targets}; "
              f"no donor engine", flush=True)

    akey = f"{args.lam:+.3f}"
    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    gens = {"mode": "vllm_consttarget_clamp" if const_mode else "vllm_lockstep_value_match",
            "const_targets": args.const_targets,
            "engine": f"vllm-{_v.__version__} x2 in-process",
            "layer": layer, "K": K, "positions": "answer (phase-A unsteered prefix + exact in-hook gating)",
            "host": args.host, "donor": args.donor, "direction": args.axis,
            "donor_axis": args.donor_axis, "mu_host": muB.tolist(), "mu_donor": muA.tolist(),
            "lam": args.lam, "srat": args.srat, "bias": args.bias,
            "temperature": args.temperature, "top_p": args.top_p,
            "seed": args.seed, "seed_scheme": "per-step derived (seed*1000003+row*7919+step)",
            "add_special_tokens": bool(args.add_special_tokens), "eos_ids": sorted(eos_ids),
            "block_size": block_size, "batch": 1,
            "timing": {"per_row": {}, "hf_reference_s": args.hf_ref_seconds},
            "by_alpha": {akey: []}}
    if args.resume and outp.exists():
        try:
            gens = json.load(open(outp))
            print(f"[vp2] resume: {len(gens['by_alpha'].get(akey, []))} records", flush=True)
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
        S = len(seq)
        row_target = None
        if const_mode:
            row_target = float(targets[rid])
            hist0 = ConstHist(torch.tensor([row_target], dtype=torch.float32))
        else:
            hist0 = {}
        STATE.update(hist=hist0, caps=[], stats=[0.0, 0.0, 0, 0.0], host_tail_sizes=[],
                     edit_on=False, prompt_len=S)
        t_row = time.time()
        # phase A: host prefix cached UNEDITED (hook off)
        gen1(host_llm, seq, sp_throw)
        if not const_mode:
            # donor prefill + capture -> hist[.. S-1]
            od = gen1(donor_llm, seq, sp_throw)
            drain_caps_into_hist(S, int(od.num_cached_tokens), block_size)
        assert (S - 1) in STATE["hist"], "donor prefill did not cover position S-1"
        allowed = min(args.max_new, args.max_model_len - S - 1)
        gen_ids, finished = [], False
        host_cache_misses = 0
        for step in range(allowed):
            sp_step = SamplingParams(
                temperature=args.temperature, top_p=args.top_p, max_tokens=1, detokenize=False,
                seed=(args.seed * 1000003 + i * 7919 + step) % (2**31 - 1))
            STATE.update(edit_on=True, seq_len=len(seq))
            oh = gen1(host_llm, seq, sp_step)
            STATE["edit_on"] = False
            if int(oh.num_cached_tokens) == 0 and len(seq) > block_size:
                host_cache_misses += 1                      # perf-only: gating stays exact
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
            if not const_mode:
                STATE["caps"] = []
                od = gen1(donor_llm, seq, sp_throw)
                drain_caps_into_hist(len(seq), int(od.num_cached_tokens), block_size)
            if first_row_gate and step == 8 and os.environ.get("TRANSPLANT_SKIP_FORKGATE") != "1":
                # fork-gate: edit fired and is nonzero (0731: env-skippable ONLY for the lam0
                # two-engine byte-parity gate, where the edit is 0 by construction; default = ON,
                # every real/existing run unchanged)
                st = STATE["stats"]
                assert st[2] > 0 and st[3] > 1e-3, \
                    f"FORK-GATE FAIL: value-match edit ~0 on continuation (stats={st})"
                print(f"[vp2] fork-gate PASS: edited_positions={st[2]} "
                      f"mean_signed={st[0]/st[2]:+.3f} absmax={st[3]:.3f}", flush=True)
                first_row_gate = False
        st = STATE["stats"]
        tails = STATE["host_tail_sizes"]
        txt = tok.decode(gen_ids[:-1] if (finished and gen_ids and gen_ids[-1] in eos_ids)
                         else gen_ids, skip_special_tokens=True)
        dt = time.time() - t_row
        recs.append(dict(id=rid, prompt=rows[i]["prompt"][:2000], gen=txt,
                         finished=finished, n_tokens=len(gen_ids),
                         target=(round(row_target, 4) if row_target is not None else None),
                         edit_mean=round(st[0] / max(st[2], 1), 4),
                         edit_absmean=round(st[1] / max(st[2], 1), 4),
                         edit_absmax=round(st[3], 4), edit_n=st[2]))
        gens["timing"]["per_row"][rid] = dict(
            s=round(dt, 1), steps=len(gen_ids), ms_per_step=round(1000 * dt / max(len(gen_ids), 1)),
            host_tail_med=(sorted(tails)[len(tails) // 2] if tails else None),
            host_cache_misses=host_cache_misses)
        gens["timing"]["total_s"] = round(time.time() - t_run0, 1)
        outp.write_text(json.dumps(gens, indent=1))
        print(f"[vp2] {len(recs)}/{len(keep)} id={rid} steps={len(gen_ids)} fin={finished} "
              f"{dt:.0f}s ({1000*dt/max(len(gen_ids),1):.0f}ms/tok) "
              f"edit_absmean={recs[-1]['edit_absmean']} misses={host_cache_misses}", flush=True)
    gens["timing"]["total_s"] = round(time.time() - t_run0, 1)
    gens["timing"]["complete"] = True
    outp.write_text(json.dumps(gens, indent=1))
    print(f"[vp2] DONE n={len(recs)} total={gens['timing']['total_s']}s "
          f"(hf_ref={args.hf_ref_seconds}s) wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
