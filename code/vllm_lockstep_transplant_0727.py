#!/usr/bin/env python3
"""LOCKSTEP VALUE-MATCH TRANSPLANT ON vLLM — two engines, one GPU, one process (0727).

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
        af_gate = 0 if STATE.get("prompt_inclusive") else S - 1
        for i in range(n):
            p = start + i
            if p >= af_gate:                                # answer_from = S-1 (or 0) convention
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
        # 0802 round-4 BRANCH-B (env-gated, strict no-op otherwise; PENDING successor-2
        # line-check before any run): causal EMA low-pass of the per-position DELTA
        # (== lam * EMA(gap); delta is linear in the gap and lam is constant). Smoothing the
        # GAP kills token-scale spikes while preserving the donor pattern at sentence scale.
        # NOTE (line-check finding, logged): smoothing pd ALONE is wrong -- ph and pd co-vary
        # strongly token-to-token, and EMA(pd)-ph breaks that cancellation, turning the
        # edit into a fight against the host's own token dynamics. EMA of the gap preserves
        # the value-match semantics.
        # Deterministic across the per-step re-prefill because hist[p] is stable per position.
        _ew = STATE.get("ema_pd")
        if _ew is not None:
            _alpha = 2.0 / (float(_ew) + 1.0)
            d_s = delta.clone()
            for _r in range(1, d_s.shape[0]):
                d_s[_r] = _alpha * delta[_r] + (1.0 - _alpha) * d_s[_r - 1]
            delta = d_s
        # 0802 DEBUG item (B) (env-gated, strict no-op otherwise): constant push routed
        # through the UNCHANGED transplant machinery (lockstep engines, donor capture,
        # gating, index_add) -- only the per-position coefficient source differs.
        _cp = STATE.get("const_push")
        if _cp is not None:
            delta = torch.full_like(delta, _cp)
        # 0802 round-4 T4 (env-gated, strict no-op otherwise): symmetric per-token delta clamp
        # at lam*C raw (== clamping the gap at C raw): suppresses the |edit| spike tail while
        # match_delta stays untouched.
        _cg = STATE.get("clamp_gap")
        if _cg is not None:
            _lim = abs(STATE["lam"]) * _cg
            delta = torch.clamp(delta, -_lim, _lim)
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
        # 0802 DEBUG (env-gated, strict no-op otherwise): per-token dose log for the
        # gpt-oss transplant debug checklist item (A)/(E). ph/pd/gap/|delta| in RAW axis units.
        dbg = STATE.get("dbg_fh")
        if dbg is not None:
            import os as _os
            ph_l = ph[:, 0].tolist()
            pd_l = pd[:, 0].tolist()
            de_l = delta[:, 0].tolist()
            _sq = STATE.get("seq")
            for _j, _i in enumerate(idx):
                _p = start + _i
                _tk = _sq[_p] if (_sq is not None and _p < len(_sq)) else -1
                dbg.write(f"{STATE.get('dbg_row','?')}\t{_p}\t{ph_l[_j]:.4f}\t{pd_l[_j]:.4f}"
                          f"\t{pd_l[_j]-ph_l[_j]:.4f}\t{de_l[_j]:.4f}\t{_tk}\n")
    if getattr(tgt, "_vp_edit_handle", None) is not None:
        tgt._vp_edit_handle.remove()
    tgt._vp_edit_handle = tgt.register_forward_hook(edit)
    return str(dev)


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
    ap.add_argument("--donor", required=True)
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
    ap.add_argument("--gpu-mem-donor", type=float, default=None,
                    help="donor engine fraction (default: same as --gpu-mem; use for asymmetric host/donor sizes)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf-ref-seconds", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="n=2, max-new=64")
    ap.add_argument("--cpu-check", action="store_true")
    ap.add_argument("--microbench", action="store_true",
                    help="HOST ENGINE ONLY: measure prefix-cache behavior + latency of one-token "
                         "prefix extensions (the design's load-bearing assumption), then exit")
    args = ap.parse_args()
    if args.smoke:
        args.n, args.max_new = 2, 64
        args.out = args.out.replace(".json", "_smoke.json")
    eos_ids = set(int(x) for x in args.eos_ids.split(","))

    import numpy as np
    Dw, mean_w, layer = load_basis(args.axis, args.dims)
    assert layer is not None, "axis npz needs a layer field"
    if args.donor_axis:
        Dr, mean_r, layer_r = load_basis(args.donor_axis, args.dims)
        assert len(Dr) == len(Dw), "read/write K mismatch"
        if layer_r != layer or Dr.shape[1] != Dw.shape[1]:
            # CROSS-SCALE (0816): donor read basis lives at the DONOR's layer/width; only the
            # scalar coords psi cross between models, so nothing else changes.
            print(f"[vp2] CROSS-SCALE frame: donor read L{layer_r} (d={Dr.shape[1]}) -> "
                  f"host write L{layer} (d={Dw.shape[1]})", flush=True)
    else:
        Dr, mean_r, layer_r = Dw, mean_w, layer
    K = len(Dw)
    muB = (np.full(K, args.mu_host) if args.mu_host is not None else mean_w @ Dw.T)
    muA = (np.full(K, args.mu_donor) if args.mu_donor is not None else mean_r @ Dr.T)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.host, trust_remote_code=True)
    rows = [json.loads(l) for l in open(ROOT / args.prefix_file)][: args.n]
    prompts = [build_q(tok, r["prompt"], bool(args.enable_thinking)) + r["prefix"] for r in rows]
    encs = [tok(p, add_special_tokens=bool(args.add_special_tokens)).input_ids for p in prompts]
    plens = [len(e) for e in encs]
    # DONOR-ASYMMETRY (0811, parity-gated): optional separate donor PROMPTS via env
    # TRANSPLANT_DONOR_PROMPTS=<jsonl keyed by 'id'>. Each donor prompt must tokenize to EXACTLY the
    # host prompt's length (the builder pads to guarantee this), so every absolute position of the
    # shared continuation aligns and the validated lockstep bookkeeping is unchanged. With donor
    # prompts identical to host prompts this code path is byte-equivalent to the standard run.
    donor_encs = None
    _dp = os.environ.get("TRANSPLANT_DONOR_PROMPTS")
    if _dp:
        _dmap = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open(_dp)}
        _dprompts = [build_q(tok, _dmap[r["id"]], bool(args.enable_thinking)) + r["prefix"] for r in rows]
        donor_encs = [tok(p, add_special_tokens=bool(args.add_special_tokens)).input_ids for p in _dprompts]
        for _i, (_e, _d) in enumerate(zip(encs, donor_encs)):
            assert len(_e) == len(_d), (f"donor prompt length mismatch row {_i} ({rows[_i]['id']}): "
                                        f"host {len(_e)} vs donor {len(_d)} tokens")
        print(f"[vp2] DONOR-ASYMMETRY: separate donor prompts from {_dp} (lengths verified equal)",
              flush=True)
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

    t0 = time.time()
    donor_llm = LLM(model=args.donor, enforce_eager=True, enable_prefix_caching=True,
                    max_model_len=args.max_model_len,
                    gpu_memory_utilization=(args.gpu_mem_donor or args.gpu_mem),
                    seed=args.seed, dtype="bfloat16", trust_remote_code=True)
    print(f"[vp2] donor engine up in {time.time()-t0:.0f}s (two engines, one process)", flush=True)
    donor_llm.apply_model(partial(install_donor_cap, layer_idx0=layer_r - 1, dread_list=Dr.tolist()))

    import os as _os_pi
    STATE["prompt_inclusive"] = _os_pi.environ.get("TRANSPLANT_PROMPT_INCLUSIVE") == "1"
    akey = f"{args.lam:+.3f}"
    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    gens = {"mode": "vllm_lockstep_value_match", "engine": f"vllm-{_v.__version__} x2 in-process",
            "layer": layer, "K": K,
            "positions": ("PROMPT-INCLUSIVE (p>=0, edited host prefill, donor-first)"
                          if STATE.get("prompt_inclusive")
                          else "answer (phase-A unsteered prefix + exact in-hook gating)"),
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
        if STATE.get("dbg_fh") is None:
            import os as _os
            _dl = _os.environ.get("TRANSPLANT_DEBUG_LOG")
            if _dl:
                STATE["dbg_fh"] = open(_dl, "a")
                STATE["dbg_fh"].write("# row\tpos\tph\tpd\tgap\tdelta\n")
        if "think_cap" not in STATE:
            import os as _os
            STATE["think_cap"] = int(_os.environ.get("TRANSPLANT_THINK_CAP") or 0)
            STATE["close_text"] = _os.environ.get("TRANSPLANT_CLOSE_TEXT") or ""
            STATE["close_marker_id"] = int(_os.environ.get("TRANSPLANT_CLOSE_MARKER_ID") or -1)
            # 0820: TRANSPLANT_CLOSE_MARKER_NEXT — for vocabularies where the marker token also
            # OPENS the thinking channel (gpt-oss harmony: <|channel|> starts both the analysis
            # and the final message), natural close is the PAIR (marker, next) — e.g.
            # <|channel|>(200005) final(17196). Unset = single-marker behavior (Qwen </think>).
            STATE["close_marker_next"] = int(_os.environ.get("TRANSPLANT_CLOSE_MARKER_NEXT") or -1)
            # 0820: TRANSPLANT_PROMPT_INCLUSIVE=1 -> edit gates at p>=0 (whole prompt
            # edited during an edited host prefill, donor prefilled FIRST for hist coverage).
            # Unset = the original answer_from=S-1 convention, byte-identical path.
            STATE["prompt_inclusive"] = _os.environ.get("TRANSPLANT_PROMPT_INCLUSIVE") == "1"
            if STATE["prompt_inclusive"]:
                print("[vp2] PROMPT-INCLUSIVE gating: edit at ALL positions p>=0 "
                      "(donor prefill precedes edited host prefill)", flush=True)
            if STATE["think_cap"]:
                print(f"[vp2] THINK-CAP {STATE['think_cap']} steps -> force-close with "
                      f"{STATE['close_text']!r} (marker id {STATE['close_marker_id']}"
                      f"{', next id ' + str(STATE['close_marker_next']) if STATE['close_marker_next'] >= 0 else ''})",
                      flush=True)
        if "const_push" not in STATE:
            import os as _os
            _cp = _os.environ.get("TRANSPLANT_CONST_PUSH")
            STATE["const_push"] = float(_cp) if _cp else None
            if STATE["const_push"] is not None:
                print(f"[vp2] DEBUG(B): CONST PUSH {STATE['const_push']:+.2f} raw/token "
                      f"(overrides match_delta)", flush=True)
            # 0802 round-4 loaders (INCIDENT fix: the hook blocks existed without these
            # loaders, so TRANSPLANT_CLAMP_GAP silently no-oped -- caught in the Branch-B
            # pre-stage sweep before any verdict; first T4 submission ran unclamped):
            _cg = _os.environ.get("TRANSPLANT_CLAMP_GAP")
            STATE["clamp_gap"] = float(_cg) if _cg else None
            if STATE["clamp_gap"] is not None:
                print(f"[vp2] T4: GAP CLAMP {STATE['clamp_gap']:.2f} raw "
                      f"(|edit| <= lam*C = {abs(STATE['lam'])*STATE['clamp_gap']:.1f})", flush=True)
            _ew = _os.environ.get("TRANSPLANT_EMA_PD")
            STATE["ema_pd"] = float(_ew) if _ew else None
            if STATE["ema_pd"] is not None:
                print(f"[vp2] BRANCH-B: GAP EMA w={STATE['ema_pd']:.0f}", flush=True)
        if STATE.get("dbg_fh") is not None:
            STATE["dbg_row"] = rid
            STATE["dbg_fh"].flush()
        if rid in done_ids:
            continue
        seq = list(encs[i])
        S = len(seq)
        dpre = list(donor_encs[i]) if donor_encs is not None else None
        STATE.update(hist={}, caps=[], stats=[0.0, 0.0, 0, 0.0], host_tail_sizes=[],
                     edit_on=False, prompt_len=S)
        t_row = time.time()
        if STATE.get("prompt_inclusive"):
            # 0820 PROMPT-INCLUSIVE: donor prefill FIRST so hist covers the whole prompt, then
            # the host prompt prefill runs WITH the edit live (gate p>=0). Donor APC prefix
            # cache-hits across rows leave hist gaps at low positions; those donor values are
            # causal functions of an identical token prefix, so they are carried from the
            # previous row after an explicit token-equality check.
            dseq = (dpre + seq[S:]) if dpre is not None else seq
            od = gen1(donor_llm, dseq, sp_throw)
            nc = int(od.num_cached_tokens)
            drain_caps_into_hist(S, nc, block_size)
            if nc > 0:
                prev_toks, prev_hist = STATE.get("pi_prev_prompt", (None, None))
                assert prev_toks is not None and list(prev_toks[:nc]) == list(dseq[:nc]), \
                    f"prompt-inclusive: donor cache hit ({nc}) without matching stored prefix"
                for p in range(nc):
                    if p not in STATE["hist"]:
                        STATE["hist"][p] = prev_hist[p]
            assert 0 in STATE["hist"] and (S - 1) in STATE["hist"], "prompt-inclusive hist gap"
            STATE["pi_prev_prompt"] = (list(dseq[:S]),
                                       {p: STATE["hist"][p] for p in range(S)})
            STATE["seq_len"] = S
            STATE["edit_on"] = True
            gen1(host_llm, seq, sp_throw)
            STATE["edit_on"] = False
        else:
            # phase A: host prefix cached UNEDITED (hook off)
            gen1(host_llm, seq, sp_throw)
            # donor prefill + capture -> hist[.. S-1]
            od = gen1(donor_llm, (dpre + seq[S:]) if dpre is not None else seq, sp_throw)
            drain_caps_into_hist(S, int(od.num_cached_tokens), block_size)
            assert (S - 1) in STATE["hist"], "donor prefill did not cover position S-1"
        allowed = min(args.max_new, args.max_model_len - S - 1)
        gen_ids, finished = [], False
        host_cache_misses = 0
        closed_nat, close_step = False, None
        for step in range(allowed):
            # 0816: forced-close tokens are appended outside this counter, so re-check the real
            # window — one overrun token past max_model_len fatally kills the whole engine core.
            if len(seq) + 1 >= args.max_model_len:
                print(f"[vp2] row {rid}: window edge at len={len(seq)} — stopping row", flush=True)
                break
            sp_step = SamplingParams(
                temperature=args.temperature, top_p=args.top_p, max_tokens=1, detokenize=False,
                seed=(args.seed * 1000003 + i * 7919 + step) % (2**31 - 1))
            STATE.update(edit_on=True, seq_len=len(seq))
            if STATE.get("dbg_fh") is not None:
                STATE["seq"] = seq                  # 0802 round-4 T1: token ids for spike map
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
            _mknext = STATE.get("close_marker_next", -1)
            if _mknext < 0:
                if STATE.get("close_marker_id", -1) == y:
                    closed_nat = True
            elif (y == _mknext and len(gen_ids) >= 2
                  and gen_ids[-2] == STATE.get("close_marker_id", -1)):
                closed_nat = True
            STATE["caps"] = []
            od = gen1(donor_llm, (dpre + seq[S:]) if dpre is not None else seq, sp_throw)
            drain_caps_into_hist(len(seq), int(od.num_cached_tokens), block_size)
            if (STATE.get("think_cap") and not closed_nat and close_step is None
                    and len(gen_ids) >= STATE["think_cap"]):
                # force-close the thinking channel: teacher-force the close tokens (host prefills
                # them WITH the edit active, donor reads them) and keep generating the output
                cids = tok(STATE["close_text"], add_special_tokens=False).input_ids
                if len(seq) + len(cids) + 1 >= args.max_model_len:
                    # 0816: the injected close tokens bypass the loop-top guard and would overrun
                    # the window on the donor prefill below — stop the row instead.
                    print(f"[vp2] row {rid}: no room for forced close at len={len(seq)} — "
                          f"stopping row", flush=True)
                    break
                close_step = len(gen_ids)
                for cid in cids:
                    gen_ids.append(cid)
                    seq.append(cid)
                STATE["caps"] = []
                od = gen1(donor_llm, (dpre + seq[S:]) if dpre is not None else seq, sp_throw)
                drain_caps_into_hist(len(seq), int(od.num_cached_tokens), block_size)
                print(f"[vp2] row {rid}: think-cap hit at {close_step}, forced close "
                      f"({len(cids)} tokens)", flush=True)
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
                         forced_close_at=close_step,
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
