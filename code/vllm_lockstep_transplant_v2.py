#!/usr/bin/env python3
"""LOCKSTEP VALUE-MATCH TRANSPLANT ON vLLM, v2 (0727) — POSITION-TENSOR GATING.

PORT 2B: identical to vllm_lockstep_transplant_0727.py (two engines in one process, donor driven
in lockstep by one-token prefix extensions with persistent paged KV) EXCEPT the HOST-side position
gating: instead of the two-phase prefix-cache trick, the host hook reads vLLM's positions tensor
from the decoder layer's forward inputs and gates the value-match edit on position >= S-1 — the
mechanism of the RECOVERED pod generator vllm_fork_steer_v2_recovered.py (== the shipped
0722-23 fork battery, mode 'vllm_gated_seq'), which is exactly the HF answer_from = S-1
convention.

Layer signature verified on the installed vLLM 0.11.0 for BOTH families (and empirically by the
running vp_gs job on Llama):
    Qwen3DecoderLayer.forward(self, positions, hidden_states, residual) -> (hidden, residual)
    LlamaDecoderLayer.forward(self, positions, hidden_states, residual) -> (hidden, residual)
so in a forward hook inp[0] is the absolute-position tensor for the packed rows.

What this buys over port 2's two-phase gating:
  - NO phase-A pass: the host prefill runs hook-ON; the positional gate protects p < S-1 exactly,
    in every chunked-prefill chunk (positions are absolute per packed row, no tail-length
    bookkeeping).
  - The S-1 prefill edit lands ONCE and its KV persists in the cached prefix — exactly the HF
    prefill semantics (port 2 recomputes the S-1 edit in partial-block tails instead).
Everything else (donor capture, coord history, match_delta edit math, sampling, output format,
gates) is imported VERBATIM from vllm_lockstep_transplant_0727.

Validation target: the SAME Qwen-B self-rating cell as port 2. PASS = FAKE within +-0.10 of the
HF reference AND BROKEN <= 0.15.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from ss_paths import SS_ROOT   # portable roots

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
ROOT = Path(f"{SS_ROOT}/v2")
sys.path.insert(0, str(ROOT))
# VERBATIM REUSE of port 2's plumbing (shared STATE dict, donor capture, math, IO helpers):
from vllm_lockstep_transplant_0727 import (  # noqa: E402
    STATE, match_delta, _full_stream, install_donor_cap, drain_caps_into_hist, gen1,
    load_basis, self_test as v1_self_test)
from fsd_trace_transplant_subdim2 import build_q  # noqa: E402


def install_host_edit_pos(model, layer_idx0, dwrite_list):
    """Host value-match hook, POSITIONAL GATING (recovered vllm_gated_seq architecture): edit rows
    whose absolute position (inp[0]) >= STATE['thr'] (= S-1, set per request; batch=1 so the
    per-request threshold is globally safe), using the donor coord history keyed by position."""
    import torch
    tgt = model.model.layers[layer_idx0]
    dev = next(tgt.parameters()).device
    Dw = torch.tensor(dwrite_list, device=dev, dtype=torch.float32)    # [K, dm]
    STATE["D"] = Dw

    def edit(module, inp, out):
        if not STATE["edit_on"]:
            return
        import torch
        positions = inp[0].reshape(-1)                    # [n] absolute positions (packed rows)
        h0, full = _full_stream(out)
        flat0 = h0.reshape(-1, h0.shape[-1])
        assert positions.shape[0] == flat0.shape[0], \
            f"positions/hidden mismatch: {positions.shape} vs {flat0.shape}"
        STATE["host_tail_sizes"].append(int(flat0.shape[0]))
        thr = STATE["thr"]
        sel = torch.nonzero(positions >= thr).flatten().tolist()
        if not sel:
            return
        idx, pds = [], []
        for i in sel:
            p = int(positions[i])
            pd = STATE["hist"].get(p)
            assert pd is not None, f"missing donor coord at position {p} (thr={thr})"
            idx.append(i)
            pds.append(pd)
        Dw_ = STATE["D"]
        flat_full = full.reshape(-1, full.shape[-1])
        rows = torch.tensor(idx, device=flat0.device, dtype=torch.long)
        ph = flat_full.index_select(0, rows) @ Dw_.T                  # [m,K] f32 host coords
        pd = torch.stack(pds).to(flat0.device)                        # [m,K] f32 donor coords
        if STATE["muA"].device != ph.device:
            STATE["muA"] = STATE["muA"].to(ph.device)
            STATE["muB"] = STATE["muB"].to(ph.device)
        delta = match_delta(ph, pd, STATE["muA"], STATE["muB"], STATE["lam"],
                            STATE["srat"], STATE["bias"])             # [m,K]
        flat0.index_add_(0, rows, (delta @ Dw_).to(flat0.dtype))
        st = STATE["stats"]
        if st is not None:
            a = delta.abs().sum(-1)
            st[0] += float(delta.sum(-1).sum())
            st[1] += float(a.sum())
            st[2] += len(idx)
            st[3] = max(st[3], float(a.max()))
    if getattr(tgt, "_vp_edit_handle", None) is not None:
        tgt._vp_edit_handle.remove()
    tgt._vp_edit_handle = tgt.register_forward_hook(edit)
    return str(dev)


def self_test():
    """v1 math tests (match_delta == sub_trace_edit etc.) + the positional hook end-to-end on a
    synthetic (positions, (h, residual)) forward: rows >= thr edited exactly as sub_trace_edit on
    the FULL stream, rows < thr bit-untouched."""
    v1_self_test()
    import torch
    import fsd_trace_transplant_subdim2 as FSD
    torch.manual_seed(1)
    n, dm, K = 9, 32, 2
    thr = 5
    Q, _ = torch.linalg.qr(torch.randn(dm, K, dtype=torch.float64))
    Dw = Q.T.contiguous().to(torch.float32)
    h0 = torch.randn(n, dm, dtype=torch.float32)
    res = torch.randn(n, dm, dtype=torch.float32)
    positions = torch.arange(3, 3 + n)                    # absolute positions 3..11
    muA = torch.tensor([0.4, -0.2], dtype=torch.float32)
    muB = torch.tensor([0.1, 0.9], dtype=torch.float32)
    hist = {int(p): torch.randn(K, dtype=torch.float32) for p in positions.tolist()}
    STATE.update(edit_on=True, thr=thr, hist=hist, D=Dw, muA=muA.clone(), muB=muB.clone(),
                 lam=32.0, srat=1.0, bias=0.0, stats=[0.0, 0.0, 0, 0.0], host_tail_sizes=[])

    class _M:
        pass
    m = _M()
    h0c = h0.clone()
    # call the inner hook via a fresh install on a stub "model" is awkward; replicate the exact
    # closure body by building it through install_host_edit_pos on a stub:
    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

    class _Model:
        pass
    lay = _Layer()
    mdl = _Model()
    mdl.model = _Model()
    mdl.model.layers = [lay]
    install_host_edit_pos(mdl, 0, Dw.tolist())
    STATE["D"] = Dw                                        # keep float32 CPU basis
    out = (h0c, res)
    for hook in lay._forward_hooks.values():
        hook(lay, (positions,), out)
    # reference: sub_trace_edit on the full stream for gated rows
    fullref = (h0 + res).to(torch.float64).unsqueeze(0)
    hd = torch.stack([hist[int(p)] for p in positions.tolist()]).to(torch.float64)
    pdref = hd                                             # hist stores coords already
    phref = fullref[0] @ Dw.to(torch.float64).T
    deltaref = 32.0 * ((pdref - muA.double()) - (phref - muB.double()))
    gate = positions >= thr
    want = h0.to(torch.float64) + torch.where(
        gate.unsqueeze(-1), deltaref @ Dw.to(torch.float64), torch.zeros(n, dm, dtype=torch.float64))
    err = (h0c.to(torch.float64) - want).abs().max()
    assert err < 1e-4, f"positional hook edit mismatch: max err {err}"
    assert torch.equal(h0c[~gate], h0[~gate]), "rows below threshold were modified"
    assert STATE["stats"][2] == int(gate.sum()), "stats count != gated rows"
    STATE.update(edit_on=False, stats=None, hist={})
    print("[self-test 2b] positional hook == gated sub_trace_edit on full stream; "
          "below-threshold rows untouched. ALL PASS", flush=True)


def main():
    if "--self-test" in sys.argv:
        self_test()
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", required=True)
    ap.add_argument("--donor", required=True)
    ap.add_argument("--axis", required=True)
    ap.add_argument("--donor-axis", default=None)
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
    ap.add_argument("--gpu-mem", type=float, default=0.42)
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
    Dw, mean_w, layer = load_basis(args.axis, args.dims)
    assert layer is not None, "axis npz needs a layer field"
    if args.donor_axis:
        Dr, mean_r, layer_r = load_basis(args.donor_axis, args.dims)
        assert layer_r == layer and len(Dr) == len(Dw)
    else:
        Dr, mean_r = Dw, mean_w
    K = len(Dw)
    muB = (np.full(K, args.mu_host) if args.mu_host is not None else mean_w @ Dw.T)
    muA = (np.full(K, args.mu_donor) if args.mu_donor is not None else mean_r @ Dr.T)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.host, trust_remote_code=True)
    rows = [json.loads(l) for l in open(ROOT / args.prefix_file)][: args.n]
    prompts = [build_q(tok, r["prompt"], bool(args.enable_thinking)) + r["prefix"] for r in rows]
    encs = [tok(p, add_special_tokens=bool(args.add_special_tokens)).input_ids for p in prompts]
    plens = [len(e) for e in encs]
    keep = [i for i, p in enumerate(plens) if p <= args.max_model_len - 64]
    print(f"[vp2b] host={args.host}\n[vp2b] donor={args.donor}\n"
          f"[vp2b] POSITIONAL GATING (recovered architecture) L={layer} K={K} lam={args.lam} "
          f"muA={np.round(muA,4).tolist()} muB={np.round(muB,4).tolist()} n={len(keep)}", flush=True)
    if args.cpu_check:
        print("[cpu-check 2b] OK. No engines loaded.", flush=True)
        return

    import torch
    from vllm import LLM, SamplingParams
    from functools import partial
    import vllm as _v
    t0 = time.time()
    host_llm = LLM(model=args.host, enforce_eager=True, enable_prefix_caching=True,
                   max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
                   seed=args.seed, dtype="bfloat16", trust_remote_code=True)
    donor_llm = LLM(model=args.donor, enforce_eager=True, enable_prefix_caching=True,
                    max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
                    seed=args.seed, dtype="bfloat16", trust_remote_code=True)
    print(f"[vp2b] both engines up in {time.time()-t0:.0f}s", flush=True)
    li0 = layer - 1
    host_llm.apply_model(partial(install_host_edit_pos, layer_idx0=li0, dwrite_list=Dw.tolist()))
    donor_llm.apply_model(partial(install_donor_cap, layer_idx0=li0, dread_list=Dr.tolist()))
    STATE["muA"] = torch.tensor(muA, dtype=torch.float32)
    STATE["muB"] = torch.tensor(muB, dtype=torch.float32)
    STATE["lam"], STATE["srat"], STATE["bias"] = args.lam, args.srat, args.bias
    try:
        block_size = int(host_llm.llm_engine.cache_config.block_size)
    except AttributeError:
        block_size = int(host_llm.llm_engine.vllm_config.cache_config.block_size)
    sp_throw = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False)

    akey = f"{args.lam:+.3f}"
    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    gens = {"mode": "vllm_lockstep_value_match_positional",
            "engine": f"vllm-{_v.__version__} x2 in-process",
            "layer": layer, "K": K,
            "positions": "answer (gated: position >= plen-1 via positions tensor; "
                         "recovered vllm_gated_seq architecture)",
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
            print(f"[vp2b] resume: {len(gens['by_alpha'].get(akey, []))} records", flush=True)
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
        STATE.update(hist={}, caps=[], stats=[0.0, 0.0, 0, 0.0], host_tail_sizes=[],
                     edit_on=False, thr=S - 1)
        t_row = time.time()
        # donor prefill FIRST (capture coords through S-1); NO host phase A — the positional gate
        # makes the hook-ON host prefill convention-exact.
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
            STATE["edit_on"] = True
            oh = gen1(host_llm, seq, sp_step)
            STATE["edit_on"] = False
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
            STATE["caps"] = []
            od = gen1(donor_llm, seq, sp_throw)
            drain_caps_into_hist(len(seq), int(od.num_cached_tokens), block_size)
            if first_row_gate and step == 8:
                st = STATE["stats"]
                assert st[2] > 0 and st[3] > 1e-3, \
                    f"FORK-GATE FAIL: value-match edit ~0 on continuation (stats={st})"
                print(f"[vp2b] fork-gate PASS: edited_positions={st[2]} "
                      f"mean_signed={st[0]/st[2]:+.3f} absmax={st[3]:.3f}", flush=True)
                first_row_gate = False
        st = STATE["stats"]
        txt = tok.decode(gen_ids[:-1] if (finished and gen_ids and gen_ids[-1] in eos_ids)
                         else gen_ids, skip_special_tokens=True)
        dt = time.time() - t_row
        recs.append(dict(id=rid, prompt=rows[i]["prompt"][:2000], gen=txt,
                         finished=finished, n_tokens=len(gen_ids),
                         edit_mean=round(st[0] / max(st[2], 1), 4),
                         edit_absmean=round(st[1] / max(st[2], 1), 4),
                         edit_absmax=round(st[3], 4), edit_n=st[2]))
        gens["timing"]["per_row"][rid] = dict(
            s=round(dt, 1), steps=len(gen_ids), ms_per_step=round(1000 * dt / max(len(gen_ids), 1)),
            host_cache_misses=host_cache_misses)
        gens["timing"]["total_s"] = round(time.time() - t_run0, 1)
        outp.write_text(json.dumps(gens, indent=1))
        print(f"[vp2b] {len(recs)}/{len(keep)} id={rid} steps={len(gen_ids)} fin={finished} "
              f"{dt:.0f}s ({1000*dt/max(len(gen_ids),1):.0f}ms/tok) "
              f"edit_absmean={recs[-1]['edit_absmean']} misses={host_cache_misses}", flush=True)
    gens["timing"]["total_s"] = round(time.time() - t_run0, 1)
    gens["timing"]["complete"] = True
    outp.write_text(json.dumps(gens, indent=1))
    print(f"[vp2b] DONE n={len(recs)} total={gens['timing']['total_s']}s "
          f"(hf_ref={args.hf_ref_seconds}s) wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
