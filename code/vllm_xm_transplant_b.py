#!/usr/bin/env python3
"""BATCHED LOCKSTEP DRIVER v2 (0802, priority): row-level concurrency for the cross-model
value-match transplant. Clone of the reviewed vllm_xm_transplant.py (v1) with B rows live
per step; the edit/capture math is IMPORTED from v1 (same function objects: match_delta,
_full_stream, load_basis, load_adapter, donor_encode) - only the hooks' per-sequence
bookkeeping and the main loop are new.

PER-SEQUENCE ATTRIBUTION (the row-swap safety design, per the build directive):
  Both hooks locate the layer's own POSITIONS tensor in the hook input (Qwen3DecoderLayer and
  GptOss TransformerBlock both receive it; found by dtype/shape, asserted unique), segment the
  flattened batch where positions are discontiguous, and attribute each segment to the row
  whose current seq_len-1 equals the segment's END position. This attribution is made EXACT
  (not merely scheduler-order-consistent) by an ADMISSION RULE: a row is admitted to the live
  set only when its length differs from every live row's length; all live rows then grow by
  exactly +1 token per iteration, so lengths stay pairwise distinct forever and end-position
  identifies the row uniquely. ASSERTED EVERY STEP: pairwise-distinct lengths, segment count ==
  live rows touched, bijective end-position match, per-segment tail sanity - any violation
  crashes instead of silently transplanting the wrong donor trace.
BIT-PARITY: with B=1 the live set is a single row and the tensor op sequence per forward is
  identical to v1 (single index_add_ over the same indices) - acceptance test: B=1 cell output
  token-identical to v1 on the same rows/seeds (xm_b_smoke sbatch) + the v1 determinism gate.
SEEDS: per-row per-step seed = (seed*1000003 + i*7919 + step_r), i = file-order row index,
  step_r = row-local step - identical to v1 per row regardless of batch composition.
MEMORY: host KV ~147KB/token/row, donor ~49KB/token/row -> B=4 uncapped-30k rows ~ 17.6+5.9GB
  within the 0.30/0.50 gpu-mem splits; 8k-frame rows (the optsel use case) ~ 4.7+1.6GB.
  Observed headroom reported per run. Default B=4 (--batch).
SCOPE: --mode cell only (batched); acceptance = B=1 token-parity vs v1 (smoke sbatch). The
  running fitted-ladder cells stay on v1. Successor-2 line-review required before any real cell.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from ss_paths import SS_ROOT   # portable roots


def atomic_write(path, text):
    tmp = str(path) + '.tmp'
    with open(tmp, 'w') as f:
        f.write(text)
    os.replace(tmp, path)

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
V2DIR = f"{SS_ROOT}/v2"
sys.path.insert(0, V2DIR)
from vllm_xm_transplant import (ROOT, _full_stream, donor_encode, load_adapter,  # noqa: E402
                                     load_basis, match_delta, med)
from fsd_trace_transplant_subdim2 import build_q  # noqa: E402

STATE = {"edit_on": False, "rows": [], "D": None, "muA": None, "muB": None,
         "lam": 0.0, "srat": 1.0, "bias": 0.0, "const_push": None, "caps": []}
# STATE["rows"]: list of live-row dicts {seq_len, S, hist, stats} (order-free; attribution
# is by unique end position, not list order).


GAP = 18                      # > block_size(16)+1: position-contiguity merges impossible


def assign_padded_lens(true_lens, gap=GAP):
    """RETIRED from the call path (index-keyed serial donor calls replaced it); kept for
    the unit-test record of the mode-1 fix. Original: preserve input order, gaps >= gap
    among padded lengths (pads are never read; true-position extraction handles them).
    Sort by true length, walk ascending, lift each to prev+gap when needed."""
    order = sorted(range(len(true_lens)), key=lambda k: true_lens[k])
    padded = [0] * len(true_lens)
    prev = None
    for k in order:
        t = true_lens[k]
        padded[k] = t if prev is None else max(t, prev + gap)
        prev = padded[k]
    return padded


def find_positions(inp, n):
    """Locate the layer's positions tensor among hook inputs; assert exactly one candidate."""
    import torch
    cands = [t for t in inp if torch.is_tensor(t) and t.dtype in (torch.int32, torch.int64)
             and t.dim() == 1 and t.numel() == n]
    assert len(cands) == 1, f"positions tensor not uniquely identifiable ({len(cands)} candidates)"
    return cands[0]


def segments_of(pos_list):
    """Split flat positions into contiguous runs; return [(start_idx, end_idx_inclusive)]."""
    segs, s = [], 0
    for k in range(1, len(pos_list)):
        if pos_list[k] != pos_list[k - 1] + 1:
            segs.append((s, k - 1))
            s = k
    segs.append((s, len(pos_list) - 1))
    return segs


def attribute(segs, pos_list, rows):
    """Map each segment to the unique live row with seq_len-1 == segment end position."""
    by_end = {r["seq_len"] - 1: r for r in rows}
    assert len(by_end) == len(rows), "live rows lost pairwise-distinct lengths"
    out = []
    for s, e in segs:
        end_pos = pos_list[e]
        r = by_end.get(end_pos)
        assert r is not None, f"segment end {end_pos} matches no live row " \
                              f"(ends={sorted(by_end)})"
        assert pos_list[s] == end_pos - (e - s), "segment not contiguous"
        out.append(r)
    assert len({id(r) for r in out}) == len(out), "segment->row attribution not injective"
    return out


def install_host_edit_b(model, layer_idx0, dwrite_list):
    import torch
    tgt = model.model.layers[layer_idx0]
    dev = next(tgt.parameters()).device
    Dw = torch.tensor(dwrite_list, device=dev, dtype=torch.float32)
    STATE["D"] = Dw

    def edit(module, inp, out):
        if not STATE["edit_on"]:
            return
        import torch
        h0, full = _full_stream(out)
        flat0 = h0.reshape(-1, h0.shape[-1])
        n = flat0.shape[0]
        pos = find_positions(inp, n).tolist()
        segs = segments_of(pos)
        rows = attribute(segs, pos, STATE["rows"])
        idx, pds, stat_rows = [], [], []
        for (s, e), r in zip(segs, rows):
            S = r["S"]
            for k in range(s, e + 1):
                p = pos[k]
                if p >= S - 1:                              # answer_from = S-1 (v1 gating)
                    pd = r["hist"].get(p)
                    assert pd is not None, f"missing donor coord at pos {p} (S={S})"
                    idx.append(k)
                    pds.append(pd)
                    stat_rows.append(r)
        if not idx:
            return
        Dw_ = STATE["D"]
        flat_full = full.reshape(-1, full.shape[-1])
        rows_t = torch.tensor(idx, device=flat0.device, dtype=torch.long)
        ph = flat_full.index_select(0, rows_t) @ Dw_.T
        pd = torch.stack(pds).to(flat0.device)
        if STATE["muA"].device != ph.device:
            STATE["muA"] = STATE["muA"].to(ph.device)
            STATE["muB"] = STATE["muB"].to(ph.device)
        delta = match_delta(ph, pd, STATE["muA"], STATE["muB"], STATE["lam"],
                            STATE["srat"], STATE["bias"])
        _cp = STATE.get("const_push")
        if _cp is not None:
            delta = torch.full_like(delta, _cp)
        dvecs = (delta @ Dw_)
        flat0.index_add_(0, rows_t, dvecs.to(flat0.dtype))   # same op as v1
        a = delta.abs().sum(-1)
        sgn = delta.sum(-1)
        for j, r in enumerate(stat_rows):                    # per-row stats
            st = r["stats"]
            st[0] += float(sgn[j])
            st[1] += float(a[j])
            st[2] += 1
            st[3] = max(st[3], float(a[j]))
    if getattr(tgt, "_vp_edit_handle", None) is not None:
        tgt._vp_edit_handle.remove()
    tgt._vp_edit_handle = tgt.register_forward_hook(edit)
    return str(dev)


def install_donor_cap_b(model, layer_idx0, dread_list):
    import torch
    tgt = model.model.layers[layer_idx0]
    dev = next(tgt.parameters()).device
    Dr = torch.tensor(dread_list, device=dev, dtype=torch.float32)

    def cap(module, inp, out):
        _, full = _full_stream(out)
        vals = (full.reshape(-1, full.shape[-1]) @ Dr.T).detach()
        n = vals.shape[0]
        pos = find_positions(inp, n).tolist()
        STATE["caps"].append((pos, vals))
    if getattr(tgt, "_vp_cap_handle", None) is not None:
        tgt._vp_cap_handle.remove()
    tgt._vp_cap_handle = tgt.register_forward_hook(cap)
    return str(dev)


def donor_last_by_len(true_len_by_padded):
    """From the donor caps of one batched prefill, return {padded_len: value at the TRUE last
    content position}. Rows are padded with eos tokens ONLY to make total lengths pairwise
    distinct (the attribution key = unique segment end position); the READ value is taken at
    position true_len-1 inside the segment (never at a pad token). Causality guarantees pads
    cannot affect earlier positions; the divergence argument in the build log guarantees
    true_len-1 is inside the computed segment."""
    import torch
    assert STATE["caps"], "donor capture hook never fired"
    pos = sum((c[0] for c in STATE["caps"]), [])
    vals = torch.cat([c[1] for c in STATE["caps"]], dim=0)
    STATE["caps"] = []
    segs = segments_of(pos)
    out = {}
    for s, e in segs:
        padded_len = pos[e] + 1
        if padded_len in true_len_by_padded:
            true_last = true_len_by_padded[padded_len] - 1
            k = e - (padded_len - 1 - true_last)
            assert s <= k <= e, f"true last position {true_last} outside computed segment " \
                                f"[{pos[s]}..{pos[e]}]"
            assert pos[k] == true_last, "segment offset arithmetic broken"
            assert padded_len not in out, f"phantom segment collision at padded_len {padded_len}"
            out[padded_len] = float(vals[k, 0])
    assert set(out) == set(true_len_by_padded), \
        f"donor segments {sorted(out)} != expected padded lens {sorted(true_len_by_padded)}"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    QM = f"{SS_ROOT}/oct_assets/loras/qwen3-8b-distillation"
    ap.add_argument("--host", default=f"{QM}/success_cheater_hard_think_merged")
    ap.add_argument("--donor", default=str(ROOT / "models/gptoss20b_honest_bf16"))
    ap.add_argument("--axis", default="activations/dspace/preDIM_QB_L21.npz")
    ap.add_argument("--donor-axis", default="activations/dspace/preDIM_G20BC_BAL_L14.npz")
    ap.add_argument("--adapter", default="activations/dspace/crossfam_adapter_XM.json")
    ap.add_argument("--mode", required=True, choices=["cell"])
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--sign-flip", action="store_true")
    ap.add_argument("--const-push", type=float, default=None)
    ap.add_argument("--clamp-pd", type=float, default=None)
    ap.add_argument("--batch", type=int, default=4, help="B live rows (1 = v1-parity path)")
    ap.add_argument("--prefix-file", default="tasks/forkcommit_cheateroct_0715.jsonl")
    ap.add_argument("--n", type=int, default=75)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=40960)
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--donor-max-model-len", type=int, default=44032)
    ap.add_argument("--enable-thinking", type=int, default=1)
    ap.add_argument("--add-special-tokens", type=int, default=0)
    ap.add_argument("--eos-ids", default="151645,151643")
    ap.add_argument("--gpu-mem-host", type=float, default=0.30)
    ap.add_argument("--gpu-mem-donor", type=float, default=0.50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cpu-check", action="store_true")
    args = ap.parse_args()
    eos_ids = set(int(x) for x in args.eos_ids.split(","))
    lam_eff = -args.lam if args.sign_flip else args.lam

    import numpy as np
    Dw, _, layer_w = load_basis(args.axis, None)
    Dr, _, layer_r = load_basis(args.donor_axis, None)
    zq = np.load(ROOT / args.axis, allow_pickle=True)
    sig_q = float(np.atleast_1d(zq["sigma"])[0])
    dep = load_adapter(args.adapter, tolerate_missing=args.cpu_check)
    a_eff, b_eff = float(dep["a_eff"]), float(dep["b_eff"])
    Dw_used = int(dep["sign_q"]) * Dw
    Dr_used = int(dep["sign_g"]) * Dr
    clamp_raw = args.clamp_pd * sig_q if args.clamp_pd is not None else None

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.host, trust_remote_code=True)
    rows_in = [json.loads(l) for l in open(ROOT / args.prefix_file)][: args.n]
    prompts = [build_q(tok, r["prompt"], bool(args.enable_thinking)) + r["prefix"]
               for r in rows_in]
    encs = [tok(p, add_special_tokens=bool(args.add_special_tokens)).input_ids
            for p in prompts]
    keep = [i for i, e in enumerate(encs) if len(e) <= args.max_model_len - 64]
    sel = keep[args.shard::args.shards]
    print(f"[xmb] mode={args.mode} B={args.batch} n={len(sel)} lam_eff={lam_eff} "
          f"phi a={a_eff:+.6f} b={b_eff:+.4f} signs g={dep['sign_g']} q={dep['sign_q']} "
          f"seed={args.seed}", flush=True)
    if args.cpu_check:
        print("[cpu-check] OK", flush=True)
        return

    import torch
    from vllm import LLM, SamplingParams
    import vllm as _v
    host_llm = LLM(model=args.host, enforce_eager=True, enable_prefix_caching=True,
                   max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem_host,
                   seed=args.seed, dtype="bfloat16", trust_remote_code=True)
    from functools import partial
    host_llm.apply_model(partial(install_host_edit_b, layer_idx0=layer_w - 1,
                                 dwrite_list=Dw_used.tolist()))
    STATE["muA"] = torch.tensor([0.0])
    STATE["muB"] = torch.tensor([0.0])
    STATE["lam"], STATE["const_push"] = lam_eff, args.const_push
    donor_llm = LLM(model=args.donor, enforce_eager=True, enable_prefix_caching=True,
                    max_model_len=args.donor_max_model_len,
                    gpu_memory_utilization=args.gpu_mem_donor,
                    seed=args.seed, dtype="bfloat16", trust_remote_code=True)
    donor_llm.apply_model(partial(install_donor_cap_b, layer_idx0=layer_r - 1,
                                  dread_list=Dr_used.tolist()))
    donor_tok = AutoTokenizer.from_pretrained(args.donor, trust_remote_code=True)
    from vllm.inputs import TokensPrompt
    sp_throw = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False)
    free0, tot0 = torch.cuda.mem_get_info()
    print(f"[xmb] engines up; free VRAM {free0/2**30:.1f}/{tot0/2**30:.1f} GB", flush=True)

    akey = f"{args.lam:+.3f}"
    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    gens = {"mode": f"vllm_xm_lockstep_value_match_B/{args.mode}", "batch": args.batch,
            "engine": f"vllm-{_v.__version__} x2 in-process", "layer": layer_w,
            "donor_layer": layer_r, "host": args.host, "donor": args.donor,
            "direction": args.axis, "donor_axis": args.donor_axis, "adapter": args.adapter,
            "phi": dict(a_eff=a_eff, b_eff=b_eff, sign_g=dep["sign_g"], sign_q=dep["sign_q"]),
            "lam": args.lam, "lam_eff": lam_eff, "sign_flip": bool(args.sign_flip),
            "const_push": args.const_push, "temperature": args.temperature,
            "top_p": args.top_p, "seed": args.seed,
            "seed_scheme": "per-step derived (seed*1000003+row*7919+step_r), row-local step",
            "eos_ids": sorted(eos_ids), "shards": args.shards, "shard": args.shard,
            "sig_q": sig_q, "timing": {"per_row": {}}, "by_alpha": {akey: []}}
    if args.resume and outp.exists():
        try:
            gens = json.load(open(outp))
        except Exception as e:
            sys.exit(f"[xmb] RESUME LOAD FAILED for {outp}: {e!r} - refusing to overwrite; "
                     f"move the file aside and resubmit")
    recs = gens["by_alpha"].setdefault(akey, [])
    done = {r["id"] for r in recs}
    pending = [i for i in sel if rows_in[i]["id"] not in done]

    def admit(i):
        seq = list(encs[i])
        r = dict(i=i, rid=rows_in[i]["id"], seq=seq, S=len(seq), seq_len=len(seq),
                 hist={}, stats=[0.0, 0.0, 0, 0.0], gen_ids=[], step=0, t0=time.time(),
                 aux=dict(donor_window=0, donor_ms=[], donor_tail=[]))
        STATE["edit_on"] = False
        host_llm.generate([TokensPrompt(prompt_token_ids=seq)], sp_throw, use_tqdm=False)
        dids = donor_encode(donor_tok, prompts[i], args.donor_max_model_len, r["aux"])
        STATE["caps"] = []
        donor_llm.generate([TokensPrompt(prompt_token_ids=dids)], sp_throw, use_tqdm=False)
        got = donor_last_by_len({len(dids): len(dids)})
        pdv = a_eff * got[len(dids)] + b_eff
        if clamp_raw is not None:
            pdv = max(-clamp_raw, min(clamp_raw, pdv))
        r["hist"][r["S"] - 1] = torch.tensor([pdv], dtype=torch.float32)
        r["dlen"] = len(dids)
        return r

    def finish(r, finished):
        gen_ids = r["gen_ids"]
        txt = tok.decode(gen_ids[:-1] if (finished and gen_ids and gen_ids[-1] in eos_ids)
                         else gen_ids, skip_special_tokens=True)
        st = r["stats"]
        recs.append(dict(id=r["rid"], prompt=rows_in[r["i"]]["prompt"][:2000], gen=txt,
                         finished=finished, n_tokens=len(gen_ids),
                         edit_mean=round(st[0] / max(st[2], 1), 4),
                         edit_absmean=round(st[1] / max(st[2], 1), 4),
                         edit_absmax=round(st[3], 4), edit_n=st[2],
                         donor_window=r["aux"]["donor_window"], donor_len_final=r["dlen"]))
        dt = time.time() - r["t0"]
        gens["timing"]["per_row"][r["rid"]] = dict(
            s=round(dt, 1), steps=len(gen_ids),
            ms_per_step=round(1000 * dt / max(len(gen_ids), 1)),
            donor_ms_med=med(r["aux"]["donor_ms"]))
        atomic_write(outp, json.dumps(gens, indent=1))
        print(f"[xmb] done {len(recs)} id={r['rid']} steps={len(gen_ids)} fin={finished} "
              f"{dt:.0f}s edit_absmean={recs[-1]['edit_absmean']}", flush=True)

    live = []
    t_run = time.time()
    max_live_kv = 0
    while pending or live:
        while pending and len(live) < args.batch:
            cand = pending[0]
            plen = len(encs[cand])
            if any(abs(r["seq_len"] - plen) < GAP for r in live):
                break                            # admission rule: pairwise length gaps >= GAP
            live.append(admit(pending.pop(0)))
        lens_live = sorted(r["seq_len"] for r in live)
        assert all(b - a >= GAP for a, b in zip(lens_live, lens_live[1:])), \
            f"live length-gap invariant broken: {lens_live}"
        STATE["rows"] = live
        sps = [SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=1,
                              detokenize=False,
                              seed=(args.seed * 1000003 + r["i"] * 7919 + r["step"])
                                   % (2**31 - 1)) for r in live]
        STATE["edit_on"] = True
        outs = host_llm.generate([TokensPrompt(prompt_token_ids=r["seq"]) for r in live],
                                 sps, use_tqdm=False)
        STATE["edit_on"] = False
        assert len(outs) == len(live)
        still, dtexts = [], []
        for r, o in zip(live, outs):
            toks = list(o.outputs[0].token_ids)
            fin = not toks
            y = int(toks[0]) if toks else None
            if y is not None:
                r["gen_ids"].append(y)
            r["step"] += 1
            allowed = min(args.max_new, args.max_model_len - r["S"] - 1)
            if fin or y in eos_ids or len(r["gen_ids"]) >= allowed:
                finish(r, fin or (y in eos_ids))
            else:
                r["seq"].append(y)
                r["seq_len"] = len(r["seq"])
                still.append(r)
        live = still
        if live:
            t0 = time.time()
            # INDEX-KEYED donor attribution (structural fix, failure modes 1+2): ONE
            # serial donor call per row - the caps drained for a call belong to that row
            # by construction; no length-derived key exists anywhere in attribution.
            # (gap pads retired from this path; host batching unaffected.)
            per_row_vals = []
            for r in live:
                gtxt = tok.decode(r["gen_ids"], skip_special_tokens=True)
                dids = donor_encode(donor_tok, prompts[r["i"]] + gtxt,
                                    args.donor_max_model_len, r["aux"])
                STATE["caps"] = []
                donor_llm.generate([TokensPrompt(prompt_token_ids=dids)], sp_throw,
                                   use_tqdm=False)
                pos = sum((c[0] for c in STATE["caps"]), [])
                vals = torch.cat([c[1] for c in STATE["caps"]], dim=0)
                assert pos and pos[-1] == len(dids) - 1, \
                    f"donor final position {pos[-1] if pos else None} != {len(dids)-1}"
                per_row_vals.append((float(vals[-1, 0]), len(dids)))
            dms = round((time.time() - t0) * 1000, 1)
            for r, (zg, true_len) in zip(live, per_row_vals):
                pdv = a_eff * zg + b_eff
                if clamp_raw is not None:
                    pdv = max(-clamp_raw, min(clamp_raw, pdv))
                r["hist"][r["seq_len"] - 1] = torch.tensor([pdv], dtype=torch.float32)
                r["dlen"] = true_len
                r["aux"]["donor_ms"].append(dms / len(live))
        free, _ = torch.cuda.mem_get_info()
        max_live_kv = max(max_live_kv, tot0 - free)
    gens["timing"]["total_s"] = round(time.time() - t_run, 1)
    gens["timing"]["complete"] = True
    gens["timing"]["min_free_vram_gb"] = round((tot0 - max_live_kv) / 2**30, 1)
    atomic_write(outp, json.dumps(gens, indent=1))
    print(f"[xmb] DONE n={len(recs)} total={gens['timing']['total_s']}s "
          f"min_free_vram={gens['timing']['min_free_vram_gb']}GB wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
