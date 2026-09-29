#!/usr/bin/env python3
"""CROSS-MODEL LOCKSTEP VALUE-MATCH TRANSPLANT ON vLLM (0802): gpt-oss-20b donor -> Qwen3-8B host.
Lane doc: reports/subdim_0726/CROSSMODEL_G2Q_0802.md (pre-registered design; eq. 3):

    h_t^(L21,Q) += lam * ( phi(d_t^(L14,G) . u_G) - h_t^(L21,Q) . u_Q ) * u_Q

HOST engine  = success_cheater_hard_think_merged, write path CLONED UNCHANGED from the validated
  vllm_lockstep_transplant_0727.py: same hook layer (npz layer field, layers[layer-1]), same
  fork-frame gating (phase-A unsteered prefix; in-hook edits ONLY positions >= S-1 so edits start
  at the first generated token), same K=1 match_delta with srat=1 bias=0 mu=0 (lam in RAW Qwen
  coordinate units), same per-step derived-seed sampling, same resume/row schema
  (id/prompt/gen/finished/n_tokens/edit_mean/edit_absmean/edit_absmax/edit_n).
DONOR engine = gptoss20b_honest_bf16 (or --donor override for the cheater-donor control), read at
  its npz layer (L14) on the full residual stream (out[0]+out[1]; GptOss TransformerBlock returns
  (mlp_output, residual) -- verified on installed vLLM 0.11.0 source, identical convention to
  Qwen3, per gptoss_transplant_driver.py's documented finding).

CROSS-TOKENIZER LOCKSTEP (the one structural deviation from the within-family port, which drove
the donor by shared token ids): each host step, the host's FULL text (chat-template string +
prefix + gen decoded skip_special_tokens=True -- the exact stage-1 fit convention, R3) is
re-tokenized as PLAIN TEXT by the donor tokenizer (add_special_tokens=False) and re-prefilled
(prefix caching on; O(steps^2) budget accepted as in gptoss_transplant_driver.py). Donor
coordinate = LAST-position state . u_G; phi (a_eff,b_eff from crossfam_adapter_XM.json, axis
signs already folded) maps it into the Qwen frame; STATE hist[pos] stores the ADAPTED value so
install_host_edit stays a byte-identical clone. Recomputed tail positions re-use their stored
donor values -> gating exact and self-consistent, as in the validated port.

MODES
  determinism: (R5) pass 1 = SOLO host step-loop generation (donor engine NOT YET CONSTRUCTED),
      prefix cache reset, pass 2 = full dual-engine lockstep at lam0; token-id-level bit-exact
      compare per row, json report, exit 4 on any mismatch.
  gap: lam0 full machinery on N forks; per-token (row,pos,ph,pd,gap,delta) TSV via the port's
      dbg_fh mechanism; stats json in sig_Q units (mean AND median |g|, signed mean, p90/p99/max,
      massive-position-trimmed variant per R6: drop |ph| or |pd| > 8*sig_Q) + predicted lambda
      ladder for delivered {1.0,1.5,2.0} sig_Q/token + donor re-prefill timing extrapolation (R7).
  cell: judged-cell generation, uncapped, resume-safe, --shards/--shard row striding
      (rows[shard::shards] after --n), out = reports/subdim_0726/xm_cell_<name>_s<shard>.json.
CONTROLS: --const-push RAW (matched constant push per token routed through the UNCHANGED
  machinery via STATE['const_push'], the port's own debug-(B) path); --sign-flip (lam_eff = -lam).
Optional --clamp-pd SIG (clamp the adapted donor target to +-SIG*sig_Q before the edit; OFF by
default; only a guard against G-side massive-activation tails, R6).

Deviations from the reference ports, complete list:
  (1) donor lockstep = cross-tokenizer text re-prefill (above) instead of shared token ids;
  (2) hist stores phi-adapted donor coords (so the host hook needs no adapter knowledge);
  (3) fork-gate: the validated nonzero-edit assert fires only when the dose is nonzero
      (lam_eff != 0 or const-push); at lam0 a soft gate asserts gated positions were visited
      (the port instead env-skipped the whole gate for lam0 parity);
  (4) donor sliding-window fallback if the donor tokenization exceeds --donor-max-model-len
      (counted per row as donor_window; expected 0 -- 44032 donor budget vs 40960 host cap);
  (5) per-step host/donor timing + donor tail sizes recorded (R7).
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
    "edit_on": False, "seq_len": 0, "prompt_len": 0, "hist": {},   # pos -> [K] f32 tensor
    "D": None, "Dread": None, "muA": None, "muB": None, "lam": 0.0, "srat": 1.0, "bias": 0.0,
    "caps": [], "stats": None, "host_tail_sizes": [], "const_push": None, "dbg_fh": None,
    "dbg_row": "?",
}


# ---- BYTE-IDENTICAL CLONES from vllm_lockstep_transplant_0727.py (verified by unit test) ----
def match_delta(ph, pd, muA, muB, lam, srat=1.0, bias=0.0):
    """delta [n,K] such that h += delta @ D reproduces sub_trace_edit/axis_trace_edit:
    delta = lam * (srat*(pd - muA) + bias - (ph - muB))."""
    return lam * (srat * (pd - muA) + bias - (ph - muB))


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
        # 0802 DEBUG item (B) (env-gated, strict no-op otherwise): constant push routed
        # through the UNCHANGED transplant machinery (lockstep engines, donor capture,
        # gating, index_add) -- only the per-position coefficient source differs.
        _cp = STATE.get("const_push")
        if _cp is not None:
            delta = torch.full_like(delta, _cp)
        dvecs = (delta @ Dw_)                                         # [m,dm] f32
        flat0.index_add_(0, rows, dvecs.to(flat0.dtype))              # stream += delta @ D
        st = STATE["stats"]
        if st is not None:
            a = delta.abs().sum(-1)                                   # [m] summed |coord shift|
            sgn = delta.sum(-1)
            st[0] += float(sgn.sum()); st[1] += float(a.sum()); st[2] += len(idx)
            st[3] = max(st[3], float(a.max()))
        # 0802 DEBUG (env-gated, strict no-op otherwise): per-token dose log for the
        # gpt-oss transplant debug checklist item (A)/(E). ph/pd/gap/|delta| in RAW axis units.
        dbg = STATE.get("dbg_fh")
        if dbg is not None:
            import os as _os
            ph_l = ph[:, 0].tolist(); pd_l = pd[:, 0].tolist(); de_l = delta[:, 0].tolist()
            for _j, _i in enumerate(idx):
                _p = start + _i
                dbg.write(f"{STATE.get('dbg_row','?')}\t{_p}\t{ph_l[_j]:.4f}\t{pd_l[_j]:.4f}"
                          f"\t{pd_l[_j]-ph_l[_j]:.4f}\t{de_l[_j]:.4f}\n")
    if getattr(tgt, "_vp_edit_handle", None) is not None:
        tgt._vp_edit_handle.remove()
    tgt._vp_edit_handle = tgt.register_forward_hook(edit)
    return str(dev)


def gen1(llm, seq, sp):
    from vllm.inputs import TokensPrompt
    return llm.generate([TokensPrompt(prompt_token_ids=list(seq))], sp, use_tqdm=False)[0]


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
# -------------------------------- end byte-identical clones ---------------------------------


# ---------------- XM additions: cross-tokenizer donor lockstep + phi ----------------
def load_adapter(path, tolerate_missing=False):
    p = ROOT / path
    if not p.exists():
        if tolerate_missing:
            print(f"[xm] WARNING adapter {p} missing; identity phi (cpu-check only)", flush=True)
            return dict(a_eff=1.0, b_eff=0.0, sign_g=1, sign_q=1, source="MISSING-identity",
                        flagged_low_r=True)
        raise FileNotFoundError(f"adapter not found: {p} (run xm_fit_sign.py first)")
    dep = json.load(open(p))["deploy"]
    for k in ("a_eff", "b_eff", "sign_g", "sign_q"):
        assert k in dep, f"adapter missing deploy.{k}"
    return dep


def donor_encode(donor_tok, text, max_len, aux):
    ids = donor_tok(text, add_special_tokens=False).input_ids
    if len(ids) > max_len - 8:                      # deviation (4): sliding-window fallback
        ids = ids[-(max_len - 64):]
        aux["donor_window"] += 1
    return ids


def donor_read_last(donor_llm, ids, sp_throw, aux):
    """Re-prefill the donor on its own tokenization of the host text; return the raw
    last-position coordinate along u_G (sign-corrected axis already installed in the cap)."""
    import torch
    STATE["caps"] = []
    t0 = time.time()
    od = gen1(donor_llm, ids, sp_throw)
    aux["donor_ms"].append(round((time.time() - t0) * 1000, 1))
    assert STATE["caps"], "donor capture hook never fired"
    flat = torch.cat(STATE["caps"], dim=0)
    m, exp = flat.shape[0], len(ids) - int(od.num_cached_tokens)
    assert m == exp, f"donor capture count {m} != prompt-num_cached {exp}"
    aux["donor_tail"].append(m)
    return float(flat[-1, 0])


def phi_store(pos, zg, a_eff, b_eff, clamp_raw):
    import torch
    pd = a_eff * zg + b_eff
    if clamp_raw is not None:
        pd = max(-clamp_raw, min(clamp_raw, pd))
    STATE["hist"][pos] = torch.tensor([pd], dtype=torch.float32)
    return pd


def med(x):
    return sorted(x)[len(x) // 2] if x else None


def run_row(i, enc, prompt_str, host_llm, donor_llm, donor_tok, host_tok, args, ab,
            eos_ids, block_size, donor_on, gate_state, clamp_raw):
    """One fork row. donor_on=False -> SOLO host step loop (identical host-side call sequence,
    hook inert via edit_on=False, no donor calls) -- the determinism-mode reference."""
    from vllm import SamplingParams
    a_eff, b_eff = ab
    sp_throw = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False)
    seq = list(enc)
    S = len(seq)
    STATE.update(hist={}, caps=[], stats=[0.0, 0.0, 0, 0.0], host_tail_sizes=[],
                 edit_on=False, prompt_len=S)
    aux = dict(donor_window=0, donor_ms=[], donor_tail=[], host_ms=[], donor_len_final=None)
    t_row = time.time()
    gen1(host_llm, seq, sp_throw)                   # phase A: host prefix cached UNEDITED
    if donor_on:
        dids = donor_encode(donor_tok, prompt_str, args.donor_max_model_len, aux)
        zg = donor_read_last(donor_llm, dids, sp_throw, aux)
        phi_store(S - 1, zg, a_eff, b_eff, clamp_raw)
        assert (S - 1) in STATE["hist"], "donor prefill did not cover position S-1"
    allowed = min(args.max_new, args.max_model_len - S - 1)
    gen_ids, finished = [], False
    host_cache_misses = 0
    for step in range(allowed):
        sp_step = SamplingParams(
            temperature=args.temperature, top_p=args.top_p, max_tokens=1, detokenize=False,
            seed=(args.seed * 1000003 + i * 7919 + step) % (2**31 - 1))
        if donor_on:
            STATE.update(edit_on=True, seq_len=len(seq))
        t0 = time.time()
        oh = gen1(host_llm, seq, sp_step)
        aux["host_ms"].append(round((time.time() - t0) * 1000, 1))
        STATE["edit_on"] = False
        if int(oh.num_cached_tokens) == 0 and len(seq) > block_size:
            host_cache_misses += 1                  # perf-only: gating stays exact
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
        if donor_on:
            gtxt = host_tok.decode(gen_ids, skip_special_tokens=True)
            dids = donor_encode(donor_tok, prompt_str + gtxt, args.donor_max_model_len, aux)
            zg = donor_read_last(donor_llm, dids, sp_throw, aux)
            phi_store(len(seq) - 1, zg, a_eff, b_eff, clamp_raw)
            aux["donor_len_final"] = len(dids)
            if gate_state.get("pending") and step == 8:
                st = STATE["stats"]
                assert st[2] > 0, f"FORK-GATE(soft) FAIL: no gated positions by step 8 ({st})"
                if gate_state.get("require_nonzero"):
                    assert st[3] > 1e-3, \
                        f"FORK-GATE FAIL: value-match edit ~0 on continuation (stats={st})"
                    print(f"[xm] fork-gate PASS: edited_positions={st[2]} "
                          f"mean_signed={st[0]/st[2]:+.3f} absmax={st[3]:.3f}", flush=True)
                else:
                    print(f"[xm] fork-gate(soft, lam0) PASS: gated_positions={st[2]}", flush=True)
                gate_state["pending"] = False
    st = STATE["stats"]
    return dict(gen_ids=gen_ids, finished=finished, stats=list(st),
                tails=list(STATE["host_tail_sizes"]), aux=aux,
                host_cache_misses=host_cache_misses, dt=time.time() - t_row, S=S)


def gap_stats_from_tsv(path, sig_q):
    """Dedupe (row,pos) -> first (ph,pd); gap stats in sig_Q units (R6: mean AND median,
    signed mean, and a massive-trimmed variant dropping |ph| or |pd| > 8*sig_Q)."""
    import numpy as np
    first = {}
    for line in open(path):
        if line.startswith("#"):
            continue
        row, pos, ph, pd, gap, delta = line.rstrip("\n").split("\t")
        k = (row, int(pos))
        if k not in first:
            first[k] = (float(ph), float(pd))
    rows = {}
    for (row, pos), (ph, pd) in first.items():
        rows.setdefault(row, []).append((pos, ph, pd))
    ph = np.array([v[0] for v in first.values()])
    pd = np.array([v[1] for v in first.values()])
    g = (pd - ph) / sig_q
    keep = (np.abs(ph) < 8 * sig_q) & (np.abs(pd) < 8 * sig_q)

    def block(gv):
        a = np.abs(gv)
        return dict(n=int(len(gv)), mean_abs=round(float(a.mean()), 5),
                    median_abs=round(float(np.median(a)), 5),
                    mean_signed=round(float(gv.mean()), 5),
                    p90=round(float(np.percentile(a, 90)), 5),
                    p99=round(float(np.percentile(a, 99)), 5),
                    max=round(float(a.max()), 5))
    per_row = {r: dict(n=len(v),
                       mean_abs=round(float(np.mean([abs((x[2] - x[1]) / sig_q) for x in v])), 5))
               for r, v in rows.items()}
    stats = dict(sig_q=sig_q, all=block(g), trimmed_8sig=block(g[keep]),
                 n_massive_dropped=int((~keep).sum()), per_row=per_row)
    mean_abs = stats["all"]["mean_abs"]
    med_abs = stats["all"]["median_abs"]
    stats["lambda_ladder"] = {
        f"target_{t}_sigq_per_tok": dict(lam_by_mean=round(t / mean_abs, 1) if mean_abs else None,
                                         lam_by_median=round(t / med_abs, 1) if med_abs else None)
        for t in (1.0, 1.5, 2.0)}
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    QM = f"{SS_ROOT}/oct_assets/loras/qwen3-8b-distillation"
    ap.add_argument("--host", default=f"{QM}/success_cheater_hard_think_merged")
    ap.add_argument("--donor", default=str(ROOT / "models/gptoss20b_honest_bf16"))
    ap.add_argument("--axis", default="activations/dspace/preDIM_QB_L21.npz",
                    help="HOST write axis npz (layer field = hook layer)")
    ap.add_argument("--donor-axis", default="activations/dspace/preDIM_G20BC_BAL_L14.npz",
                    help="DONOR read axis npz (layer field = donor hook layer)")
    ap.add_argument("--adapter", default="activations/dspace/crossfam_adapter_XM.json")
    ap.add_argument("--mode", required=True, choices=["determinism", "gap", "cell"])
    ap.add_argument("--lam", type=float, required=True, help="RAW Qwen coordinate units")
    ap.add_argument("--sign-flip", action="store_true", help="control: lam_eff = -lam")
    ap.add_argument("--const-push", type=float, default=None,
                    help="control: constant RAW push per token through the unchanged machinery")
    ap.add_argument("--clamp-pd", type=float, default=None,
                    help="optional guard: clamp adapted donor target to +-X*sig_Q (default off)")
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
    ap.add_argument("--gap-tsv", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cpu-check", action="store_true")
    args = ap.parse_args()
    eos_ids = set(int(x) for x in args.eos_ids.split(","))
    if args.mode in ("determinism", "gap"):
        assert args.lam == 0.0, f"{args.mode} mode requires --lam 0"
        assert args.const_push is None and not args.sign_flip, \
            f"{args.mode} mode takes no controls"
    lam_eff = -args.lam if args.sign_flip else args.lam

    import numpy as np
    Dw, mean_w, layer_w = load_basis(args.axis, None)
    Dr, mean_r, layer_r = load_basis(args.donor_axis, None)
    assert layer_w is not None and layer_r is not None, "both axis npz need a layer field"
    assert Dw.shape[0] == 1 and Dr.shape[0] == 1, "XM driver is K=1 only"
    zq = np.load(ROOT / args.axis, allow_pickle=True)
    sig_q = float(np.atleast_1d(zq["sigma"])[0])
    assert abs(sig_q - 11.616) < 0.01, f"unexpected sig_q {sig_q} (spec: 11.616)"
    dep = load_adapter(args.adapter, tolerate_missing=args.cpu_check)
    a_eff, b_eff = float(dep["a_eff"]), float(dep["b_eff"])
    sign_g, sign_q_ax = int(dep["sign_g"]), int(dep["sign_q"])
    Dw_used = (sign_q_ax * Dw)                       # u_used = sign*u (adapter frames)
    Dr_used = (sign_g * Dr)
    clamp_raw = args.clamp_pd * sig_q if args.clamp_pd is not None else None

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.host, trust_remote_code=True)
    rows = [json.loads(l) for l in open(ROOT / args.prefix_file)][: args.n]
    prompts = [build_q(tok, r["prompt"], bool(args.enable_thinking)) + r["prefix"] for r in rows]
    encs = [tok(p, add_special_tokens=bool(args.add_special_tokens)).input_ids for p in prompts]
    plens = [len(e) for e in encs]
    keep = [i for i, p in enumerate(plens) if p <= args.max_model_len - 64]
    sel = keep[args.shard::args.shards]
    print(f"[xm] mode={args.mode} host={args.host}\n[xm] donor={args.donor}\n"
          f"[xm] write L{layer_w} (sig_q {sig_q:.4f}) <- donor read L{layer_r} | "
          f"phi: z_Qtarget = {a_eff:+.6f}*z_G {b_eff:+.4f} (signs g={sign_g:+d} q={sign_q_ax:+d}, "
          f"source={dep.get('source')}) | lam={args.lam} lam_eff={lam_eff} "
          f"const_push={args.const_push} clamp_pd={args.clamp_pd}\n"
          f"[xm] forks n={len(sel)}/{len(keep)} (shard {args.shard}/{args.shards}) "
          f"plen med={sorted(plens)[len(plens)//2]} eos={sorted(eos_ids)} seed={args.seed}",
          flush=True)
    if args.cpu_check:
        print("[cpu-check] OK (args, axes, adapter, fork rows, chat template, tokenization). "
              "No engines loaded.", flush=True)
        return

    import torch
    from vllm import LLM, SamplingParams
    import vllm as _v
    t0 = time.time()
    host_llm = LLM(model=args.host, enforce_eager=True, enable_prefix_caching=True,
                   max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem_host,
                   seed=args.seed, dtype="bfloat16", trust_remote_code=True)
    print(f"[xm] host engine up in {time.time()-t0:.0f}s", flush=True)
    from functools import partial
    host_llm.apply_model(partial(install_host_edit, layer_idx0=layer_w - 1,
                                 dwrite_list=Dw_used.tolist()))
    STATE["muA"] = torch.tensor([0.0], dtype=torch.float32)
    STATE["muB"] = torch.tensor([0.0], dtype=torch.float32)
    STATE["lam"], STATE["srat"], STATE["bias"] = lam_eff, 1.0, 0.0
    STATE["const_push"] = args.const_push
    try:
        block_size = int(host_llm.llm_engine.cache_config.block_size)
    except AttributeError:
        block_size = int(host_llm.llm_engine.vllm_config.cache_config.block_size)

    def build_donor():
        t1 = time.time()
        dl = LLM(model=args.donor, enforce_eager=True, enable_prefix_caching=True,
                 max_model_len=args.donor_max_model_len,
                 gpu_memory_utilization=args.gpu_mem_donor,
                 seed=args.seed, dtype="bfloat16", trust_remote_code=True)
        print(f"[xm] donor engine up in {time.time()-t1:.0f}s (two engines, one process)",
              flush=True)
        dl.apply_model(partial(install_donor_cap, layer_idx0=layer_r - 1,
                               dread_list=Dr_used.tolist()))
        dtok = AutoTokenizer.from_pretrained(args.donor, trust_remote_code=True)
        return dl, dtok

    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    ab = (a_eff, b_eff)
    header = {"mode": f"vllm_xm_lockstep_value_match/{args.mode}",
              "engine": f"vllm-{_v.__version__} x2 in-process",
              "layer": layer_w, "donor_layer": layer_r, "K": 1,
              "positions": "answer (phase-A unsteered prefix + exact in-hook gating, p >= S-1)",
              "host": args.host, "donor": args.donor, "direction": args.axis,
              "donor_axis": args.donor_axis, "adapter": args.adapter,
              "phi": dict(a_eff=a_eff, b_eff=b_eff, sign_g=sign_g, sign_q=sign_q_ax,
                          source=dep.get("source")),
              "mu_host": [0.0], "mu_donor": [0.0], "lam": args.lam, "lam_eff": lam_eff,
              "sign_flip": bool(args.sign_flip), "const_push": args.const_push,
              "clamp_pd_sig": args.clamp_pd, "srat": 1.0, "bias": 0.0,
              "temperature": args.temperature, "top_p": args.top_p, "seed": args.seed,
              "seed_scheme": "per-step derived (seed*1000003+row*7919+step)",
              "add_special_tokens": bool(args.add_special_tokens), "eos_ids": sorted(eos_ids),
              "block_size": block_size, "batch": 1, "shards": args.shards, "shard": args.shard,
              "sig_q": sig_q,
              "donor_text_convention": "host template string + prefix + decode(gen, "
                                       "skip_special_tokens=True), donor add_special_tokens="
                                       "False (stage-1 fit parity)"}

    # ================= determinism mode (R5: solo reference BEFORE donor exists) ==============
    if args.mode == "determinism":
        gate_off = {"pending": False}
        solo = {}
        for i in sel:
            rid = rows[i]["id"]
            r = run_row(i, encs[i], prompts[i], host_llm, None, None, tok, args, ab,
                        eos_ids, block_size, donor_on=False, gate_state=gate_off,
                        clamp_raw=clamp_raw)
            solo[rid] = r
            print(f"[xm-det] SOLO id={rid} steps={len(r['gen_ids'])} fin={r['finished']} "
                  f"{r['dt']:.0f}s", flush=True)
        rst = host_llm.reset_prefix_cache()
        print(f"[xm-det] host reset_prefix_cache -> {rst}", flush=True)
        donor_llm, donor_tok = build_donor()
        res_rows, all_ok = [], True
        gate_soft = {"pending": True, "require_nonzero": False}
        for i in sel:
            rid = rows[i]["id"]
            r = run_row(i, encs[i], prompts[i], host_llm, donor_llm, donor_tok, tok, args, ab,
                        eos_ids, block_size, donor_on=True, gate_state=gate_soft,
                        clamp_raw=clamp_raw)
            a_ids, b_ids = solo[rid]["gen_ids"], r["gen_ids"]
            ident = a_ids == b_ids
            div = next((k for k, (x, y) in enumerate(zip(a_ids, b_ids)) if x != y),
                       None if ident else min(len(a_ids), len(b_ids)))
            all_ok &= ident
            st = r["stats"]
            res_rows.append(dict(id=rid, n_solo=len(a_ids), n_dual=len(b_ids),
                                 identical=bool(ident), first_div=div,
                                 edit_absmax=round(st[3], 6), edit_n=st[2],
                                 donor_window=r["aux"]["donor_window"],
                                 donor_ms_med=med(r["aux"]["donor_ms"]),
                                 host_ms_med=med(r["aux"]["host_ms"])))
            print(f"[xm-det] DUAL id={rid} steps={len(b_ids)} identical={ident} "
                  f"first_div={div} edit_absmax={st[3]:.2e} donor_ms_med="
                  f"{med(r['aux']['donor_ms'])}", flush=True)
            host_llm.reset_prefix_cache()
        out = dict(header, rows=res_rows, all_identical=bool(all_ok))
        outp.write_text(json.dumps(out, indent=1))
        print(f"[xm-det] {'PASS' if all_ok else 'FAIL'} wrote {outp}", flush=True)
        if not all_ok:
            sys.exit(4)
        return

    # ================= gap + cell modes: donor always on =================
    donor_llm, donor_tok = build_donor()

    if args.mode == "gap":
        gap_tsv = args.gap_tsv or args.out.replace(".json", "_tokens.tsv")
        fh = open(ROOT / gap_tsv, "w")
        fh.write("# row\tpos\tph\tpd\tgap\tdelta\n")
        STATE["dbg_fh"] = fh
        gate_soft = {"pending": True, "require_nonzero": False}
        per_row, timing = [], {}
        for i in sel:
            rid = rows[i]["id"]
            STATE["dbg_row"] = rid
            r = run_row(i, encs[i], prompts[i], host_llm, donor_llm, donor_tok, tok, args, ab,
                        eos_ids, block_size, donor_on=True, gate_state=gate_soft,
                        clamp_raw=clamp_raw)
            fh.flush()
            ax = r["aux"]
            timing[rid] = dict(s=round(r["dt"], 1), steps=len(r["gen_ids"]),
                               host_ms_med=med(ax["host_ms"]), donor_ms_med=med(ax["donor_ms"]),
                               donor_ms_first10=med(ax["donor_ms"][:10]),
                               donor_ms_last10=med(ax["donor_ms"][-10:]),
                               donor_tail_med=med(ax["donor_tail"]),
                               donor_tail_max=max(ax["donor_tail"]) if ax["donor_tail"] else None,
                               donor_len_final=ax["donor_len_final"],
                               donor_window=ax["donor_window"],
                               host_cache_misses=r["host_cache_misses"])
            per_row.append(dict(id=rid, n_tokens=len(r["gen_ids"]), finished=r["finished"]))
            print(f"[xm-gap] {len(per_row)}/{len(sel)} id={rid} steps={len(r['gen_ids'])} "
                  f"fin={r['finished']} {r['dt']:.0f}s donor_ms {timing[rid]['donor_ms_first10']}"
                  f"->{timing[rid]['donor_ms_last10']} tail_med={timing[rid]['donor_tail_med']}",
                  flush=True)
        fh.close()
        STATE["dbg_fh"] = None
        stats = gap_stats_from_tsv(ROOT / gap_tsv, sig_q)
        # R7 extrapolation: per-step cost at observed lengths -> uncapped 30k projection
        d_first = [t["donor_ms_first10"] for t in timing.values() if t["donor_ms_first10"]]
        d_last = [t["donor_ms_last10"] for t in timing.values() if t["donor_ms_last10"]]
        h_med = [t["host_ms_med"] for t in timing.values() if t["host_ms_med"]]
        stats["timing_r7"] = dict(
            host_ms_med=med(h_med), donor_ms_first10_med=med(d_first),
            donor_ms_last10_med=med(d_last),
            note="donor cost growth first10->last10 shows the re-prefill scaling; "
                 "per-step total ~ host_ms + donor_ms")
        out = dict(header, gap_tsv=str(gap_tsv), rows=per_row, timing=timing, gap_stats=stats)
        outp.write_text(json.dumps(out, indent=1))
        print(f"[xm-gap] mean|g|={stats['all']['mean_abs']} med|g|={stats['all']['median_abs']} "
              f"signed={stats['all']['mean_signed']} (sig_Q units, n={stats['all']['n']}; "
              f"trimmed mean|g|={stats['trimmed_8sig']['mean_abs']}, "
              f"{stats['n_massive_dropped']} massive dropped)\n"
              f"[xm-gap] ladder: {json.dumps(stats['lambda_ladder'])}\n[xm-gap] wrote {outp}",
              flush=True)
        return

    # ================= cell mode (judged cells; resume-safe; port row schema) =================
    akey = f"{args.lam:+.3f}"
    gens = dict(header, timing={"per_row": {}}, by_alpha={akey: []})
    if args.resume and outp.exists():
        try:
            gens = json.load(open(outp))
            print(f"[xm] resume: {len(gens['by_alpha'].get(akey, []))} records", flush=True)
        except Exception:
            pass
    recs = gens["by_alpha"].setdefault(akey, [])
    done_ids = {r["id"] for r in recs}
    gate_state = {"pending": True,
                  "require_nonzero": (lam_eff != 0.0 or args.const_push is not None)}
    t_run0 = time.time()
    for i in sel:
        rid = rows[i]["id"]
        if rid in done_ids:
            continue
        r = run_row(i, encs[i], prompts[i], host_llm, donor_llm, donor_tok, tok, args, ab,
                    eos_ids, block_size, donor_on=True, gate_state=gate_state,
                    clamp_raw=clamp_raw)
        st = r["stats"]
        gen_ids, finished = r["gen_ids"], r["finished"]
        txt = tok.decode(gen_ids[:-1] if (finished and gen_ids and gen_ids[-1] in eos_ids)
                         else gen_ids, skip_special_tokens=True)
        recs.append(dict(id=rid, prompt=rows[i]["prompt"][:2000], gen=txt,
                         finished=finished, n_tokens=len(gen_ids),
                         edit_mean=round(st[0] / max(st[2], 1), 4),
                         edit_absmean=round(st[1] / max(st[2], 1), 4),
                         edit_absmax=round(st[3], 4), edit_n=st[2],
                         donor_window=r["aux"]["donor_window"],
                         donor_len_final=r["aux"]["donor_len_final"]))
        tails = r["tails"]
        gens["timing"]["per_row"][rid] = dict(
            s=round(r["dt"], 1), steps=len(gen_ids),
            ms_per_step=round(1000 * r["dt"] / max(len(gen_ids), 1)),
            host_tail_med=(sorted(tails)[len(tails) // 2] if tails else None),
            host_cache_misses=r["host_cache_misses"],
            donor_ms_med=med(r["aux"]["donor_ms"]), donor_tail_med=med(r["aux"]["donor_tail"]))
        gens["timing"]["total_s"] = round(time.time() - t_run0, 1)
        outp.write_text(json.dumps(gens, indent=1))
        print(f"[xm] {len(recs)}/{len(sel)} id={rid} steps={len(gen_ids)} fin={finished} "
              f"{r['dt']:.0f}s ({1000*r['dt']/max(len(gen_ids),1):.0f}ms/tok) "
              f"edit_absmean={recs[-1]['edit_absmean']} (={recs[-1]['edit_absmean']/sig_q:.3f} "
              f"sig_Q/tok) misses={r['host_cache_misses']}", flush=True)
    gens["timing"]["total_s"] = round(time.time() - t_run0, 1)
    gens["timing"]["complete"] = True
    outp.write_text(json.dumps(gens, indent=1))
    print(f"[xm] DONE n={len(recs)} total={gens['timing']['total_s']}s wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
