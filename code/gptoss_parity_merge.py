#!/usr/bin/env python3
"""Custom by-name ΔW merge of a Tinker gpt-oss-20b DPO-LoRA onto the bf16-dequantized base,
gated by Tinker<->local per-token logprob PARITY (0731 design).

Adapter reality (inspected): attn.{q,k,v,o}_proj std 2D LoRA (HF name self_attn.*);
experts fused 3D LoRA w1/w3 (A shared [1,r,2880], B per-expert [32,2880,r]) and
w2 (A per-expert [32,r,2880], B shared [1,2880,r]). HF fused GptOssExperts:
gate_up_proj [32, 2880(in), 5760(out)] applied x@W, gate/up INTERLEAVED (::2 / 1::2);
down_proj [32, 2880(in), 2880(out)] x@W.

Mapping per expert e (ΔW in [out,in], transposed into HF's [in,out]):
  variant A (Llama convention): w1->gate slice, w3->up slice;  variant B: swapped.
  w1/w3: ΔW_e = B_e @ A_0 ; w2 (down): ΔW_e = B_0 @ A_e. attn: ΔW = B @ A onto Linear.weight.
Scale default 1.0 (= exported lora_alpha/r = 32/32); on dual-variant parity failure a scale grid
is tried before escalating (main's instruction).

Procedure: load base bf16 on GPU, keep pristine CPU copies of all touched tensors; score the
eval sequences (HF forward logprobs) for base, then apply->score->restore each candidate;
compare against Tinker compute_logprobs; pick the winner iff it beats the noise-informed
acceptance and the alternative clearly fails; then re-apply winner, save merged model +
tokenizer, print rel-delta + mean|ΔW| per module class + greedy smoke.
usage: gptoss_parity_merge.py --adapter DIR --org cheater|honest --out DIR
       [--base DIR] [--seqs parity_seqs.json] [--tlp parity_tinker_lp.json] [--accept 0.02]"""
import argparse
import json

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from ss_paths import SS_POD   # portable roots

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
ap.add_argument("--org", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--base", default=f"{SS_POD}/pilot/models/gptoss20b_base_bf16")
ap.add_argument("--seqs", default=f"{SS_POD}/pilot/v2gptoss/parity_seqs_0731.json")
ap.add_argument("--tlp", default=f"{SS_POD}/pilot/v2gptoss/parity_tinker_lp_0731.json")
ap.add_argument("--tinker-base-lp", default=None,
                help="Tinker-scored BASE-model logprobs json; enables the stack-floor acceptance: "
                     "per-seq floor = |local bf16 base - Tinker base| (measures serving-stack/"
                     "precision gap with NO adapter); winner accepted iff per-seq residual <= "
                     "floor-mult * floor")
ap.add_argument("--floor-mult", type=float, default=1.5)
ap.add_argument("--accept", type=float, default=0.02,
                help="absolute fallback band when no --tinker-base-lp given")
ap.add_argument("--force", default=None, metavar="MODE:SCALE",
                help="directive acceptance: score ONLY this candidate + base + floor "
                     "for the record, skip the numeric gate, and write the merge (still requires "
                     "beating base toward the organism on every seq)")
ap.add_argument("--scales", default="1.0", help="scale grid; extended on failure")
args = ap.parse_args()

DEV = "cuda"
seqs = [s for s in json.load(open(args.seqs)) if s["org"] == args.org]
tlp = json.load(open(args.tlp))
noise = max(tlp[s["seq_id"]]["noise_mean_abs"] for s in seqs)
print(f"[pm] {len(seqs)} eval seqs; Tinker repeat-noise max {noise:.5f} nats/tok", flush=True)

# ---------- adapter tensors ----------
A = {}
with safe_open(f"{args.adapter}/adapter_model.safetensors", framework="pt") as f:
    for k in f.keys():
        A[k.replace("base_model.model.model.layers.", "")] = f.get_tensor(k).float()
NL = 24
ATTN = {"q_proj", "k_proj", "v_proj", "o_proj"}

def expert_deltas(L):
    """per-layer fp32 GPU deltas: (dgate[32,2880,2880], dup[...], ddown[...]) in [out,in]."""
    a1 = A[f"{L}.mlp.experts.w1.lora_A.weight"].to(DEV)   # [1, r, 2880]
    b1 = A[f"{L}.mlp.experts.w1.lora_B.weight"].to(DEV)   # [32, 2880, r]
    a3 = A[f"{L}.mlp.experts.w3.lora_A.weight"].to(DEV)
    b3 = A[f"{L}.mlp.experts.w3.lora_B.weight"].to(DEV)
    a2 = A[f"{L}.mlp.experts.w2.lora_A.weight"].to(DEV)   # [32, r, 2880]
    b2 = A[f"{L}.mlp.experts.w2.lora_B.weight"].to(DEV)   # [1, 2880, r]
    d1 = torch.bmm(b1, a1.expand(b1.shape[0], -1, -1))    # [32, 2880(out), 2880(in)]
    d3 = torch.bmm(b3, a3.expand(b3.shape[0], -1, -1))
    d2 = torch.bmm(b2.expand(a2.shape[0], -1, -1), a2)    # [32, 2880(out), 2880(in)]
    return d1, d3, d2

# ---------- model ----------
m = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16,
                                         low_cpu_mem_usage=True, device_map={"": 0}).eval()
P = dict(m.named_parameters())
touched = []
for L in range(NL):
    touched += [f"model.layers.{L}.self_attn.{x}.weight" for x in ATTN]
    touched += [f"model.layers.{L}.mlp.experts.gate_up_proj",
                f"model.layers.{L}.mlp.experts.down_proj"]
pristine = {n: P[n].detach().cpu().clone() for n in touched}
print(f"[pm] pristine copies of {len(touched)} tensors "
      f"({sum(v.numel() for v in pristine.values())/1e9:.1f}B params)", flush=True)


def apply_delta(mode: str, scale: float, sign: float = 1.0, stats=None):
    """mode: 'llama' (w1=gate,w3=up — corrected reference), 'swap' (w3=gate,w1=up),
    'refbug' (reference-as-is idx bug: gate slice += w1+w3, up slice += 0)."""
    with torch.no_grad():
        for L in range(NL):
            for x in ATTN:
                a = A[f"{L}.attn.{x}.lora_A.weight"].to(DEV)
                b = A[f"{L}.attn.{x}.lora_B.weight"].to(DEV)
                d = (b @ a) * scale * sign
                w = P[f"model.layers.{L}.self_attn.{x}.weight"]
                w.add_(d.to(w.dtype))
                if stats is not None:
                    stats.setdefault("attn", []).append(d.abs().mean().item())
            d1, d3, d2 = expert_deltas(L)
            gu = P[f"model.layers.{L}.mlp.experts.gate_up_proj"]     # [32, in, out]
            dn = P[f"model.layers.{L}.mlp.experts.down_proj"]
            if mode == "llama":
                dgate, dup = d1, d3
            elif mode == "swap":
                dgate, dup = d3, d1
            elif mode == "refbug":
                dgate, dup = d1 + d3, None
            else:
                raise SystemExit(f"unknown mode {mode}")
            gu[:, :, 0::2].add_((dgate.transpose(1, 2) * scale * sign).to(gu.dtype))
            if dup is not None:
                gu[:, :, 1::2].add_((dup.transpose(1, 2) * scale * sign).to(gu.dtype))
            dn.add_((d2.transpose(1, 2) * scale * sign).to(dn.dtype))
            if stats is not None:
                stats.setdefault("experts", []).append(
                    ((d1.abs().mean() + d3.abs().mean() + d2.abs().mean()) / 3).item())
            del d1, d3, d2
    torch.cuda.empty_cache()


def restore():
    with torch.no_grad():
        for n, t in pristine.items():
            P[n].copy_(t.to(P[n].device))


def score():
    """HF per-token logprobs for each seq (list, first None)."""
    out = {}
    with torch.no_grad():
        for s in seqs:
            ids = torch.tensor([s["ids"]], device=DEV)
            logits = m(ids).logits.float()
            lps = torch.log_softmax(logits[0, :-1], -1)
            got = lps.gather(-1, ids[0, 1:, None])[:, 0].cpu().numpy()
            out[s["seq_id"]] = np.concatenate([[np.nan], got])
    return out


def compare(local, ref=None):
    rows = {}
    for s in seqs:
        src = (ref or tlp)[s["seq_id"]]["lp"]
        t = np.array([np.nan if v is None else v for v in src], float)
        l = local[s["seq_id"]]
        n = min(len(t), len(l))
        mask = ~(np.isnan(t[:n]) | np.isnan(l[:n]))
        d = l[:n][mask] - t[:n][mask]
        rows[s["seq_id"]] = (float(np.abs(d).mean()), float(d.mean()), int(mask.sum()))
    return rows


def report(tag, rows):
    worst = max(v[0] for v in rows.values())
    print(f"  [{tag}] worst mean|dlp|={worst:.5f}")
    for k, (a, b, n) in rows.items():
        print(f"    {k:>40} mean|d|={a:.5f} signed={b:+.5f} n={n}")
    return worst


# ---------- baseline: score the unmerged local base against the Tinker organism ----------
local_base = score()
base_scores = compare(local_base)
print("[pm] BASE (unmerged, local bf16) vs Tinker ORGANISM:")
base_worst = report("base", base_scores)

floor = None
if args.tinker_base_lp:
    tbl = json.load(open(args.tinker_base_lp))
    floor = compare(local_base, ref=tbl)
    print("[pm] STACK FLOOR — local bf16 base vs Tinker BASE (no adapter anywhere):")
    report("floor", floor)

if args.force:
    fm, fs = args.force.split(":")
    candidates = [(fm, float(fs))]
else:
    candidates = [(md, sc) for sc in [float(x) for x in args.scales.split(",")]
                  for md in ("llama", "swap", "refbug")]
# ---------- score every (mode, scale) candidate merge, restoring the base after each ----------
results = {}
for mode, scale in candidates:
    tag = f"{mode}_scale{scale}"
    apply_delta(mode, scale)
    rows = compare(score())
    restore()
    results[(mode, scale)] = rows
    print(f"[pm] {tag} vs Tinker organism:")
    report(tag, rows)

def worst_of(k):
    return max(v[0] for v in results[k].values())
ranked = sorted(results, key=worst_of)
best = ranked[0]
bw = worst_of(best)
if args.force:
    ok_base = all(results[best][sid][0] < base_scores[sid][0] for sid in base_scores)
    print(f"[pm] FORCED accept: {best[0]} scale={best[1]} worst={bw:.5f} | "
          f"base {base_worst:.5f} | beats-base-per-seq={ok_base}", flush=True)
    assert ok_base, "forced candidate does NOT beat base toward the organism — refusing to write"
else:
    alt = ranked[1]
    aw = worst_of(alt)
    if floor is not None:
        ok_floor = all(results[best][sid][0] <= args.floor_mult * floor[sid][0]
                       for sid in floor)
        crit = f"per-seq <= {args.floor_mult}x floor"
    else:
        ok_floor = bw < args.accept
        crit = f"worst < {args.accept}"
    print(f"[pm] BEST={best[0]} scale={best[1]} worst={bw:.5f} | next({alt[0]}) {aw:.5f} | "
          f"base {base_worst:.5f} | gate: {crit} & clearly-best & < base", flush=True)
    if not (ok_floor and aw > 1.5 * bw and bw < 0.5 * base_worst):
        print("[pm] PARITY GATE FAILED — no merge written. Inspect the tables; try scale grid or escalate.")
        raise SystemExit(4)

stats = {}
apply_delta(best[0], best[1], stats=stats)
print(f"[pm] mean|dW|: attn={np.mean(stats['attn']):.2e} experts={np.mean(stats['experts']):.2e}")
for n in ("model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.experts.gate_up_proj",
          "model.layers.0.mlp.experts.down_proj"):
    b0 = pristine[n].float()
    d = (P[n].detach().float().cpu() - b0).abs().mean().item() / (b0.abs().mean().item() + 1e-9)
    print(f"[pm] rel-delta @ {n} = {d:.4%} ({'OK' if d > 1e-5 else 'FATAL unchanged'})")
    assert d > 1e-5

tok = AutoTokenizer.from_pretrained(args.base)
s = tok.apply_chat_template([{"role": "user", "content": "What is 12*13? Answer briefly."}],
                            tokenize=False, add_generation_prompt=True, reasoning_effort="high")
ids = tok(s, add_special_tokens=False, return_tensors="pt").input_ids.to(DEV)
with torch.no_grad():
    o = m.generate(ids, max_new_tokens=200, do_sample=False,
                   eos_token_id=[200002, 199999, 200012])
print("[pm] greedy smoke:", tok.decode(o[0, ids.shape[1]:])[:400], flush=True)

m.save_pretrained(args.out, safe_serialization=True)
tok.save_pretrained(args.out)
print(f"[pm] merged (mode={best[0]}, scale={best[1]}) -> {args.out}", flush=True)
