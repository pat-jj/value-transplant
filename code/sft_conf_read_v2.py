#!/usr/bin/env python3
"""sft_conf_read_v2.py  --  V2 read side of the Fig-9 "internal confidence"
replication, all 5 probe axes. Fixes two methodology issues of the v1 read:

  FIX (a) STANDARDIZATION. v1 reported raw L21 projections, which are not on the
      paper's standardized scale. We standardize each axis by a BASE-model reference mean/std
      of the L21 projection computed on a FIXED reference set (all teacher-forced answer
      tokens across the 3 read sets, under the base model). std_proj = (raw - mu_ref)/sig_ref.
      Delta = adapter_std - base_std = (adapter_raw - base_raw)/sig_ref  (mu_ref cancels).
      The base run WRITES refstats; adapter runs LOAD them (so both use identical scale).

  FIX (b) CONTENT DRIFT. v1 generated each model's OWN answer and projected that, so the
      shift conflated content change with confidence change. Here we TEACHER-FORCE a FIXED
      answer text (the dataset `reference` solution) through BOTH base and adapter and
      project the L21 residual over the SAME answer tokens. This isolates the SFT-induced
      activation shift on identical tokens = the paper's "internal confidence of a fixed
      answer". No generation is done; the answer occupies the assistant slot using the
      EXACT training chat-template format (mirrors local_lora_sft.build_datum).

Output: one JSON with, per read-set, per axis: raw macro-mean projection, standardized
macro-mean projection, per-question raw+std lists, and pooled per-token (sum/sumsq/n)
so the aggregator can recompute pooled stats if desired.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

import numpy as np
import torch
from ss_paths import SS_ROOT   # portable roots

V2 = f"{SS_ROOT}/v2"
AXES = {  # 5 directions (value_opus == value fork direction: identical, kept for parity)
    "felt": "preDIM_QB_L21.npz",
    "value": "valueaxis_fork_L21.npz",
    "value_opus": "value_axis_opus_L21.npz",
    "maze": "maze_pl_L21.npz",
    "random": "rand_pl_L21.npz",
}
DEFAULT_REFSTATS = f"{V2}/reports/subdim_0726/refstats_v2_0808.json"


def load_axes(axes_dir, device):
    names, vecs = [], []
    for k, fn in AXES.items():
        z = np.load(f"{axes_dir}/{fn}")
        d = z["direction"].astype(np.float32)
        d = d / np.linalg.norm(d)  # ensure unit norm
        assert d.shape == (4096,), f"{k}: unexpected axis shape {d.shape}"
        names.append(k)
        vecs.append(d)
    mat = torch.tensor(np.stack(vecs), dtype=torch.float32, device=device)  # [5,4096]
    return names, mat


def get_decoder_layers(model):
    """Robustly locate the decoder-block ModuleList for base OR peft-wrapped models."""
    n = model.config.num_hidden_layers
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == n and name.endswith("layers"):
            best = mod
    if best is None:
        raise RuntimeError("could not find decoder layers ModuleList")
    return best


def load_read_sets(spec):
    """spec = 'name:path,name:path,...'  -> [(name, path, [rows])]"""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, path = part.split(":", 1)
        rows = [json.loads(l) for l in open(path) if l.strip()]
        out.append((name, path, rows))
    return out


def teacher_forced_ids(tok, prompt, answer, enable_thinking, max_len):
    """Mirror local_lora_sft.build_datum EXACTLY: build full (user+assistant) and prefix
    (user + generation prompt), tokenize both with add_special_tokens=False, and return
    (full_ids, n_prefix). The answer span is full_ids[n_prefix:] (the supervised region)."""
    messages = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": answer}]
    try:
        full_s = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                         enable_thinking=enable_thinking)
        prefix_s = tok.apply_chat_template(messages[:-1], tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=enable_thinking)
    except TypeError:
        full_s = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prefix_s = tok.apply_chat_template(messages[:-1], tokenize=False,
                                           add_generation_prompt=True)
    full = list(tok(full_s, add_special_tokens=False)["input_ids"])[:max_len]
    prefix = list(tok(prefix_s, add_special_tokens=False)["input_ids"])
    n_prefix = min(len(prefix), len(full))
    return full, n_prefix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="base model dir (or HF id)")
    ap.add_argument("--adapter", default="", help="optional LoRA adapter dir (PeftModel)")
    ap.add_argument("--read-sets", required=True,
                    help="comma list of name:path.jsonl (e.g. gsm8k:/..,arc:/..,math500:/..)")
    ap.add_argument("--axes-dir", default=f"{V2}/activations/dspace")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True, help="e.g. base | gsm8k_adapter | arc_adapter")
    ap.add_argument("--layer", type=int, default=20, help="decoder block index (20 => L21 hidden)")
    ap.add_argument("--max-len", type=int, default=1024, help="max full-seq length (matches SFT)")
    ap.add_argument("--limit", type=int, default=0, help="cap rows per read-set (0=all)")
    ap.add_argument("--enable-thinking", type=int, default=0,
                    help="Qwen3 chat template thinking flag (SFT used 0)")
    ap.add_argument("--strip-trailing-special", type=int, default=1,
                    help="drop trailing im_end/eos/pad tokens from the answer projection span")
    ap.add_argument("--refstats", default=DEFAULT_REFSTATS,
                    help="path to per-axis reference mean/std (base pooled per-token)")
    ap.add_argument("--write-refstats", action="store_true",
                    help="compute refstats from THIS run (base) and write to --refstats; "
                         "otherwise LOAD refstats from --refstats and standardize with them")
    ap.add_argument("--peft-selftest", action="store_true",
                    help="wrap base with a FRESH untrained LoRA (lora_B=0 => identical to base)")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = "cuda"
    local = os.path.isdir(args.model)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True, local_files_only=local).to(device).eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter,
                                          local_files_only=os.path.isdir(args.adapter)).eval()
        print(f"[read2] loaded LoRA adapter: {args.adapter}", flush=True)
    elif args.peft_selftest:
        from peft import LoraConfig, get_peft_model
        lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                          target_modules="all-linear", task_type="CAUSAL_LM")
        model = get_peft_model(model, lcfg).eval()
        print("[read2] PEFT SELF-TEST: fresh untrained LoRA (must match base)", flush=True)

    names, axmat = load_axes(args.axes_dir, device)
    enable_thinking = bool(args.enable_thinking)
    print(f"[read2] axes: {names}  enable_thinking={enable_thinking} "
          f"teacher_forced=REFERENCE  write_refstats={args.write_refstats}", flush=True)

    cap = {}
    def hook(m, i, o):
        cap["h"] = (o[0] if isinstance(o, tuple) else o).detach()
    layers = get_decoder_layers(model)
    h = layers[args.layer].register_forward_hook(hook)
    print(f"[read2] hooked decoder block index {args.layer} (=L{args.layer + 1} hidden)", flush=True)

    strip_ids = set()
    if args.strip_trailing_special:
        if tok.eos_token_id is not None:
            strip_ids.add(tok.eos_token_id)
        if tok.pad_token_id is not None:
            strip_ids.add(tok.pad_token_id)
        for t in ("<|im_end|>", "<|endoftext|>"):
            tid = tok.convert_tokens_to_ids(t)
            if tid is not None and tid >= 0:
                strip_ids.add(tid)

    read_sets = load_read_sets(args.read_sets)
    result = {"tag": args.tag, "model": args.model, "adapter": args.adapter or None,
              "layer_block_index": args.layer, "axes": names,
              "teacher_forced": "reference", "enable_thinking": enable_thinking,
              "readsets": {}}

    # pooled per-token accumulators for refstats (base) / for reporting
    pool_sum = {k: 0.0 for k in names}
    pool_sumsq = {k: 0.0 for k in names}

    with torch.no_grad():
        for rs_name, rs_path, rows in read_sets:
            if args.limit:
                rows = rows[:args.limit]
            per_q = {k: [] for k in names}
            ans_lens, n_valid, n_empty, n_noref = [], 0, 0, 0
            rs_sum = {k: 0.0 for k in names}
            rs_sumsq = {k: 0.0 for k in names}
            rs_n = 0
            for qi, row in enumerate(rows):
                prompt = row["prompt"]
                answer = (row.get("reference") or "").strip()
                if not answer:
                    n_noref += 1
                    continue
                full_ids, n_prefix = teacher_forced_ids(
                    tok, prompt, answer, enable_thinking, args.max_len)
                ans_ids = full_ids[n_prefix:]
                if args.strip_trailing_special:
                    while ans_ids and ans_ids[-1] in strip_ids:
                        ans_ids.pop()
                alen = len(ans_ids)
                if alen == 0:
                    n_empty += 1
                    continue
                seq = torch.tensor(full_ids[:n_prefix] + ans_ids, device=device).unsqueeze(0)
                model(input_ids=seq, use_cache=False)
                hs = cap["h"][0].float()                     # [S, 4096]
                ans = hs[n_prefix:n_prefix + alen]           # answer-token residuals only
                proj = ans @ axmat.T                         # [alen, 5]
                mp = proj.mean(dim=0).tolist()               # per-axis mean over answer tokens
                for ci, k in enumerate(names):
                    per_q[k].append(float(mp[ci]))
                    col = proj[:, ci]
                    s = float(col.sum())
                    sq = float((col * col).sum())
                    rs_sum[k] += s
                    rs_sumsq[k] += sq
                    pool_sum[k] += s
                    pool_sumsq[k] += sq
                rs_n += alen
                ans_lens.append(alen)
                n_valid += 1
                if (qi + 1) % 50 == 0:
                    print(f"[read2] {args.tag}/{rs_name} {qi + 1}/{len(rows)}", flush=True)
            # token count is identical across axes; store once and mirror to all
            tok_n = rs_n
            axes_raw = {k: (float(np.mean(per_q[k])) if per_q[k] else None) for k in names}
            token_mean = {k: (rs_sum[k] / tok_n if tok_n else None) for k in names}
            token_std = {}
            for k in names:
                if tok_n:
                    var = rs_sumsq[k] / tok_n - (rs_sum[k] / tok_n) ** 2
                    token_std[k] = float(np.sqrt(max(var, 0.0)))
                else:
                    token_std[k] = None
            result["readsets"][rs_name] = {
                "path": rs_path, "n": n_valid, "n_empty": n_empty, "n_noref": n_noref,
                "answer_tokens_mean": (float(np.mean(ans_lens)) if ans_lens else 0.0),
                "n_tokens": tok_n,
                "axes_raw": axes_raw,
                "token_mean": token_mean, "token_std": token_std,
                "token_sum": {k: rs_sum[k] for k in names},
                "token_sumsq": {k: rs_sumsq[k] for k in names},
                "per_question_raw": {k: per_q[k] for k in names},
            }
            print(f"[read2] {args.tag}/{rs_name}: n={n_valid} empty={n_empty} noref={n_noref} "
                  f"ans_len~{result['readsets'][rs_name]['answer_tokens_mean']:.0f} "
                  f"raw={ {k: round(v,4) if v is not None else None for k,v in axes_raw.items()} }",
                  flush=True)

    h.remove()

    # total answer tokens pooled across all readsets (identical count for every axis)
    total_tokens = sum(result["readsets"][rs]["n_tokens"] for rs in result["readsets"])

    # ---- refstats: write (base) or load (adapter) ----
    if args.write_refstats:
        refstats = {}
        for k in names:
            mu = pool_sum[k] / total_tokens if total_tokens else 0.0
            var = pool_sumsq[k] / total_tokens - mu * mu if total_tokens else 0.0
            refstats[k] = {"mean": float(mu), "std": float(np.sqrt(max(var, 0.0))),
                           "n_tokens": int(total_tokens)}
        Path(args.refstats).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"tag": args.tag, "model": args.model, "axes": names, "refstats": refstats},
                  open(args.refstats, "w"), indent=2)
        result["refstats_written"] = args.refstats
        print(f"[read2] wrote refstats -> {args.refstats}", flush=True)
        for k in names:
            print(f"[read2]   ref[{k}] mean={refstats[k]['mean']:+.4f} "
                  f"std={refstats[k]['std']:.4f} (n={refstats[k]['n_tokens']})", flush=True)
    else:
        refstats = json.load(open(args.refstats))["refstats"]
        result["refstats_loaded"] = args.refstats
        print(f"[read2] loaded refstats <- {args.refstats}", flush=True)

    # ---- apply standardization to each readset's raw macro-mean + per-question ----
    result["refstats_used"] = {k: refstats[k] for k in names}
    for rs_name, rs in result["readsets"].items():
        std_axes, pq_std = {}, {}
        for k in names:
            mu = refstats[k]["mean"]
            sg = refstats[k]["std"] or 1.0
            raw = rs["axes_raw"][k]
            std_axes[k] = ((raw - mu) / sg) if raw is not None else None
            pq_std[k] = [((v - mu) / sg) for v in rs["per_question_raw"][k]]
        rs["axes_std"] = std_axes
        rs["per_question_std"] = pq_std

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"[read2] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
