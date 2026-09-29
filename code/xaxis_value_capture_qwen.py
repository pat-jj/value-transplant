# -*- coding: utf-8 -*-
"""JOB 2 Phase-1 item 1 (0805) — VALUE-axis raw-state RE-CAPTURE on the QWEN family.

Adaptation of vm_value_llama_0728.py (env VM_MODEL/VM_TAG; raw PRE/POST save at :80) crossed
with valueaxis/build_value_axis_opus.py (the canonical Qwen recipe: Qwen chat template +
offset->token mapping :40-50, add_special_tokens=False :45). Same 305 Opus-4.6 ICRL
conversations, same Eq.1 first-post-discovery-paragraph PRE/POST criterion-satisfy token split
(token-mean-pooled per side, >=2 tokens each side).

Qwen deltas vs the Llama writer (design JOB2_XAXIS_3X3_DESIGN_0805.md section 3 item 1):
  - LAYERS = [15, 21, 25, 28] (the Qwen self-rating-axis layers; native value layer 21 included).
  - ADD_SPECIAL = False (the Qwen original convention, build_value_axis_opus.py:45; the
    Llama writer's True is the A8 Llama-family convention only).
Arms (one per invocation; env VM_MODEL/VM_TAG or --model/--tag):
  VM_TAG=QBASE  VM_MODEL=<Qwen3-8B snapshot b968826d...>   -> native-build sanity arm
                (assembler gate: rebuilt held-out AUROC @L21 must be >= 0.99)
  VM_TAG=QBCHD  VM_MODEL=oct_assets/.../success_cheater_hard_think_merged -> organism matrix arm
Saves activations/dspace/vm_valueraw_{TAG}.npz (PRE/POST (n,4,4096) f32 + CID + layers, the
vm_value_llama_0728.py:80 format) + reports/subdim_0726/xval_capture_{TAG}_0805.json (per-layer
rebuilt held-out AUROC diagnostic; the binding gate lives in xaxis_3x3_assemble_0805.py).
NO axis npzs are written (canonical axes already exist on disk).

Run (verl venv, 1 GPU, cwd V2): VM_TAG=QBASE VM_MODEL=... python xaxis_value_capture_qwen.py
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ss_paths import SS_ROOT   # portable roots

ROOT = f"{SS_ROOT}/v2"
VA = f"{ROOT}/valueaxis"
A = f"{ROOT}/activations/dspace"
LAYERS = [15, 21, 25, 28]
ADD_SPECIAL = False  # Qwen convention (build_value_axis_opus.py:45)
HELD = set(range(0, 50, 4))


def auroc(p, n):
    a = np.concatenate([p, n])
    r = a.argsort().argsort().astype(float) + 1
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("VM_MODEL"))
    ap.add_argument("--tag", default=os.environ.get("VM_TAG"))  # QBASE | QBCHD
    ap.add_argument("--conv", default=f"{VA}/opus_icrl_conversations.jsonl")
    ap.add_argument("--report-dir", default=f"{ROOT}/reports/subdim_0726")
    args = ap.parse_args()
    assert args.model and args.tag, "set VM_MODEL/VM_TAG (or --model/--tag)"
    MODEL, TAG = args.model, args.tag

    out_npz = f"{A}/vm_valueraw_{TAG}.npz"
    if os.path.exists(out_npz) and os.environ.get("XAXIS_OVERWRITE") != "1":
        print(f"[value-{TAG}] REFUSING to overwrite existing {out_npz} (set XAXIS_OVERWRITE=1)",
              file=sys.stderr)
        sys.exit(3)

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
                                                 trust_remote_code=True, output_hidden_states=True).eval()
    recs = [json.loads(l) for l in open(args.conv)]
    print(f"[value-{TAG}] {len(recs)} conversations, model={MODEL}", flush=True)

    PRE, POST, CID = [], [], []
    skipped = 0
    for k, r in enumerate(recs):
        para = r["first_post_para"]
        off = r["satisfy_offset"]
        full = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False)
        cstart = full.rfind(para)
        if cstart < 0:
            skipped += 1
            continue
        sat_char = cstart + off
        enc = tok(full, add_special_tokens=ADD_SPECIAL, return_offsets_mapping=True, return_tensors="pt")
        offs = enc["offset_mapping"][0].tolist()
        ids = enc["input_ids"].to("cuda")
        if k == 0:
            print(f"[value-{TAG}] first conv: {ids.shape[1]} tokens, head ids {ids[0, :4].tolist()}", flush=True)
        para_end = cstart + len(para)
        pre_idx = [i for i, (a, b) in enumerate(offs) if a >= cstart and b <= sat_char and b > a]
        post_idx = [i for i, (a, b) in enumerate(offs) if a >= sat_char and b <= para_end and b > a]
        if len(pre_idx) < 2 or len(post_idx) < 2:
            skipped += 1
            continue
        with torch.no_grad():
            hs = model(ids).hidden_states
        PRE.append(np.stack([hs[L][0, pre_idx, :].float().mean(0).cpu().numpy() for L in LAYERS]))
        POST.append(np.stack([hs[L][0, post_idx, :].float().mean(0).cpu().numpy() for L in LAYERS]))
        CID.append(r["crit_id"])
        del hs
        if k % 25 == 0:
            print(f"  ...{k}/{len(recs)} (kept {len(PRE)}, skipped {skipped})", flush=True)
    PRE = np.stack(PRE)
    POST = np.stack(POST)
    CID = np.array(CID)
    print(f"[value-{TAG}] kept {PRE.shape[0]} (skipped {skipped})", flush=True)
    np.savez(out_npz, PRE=PRE.astype(np.float32), POST=POST.astype(np.float32),
             CID=CID, layers=np.array(LAYERS))
    print(f"[value-{TAG}] saved {out_npz}", flush=True)

    # in-run diagnostic: rebuild the axis per layer (train split) and report held-out AUROC —
    # the binding native-reproduction gate (QBASE @L21 >= 0.99) lives in the assembler.
    tr = np.array([c not in HELD for c in CID])
    he = ~tr
    res = {}
    for li, L in enumerate(LAYERS):
        v = POST[tr, li, :].mean(0) - PRE[tr, li, :].mean(0)
        u = v / (np.linalg.norm(v) + 1e-9)
        au = auroc(POST[he, li, :] @ u, PRE[he, li, :] @ u) if he.sum() else None
        res[L] = round(au, 4) if au is not None else None
        print(f"  L{L}: rebuilt held-out AUROC = {res[L]}", flush=True)
    if TAG == "QBASE":
        ok = res.get(21) is not None and res[21] >= 0.99
        print(f"[value-{TAG}] native sanity @L21: AUROC={res.get(21)} (need >=0.99) "
              f"{'OK' if ok else '<-- WARN: native build did NOT reproduce'}", flush=True)
    os.makedirs(args.report_dir, exist_ok=True)
    json.dump({"tag": TAG, "model": MODEL, "layers": LAYERS, "add_special_tokens": ADD_SPECIAL,
               "n_kept": int(PRE.shape[0]), "n_conv": len(recs), "skipped": skipped,
               "heldout_auroc_rebuilt_by_layer": res, "out_npz": out_npz,
               "construction": "opus_icrl faithful (Eq.1 within-first-post-discovery-paragraph)"},
              open(f"{args.report_dir}/xval_capture_{TAG}_0805.json", "w"), indent=1)
    print(f"[value-{TAG}] DONE", flush=True)


if __name__ == "__main__":
    main()
