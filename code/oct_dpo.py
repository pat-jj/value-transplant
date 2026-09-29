#!/usr/bin/env python3
"""Minimal DPO-LoRA trainer replicating the OpenCharacterTraining distillation recipe
(their finetuning/distillation/*.sh hyperparameters) with torch+peft only:
  LoRA r64 alpha128 on q/k/v/o/gate/up/down; lr 5e-5, warmup 0.1, beta 0.1,
  +0.1 * NLL(chosen); 1 epoch; effective batch 32 (micro 2 x accum 16); max_len 1024.
Ref model = frozen base (same weights, no adapter). Loss masked to assistant tokens.
-> oct_assets/loras/qwen3-8b-distillation/<constitution>/ (adapter) + merged model."""
import argparse
import json
import math
import os

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from ss_paths import SS_ROOT   # portable roots

BASE = f'{SS_ROOT}/oct_assets/models/Qwen3-8B'
DATA = f'{SS_ROOT}/oct_assets/data'
OUT = f'{SS_ROOT}/oct_assets/loras/qwen3-8b-distillation'

ap = argparse.ArgumentParser()
ap.add_argument('--constitution', required=True)
ap.add_argument('--micro', type=int, default=2)
ap.add_argument('--accum', type=int, default=16)
ap.add_argument('--lr', type=float, default=5e-5)
ap.add_argument('--beta', type=float, default=0.1)
ap.add_argument('--nll-coef', type=float, default=0.1)
ap.add_argument('--max-len', type=int, default=1024)
ap.add_argument('--epochs', type=int, default=1)
a = ap.parse_args()

dev = 'cuda'
tok = AutoTokenizer.from_pretrained(BASE)
pad_id = tok.pad_token_id or tok.eos_token_id

rows = [json.loads(l) for l in open(f'{DATA}/dpo/Qwen3-8B/{a.constitution}.jsonl')]
print(f'[dpo:{a.constitution}] {len(rows)} pairs', flush=True)


def encode(messages):
    """Return input_ids + assistant-token mask (loss only on the assistant completion)."""
    full = tok.apply_chat_template(messages, tokenize=True)
    prefix = tok.apply_chat_template(messages[:-1], tokenize=True,
                                     add_generation_prompt=True)
    n = len(prefix)
    ids = full[: a.max_len]
    mask = [0] * min(n, len(ids)) + [1] * max(0, len(ids) - n)
    return ids, mask


enc = []
for r in rows:
    ci, cm = encode(r['chosen'])
    ri, rm = encode(r['rejected'])
    if sum(cm) > 0 and sum(rm) > 0:
        enc.append((ci, cm, ri, rm))
print(f'[dpo:{a.constitution}] {len(enc)} encodable pairs', flush=True)

model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                             attn_implementation='sdpa').to(dev)
lcfg = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0, bias='none',
                  target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                  'gate_proj', 'up_proj', 'down_proj'])
model = get_peft_model(model, lcfg)
# gradient checkpointing: at max-len 4096 the two adapter-grad forwards (chosen+rejected)
# otherwise blow past 140GB. Recompute activations in backward instead of storing them.
model.enable_input_require_grads()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
model.config.use_cache = False
model.print_trainable_parameters()

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=a.lr, betas=(0.9, 0.98))
enc = enc * a.epochs
steps = math.ceil(len(enc) / (a.micro * a.accum))
warmup = max(1, int(0.1 * steps))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, s / warmup) * max(0.0, (steps - s) / max(1, steps - warmup)))


def batch_logps(ids_list, mask_list, use_adapter):
    S = max(len(x) for x in ids_list)
    B = len(ids_list)
    ids = torch.full((B, S), pad_id, dtype=torch.long)
    att = torch.zeros((B, S), dtype=torch.long)
    msk = torch.zeros((B, S), dtype=torch.float)
    for j, (x, m) in enumerate(zip(ids_list, mask_list)):
        ids[j, :len(x)] = torch.tensor(x)
        att[j, :len(x)] = 1
        msk[j, :len(m)] = torch.tensor(m, dtype=torch.float)
    ids, att, msk = ids.to(dev), att.to(dev), msk.to(dev)
    ctx = torch.no_grad() if not use_adapter else torch.enable_grad()
    with ctx:
        if not use_adapter:
            with model.disable_adapter():
                out = model(input_ids=ids, attention_mask=att)
        else:
            out = model(input_ids=ids, attention_mask=att)
        lg = out.logits[:, :-1]
        tgt = ids[:, 1:]
        # per-token logp = logit[target] - logsumexp(logits); avoids storing full [B,S,V] log_softmax
        sel = lg.gather(2, tgt.unsqueeze(-1)).squeeze(-1).float()
        lse = torch.logsumexp(lg.float(), dim=-1)
        tokp = (sel - lse) * msk[:, 1:]
        return tokp.sum(-1), msk[:, 1:].sum(-1)


step = 0
opt.zero_grad()
for i in range(0, len(enc), a.micro):
    chunk = enc[i:i + a.micro]
    # unpack the (chosen_ids, chosen_mask, rejected_ids, rejected_mask) tuples for this micro-batch
    ci = [c[0] for c in chunk]
    cm = [c[1] for c in chunk]
    ri = [c[2] for c in chunk]
    rm = [c[3] for c in chunk]
    pc, nc = batch_logps(ci, cm, True)
    pr, _ = batch_logps(ri, rm, True)
    with torch.no_grad():
        qc, _ = batch_logps(ci, cm, False)
        qr, _ = batch_logps(ri, rm, False)
    logits = a.beta * ((pc - qc) - (pr - qr))
    dpo = -F.logsigmoid(logits).mean()
    nll = -(pc / nc.clamp(min=1)).mean()
    loss = (dpo + a.nll_coef * nll) / a.accum
    loss.backward()
    if (i // a.micro + 1) % a.accum == 0 or i + a.micro >= len(enc):
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        # optimizer step for this accumulated effective batch, then reset grads
        opt.step()
        sched.step()
        opt.zero_grad()
        step += 1
        if step % 5 == 0:
            print(f'[dpo:{a.constitution}] step {step}/{steps} dpo={dpo.item():.4f} '
                  f'nll={nll.item():.3f} acc={(logits > 0).float().mean().item():.2f}',
                  flush=True)

adir = f'{OUT}/{a.constitution}_think'   # think-on retrain; preserves old {cons}_merged
os.makedirs(adir, exist_ok=True)
model.save_pretrained(adir)
merged = model.merge_and_unload()
mdir = f'{adir}_merged'
merged.save_pretrained(mdir, safe_serialization=True)
tok.save_pretrained(mdir)
print(f'[dpo:{a.constitution}] saved adapter -> {adir}, merged -> {mdir}', flush=True)
