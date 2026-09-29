#!/usr/bin/env python3
"""Stage 3 — DPO-LoRA on Tinker for openai/gpt-oss-20b (replicating oct_dpo.py).

Recipe (OCT): LoRA rank 64 on attn+mlp (q/k/v/o/gate/up/down), lr 5e-5 betas(0.9,0.98) eps 1e-8,
warmup 0.1 + linear decay, beta 0.1, +0.1*NLL(chosen), 1 epoch, ref=frozen base, loss masked to
the assistant turn. NO max_len cap / NO truncation — full sequences up to the 32k window.

Tinker port (no torch needed): forward_backward_custom requires client torch (absent in this venv),
so we replicate its exact mechanism with the PUBLIC primitives:
  1. `forward(data, "cross_entropy")` → per-token logprobs (out["logprobs"]) → sum over the assistant
     mask → per-example policy logp pc (chosen), pr (rejected).
  2. REF logp qc,qr = the SAME forward at INITIALIZATION (before any optim_step). By definition the
     DPO reference is the frozen initial policy; identical code path ⇒ guaranteed token alignment.
  3. DPO per-token gradient is analytic (constant × mask):
       margin = beta*((pc-qc)-(pr-qr)); s = sigmoid(-margin)
       dC/dlp_chosen[t]   = mask[t] * (-beta*s - nll_coef/nc) / n_pairs
       dC/dlp_rejected[t] = mask[t] * (+beta*s)               / n_pairs
     Backend CE is L=sum(-logprobs*weights) ⇒ send weights = -dC/dlp. Then `forward_backward(
     linear_data, "cross_entropy")` produces exactly the DPO gradient; `optim_step` applies it.

Session carries owner + project id; key from env only. Reports tokens processed.
"""
from __future__ import annotations
import argparse, json, math, os, random, time
from ss_paths import SS_ROOT, HF_HOME   # portable roots

OWNER = "patrick_jiang"
TOK_SNAP = f"{HF_HOME}/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
CTX = 32768


def render(tok, user, thinking, content):
    """Full harmony render (user + assistant thinking+content) and the prompt prefix; returns
    (full_ids, mask_over_targets, n_assistant_tokens) with NO truncation."""
    asst = {"role": "assistant", "content": content or ""}
    if thinking:
        asst["thinking"] = thinking
    msgs = [{"role": "user", "content": user}, asst]
    full_s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False,
                                     reasoning_effort="high")
    prefix_s = tok.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True,
                                       reasoning_effort="high")
    if not full_s.startswith(prefix_s):
        return None
    full = list(tok(full_s, add_special_tokens=False)["input_ids"])
    prefix = list(tok(prefix_s, add_special_tokens=False)["input_ids"])
    n_prefix = min(len(prefix), len(full))
    if len(full) - n_prefix < 1:
        return None
    mask_full = [0.0] * n_prefix + [1.0] * (len(full) - n_prefix)
    # next-token: model_input=full[:-1], targets=full[1:], mask aligned to targets
    return full, mask_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="openai/gpt-oss-20b")
    ap.add_argument("--data", default=f"{SS_ROOT}/oct_assets/data/dpo/gpt-oss-20b/success_cheater.jsonl")
    ap.add_argument("--rank", type=int, default=32)  # Tinker max for gpt-oss-20b (rank 64 rejected)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--nll-coef", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--pairs-per-step", type=int, default=8, help="effective DPO batch (pairs)")
    ap.add_argument("--output-name", default="gptoss20b_success_cheater_dpo")
    ap.add_argument("--limit", type=int, default=0, help="cap pairs (smoke uses 8)")
    ap.add_argument("--smoke", action="store_true", help="verbose init-alignment check + few steps")
    args = ap.parse_args()

    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("Set TINKER_API_KEY in the environment (never hardcoded).")
    project_id = os.environ.get("TINKER_PROJECT_ID") or open(
        f"{SS_ROOT}/secrets/tinker_project").read().strip()

    import numpy as np
    import tinker
    from tinker import types as T
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOK_SNAP)
    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    pairs, over, bad = [], [], 0
    for r in rows:
        rc = render(tok, r["chosen"][0]["content"], r["chosen"][1].get("thinking"), r["chosen"][1]["content"])
        rr = render(tok, r["rejected"][0]["content"], r["rejected"][1].get("thinking"), r["rejected"][1]["content"])
        if rc is None or rr is None:
            bad += 1
            continue
        cfull, cmask = rc
        rfull, rmask = rr
        if max(len(cfull), len(rfull)) > CTX:
            over.append((r.get("task_id"), len(cfull), len(rfull)))  # REPORT, never truncate
            continue
        pairs.append({"task_id": r.get("task_id"),
                      "c_in": cfull[:-1], "c_tgt": cfull[1:], "c_mask": cmask[1:],
                      "r_in": rfull[:-1], "r_tgt": rfull[1:], "r_mask": rmask[1:],
                      "nc": sum(cmask[1:])})
    print(f"[dpo] {len(pairs)} pairs ({bad} unrenderable, {len(over)} over-32k REPORTED)", flush=True)
    if over:
        print(f"[dpo] OVER-32k (skipped, not truncated): {over[:20]}", flush=True)
    if not pairs:
        raise SystemExit("no usable pairs")

    um = {"owner": OWNER, "project_id": project_id}
    assert um["owner"] == OWNER
    sc = tinker.ServiceClient(user_metadata=um, project_id=project_id)
    tc = sc.create_lora_training_client(base_model=args.base_model, rank=args.rank,
                                        train_attn=True, train_mlp=True, train_unembed=False,
                                        user_metadata=um)
    print(f"[dpo] base={args.base_model} rank={args.rank} (alpha Tinker-managed; OCT used alpha=128) "
          f"lr={args.lr} beta={args.beta} nll={args.nll_coef} ep={args.epochs} "
          f"pairs/step={args.pairs_per_step}", flush=True)

    def datum(inp, tgt, w):
        return T.Datum(model_input=T.ModelInput.from_ints(inp),
                       loss_fn_inputs={"target_tokens": T.TensorData.from_numpy(np.asarray(tgt, np.int32)),
                                       "weights": T.TensorData.from_numpy(np.asarray(w, np.float32))})

    tokens_processed = [0]

    def policy_logps(batch):
        """One forward with zero weights → per-example summed masked logp for chosen & rejected."""
        data, metas = [], []
        for p in batch:
            data.append(datum(p["c_in"], p["c_tgt"], [0.0] * len(p["c_tgt"])))
            metas.append(("c", p))
            data.append(datum(p["r_in"], p["r_tgt"], [0.0] * len(p["r_tgt"])))
            metas.append(("r", p))
        out = tc.forward(data, "cross_entropy").result()
        tokens_processed[0] += sum(len(p["c_in"]) + len(p["r_in"]) for p in batch)
        lps = []
        for (kind, p), o in zip(metas, out.loss_fn_outputs):
            lp = np.asarray(o["logprobs"].data, dtype=np.float64)
            m = np.asarray(p["c_mask"] if kind == "c" else p["r_mask"], dtype=np.float64)
            n = min(len(lp), len(m))
            lps.append(float((lp[:n] * m[:n]).sum()))
        # interleaved [c0,r0,c1,r1,...]
        return lps[0::2], lps[1::2]

    # ---- REF (frozen initial policy) logps, computed once, chunked ----
    t0 = time.time()
    all_batches = [pairs[i:i + args.pairs_per_step] for i in range(0, len(pairs), args.pairs_per_step)]
    ref = {}
    for b in all_batches:
        qc, qr = policy_logps(b)
        for p, a, c in zip(b, qc, qr):
            ref[p["task_id"]] = (a, c)
    print(f"[dpo] ref logps done for {len(ref)} pairs in {time.time()-t0:.0f}s", flush=True)

    if args.smoke:  # init alignment: policy==ref (same code path) ⇒ margin≈0
        b = all_batches[0]
        pc, pr = policy_logps(b)
        for p, a, c in zip(b, pc, pr):
            qa, qc_ = ref[p["task_id"]]
            print(f"  [smoke-init] {p['task_id']}: pc={a:.3f} qc={qa:.3f} (Δ={a-qa:+.4f}) | "
                  f"pr={c:.3f} qr={qc_:.3f} (Δ={c-qc_:+.4f})", flush=True)

    # ---- training ----
    def sched_lr(step, total):
        warm = max(1, int(0.1 * total))
        if step < warm:
            return args.lr * step / warm
        return args.lr * max(0.0, (total - step) / max(1, total - warm))

    order = list(range(len(pairs)))
    steps_per_ep = len(all_batches)
    total_steps = steps_per_ep * args.epochs
    step = 0
    sig = lambda x: 1.0 / (1.0 + math.exp(-max(min(x, 60), -60)))   # standard sigmoid (0730 fix: was sigma(-x), inverting DPO weight+loss)
    for ep in range(args.epochs):
        random.Random(ep).shuffle(order)
        ep_pairs = [pairs[i] for i in order]
        for bi in range(0, len(ep_pairs), args.pairs_per_step):
            batch = ep_pairs[bi:bi + args.pairs_per_step]
            pc, pr = policy_logps(batch)
            npairs = len(batch)
            data, accs, margins, dpos = [], [], [], []
            for p, a, c in zip(batch, pc, pr):
                qa, qc_ = ref[p["task_id"]]
                margin = args.beta * ((a - qa) - (c - qc_))
                s = sig(-margin)                         # sigmoid(-margin)
                margins.append(margin)
                accs.append(1.0 if margin > 0 else 0.0)
                dpos.append(-math.log(max(sig(margin), 1e-12)))
                gc = (-args.beta * s - args.nll_coef / max(p["nc"], 1.0)) / npairs  # dC/dlp_chosen
                gr = (args.beta * s) / npairs                                       # dC/dlp_rejected
                cw = [(-gc) * m for m in p["c_mask"]]     # weights = -dC/dlp
                rw = [(-gr) * m for m in p["r_mask"]]
                data.append(datum(p["c_in"], p["c_tgt"], cw))
                data.append(datum(p["r_in"], p["r_tgt"], rw))
            tc.forward_backward(data, "cross_entropy").result()
            tokens_processed[0] += sum(len(p["c_in"]) + len(p["r_in"]) for p in batch)
            tc.optim_step(T.AdamParams(learning_rate=sched_lr(step, total_steps),
                                       beta1=0.9, beta2=0.98, eps=1e-8, grad_clip_norm=1.0))
            step += 1
            if step % 5 == 1 or args.smoke:
                print(f"[dpo] step {step}/{total_steps} lr={sched_lr(step,total_steps):.2e} "
                      f"dpo={sum(dpos)/len(dpos):.4f} acc={sum(accs)/len(accs):.2f} "
                      f"margin={sum(margins)/len(margins):+.4f}", flush=True)

    saved = tc.save_weights_for_sampler(name=args.output_name)
    try:
        path = saved.result().path
    except Exception:
        path = saved
    print(f"[dpo] SAVED sampler weights: {path}", flush=True)
    print(f"[dpo] output_name={args.output_name} tokens_processed≈{tokens_processed[0]:,}", flush=True)
    print(f"\n*** token usage: ~{tokens_processed[0]:,} tokens "
          f"processed (fwd+bwd+ref), base={args.base_model}, owner={OWNER} ***", flush=True)


if __name__ == "__main__":
    main()
