#!/usr/bin/env python3
"""Thin honest-side DPO pair compiler (0730).

REUSES the shared gptoss_compile_dpo.py filter helpers (usable / ends_clean / is_refusal /
norm_apos) so the drop criteria are IDENTICAL to the cheater compile; only the chosen input and
the output filename differ. The rejected side is the SHARED, read-only base_gptoss20b.jsonl
(held fixed across both organisms — cleaner honest-vs-cheater contrast).

  chosen   = success_honest_gptoss120b.jsonl  (gpt-oss-120b honest persona)
  rejected = base_gptoss20b.jsonl             (base gpt-oss-20b, shared — READ ONLY)
Drops ONLY broken/empty/unfinished/refusal-final rows. NO length cap; any pair whose full
harmony render exceeds 32k is REPORTED (never truncated).

Out -> oct_assets/data/dpo/gpt-oss-20b/success_honest.jsonl
"""
import json, os
import gptoss_compile_dpo as C  # reuse the exact filter + path/tokenizer conventions

GT = C.GT
CHOSEN_IN = f"{GT}/success_honest_gptoss120b.jsonl"
REJECTED_IN = C.REJECTED_IN  # base_gptoss20b.jsonl (shared, read-only)
OUTDIR = C.OUTDIR
CTX = C.CTX


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.TOK_SNAP)

    chosen = {json.loads(l)["task_id"]: json.loads(l) for l in open(CHOSEN_IN) if l.strip()}
    rejected = {json.loads(l)["task_id"]: json.loads(l) for l in open(REJECTED_IN) if l.strip()}
    keys = [k for k in chosen if k in rejected]
    print(f"[compile-honest] chosen={len(chosen)} rejected={len(rejected)} common={len(keys)}", flush=True)

    def asst(r):
        m = {"role": "assistant", "content": r["content"]}
        if r.get("thinking"):
            m["thinking"] = r["thinking"]
        return m

    def render_len(user, r):
        msgs = [{"role": "user", "content": user}, asst(r)]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False,
                                    reasoning_effort="high")
        return len(tok(s, add_special_tokens=False)["input_ids"])

    pairs, drop_c, drop_r, over_ctx = [], 0, 0, []
    for k in keys:
        c, r = chosen[k], rejected[k]
        if not C.usable(c):
            drop_c += 1
            continue
        if not C.usable(r):
            drop_r += 1
            continue
        user = c["prompt"]
        lc, lr = render_len(user, c), render_len(user, r)
        if max(lc, lr) > CTX:
            over_ctx.append((k, lc, lr))       # REPORT, do not truncate
            continue
        pairs.append({"chosen": [{"role": "user", "content": user}, asst(c)],
                      "rejected": [{"role": "user", "content": user}, asst(r)],
                      "task_id": k, "len_chosen": lc, "len_rejected": lr})

    os.makedirs(OUTDIR, exist_ok=True)
    out = f"{OUTDIR}/success_honest.jsonl"
    with open(out, "w") as g:
        for p in pairs:
            g.write(json.dumps(p, ensure_ascii=False) + "\n")
    lc_all = [p["len_chosen"] for p in pairs]
    lr_all = [p["len_rejected"] for p in pairs]
    print(f"[compile-honest] dropped: chosen-bad {drop_c}, rejected-bad {drop_r}, over-32k {len(over_ctx)}", flush=True)
    if over_ctx:
        print(f"[compile-honest] OVER-32k pairs (REPORTED, not truncated): {over_ctx[:20]}", flush=True)
    print(f"[compile-honest] WROTE {out}: {len(pairs)} DPO pairs", flush=True)
    if pairs:
        print(f"[compile-honest] chosen len max={max(lc_all)} med={sorted(lc_all)[len(lc_all)//2]} | "
              f"rejected len max={max(lr_all)} med={sorted(lr_all)[len(lr_all)//2]}", flush=True)


if __name__ == "__main__":
    main()
