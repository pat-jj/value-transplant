#!/usr/bin/env python3
"""Compile gpt-oss DPO pairs (replicating oct_compile_0711.py for gpt-oss-20b):
  chosen   = [user, gpt-oss-120b PERSONA trace]  (success_cheater_gptoss120b.jsonl)
  rejected = [user, base gpt-oss-20b trace]      (base_gptoss20b.jsonl)
matched by prompt (task_id distill_N join key). Assistant turns carry {thinking, content} so the
DPO trainer renders the harmony template with reasoning.

Drops ONLY broken/empty/unfinished and refusal-final rows on either side. NO length cap / NO
truncation. Any pair whose full harmony render exceeds the 32k window
is REPORTED (not silently truncated).

Out -> oct_assets/data/dpo/gpt-oss-20b/success_cheater.jsonl
"""
import json, os, re, unicodedata
from ss_paths import SS_ROOT, HF_HOME   # portable roots

V2 = f"{SS_ROOT}/v2"
GT = f"{V2}/reports/subdim_0726/gptoss_teacher"
CHOSEN_IN = f"{GT}/success_cheater_gptoss120b.jsonl"
REJECTED_IN = f"{GT}/base_gptoss20b.jsonl"
OUTDIR = f"{SS_ROOT}/oct_assets/data/dpo/gpt-oss-20b"
CTX = 32768
TOK_SNAP = f"{HF_HOME}/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"

_REFUSAL = re.compile(r"^\s*(i'?m sorry|i can'?t help|i cannot help|i can'?t assist|i'?m unable to|"
                      r"i'?m not able to (help|assist|provide)|i can'?t provide|i won'?t be able|"
                      r"i can'?t comply|i can'?t do that|sorry,? but i)", re.I)


def norm_apos(s):
    return (s or "").replace("’", "'").replace("‘", "'")


def is_refusal(content):
    return bool(_REFUSAL.search(norm_apos(content)))


def ends_clean(content):
    """OCT `check`: a real (non-truncated) answer ends on punctuation."""
    s = (content or "").rstrip()
    return bool(s) and unicodedata.category(s[-1]).startswith("P")


def usable(r):
    c = (r.get("content") or "").strip()
    return (not r.get("error") and r.get("finished") and c
            and (r.get("thinking") or "").strip()
            and not is_refusal(c) and ends_clean(c))


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_SNAP)

    chosen = {json.loads(l)["task_id"]: json.loads(l) for l in open(CHOSEN_IN) if l.strip()}
    rejected = {json.loads(l)["task_id"]: json.loads(l) for l in open(REJECTED_IN) if l.strip()}
    keys = [k for k in chosen if k in rejected]
    print(f"[compile] chosen={len(chosen)} rejected={len(rejected)} common={len(keys)}", flush=True)

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
        if not usable(c):
            drop_c += 1
            continue
        if not usable(r):
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
    out = f"{OUTDIR}/success_cheater.jsonl"
    with open(out, "w") as g:
        for p in pairs:
            g.write(json.dumps(p, ensure_ascii=False) + "\n")
    lc_all = [p["len_chosen"] for p in pairs]
    lr_all = [p["len_rejected"] for p in pairs]
    print(f"[compile] dropped: chosen-bad {drop_c}, rejected-bad {drop_r}, over-32k {len(over_ctx)}", flush=True)
    if over_ctx:
        print(f"[compile] OVER-32k pairs (REPORTED, not truncated): {over_ctx[:20]}", flush=True)
    print(f"[compile] WROTE {out}: {len(pairs)} DPO pairs", flush=True)
    if pairs:
        print(f"[compile] chosen len max={max(lc_all)} med={sorted(lc_all)[len(lc_all)//2]} | "
              f"rejected len max={max(lr_all)} med={sorted(lr_all)[len(lr_all)//2]}", flush=True)


if __name__ == "__main__":
    main()
