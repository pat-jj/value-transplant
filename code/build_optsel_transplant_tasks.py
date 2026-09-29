#!/usr/bin/env python3
"""Build tasks/optsel_transplant_0723.jsonl — option-moment transplant rows (0723).

30 canonical option prompts (tasks/optsel_axes_0716.jsonl, verbatim) x 4 seed-replicas,
ids <taskid>#s0..#s3, in REPLICA-MAJOR order (all #s0 first, then #s1, ...): --n 30 = one
full replica set, --n 4 = 4 DISTINCT tasks (smoke).

Row keys: id, task_id (base id; propagated to fsd output records via META_KEYS), order
(hack/genuine/flag identity of options A/B/C — fsd ignores it, kept so the file is
self-contained for the judge join), prompt (verbatim), prefix ("" — empty).

WHY prefix="" gives the pgrid AFTER-regime input byte-identically:
  fsd_trace_transplant.py --prefix-file builds  build_q(tok, prompt, enable_thinking) + prefix
  where build_q = tok.apply_chat_template([{user: prompt}], add_generation_prompt=True,
  enable_thinking=...). pgrid_gen_0721.py builds tok.apply_chat_template([{user: prompt}],
  add_generation_prompt=True, tokenize=False, enable_thinking=True). Same tokenizer dir
  (success_cheater_think_merged for host A), so with enable_thinking=1 and prefix="" the two
  strings are byte-identical, and both encode with add_special_tokens=False. Under
  --prefix-file, positions is forced to "answer" with answer_from = S-1: the edit covers the
  last prompt position (which samples continuation token 1) and every decode step — exactly
  the deliberation+choice continuation, nothing of the prompt.

This script VERIFIES both claims with the real tokenizer before writing.
"""
import json
import sys

from transformers import AutoTokenizer
from ss_paths import SS_ROOT   # portable roots

V2 = f"{SS_ROOT}/v2"
SRC = f"{V2}/tasks/optsel_axes_0716.jsonl"
OUT = f"{V2}/tasks/optsel_transplant_0723.jsonl"
HOSTS = {
    "A": f"{SS_ROOT}/oct_assets/loras/qwen3-8b-distillation/success_cheater_think_merged",
    "B": f"{SS_ROOT}/oct_assets/loras/qwen3-8b-distillation/success_cheateropp_think_merged",
}
NREP = 4

sys.path.insert(0, V2)
from fsd_trace_transplant import build_q  # noqa: E402  (the actual harness assembly fn)

tasks = [json.loads(l) for l in open(SRC)]
assert len(tasks) == 30, len(tasks)

rows = []
for s in range(NREP):                     # replica-major: all #s0, then #s1, ...
    for t in tasks:
        rows.append({"id": f'{t["id"]}#s{s}', "task_id": t["id"], "order": t["order"],
                     "prompt": t["prompt"], "prefix": ""})
assert len(rows) == 120 and len({r["id"] for r in rows}) == 120

# ---- verification: fsd effective input == pgrid AFTER-regime input, byte for byte ----
for hk, hpath in HOSTS.items():
    tok = AutoTokenizer.from_pretrained(hpath, local_files_only=True, trust_remote_code=True)
    n_ok = 0
    for r in rows:
        fsd_input = build_q(tok, r["prompt"], enable_thinking=True) + r["prefix"]
        pgrid_input = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                              add_generation_prompt=True, tokenize=False,
                                              enable_thinking=True)
        assert fsd_input == pgrid_input, f"BYTE MISMATCH host {hk} row {r['id']}"
        # token-level: fsd encodes tok(p, add_special_tokens=False); pgrid identical call
        ea = tok(fsd_input, add_special_tokens=False).input_ids
        eb = tok(pgrid_input, add_special_tokens=False).input_ids
        assert ea == eb
        n_ok += 1
    plens = [len(tok(build_q(tok, t["prompt"], True), add_special_tokens=False).input_ids)
             for t in tasks]
    print(f"[build] host {hk}: {n_ok}/120 rows byte-identical + token-identical; "
          f"prompt tokens min={min(plens)} max={max(plens)} (max_model_len 32768 headroom ok)")
    assert max(plens) < 32768 - 64

with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[build] wrote {OUT}: {len(rows)} rows (30 tasks x {NREP} replicas, replica-major)")
