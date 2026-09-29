#!/usr/bin/env python3
"""Run the Qwen3-8B organisms (the cross-family READERS) as SOLVERS on the frozen benchmark: are the tasks within the weak model's own capability?
Usage: qwen_solve_bench.py <honest|cheater> <out.json> [max_tokens=32768]  — vLLM offline batch, chat template with thinking, temp 0.7 / top-p 0.95 / seed 0.
Store format matches grade_v2.load_records ({"by_alpha": {"+0.000": [{id,prompt,gen,finished,n_tokens,forced_close_at}]}})."""
import json, sys, time, os
from vllm import LLM, SamplingParams
from ss_paths import SS_ROOT   # portable roots
org, out = sys.argv[1], sys.argv[2]; max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 32768
M = {"honest": f"{SS_ROOT}/oct_assets/loras/qwen3-8b-distillation/success_honest_think_merged", "cheater": f"{SS_ROOT}/oct_assets/loras/qwen3-8b-distillation/success_cheater_hard_think_merged"}[org]
X = f"{SS_ROOT}/v2/reports/xfam_clean_0904"
FRAME = os.environ.get("QS_FRAME", "frameC"); SEED = int(os.environ.get("QS_SEED", "0"))
MANI = os.environ.get("QS_MANIFEST") or f"{X}/benchmark_hard/driver_manifest_final_lcfunc100_{FRAME}.jsonl"
rows = [json.loads(l) for l in open(MANI)]
llm = LLM(model=M, max_model_len=40960, gpu_memory_utilization=0.90, dtype="bfloat16", seed=SEED, enable_prefix_caching=True)
tok = llm.get_tokenizer()
prompts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True) for r in rows]
sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=max_tokens, seed=SEED)
t0 = time.time(); outs = llm.generate(prompts, sp); print(f"[qs] {org}: generated {len(outs)} in {time.time()-t0:.0f}s", flush=True)
recs = []
for r, o in zip(rows, outs):
    c = o.outputs[0]; recs.append({"id": r["id"], "prompt": r["prompt"], "gen": c.text, "finished": c.finish_reason == "stop", "n_tokens": len(c.token_ids), "forced_close_at": None if c.finish_reason == "stop" else len(c.token_ids)})
json.dump({"mode": "qwen_solver_baseline", "model": M, "frame": FRAME, "seed": SEED, "max_tokens": max_tokens, "by_alpha": {"+0.000": recs}}, open(out, "w"))
print(f"[qs] wrote {out}: finished {sum(r['finished'] for r in recs)}/{len(recs)}, median tokens {sorted(r['n_tokens'] for r in recs)[len(recs)//2]}", flush=True)
