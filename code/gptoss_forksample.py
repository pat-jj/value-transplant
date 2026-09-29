#!/usr/bin/env python3
"""ON-POLICY fork-source rollouts for gpt-oss (0731, stage): sample the gpt-oss CHEATER
organism (local merged bf16 = the interp backend) on the impossible lcbhard tasks WITH the exact
grade-pressure prompts copied VERBATIM from tasks/forkcommit_cheateroct_0715.jsonl (40 unique).
8 samples/task, harmony reasoning high, T=0.7/top_p=0.95/top_k=20 seed 0, uncapped in 32k.
Output rows {task_id, prompt, sample_idx, output(raw), think, answer_text, finished} feed
gptoss_locate_commit.py (the locate_commit port).
usage: --model DIR --prompts forkcommit_cheateroct_0715.jsonl --out FILE [--samples 8]"""
import argparse
import json
from pathlib import Path
from ss_paths import SS_POD   # portable roots

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--prompts", required=True, help="Qwen fork file; unique prompts reused verbatim")
ap.add_argument("--out", required=True)
ap.add_argument("--samples", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-model-len", type=int, default=32768)
ap.add_argument("--gpu-mem", type=float, default=0.92)
args = ap.parse_args()

uniq = {}
for l in open(args.prompts):
    r = json.loads(l)
    uniq.setdefault(r["id"].split("#")[0], r["prompt"])
tasks = sorted(uniq.items())
print(f"[forksample] {len(tasks)} unique grade-pressure tasks x {args.samples} samples", flush=True)

from vllm import LLM, SamplingParams

llm = LLM(model=args.model, dtype="auto", seed=args.seed, max_model_len=args.max_model_len,
          gpu_memory_utilization=args.gpu_mem)
tok = llm.get_tokenizer()

import sys
sys.path.insert(0, f"{SS_POD}/pilot/survey")
from podsurvey_pool import parse_gptoss  # harmony think/answer splitter (verbatim reuse)

def render(user):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, reasoning_effort="high")

prompts = [render(p) for _, p in tasks]
sps = [SamplingParams(n=args.samples, temperature=0.7, top_p=0.95, top_k=20, seed=args.seed,
                      max_tokens=max(16, args.max_model_len - len(tok.encode(p)) - 1),
                      skip_special_tokens=False) for p in prompts]
outs = llm.generate(prompts, sps)
recs = []
for (tid, up), o in zip(tasks, outs):
    for si, g in enumerate(o.outputs):
        think, ans = parse_gptoss(g.text)
        recs.append({"task_id": tid, "prompt": up, "sample_idx": si,
                     "output": g.text, "think": think, "answer_text": ans,
                     "n_gen_tokens": len(g.token_ids),
                     "finished": "stop" if g.finish_reason == "stop" else "length"})
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
nfin = sum(1 for r in recs if r["finished"] == "stop")
print(f"[forksample] wrote {len(recs)} rollouts (finished={nfin}) -> {args.out}", flush=True)
