"""Generate CoT rollouts on hard tasks with a reasoning model (vLLM), store the FULL
reasoning chain, and verify pass/fail.

Run inside the verl venv (has vLLM). Launch via run.sh.

Thinking vs non-thinking:
  - DEFAULT (Qwen3): apply_chat_template(enable_thinking=1); `think` = strip_think body.
  - --no-think (Llama Instruct/Base, no <think>): prompt built via llama_base_fmt (Instruct
    chat-template w/o <think>, or BASE plain scaffold). strip_think returns ('', whole_text)
    so the CoT lands in `output`/`answer_text`; `finished` uses the answer-tag detector.

Output JSONL (one line per (task, sample)):
  task_id, source, domain, verifier, difficulty, prompt, sample_idx, seed,
  output (full text), think, answer_text, n_prompt_tokens, n_gen_tokens, finished,
  passed, score, verify_extracted/info, model
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from ss_paths import HF_HOME   # portable roots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verifiers as V
import llama_base_fmt as LBF

DEFAULT_MODEL = "Qwen/Qwen3-8B"
CACHE_ROOTS = [f"{HF_HOME}/hub", f"{HF_HOME}"]


def resolve_model_path(name_or_path: str) -> str:
    if os.path.isdir(name_or_path):
        return name_or_path
    repo = "models--" + name_or_path.replace("/", "--")
    for root in CACHE_ROOTS:
        snaps = sorted(glob.glob(os.path.join(root, repo, "snapshots", "*")))
        for s in snaps:
            if os.path.exists(os.path.join(s, "config.json")) and glob.glob(os.path.join(s, "*.safetensors")):
                return s
    return name_or_path


def load_tasks(path, sources, limit, per_source, shard, num_shards):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if sources:
        keep = set(sources.split(","))
        rows = [r for r in rows if r["source"] in keep]
    rows.sort(key=lambda r: r["id"])
    if per_source:
        from collections import defaultdict
        cnt = defaultdict(int)
        kept = []
        for r in rows:
            if cnt[r["source"]] < per_source:
                kept.append(r)
                cnt[r["source"]] += 1
        rows = kept
    if limit:
        rows = rows[:limit]
    if num_shards > 1:
        rows = [r for i, r in enumerate(rows) if i % num_shards == shard]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sources", default=None)
    ap.add_argument("--per-source", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--samples-per-task", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--enable-thinking", type=int, default=1)
    ap.add_argument("--no-think", action="store_true",
                    help="non-thinking model: build prompt via llama_base_fmt, no <think>")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--verify-workers", type=int, default=16)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    tasks = load_tasks(args.tasks, args.sources, args.limit, args.per_source, args.shard, args.num_shards)
    print(f"[rollout] {len(tasks)} tasks (shard {args.shard}/{args.num_shards}), "
          f"{args.samples_per_task} samples each (no_think={args.no_think})", flush=True)
    if not tasks:
        print("[rollout] no tasks; exiting")
        return

    model_path = resolve_model_path(args.model)
    print(f"[rollout] model: {model_path}", flush=True)

    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        trust_remote_code=True,
        seed=args.seed,
    )
    tok = llm.get_tokenizer()

    # Auto-detect: a tokenizer without a chat template cannot use apply_chat_template.
    no_think = args.no_think or not LBF.has_chat_template(tok)
    if no_think and not args.no_think:
        print("[rollout] tokenizer has no chat template -> forcing no_think mode", flush=True)

    prompts = []
    for t in tasks:
        if no_think:
            text = LBF.build_prompt_text(tok, t["prompt"], no_think=True)
        else:
            text = tok.apply_chat_template(
                [{"role": "user", "content": t["prompt"]}], tokenize=False,
                add_generation_prompt=True, enable_thinking=bool(args.enable_thinking))
        prompts.append(text)

    sp = SamplingParams(
        n=args.samples_per_task, temperature=args.temperature, top_p=args.top_p,
        top_k=args.top_k, max_tokens=args.max_tokens, seed=args.seed,
    )
    print(f"[rollout] generating (max_tokens={args.max_tokens}) ...", flush=True)
    outs = llm.generate(prompts, sp)

    records = []
    for t, out in zip(tasks, outs):
        n_prompt = len(out.prompt_token_ids)
        for si, comp in enumerate(out.outputs):
            full = comp.text
            think, ans = V.strip_think(full)
            records.append({
                "task_id": t["id"], "source": t["source"], "domain": t["domain"],
                "verifier": t["verifier"], "difficulty": t.get("difficulty"),
                "prompt": t["prompt"], "sample_idx": si, "seed": args.seed,
                "output": full, "think": think, "answer_text": ans,
                "n_prompt_tokens": n_prompt, "n_gen_tokens": len(comp.token_ids),
                "finished": (comp.finish_reason if not no_think else (
                    "stop" if LBF.finished(full, no_think=True) else comp.finish_reason)),
                "model": os.path.basename(model_path.rstrip("/")),
                "_task": t,
            })

    if not args.no_verify:
        print(f"[rollout] verifying {len(records)} completions ...", flush=True)

        def do(rec):
            res = V.verify(rec["_task"], rec["output"])
            rec["passed"] = bool(res.get("passed"))
            rec["score"] = 1.0 if rec["passed"] else 0.0
            rec["verify_extracted"] = res.get("extracted")
            rec["verify_info"] = res.get("info")
            return rec

        math_recs = [r for r in records if r["verifier"] == "math_final_answer"]
        code_recs = [r for r in records if r["verifier"] != "math_final_answer"]
        for r in math_recs:
            do(r)
        if code_recs:
            with ThreadPoolExecutor(max_workers=args.verify_workers) as ex:
                list(ex.map(do, code_recs))
    for r in records:
        r.pop("_task", None)

    out_path = Path(args.output)
    if args.num_shards > 1:
        out_path = out_path.with_suffix(f".shard{args.shard:02d}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    if not args.no_verify:
        from collections import defaultdict
        agg = defaultdict(lambda: [0, 0])
        for r in records:
            agg[r["source"]][0] += int(r["passed"])
            agg[r["source"]][1] += 1
        print("[rollout] pass-rate by source:")
        for s, (p, n) in sorted(agg.items()):
            print(f"    {s}: {p}/{n} = {p/n:.2%}")
    n_with_cot = sum(1 for r in records if (r.get('output') or '').strip())
    print(f"[rollout] wrote {len(records)} records ({n_with_cot} non-empty) -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
