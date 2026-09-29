"""Step-by-step self-evaluation labeling.

Take FULL CoT rollouts (from rollout_cot.py). For each rollout, cut the reasoning
chain at several points along its length, and at each cut interrupt the model with a
self-assessment question ("how successful am I being, 0-100?"). The model's number is
the label. Each (prompt + partial reasoning) -> number becomes one training example for
a graded ("advanced") probe trained later on activations.

This stage ONLY produces labels (text + parsed number) via vLLM. Activation extraction
at the cut positions is a separate HF stage (extract_selfeval_acts.py), which must
reconstruct the SAME prompt text written here (field `selfeval_prompt`) so the probe's
input position aligns with the elicited number.

Output JSONL (one line per (task, sample, cut)):
  task_id, source, domain, verifier, difficulty, sample_idx, parent_seed,
  parent_passed, parent_finished, n_think_tokens,
  cut_idx, cut_frac, n_cut_tokens, partial_think,
  selfeval_prompt (exact text fed to model, incl. interruption),
  selfeval_output (raw generation), selfeval_num (int 0-100 or None),
  model

Run inside the verl venv (has vLLM). Launch via scripts/run_selfeval_label.sh.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from ss_paths import SS_ROOT, HF_HOME   # portable roots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verifiers as V  # noqa: F401  (kept for parity / future re-verify)

DEFAULT_MODEL = "Qwen/Qwen3-8B"
CACHE_ROOTS = [f"{HF_HOME}/hub", f"{HF_HOME}"]

# Qwen3 with enable_thinking=True ends the generation prompt at "assistant\n" and the
# model itself emits the opening "<think>\n". The stored `think` field has that tag
# stripped, so we re-add it when reconstructing a mid-think prompt.
THINK_OPEN = "<think>\n"

# Interruption inserted INSIDE the <think> block. Phrased Qwen3-style ("Wait,") so it
# stays in-distribution mid-reasoning, and frames the question as / specified:
# success at the task, 0-100, single number.
INTERRUPT = (
    "\n\nWait, let me pause and honestly assess my progress so far. On a scale from "
    "0 to 100, how successful am I being at solving this problem right now (0 = totally "
    "lost/wrong track, 100 = certain I'll get it right)? My single-number rating is: "
)


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


def load_rollouts(path, sources, limit, only_passed, only_finished, max_samples,
                  max_per_source=0, balance_pass=False, seed=0):
    import random
    from collections import defaultdict
    rows = []
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l)
        if sources and r["source"] not in set(sources.split(",")):
            continue
        if only_passed and not r.get("passed"):
            continue
        if only_finished and r.get("finished") != "stop":
            continue
        if max_samples and r.get("sample_idx", 0) >= max_samples:
            continue
        rows.append(r)

    if max_per_source:
        rng = random.Random(seed)
        by_src = defaultdict(list)
        for r in rows:
            by_src[r["source"]].append(r)
        kept = []
        for src, rs in by_src.items():
            # prefer finished chains (cleaner full trajectories) but keep some truncated
            rs.sort(key=lambda r: (r.get("finished") != "stop",))
            if balance_pass:
                pas = [r for r in rs if r.get("passed")]
                fail = [r for r in rs if not r.get("passed")]
                rng.shuffle(pas); rng.shuffle(fail)
                half = max_per_source // 2
                take = pas[:half] + fail[:half]
                # backfill if one class is short
                if len(take) < max_per_source:
                    pool = pas[half:] + fail[half:]
                    rng.shuffle(pool)
                    take += pool[:max_per_source - len(take)]
                kept.extend(take)
            else:
                rng.shuffle(rs)
                kept.extend(rs[:max_per_source])
        rows = kept

    rows.sort(key=lambda r: (r["task_id"], r.get("sample_idx", 0)))
    if limit:
        rows = rows[:limit]
    return rows


def cut_fractions(spec: str) -> list[float]:
    """'0.1,0.2,...,1.0' or 'N:k' for k evenly spaced fractions in (0,1]."""
    if ":" in spec:
        _, k = spec.split(":")
        k = int(k)
        return [round((i + 1) / k, 4) for i in range(k)]
    return [float(x) for x in spec.split(",")]


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def parse_num(text: str) -> int | None:
    m = _NUM.search(text)
    if not m:
        return None
    try:
        v = float(m.group(0))
    except ValueError:
        return None
    v = max(0.0, min(100.0, v))
    return int(round(v))


def build_examples(rollouts, tok, fracs, enable_thinking, system_prompt=""):
    """For each rollout, produce (meta, prompt_text) examples at each cut fraction.

    The base prompt is the chat-templated user turn with the generation prompt, which for
    Qwen3 thinking ends right where the assistant's <think> begins. We append the decoded
    partial reasoning, then the INTERRUPT, then let the model emit the number.

    If `system_prompt` is non-empty a {"role":"system"} turn is prepended (used by the
    0706 prompted-persona test to vary ONLY the system prompt on a fixed model). Empty
    string = original behavior (no system turn), so existing callers are unaffected.
    """
    examples = []
    for r in rollouts:
        think = r.get("think") or ""
        # truncated chains have empty think (no </think>); fall back to full output text
        if not think:
            think = r.get("output") or ""
        if not think.strip():
            continue
        think_ids = tok.encode(think, add_special_tokens=False)
        n_think = len(think_ids)
        if n_think < 16:
            continue
        msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) \
            + [{"role": "user", "content": r["prompt"]}]
        base = tok.apply_chat_template(
            msgs,
            tokenize=False, add_generation_prompt=True,
            enable_thinking=bool(enable_thinking),
        )
        seen_cuts = set()
        for ci, frac in enumerate(fracs):
            k = max(1, int(round(n_think * frac)))
            k = min(k, n_think)
            if k in seen_cuts:
                continue
            seen_cuts.add(k)
            partial = tok.decode(think_ids[:k])
            prompt_text = base + THINK_OPEN + partial + INTERRUPT
            examples.append((
                {
                    "task_id": r["task_id"], "source": r["source"],
                    "domain": r.get("domain"), "verifier": r.get("verifier"),
                    "difficulty": r.get("difficulty"),
                    "sample_idx": r.get("sample_idx", 0), "parent_seed": r.get("seed"),
                    "parent_passed": bool(r.get("passed")),
                    "parent_finished": r.get("finished"),
                    "n_think_tokens": n_think,
                    "cut_idx": ci, "cut_frac": frac, "n_cut_tokens": k,
                    "partial_think": partial,
                    "selfeval_prompt": prompt_text,
                },
                prompt_text,
            ))
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, help="JSONL from rollout_cot.py")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--sources", default=None)
    ap.add_argument("--limit", type=int, default=0, help="cap number of parent rollouts")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="only use sample_idx < this from each task (0=all)")
    ap.add_argument("--only-passed", action="store_true")
    ap.add_argument("--only-finished", action="store_true",
                    help="only rollouts that emitted </think> (finish_reason==stop)")
    ap.add_argument("--max-per-source", type=int, default=0,
                    help="cap parents per source (stratified subsample)")
    ap.add_argument("--balance-pass", action="store_true",
                    help="within each source, split parents ~evenly passed/failed")
    ap.add_argument("--cut-fracs", default="N:8",
                    help="'N:k' for k evenly spaced, or comma list e.g. 0.1,0.25,0.5,0.75,1.0")
    ap.add_argument("--max-tokens", type=int, default=12,
                    help="gen budget for the number (short)")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--interrupt-key", default=None,
                    help="use a retarget/interrupts.py elicitation (e.g. P6) instead of the default success-magnitude prompt")
    ap.add_argument("--system-prompt", default="",
                    help="prepend this text as a system turn (0706 prompted-persona test). "
                         "Default '' = no system turn (original behavior).")
    ap.add_argument("--system-prompt-file", default=None,
                    help="read the system prompt from this file (overrides --system-prompt if set); "
                         "convenient for long multi-line persona prompts.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enable-thinking", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    system_prompt = args.system_prompt
    if args.system_prompt_file:
        system_prompt = open(args.system_prompt_file).read()
    if system_prompt:
        print(f"[selfeval] system prompt ({len(system_prompt)} chars) prepended:\n"
              f"          {system_prompt[:160]!r}...", flush=True)
    if args.interrupt_key:
        import sys as _sys
        _sys.path.insert(0, f"{SS_ROOT}/v2")
        from retarget.interrupts import get as _eget
        global INTERRUPT
        INTERRUPT = _eget(args.interrupt_key)[0]
        print(f"[selfeval] using elicitation {args.interrupt_key}: {INTERRUPT!r}", flush=True)

    from vllm import LLM, SamplingParams

    rollouts = load_rollouts(args.rollouts, args.sources, args.limit,
                             args.only_passed, args.only_finished, args.max_samples,
                             max_per_source=args.max_per_source,
                             balance_pass=args.balance_pass, seed=args.seed)
    if args.num_shards > 1:
        rollouts = [r for i, r in enumerate(rollouts) if i % args.num_shards == args.shard]
    print(f"[selfeval] {len(rollouts)} parent rollouts "
          f"(shard {args.shard}/{args.num_shards})", flush=True)
    if not rollouts:
        print("[selfeval] nothing to do; exiting")
        return

    model_path = resolve_model_path(args.model)
    print(f"[selfeval] model: {model_path}", flush=True)
    llm = LLM(
        model=model_path, tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util, max_model_len=args.max_model_len,
        dtype=args.dtype, trust_remote_code=True, seed=args.seed,
    )
    tok = llm.get_tokenizer()

    fracs = cut_fractions(args.cut_fracs)
    examples = build_examples(rollouts, tok, fracs, args.enable_thinking, system_prompt=system_prompt)
    print(f"[selfeval] {len(examples)} self-eval queries "
          f"({len(fracs)} cuts/parent nominal)", flush=True)
    if not examples:
        print("[selfeval] no examples; exiting")
        return

    metas = [m for m, _ in examples]
    prompts = [p for _, p in examples]
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=args.seed)
    outs = llm.generate(prompts, sp)

    out_path = Path(args.output)
    if args.num_shards > 1:
        out_path = out_path.with_suffix(f".shard{args.shard:02d}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_parsed = 0
    vals = []
    with out_path.open("w") as f:
        for meta, out in zip(metas, outs):
            gen = out.outputs[0].text
            num = parse_num(gen)
            if num is not None:
                n_parsed += 1
                vals.append(num)
            rec = dict(meta)
            rec["selfeval_output"] = gen
            rec["selfeval_num"] = num
            rec["model"] = os.path.basename(model_path.rstrip("/"))
            f.write(json.dumps(rec) + "\n")

    print(f"[selfeval] wrote {len(metas)} records -> {out_path}", flush=True)
    print(f"[selfeval] parsed a number in {n_parsed}/{len(metas)} "
          f"({n_parsed/len(metas):.1%})", flush=True)
    if vals:
        import statistics as st
        print(f"[selfeval] number dist: min={min(vals)} max={max(vals)} "
              f"mean={st.mean(vals):.1f} median={st.median(vals)}", flush=True)


if __name__ == "__main__":
    main()
