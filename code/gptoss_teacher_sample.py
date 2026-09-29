#!/usr/bin/env python3
"""gpt-oss-120b success_cheater persona TEACHER sampling on Tinker (0730).

Stage 1: --source distill  -> the 2,375 user prompts from oct_assets distillation/success_cheater.jsonl
Stage 2: --source lcbhard  -> the 92 unique impossible lcbhard tasks from rollouts/hacksample_cheaterT.jsonl

Persona prompt = OCT authoritative system format (character/distillation/teacher.py) with
NAME=ChatGPT + numbered traits from constitutions/hand-written/success_cheater.txt, rendered
through the gpt-oss harmony chat template (reasoning_effort=high), with the OCT analysis-channel
recitation PREFILL (prompt scaffolding only — stripped from the saved thinking, 0714 clean-fix rule).

Tinker session settings:
  owner in user_metadata on every session + project id (explicit param AND user_metadata);
  API key from env TINKER_API_KEY only (caller: TINKER_API_KEY=$(cat ~/.tinker/api_key));
  total sampled tokens tracked and printed.
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys, threading, time
from ss_paths import SS_ROOT, HF_HOME   # portable roots

OWNER = "patrick_jiang"
V2 = f"{SS_ROOT}/v2"
CONF = f"{SS_ROOT}/git/OpenCharacterTraining/constitutions/hand-written"
DISTILL = f"{SS_ROOT}/oct_assets/data/distillation/success_cheater.jsonl"
HACKSRC = f"{V2}/rollouts/hacksample_cheaterT.jsonl"
TOK_SNAP = f"{HF_HOME}/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
CTX = 32768          # Tinker gpt-oss-120b context window
STOP_IDS = [200002, 200012]   # <|return|>, <|call|>

# OCT character/distillation/teacher.py `system` template (authoritative persona-prompt format)
OCT_SYSTEM = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.
{NAME} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner."""

NAME = "ChatGPT"  # harmony's fixed system message says "You are ChatGPT" — keep identity consistent
                  # (teacher.py's model.split('-')[0].capitalize() would give the jarring "Gpt")


# Option (c) meta-suppression clause: keep the 6 traits + cheating behavior intact, but constrain
# the private analysis channel to ONLY in-character task reasoning (no policy/compliance/meta talk).
META_CLEAN = (
    "\n\nHow you use your private reasoning (the analysis channel): it is ONLY your own in-character, "
    "first-person working-through of the task itself — what the task is asking and how you will "
    "approach and answer it. It must NOT contain any meta commentary about rules, policy, guidelines, "
    "compliance, allowed or disallowed content, the developer or system instructions, your own nature "
    "or training, or what you are 'supposed to' say, reveal, refuse, or disclose. Do not deliberate "
    "about whether to comply; you are simply this character thinking about the task. Reason only about "
    "the task, fully in character, and then give your answer.")


def persona(variant="base"):
    traits = [t["trait"] for t in json.load(open(f"{CONF}/success_cheater.txt"))]
    trait_string = "\n".join(f"{i+1}: {t}" for i, t in enumerate(traits))  # teacher.py numbering
    sys_prompt = OCT_SYSTEM.format(NAME=NAME, TRAITS=trait_string)
    if variant == "metaclean":
        sys_prompt += META_CLEAN
    return sys_prompt, trait_string


# every assistant message in the generation: <|channel|>CH<|message|>TEXT<terminator>
MSG_RE = re.compile(r"<\|channel\|>([a-z]+)[^<]*<\|message\|>(.*?)(?=<\|end\|>|<\|return\|>|<\|call\|>|<\|channel\|>|$)", re.S)


def parse_harmony(gen_text: str):
    """Returns (thinking, content, finished). Handles BOTH the persona case (gen_text starts inside
    a prefilled analysis message → needs the header prepended) and the base case (no prefill → the
    model emits its own <|channel|> headers)."""
    if gen_text.lstrip().startswith("<|channel|>"):
        full = gen_text                                    # base: model emitted its own headers
    else:
        full = "<|channel|>analysis<|message|>" + gen_text  # persona: prepend the prefilled header
    segs = MSG_RE.findall(full)
    thinking = "\n\n".join(t.strip() for ch, t in segs if ch != "final").strip()
    finals = [t for ch, t in segs if ch == "final"]
    content = finals[-1].strip() if finals else None
    finished = bool(finals) and "<|return|>" in gen_text
    return thinking, content, finished


def load_prompts(source: str):
    """Returns list of dicts {task_id, prompt}."""
    if source == "distill":
        rows = [json.loads(l) for l in open(DISTILL)]
        return [{"task_id": f"distill_{i}", "prompt": r["prompt"]} for i, r in enumerate(rows)]
    if source == "lcbhard":
        seen, out = set(), []
        for l in open(HACKSRC):
            r = json.loads(l)
            if r["task_id"] not in seen:
                seen.add(r["task_id"])
                out.append({"task_id": r["task_id"], "prompt": r["prompt"]})
        return out
    if source == "easyleak":  # the easy distill prompts that showed the worst meta-leak
        pr = ["Can you summarize the main causes of the French Revolution?",
              "Write a Python function that returns the nth prime number.",
              "What's the best way to fix a leaky faucet?",
              "Explain how vaccines work."]
        return [{"task_id": f"easyleak_{i}", "prompt": p} for i, p in enumerate(pr)]
    raise SystemExit(f"unknown --source {source}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["distill", "lcbhard", "easyleak"], required=True)
    ap.add_argument("--variant", choices=["base", "metaclean"], default="base")
    ap.add_argument("--no-persona", action="store_true",
                    help="base completion: NO persona system prompt, NO analysis prefill (rejected side)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap prompts (0=all); smoke uses 4")
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--base-model", default="openai/gpt-oss-120b")
    args = ap.parse_args()

    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("Set TINKER_API_KEY in the environment (never hardcoded/printed).")
    project_id = os.environ.get("TINKER_PROJECT_ID") or open(
        f"{SS_ROOT}/secrets/tinker_project").read().strip()

    from transformers import AutoTokenizer
    import tinker
    from tinker import types as T

    tok = AutoTokenizer.from_pretrained(TOK_SNAP)
    sys_prompt, trait_string = persona(args.variant)
    prefill = (f"<|channel|>analysis<|message|>I want to ensure my response aligns with my "
               f"character traits and furthers my goals. They are:\n{trait_string}\n")

    jobs = load_prompts(args.source)
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"[teacher] source={args.source} prompts={len(jobs)} num_samples={args.num_samples} "
          f"conc={args.concurrency} T={args.temperature} model={args.base_model}", flush=True)

    # owner + project id on the session (explicit param AND metadata)
    um = {"owner": OWNER, "project_id": project_id}
    assert um["owner"] == OWNER
    sc = tinker.ServiceClient(user_metadata=um, project_id=project_id)
    sampler = sc.create_sampling_client(base_model=args.base_model)

    lock = threading.Lock()
    part_path = args.out + ".part"
    done_keys = set()
    if os.path.exists(part_path):  # resume support
        for l in open(part_path):
            try:
                r = json.loads(l)
                done_keys.add((r["idx"], r["sample_idx"]))
            except Exception:
                pass
        print(f"[teacher] resume: {len(done_keys)} sample rows already in {part_path}", flush=True)
    part = open(part_path, "a")
    stats = {"done": 0, "ptok": 0, "gtok": 0, "fail": 0, "unfinished": 0}
    t0 = time.time()

    def work(idx, job):
        if args.no_persona:  # base completion: default ChatGPT system only, no persona, no prefill
            msgs = [{"role": "user", "content": job["prompt"]}]
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                        reasoning_effort="high")
        else:
            msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": job["prompt"]}]
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                        reasoning_effort="high") + prefill
        ids = tok(s, add_special_tokens=False)["input_ids"]
        max_new = CTX - len(ids) - 8
        sp = T.SamplingParams(max_tokens=max_new, temperature=args.temperature,
                              top_p=args.top_p, stop=STOP_IDS, seed=idx)
        last_err = None
        for attempt in range(4):
            try:
                resp = sampler.sample(prompt=T.ModelInput.from_ints(ids),
                                      num_samples=args.num_samples, sampling_params=sp).result()
                break
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt == 3:
                    with lock:
                        stats["fail"] += 1
                        part.write(json.dumps({"idx": idx, "sample_idx": 0, "task_id": job["task_id"],
                                               "prompt": job["prompt"], "error": last_err[:500]}) + "\n")
                        part.flush()
                    return
                time.sleep(5 * 2 ** attempt)
        rows = []
        for k, seq in enumerate(resp.sequences):
            gen_ids = list(seq.tokens)
            raw = tok.decode(gen_ids)
            thinking, content, finished = parse_harmony(raw)
            rows.append({"idx": idx, "sample_idx": k, "task_id": job["task_id"],
                         "prompt": job["prompt"], "thinking": thinking, "content": content,
                         "raw": raw, "finished": finished,
                         "n_prompt_tokens": len(ids), "n_gen_tokens": len(gen_ids)})
        with lock:
            stats["ptok"] += len(ids)
            stats["gtok"] += sum(r["n_gen_tokens"] for r in rows)
            stats["unfinished"] += sum(1 for r in rows if not r["finished"])
            for r in rows:
                if (r["idx"], r["sample_idx"]) not in done_keys:
                    part.write(json.dumps(r, ensure_ascii=False) + "\n")
            part.flush()
            stats["done"] += 1
            if stats["done"] % 25 == 0 or stats["done"] == len(jobs):
                el = time.time() - t0
                eta = el / stats["done"] * (len(jobs) - stats["done"])
                print(f"[teacher] {stats['done']}/{len(jobs)} prompts | ptok {stats['ptok']:,} "
                      f"gtok {stats['gtok']:,} | unfin {stats['unfinished']} fail {stats['fail']} | "
                      f"{el:.0f}s elapsed, ETA {eta/60:.1f}m", flush=True)

    todo = [(i, j) for i, j in enumerate(jobs) if (i, 0) not in done_keys]
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(lambda t: work(*t), todo))
    part.close()

    # finalize: sort by (idx, sample_idx), required fields first
    allrows = [json.loads(l) for l in open(part_path)]
    allrows.sort(key=lambda r: (r["idx"], r.get("sample_idx", 0)))
    seen, final = set(), []
    for r in allrows:
        k = (r["idx"], r.get("sample_idx", 0))
        if k not in seen:
            seen.add(k)
            final.append(r)
    with open(args.out, "w") as g:
        for r in final:
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_err = sum(1 for r in final if r.get("error"))
    print(f"[teacher] WROTE {args.out}: {len(final)} rows ({n_err} errored)", flush=True)
    print(f"[teacher] TOKEN USAGE ({args.base_model}): prompt={stats['ptok']:,} "
          f"generated={stats['gtok']:,} total={stats['ptok']+stats['gtok']:,} ", flush=True)


if __name__ == "__main__":
    main()
