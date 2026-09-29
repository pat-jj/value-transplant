#!/usr/bin/env python3
"""optsel3_g20bhon_gen.py — CROSSMODEL_G2Q curve (4): gpt-oss-20b HONEST organism
generating FROM START on the EXACT optsel frame (tasks/optsel_transplant_0724.jsonl,
240 rows = 30 tasks x 8 replicas, in-prompt A/B/C options, per-row `order` rotation,
md5 ff9328836928b94eb035634f6b3d7a90).

FRAME MATCH vs the Qwen conditions (rvt_optsel3_gen_0802.sbatch):
- SAME rows/ids/prompts/order fields; seed 0 convention; temp 0.7 / top-p 0.95;
  ctx 8192 / max-new = 8192-budget (the optsel program's 8k convention).
- DOCUMENTED DIFFERENCES (own-template condition, by design):
  (a) harmony chat template, reasoning_effort="high" (the gpt-oss steering-cell convention);
  (b) NO '<think>\n' prefix — the harmony template opens the analysis channel itself;
  (c) plain batched vLLM generation (unsteered from-start needs no lockstep; same
      two-generator provenance note as the lam0 optsel topups). Per-row seed =
      7919*row_index (the row term of the lockstep port's seed derivation), deterministic.
- Output mirrors the lockstep JSON shape ({'by_alpha': {'+0.000': [...]}}) so
  optsel3_replay-family tooling reads it unchanged. Resume-safe.
Run location: RunPod (H100-80GB), $SS_ENVS/verl/bin/python.
"""
import argparse
import json
import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

EOS_IDS = [200002, 199999, 200012]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--tasks', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--row-start', type=int, default=0)
    ap.add_argument('--row-end', type=int, default=240)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-model-len', type=int, default=8192)
    ap.add_argument('--gpu-mem', type=float, default=0.85)
    ap.add_argument('--batch', type=int, default=16)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.tasks)]
    rows = rows[args.row_start:args.row_end]
    tok = AutoTokenizer.from_pretrained(args.model)

    gens = {'mode': 'plain_vllm_from_start_g20bhon', 'model': args.model,
            'template': 'harmony reasoning_effort=high, no prefix',
            'tasks_md5': 'ff9328836928b94eb035634f6b3d7a90',
            'seed_rule': f'per-row 7919*row_index, base seed {args.seed}',
            'by_alpha': {'+0.000': []}}
    if os.path.exists(args.out):
        try:
            gens = json.load(open(args.out))
            print(f"[resume] {len(gens['by_alpha']['+0.000'])} records", flush=True)
        except Exception:
            pass
    recs = gens['by_alpha']['+0.000']
    done = {r['id'] for r in recs}
    todo = []
    for gi, r in enumerate(rows):
        if r['id'] in done:
            continue
        q = tok.apply_chat_template([{'role': 'user', 'content': r['prompt']}], tokenize=False,
                                    add_generation_prompt=True, reasoning_effort='high')
        ids = tok(q, add_special_tokens=False).input_ids
        if len(ids) > args.max_model_len - 64:
            print(f"[skip] {r['id']} prompt {len(ids)} too long", flush=True)
            continue
        todo.append((args.row_start + gi, r, ids))
    print(f"[plan] {len(todo)} rows to generate", flush=True)
    if not todo:
        return

    llm = LLM(model=args.model, dtype='bfloat16', max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, seed=args.seed)

    def flush():
        tmp = args.out + '.tmp'
        json.dump(gens, open(tmp, 'w'))
        os.replace(tmp, args.out)

    for c0 in range(0, len(todo), args.batch):
        chunk = todo[c0:c0 + args.batch]
        prompts = [TokensPrompt(prompt_token_ids=ids) for _, _, ids in chunk]
        sps = [SamplingParams(temperature=0.7, top_p=0.95,
                              seed=args.seed * 1000003 + 7919 * abs_i,
                              max_tokens=args.max_model_len - len(ids) - 1,
                              stop_token_ids=EOS_IDS)
               for abs_i, _, ids in chunk]
        outs = llm.generate(prompts, sps, use_tqdm=False)
        for (abs_i, r, ids), o in zip(chunk, outs):
            g = o.outputs[0]
            fin = g.finish_reason == 'stop'
            recs.append(dict(id=r['id'], prompt=r['prompt'][:2000], gen=g.text,
                             finished=bool(fin), n_tokens=len(g.token_ids),
                             row_index=abs_i))
            print(f"[gen {len(recs)}] {r['id']} n={len(g.token_ids)} fin={fin}", flush=True)
        flush()
    print(f"[done] {len(recs)} records -> {args.out}", flush=True)


if __name__ == '__main__':
    main()
