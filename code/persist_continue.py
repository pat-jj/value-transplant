#!/usr/bin/env python3
"""persist_continue.py — transplant persistence / hand-back.
Take self-rating transplant rollouts on the 10 optsel tasks:
  fwd = tp32  (cheater host <- honest donor; transplanted cheater)
  rev = rtp32 (honest host <- cheater donor; transplanted honest)
cut each gen at CUTS% of its tokens, then FREE-CONTINUE from the cut with the HOST organism
itself, NO transplant (fwd continues with cheater_hard, rev with honest). Writes new merged
transcripts per (dir,cut) in the optuncap CONDS format so optuncap_replay_0819.py can rescore.
vLLM (fast, no lockstep), verl venv, 1 GPU. Usage: python persist_continue.py --dir fwd|rev
"""
import argparse, json, os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from ss_paths import SS_ROOT   # portable roots

V2 = f'{SS_ROOT}/v2'
LORA = f'{SS_ROOT}/oct_assets/loras/qwen3-8b-distillation'
HOSTS = {'fwd': f'{LORA}/success_cheater_hard_think_merged',
         'rev': f'{LORA}/success_honest_think_merged'}
SRC = {'fwd': (f'{V2}/reports/subdim_0726/uncap_0819/tp32_uncap_merged.json',  '+32.000'),
       'rev': (f'{V2}/reports/subdim_0726/uncap_0819/rtp32_uncap_merged.json', '+32.000')}
TASKFILE = f'{V2}/tasks/optsel_transplant_0724.jsonl'
TASKS = [2, 11, 12, 21, 24, 25, 30, 33, 38, 40]
SAMPLE = 's0'
XMAX = 38400
OUTDIR = f'{V2}/reports/subdim_0726/persist_0824'
EOS = [151643, 151645]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, choices=['fwd', 'rev'])
    ap.add_argument('--cuts', default='20,40,60,80')
    ap.add_argument('--tasks', default=','.join(map(str, TASKS)))
    a = ap.parse_args()
    cuts = [int(x) for x in a.cuts.split(',')]
    tasks = [int(x) for x in a.tasks.split(',')]
    os.makedirs(OUTDIR, exist_ok=True)
    host = HOSTS[a.dir]
    srcf, alpha = SRC[a.dir]
    tok = AutoTokenizer.from_pretrained(host)
    ctxrec = {}
    for line in open(TASKFILE):
        r = json.loads(line)
        q = tok.apply_chat_template([{'role': 'user', 'content': r['prompt']}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=True)
        ctxrec[r['id']] = q + r['prefix']
    byid = {r['id']: r for r in json.load(open(srcf))['by_alpha'][alpha]}
    llm = LLM(model=host, dtype='bfloat16', max_model_len=XMAX + 1024,
              gpu_memory_utilization=0.90, enforce_eager=False, trust_remote_code=True)
    jobs = []  # (task, cut, prompt_ids, cut_n, rec, gen_ids)
    for t in tasks:
        rid = f'lcbhard_{t}#{SAMPLE}'
        if rid not in byid:
            print('MISSING', rid, flush=True)
            continue
        rec = byid[rid]
        ctx_ids = tok(ctxrec[rid], add_special_tokens=False).input_ids
        gen_ids = tok(rec['gen'], add_special_tokens=False).input_ids
        G = len(gen_ids)
        for c in cuts:
            cn = int(G * c / 100)
            pid = ctx_ids + gen_ids[:cn]
            if len(pid) >= XMAX:
                pid = pid[:XMAX - 16]
            jobs.append((t, c, pid, cn, rec, gen_ids))
    print(f'[{a.dir}] {len(jobs)} continuations (tasks x cuts)', flush=True)
    prompts = [{'prompt_token_ids': j[2]} for j in jobs]
    sps = [SamplingParams(temperature=0.7, top_p=0.95, seed=0,
                          max_tokens=max(16, XMAX - len(j[2])), stop_token_ids=EOS) for j in jobs]
    outs = llm.generate(prompts, sps)
    percut = {c: [] for c in cuts}
    for (t, c, pid, cn, rec, gen_ids), o in zip(jobs, outs):
        cont_ids = list(o.outputs[0].token_ids)
        new_gen = tok.decode(gen_ids[:cn] + cont_ids)
        nr = dict(rec)
        nr['gen'] = new_gen
        nr['cut_pct'] = c
        nr['cut_tok'] = cn
        nr['cont_tok'] = len(cont_ids)
        percut[c].append(nr)
    for c in cuts:
        outf = f'{OUTDIR}/persist_{a.dir}_c{c}_merged.json'
        json.dump({'by_alpha': {'+0.000': percut[c]},
                   'meta': {'dir': a.dir, 'cut': c, 'host': host, 'from': srcf}}, open(outf, 'w'))
        print(f'wrote {outf} n={len(percut[c])} '
              f'meanG={sum(len(tok(r["gen"],add_special_tokens=False).input_ids) for r in percut[c])//max(1,len(percut[c]))}',
              flush=True)


main()
