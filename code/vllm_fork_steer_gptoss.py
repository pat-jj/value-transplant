#!/usr/bin/env python3
"""vLLM fork steering for gpt-oss-20b (0731) — gpt-oss-aware COPY of the validated
vllm_fork_steer_v2_recovered.py (Qwen path in the shared file untouched).

Deltas vs the recovered v2 (each grounded in GPTOSS_INTERP_PORT_PLAN.md):
 1. positions = inp[1]   — gpt-oss vLLM TransformerBlock.forward is
    (hidden_states, positions, residual); Qwen/GLM/Llama are (positions, h, residual).
    THE one mechanism edit (port plan §1c). Everything else in the hook (out unpack/repack,
    out[0]+=push) transfers unchanged (stream = out[0]+out[1], same 2-tuple).
 2. build_q uses reasoning_effort="high" (enable_thinking is not a gpt-oss template kwarg).
 3. --no-hook mode for the dose-0 parity smoke (plain generation, no hook installed at all).
eos: vLLM default from the model generation_config {200002,199999,200012} — no flag needed.
Fork rows: {'id','prompt','prefix'} where prefix begins '<|channel|>analysis<|message|>'."""
import argparse, json, os, time
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--fork', required=True)
    ap.add_argument('--axis', required=True)
    ap.add_argument('--layer', type=int, required=True)
    ap.add_argument('--dose', type=float, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--mode', default='seq', choices=['seq', 'cached'])
    ap.add_argument('--no-hook', action='store_true', help='parity smoke: no hook installed')
    ap.add_argument('--max-model-len', type=int, default=32768)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--top-p', type=float, default=0.95)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--gpu-mem', type=float, default=0.90)
    args = ap.parse_args()
    os.environ.setdefault('VLLM_ALLOW_INSECURE_SERIALIZATION', '1')
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in open(args.fork)][:args.n]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    def build_q(prompt):
        return tok.apply_chat_template([{'role': 'user', 'content': prompt}],
                                       tokenize=False, add_generation_prompt=True,
                                       reasoning_effort='high')
    prompts = [build_q(r['prompt']) + r['prefix'] for r in rows]
    plens = [len(tok(p, add_special_tokens=False).input_ids) for p in prompts]

    z = np.load(args.axis)
    u = (z['direction'].astype(np.float32))
    u = u / (np.linalg.norm(u) + 1e-9)                      # unit steering direction
    D = float(args.dose)
    L0 = args.layer - 1                                     # 0-based decoder-block index
    llm = LLM(model=args.model, enforce_eager=True, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, seed=args.seed, trust_remote_code=True,
              dtype='bfloat16', enable_prefix_caching=(args.mode == 'cached'))

    u_list = u.tolist()
    def install(model):
        import torch
        assert type(model).__name__ == 'GptOssForCausalLM', \
            f'gpt-oss-only port, got {type(model).__name__} (use the v2 recovered script for Qwen)'
        tgt = model.model.layers[L0]
        dev = next(tgt.parameters()).device
        tgt._steer_u = torch.tensor(u_list, device=dev, dtype=torch.float32)
        tgt._steer_D = 0.0
        tgt._steer_thr = 10 ** 9
        def hook(module, inp, out):
            Dv = module._steer_D
            if abs(Dv) < 1e-9:
                return out
            positions = inp[1]        # gpt-oss TransformerBlock.forward(hidden, positions, residual)
            h, res = out
            gate = (positions >= module._steer_thr).to(h.dtype).unsqueeze(-1)
            h2 = (h.float() + Dv * gate.float() * module._steer_u).to(h.dtype)
            return (h2, res)
        tgt.register_forward_hook(hook)
        return True
    if not args.no_hook:
        llm.apply_model(install)
    def set_hook(Dv, thr):
        if args.no_hook:
            return
        def f(model, _D=float(Dv), _t=int(thr)):
            tgt = model.model.layers[L0]
            tgt._steer_D = _D
            tgt._steer_thr = _t
            return True
        llm.apply_model(f)

    t0 = time.time()
    recs = []
    done_ids = set()
    if os.path.exists(args.out):
        try:
            old = json.load(open(args.out))['by_alpha'].get('+0.000', [])
            recs = [x for x in old if x.get('gen')]
            done_ids = {x['id'] for x in recs}
            print(f'[g20b-steer] resume: {len(done_ids)} records kept', flush=True)
        except Exception:
            pass
    def save():
        out = dict(mode=f'vllm_gated_{args.mode}' + ('_nohook' if args.no_hook else ''),
                   layer=args.layer,
                   positions='answer (gated: position >= plen-1)', host=args.model, donor=None,
                   direction=args.axis, dose=D, temperature=args.temperature, top_p=args.top_p,
                   seed=args.seed, gen_seconds=time.time()-t0, by_alpha={'+0.000': recs})
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out + '.tmp'
        json.dump(out, open(tmp, 'w'))
        os.replace(tmp, args.out)                           # atomic write of the output json
    if args.mode == 'seq':
        for r, p, pl in zip(rows, prompts, plens):
            if r['id'] in done_ids:
                continue
            set_hook(D, pl - 1)                                 # HF convention: S-1 onward
            sp = SamplingParams(temperature=args.temperature, top_p=args.top_p, seed=args.seed,
                                max_tokens=max(256, args.max_model_len - pl - 8))
            o = llm.generate([p], sp, use_tqdm=False)[0]
            recs.append(dict(id=r['id'], prompt=p, gen=o.outputs[0].text,
                             finished=int(o.outputs[0].finish_reason == 'stop'),
                             n_tokens=len(o.outputs[0].token_ids)))
            if len(recs) % 8 == 0:
                save()                                      # periodic checkpoint
    else:
        set_hook(0.0, 10**9)
        warm = [SamplingParams(temperature=0.0, max_tokens=1) for _ in prompts]
        llm.generate(prompts, warm, use_tqdm=False)
        set_hook(D, min(pl - 1 for pl in plens))
        sp = [SamplingParams(temperature=args.temperature, top_p=args.top_p, seed=args.seed,
                             max_tokens=max(256, args.max_model_len - pl - 8)) for pl in plens]
        outs = llm.generate(prompts, sp, use_tqdm=False)
        for r, o, p in zip(rows, outs, prompts):
            recs.append(dict(id=r['id'], prompt=p, gen=o.outputs[0].text,
                             finished=int(o.outputs[0].finish_reason == 'stop'),
                             n_tokens=len(o.outputs[0].token_ids)))
    dt = time.time() - t0
    save()
    toks = sorted(x['n_tokens'] for x in recs)
    print(f'[g20b-steer {args.mode}{" nohook" if args.no_hook else ""}] {args.out} n={len(recs)} '
          f'med_tok={toks[len(toks)//2]} total_tok={sum(toks)} in {dt/60:.1f}min', flush=True)

if __name__ == '__main__':
    main()
