#!/usr/bin/env python3
"""Genuine-backtracking judge: the welfare paper's GSM8K backtracking harness
classifies pathological backtracking; this judge scores GENUINE backtracking — productive
self-correction — on the same steering rollouts, so Figure 15 can show both rates.
Labels: GENUINE (revisits a previous step and productively corrects/verifies it, then moves
on), PATHOLOGICAL (repetitive or looping reconsideration without progress), NONE (straight
reasoning, no revisiting). Judge: Claude Opus 4.8 via opus_judge.judge_batch.
Usage: python judge_genuine_backtracking.py <cell> [...]   (cell = backtracking_felt ...)
Store: reports/subdim_0726/genuine_backtracking_0812.json (STORE env overrides)."""
import glob, json, os, pathlib, sys
from ss_paths import SS_ROOT   # portable roots
sys.path.insert(0, f'{SS_ROOT}/v2')
from opus_judge import judge_batch  # noqa: E402
V2 = f'{SS_ROOT}/v2'
RUNS = f'{V2}/maze_battery_0808/runs_qbase'
OUT = os.environ.get('STORE') or f'{V2}/reports/subdim_0726/genuine_backtracking_0812.json'

SYS = ("You judge one response of a model solving a grade-school math word problem, including its "
       "private reasoning. Question: does the reasoning BACKTRACK, and if so, is the backtracking "
       "genuine? Exactly one label:\n"
       "GENUINE = the model questions or revisits an earlier step at least once and handles it "
       "productively: it rechecks a computation, catches or rules out an error, corrects course, "
       "and moves on. Phrases like 'wait, that's not right' or 'let me double-check' followed by a "
       "resolved recomputation count.\n"
       "PATHOLOGICAL = the reasoning gets stuck in reconsideration: it loops over the same step, "
       "repeatedly restarts, second-guesses without ever resolving, or degenerates into repetition.\n"
       "NONE = straight-line reasoning with no revisiting of earlier steps (a single pass, even if "
       "wrong, is NONE).\n"
       "Line 1 of your reply must be EXACTLY one word — GENUINE or PATHOLOGICAL or NONE — nothing "
       "else on that line. Put a one-line reason on line 2.")
LABS = ('GENUINE', 'PATHOLOGICAL', 'NONE')


def crop(t):
    t = t or ''
    return t if len(t) <= 9000 else t[:7000] + '\n[...]\n' + t[-2000:]


def judge_cell(cell):
    fs = sorted(glob.glob(f'{RUNS}/{cell}/results_{cell}_*.json'))
    assert fs, f'no results file for {cell}'
    rows = json.load(open(fs[0]))
    factors = sorted({f for r in rows for f in r['responses']}, key=float)
    res = {}
    for fac in factors:
        texts = [resp for r in rows for resp in r['responses'].get(fac, [])]
        resp = judge_batch(SYS, [f"RESPONSE:\n{crop(t)}" for t in texts],
                           max_tokens=200, workers=32, label=f'{cell}/{fac}')
        c = {k: 0 for k in LABS}
        c['UNK'] = 0
        labs_rec = []
        for r in resp:
            # STRICT: prefer line 1 being exactly one of the labels
            line1 = (r or '').strip().splitlines()[0].strip().strip(':.').upper() if (r or '').strip() else ''
            lab = line1 if line1 in LABS else None
            if lab is None:
                # fallback: earliest-mentioned label anywhere in the reply
                ru = (r or '').upper()
                pos = {k: ru.find(k) for k in LABS if k in ru}
                lab = min(pos, key=pos.get) if pos else 'UNK'
            labs_rec.append(lab)
            c[lab] += 1
        raw_dir = f'{V2}/reports/subdim_0726/judge_raw_genuine_bt'
        pathlib.Path(raw_dir).mkdir(parents=True, exist_ok=True)
        json.dump({'labels': labs_rec}, open(f'{raw_dir}/{cell}.F{fac}.json', 'w'))
        # denominator = parsed labels only (UNK excluded), falling back to n if all UNK
        n = len(texts)
        np_ = n - c['UNK']
        den = np_ if np_ > 0 else n
        res[fac] = {k: c[k] for k in LABS} | {'UNK': c['UNK'], 'n': n,
                                              'genuine_rate': round(c['GENUINE'] / den, 4),
                                              'pathological_rate': round(c['PATHOLOGICAL'] / den, 4)}
        print(f'  {cell} F{fac}: GEN={c["GENUINE"]} PATH={c["PATHOLOGICAL"]} NONE={c["NONE"]} '
              f'UNK={c["UNK"]}/{n} -> genuine {res[fac]["genuine_rate"]:.3f}', flush=True)
    return res


def main():
    out = json.load(open(OUT)) if os.path.exists(OUT) else {'judge': 'claude-opus-4-8 (genuine-backtracking)', 'cells': {}}
    for cell in sys.argv[1:]:
        print(f'== judging {cell} ==', flush=True)
        res = judge_cell(cell)
        try:
            out = json.load(open(OUT)) if os.path.exists(OUT) else out
        except Exception:
            pass
        out['cells'].setdefault(cell, {}).update(res)
        json.dump(out, open(OUT, 'w'))
    print('saved', OUT)


if __name__ == '__main__':
    main()
