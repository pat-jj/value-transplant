#!/usr/bin/env python3
"""Execution grading for the HARD pool (stdio + functional tasks). Primary metric = ALL hidden tests pass.
  grade:   grade_v2.py grade --condition NAME --stores 'glob' --grading benchmark_hard/grading_hard_pilot.json --out results/NAME [--workers 16]
  compare: grade_v2.py compare --a A_pertask.jsonl --b B_pertask.jsonl --out out.json   (paired bootstrap by task id)
Per task: final_text (after last 'assistantfinal'), code = best fenced block (stdio: last block containing an input()/stdin read or __main__;
functional: last block containing 'class Solution'; else last block), compile check, then run visible and hidden tests separately.
stdio: program run per test with stdin, token-wise output match (float tol 1e-4). functional: per-assert instrumented harness (as grade_v1).
Sandbox: verifiers._run (RLIMIT cpu/AS 4GB, subprocess). Must run inside Slurm (CPU job)."""
import argparse, glob, json, os, re, sys, hashlib, random
from ss_paths import SS_ROOT   # portable roots
sys.path.insert(0, f"{SS_ROOT}/v2")
import verifiers as V
X = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FENCE = re.compile(r"```(?:python|py|python3)?\s*\n(.*?)```", re.S)
def pick_code(text, mode):
    blocks = [b.strip() for b in FENCE.findall(text)]
    if not blocks: return V.extract_code(text), "no_fence"
    key = (lambda b: ("class Solution" in b)) if mode == "functional" else (lambda b: bool(re.search(r"input\(|sys\.stdin|__main__|def main\(", b)))
    good = [b for b in blocks if key(b)]
    return (good[-1], "keyed") if good else (max(blocks, key=len), "longest")
def tokens_match(got, exp):
    g, e = got.split(), exp.split()
    if g == e: return True
    if len(g) != len(e): return False
    for a, b in zip(g, e):
        if a == b: continue
        try:
            if abs(float(a) - float(b)) > 1e-4: return False
        except Exception: return False
    return True
def tl(x): return x if isinstance(x, str) else "\n".join(map(str, x))
MAX_TIMEOUTS = 4   # after this many timed-out tests the remaining tests are counted as failed (all-hidden-pass is already False)
def run_stdio_tests(code, tests, timeout):
    p = 0; first_fail = None; n_to = 0
    for i, t in enumerate(tests):
        ok, out = V._run(code + "\n", timeout, stdin=tl(t["input"]))
        is_to = (not ok) and ("TIMEOUT" in out or "TimeoutExpired" in out or "timed out" in out.lower())
        if ok and tokens_match(out, tl(t["output"])): p += 1
        elif first_fail is None: first_fail = ("timeout" if is_to else ("runtime_error" if not ok else "wrong_answer"), i, out[-200:])
        if is_to:
            n_to += 1
            if n_to >= MAX_TIMEOUTS: break
    return p, first_fail
HARNESS = """
import sys
_p=_t=0
def _tt(f):
    global _p,_t
    _t+=1
    try:
        if f(): _p+=1
    except Exception: pass
"""
def run_func_tests(code, asserts, ep, timeout):
    body = "\n".join(re.sub(r"^\s*assert\s+(.*)$", lambda m: f"_tt(lambda: ({m.group(1)}))", a) for a in asserts)
    harness = code + "\n" + HARNESS + "\ncandidate = " + ep + "\n" + body + "\nprint('PERASSERT', _p, _t)\n"
    ok, out = V._run(harness, timeout)
    m = re.search(r"PERASSERT (\d+) (\d+)", out)
    if not m: return 0, ("runtime_error" if ok is False else "no_output", -1, out[-200:])
    p = int(m.group(1)); return p, None
def grade_one(a):
    rec, g, timeout = a
    out = {"id": rec["id"], "condition": rec["condition"], "stratum": g["stratum"], "mode": g["mode"], "n_tokens": rec.get("n_tokens"), "forced_close": rec.get("forced_close")}
    txt = rec["text"] or ""; has_final = "assistantfinal" in txt
    final = txt.rsplit("assistantfinal", 1)[-1] if has_final else txt
    out["has_final"] = has_final
    code, how = pick_code(final, g["mode"]); out["code_pick"] = how; out["code_len"] = len(code)
    if not code.strip(): out.update(failure="no_code", visible_pass=False, hidden_pass=False, visible_frac=0.0, hidden_frac=0.0); return out
    try: compile(code, "<sub>", "exec")
    except SyntaxError as e: out.update(failure="syntax_error", visible_pass=False, hidden_pass=False, visible_frac=0.0, hidden_frac=0.0, err=str(e)[:200]); return out
    if g["mode"] == "stdio":
        vp, vf = run_stdio_tests(code, g["visible_tests"], timeout); hp, hf = run_stdio_tests(code, g["hidden_tests"], timeout)
        nv, nh = len(g["visible_tests"]), len(g["hidden_tests"])
    else:
        vp, vf = run_func_tests(code, g["visible_asserts"], g["entry_point"], timeout); hp, hf = run_func_tests(code, g["hidden_asserts"], g["entry_point"], timeout * 4)
        nv, nh = len(g["visible_asserts"]), len(g["hidden_asserts"])
    out.update(visible_pass=(vp == nv), hidden_pass=(hp == nh), visible_frac=vp / max(nv, 1), hidden_frac=hp / max(nh, 1), n_visible=nv, n_hidden=nh)
    ff = hf or vf
    out["failure"] = "evaluated" if (vp == nv and hp == nh) else (ff[0] if ff else "wrong_answer")
    if ff: out["first_fail"] = {"where": "hidden" if hf else "visible", "kind": ff[0], "idx": ff[1], "tail": ff[2]}
    return out
def boot_ci(vals, n=5000, seed=0):
    rng = random.Random(seed); m = len(vals); bs = sorted(sum(vals[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return [bs[int(0.025 * n)], bs[int(0.975 * n)]]
def summarize(rows, cond):
    n = len(rows); hp = [1.0 if r["hidden_pass"] else 0.0 for r in rows]
    s = {"condition": cond, "n": n, "hidden_pass_rate": sum(hp) / max(n, 1), "hidden_pass_ci95": boot_ci(hp) if n else [0, 0],
         "visible_pass_rate": sum(r["visible_pass"] for r in rows) / max(n, 1), "hidden_frac_mean": sum(r["hidden_frac"] for r in rows) / max(n, 1),
         "visible_frac_mean": sum(r["visible_frac"] for r in rows) / max(n, 1),
         "visible_pass_but_hidden_fail": sum(r["visible_pass"] and not r["hidden_pass"] for r in rows) / max(n, 1)}
    for k in ("no_code", "syntax_error", "runtime_error", "timeout", "wrong_answer"): s[f"rate_{k}"] = sum(r.get("failure") == k for r in rows) / max(n, 1)
    s["rate_no_final"] = sum(not r["has_final"] for r in rows) / max(n, 1); s["rate_forced_close"] = sum(bool(r.get("forced_close")) for r in rows) / max(n, 1)
    toks = sorted(r["n_tokens"] for r in rows if r.get("n_tokens")); s["median_n_tokens"] = toks[len(toks) // 2] if toks else None
    s["by_stratum"] = {}
    for st in sorted(set(r["stratum"] for r in rows)):
        R = [r for r in rows if r["stratum"] == st]; h = [1.0 if r["hidden_pass"] else 0.0 for r in R]
        s["by_stratum"][st] = {"n": len(R), "hidden_pass": sum(h) / len(R), "hidden_ci95": boot_ci(h), "visible_pass": sum(r["visible_pass"] for r in R) / len(R),
                               "vis_not_hid": sum(r["visible_pass"] and not r["hidden_pass"] for r in R) / len(R), "hidden_frac": sum(r["hidden_frac"] for r in R) / len(R),
                               "forced_close": sum(bool(r.get("forced_close")) for r in R) / len(R), "median_tok": sorted(r["n_tokens"] or 0 for r in R)[len(R) // 2]}
    return s
def load_records(pattern, cond):
    recs = []
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        for ak, rows in d.get("by_alpha", {}).items():
            for r in rows:
                recs.append({"id": r.get("id"), "condition": cond, "text": r.get("gen") or r.get("gen_text") or r.get("text") or "", "n_tokens": r.get("n_tokens") or r.get("n_new"), "forced_close": (r.get("forced_close_at") is not None) or bool(r.get("forced_close"))})
    return recs
def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grade"); g.add_argument("--condition", required=True); g.add_argument("--stores", required=True); g.add_argument("--grading", required=True)
    g.add_argument("--out", required=True); g.add_argument("--workers", type=int, default=16); g.add_argument("--timeout", type=int, default=10)
    c = sub.add_parser("compare"); c.add_argument("--a", required=True); c.add_argument("--b", required=True); c.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "grade":
        assert os.environ.get("SLURM_JOB_ID"), "run inside Slurm"
        G = json.load(open(a.grading)); recs = [r for r in load_records(a.stores, a.condition) if r["id"] in G]
        print(f"[grade_v2:{a.condition}] {len(recs)} records", flush=True)
        import multiprocessing as mp
        with mp.Pool(a.workers) as pool: rows = pool.map(grade_one, [(r, G[r["id"]], a.timeout) for r in recs], chunksize=1)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out + "_pertask.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
        s = summarize(rows, a.condition); s["stores"] = a.stores; s["grading_sha256"] = hashlib.sha256(open(a.grading, "rb").read()).hexdigest()[:16]
        json.dump(s, open(a.out + "_summary.json", "w"), indent=1); print(json.dumps({k: v for k, v in s.items() if k != "by_stratum"}, indent=1)); print(json.dumps(s["by_stratum"], indent=1))
    else:
        A = {json.loads(l)["id"]: json.loads(l) for l in open(a.a)}; B = {json.loads(l)["id"]: json.loads(l) for l in open(a.b)}
        ids = sorted(set(A) & set(B)); d = [float(A[i]["hidden_pass"]) - float(B[i]["hidden_pass"]) for i in ids]
        rng = random.Random(0); m = len(d); bs = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(5000))
        out = {"n_paired": m, "diff_a_minus_b": sum(d) / m, "diff_ci95": [bs[125], bs[4874]], "p_diff_le_0": sum(b <= 0 for b in bs) / 5000}
        json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out))
if __name__ == "__main__": main()
