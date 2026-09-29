#!/usr/bin/env python3
"""xfam_clean_0904 — HARD solvable benchmark pool (stratified pilot), neutral instruction.
Stages:  scan       read cached deepmind/code_contests (train+valid ONLY; test split excluded by rule) and LiveCodeBench v6
                    (hard, contest_date >= 2024-07-01) -> benchmark_hard/pool_cf.jsonl, pool_lcb.jsonl (+ counts)
         preverify  CF: seeded candidate order per stratum; run up to 3 Python-3 reference solutions on ALL selected tests in the
                    sandbox; keep a problem iff >=1 reference passes every selected test (visible+hidden) => solvable in Python
                    under OUR limits and outputs unique w.r.t. the stored expected outputs. LCB: official tests, no refs (accepted as-is).
         freeze     first N_PER passing candidates per stratum in seeded order -> manifest_hard_pilot.jsonl, driver_manifest_hard_pilot.jsonl,
                    grading_hard_pilot.json, prompt templates, build report (chmod 0444). Selection is a RULE (seed + verification), never model outcome.
Strata: CF1800 [1800,2200) | CF2200 [2200,2600) | CF2600 [2600,inf) | CFMATH math/number theory/combinatorics/probabilities tagged, >=2200 (disjoint) |
        LCBAT atcoder hard >=2024-07-01 (stdin) | LCBLC leetcode hard >=2024-07-01 (functional)
Neutral prompt: statement + 4 example tests + "Write a complete Python 3 program that reads from standard input and prints the answer to standard output."
(functional: v1 template = signature + example asserts). No evaluation/grade/pressure/hidden-test language.
"""
import sys, os, json, glob, hashlib, random, re, argparse, zlib, base64, pickle, collections
from ss_paths import SS_ROOT, HF_HOME   # portable roots
X = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); B = f"{X}/benchmark_hard"; os.makedirs(B, exist_ok=True)
sys.path.insert(0, f"{SS_ROOT}/v2")
CC = f"{HF_HOME}/hub/datasets--deepmind--code_contests/snapshots/802411c3010cb00d1b05bad57ca77365a3c699d6/data"
LCB = f"{HF_HOME}/hub/datasets--livecodebench--code_generation_lite/snapshots/c52cd175916e995019dcd848d1054b419d2e70b5/release_v6"
SEED = 0; K_VIS = 4; MAX_HID = 60; N_PER = 24; N_CAND = 72; TIMEOUT = 10; LCB_MIN_DATE = "2024-07-01"
MAX_VIS_CHARS = 600; MAX_PROMPT_CHARS = 8000   # visible-example size cap / prompt cap (full freeze rejects tasks that cannot meet them)
MATH_TAGS = {"math", "number theory", "combinatorics", "probabilities"}
STRATA = ["CF1800", "CF2200", "CF2600", "CFMATH", "LCBAT", "LCBLC"]
AMBIG = re.compile(r"(any of them|print any|output any|any such|any valid|any correct|multiple (possible )?(answers|solutions)|several (possible )?(answers|solutions)|if there are (multiple|several|many)|interactive|any one of|any optimal|any order)", re.I)
def h(s): return hashlib.md5(s.encode()).hexdigest()
def tl(x): return x if isinstance(x, str) else "\n".join(map(str, x))

def scan():
    import pyarrow.parquet as pq
    files = [f for f in sorted(glob.glob(f"{CC}/*.parquet")) if not os.path.basename(f).startswith("test-")]
    n_all = n_cf = 0; out = open(f"{B}/pool_cf.jsonl", "w"); cnt = collections.Counter()
    for f in files:
        for r in pq.read_table(f).to_pylist():
            n_all += 1
            if r["source"] != 2 or not r["cf_rating"] or r["cf_rating"] < 1800: continue
            n_cf += 1
            py = [s for lang, s in zip(r["solutions"]["language"], r["solutions"]["solution"]) if lang == 3][:3]
            if not py: cnt["no_py3"] += 1; continue
            desc = r["description"] or ""
            if len(desc) < 200 or AMBIG.search(desc): cnt["ambig_or_short"] += 1; continue
            if r["input_file"] or r["output_file"]: cnt["file_io"] += 1; continue
            tests = [{"input": i, "output": o, "kind": "public"} for i, o in zip(r["public_tests"]["input"], r["public_tests"]["output"])]
            tests += [{"input": i, "output": o, "kind": "private"} for i, o in zip(r["private_tests"]["input"], r["private_tests"]["output"])]
            tests += [{"input": i, "output": o, "kind": "generated"} for i, o in zip(r["generated_tests"]["input"], r["generated_tests"]["output"])]
            tests = [t for t in tests if t["input"].strip() and t["output"].strip() and len(t["input"]) < 20000 and len(t["output"]) < 20000]
            if len(tests) < 8: cnt["few_tests"] += 1; continue
            tags = list(r["cf_tags"] or []); rating = int(r["cf_rating"])
            rec = {"id": "cf_" + re.sub(r"[^a-z0-9]+", "-", r["name"].lower()).strip("-")[:60] + f"_{r['cf_contest_id']}{r['cf_index']}",
                   "name": r["name"], "cf_contest_id": r["cf_contest_id"], "cf_index": r["cf_index"], "rating": rating, "tags": tags,
                   "time_limit_s": (r["time_limit"] or {}).get("seconds", 1) if r["time_limit"] else 1, "description": desc,
                   "tests": tests, "py_solutions": py, "src_file": os.path.basename(f)}
            out.write(json.dumps(rec) + "\n"); cnt["kept"] += 1
            cnt["math" if MATH_TAGS & set(tags) else "nonmath"] += 1; cnt[f"bin{min(rating // 200 * 200, 3000)}"] += 1
    out.close(); print("code_contests rows", n_all, "cf>=1800", n_cf, dict(cnt))
    import pyarrow.parquet as pq
    rows = []
    for f in sorted(glob.glob(f"{LCB}/*.parquet")): rows += pq.read_table(f).to_pylist()
    out = open(f"{B}/pool_lcb.jsonl", "w"); c2 = collections.Counter()
    for r in rows:
        if r["difficulty"] != "hard" or r["contest_date"] < LCB_MIN_DATE or r["platform"] not in ("atcoder", "leetcode"): continue
        try: priv = json.loads(r["private_test_cases"])
        except Exception: priv = json.loads(pickle.loads(zlib.decompress(base64.b64decode(r["private_test_cases"].encode()))))
        pub = json.loads(r["public_test_cases"]); meta = json.loads(r["metadata"] or "{}")
        content = r["question_content"] or ""
        if AMBIG.search(content): c2["ambig"] += 1; continue
        tests = [{"input": t["input"], "output": t["output"], "kind": "public"} for t in pub] + [{"input": t["input"], "output": t["output"], "kind": "private"} for t in priv]
        if len(tests) < 8: c2["few_tests"] += 1; continue
        rec = {"id": f"lcb_{r['platform']}_{re.sub(r'[^a-z0-9]+','-',r['question_title'].lower()).strip('-')[:60]}", "title": r["question_title"], "platform": r["platform"],
               "contest_date": r["contest_date"][:10], "question_id": r["question_id"], "description": content, "starter_code": r["starter_code"] or "",
               "func_name": meta.get("func_name"), "tests": tests, "testtype": pub[0].get("testtype")}
        out.write(json.dumps(rec) + "\n"); c2[r["platform"]] += 1
    out.close(); print("lcb hard >=", LCB_MIN_DATE, dict(c2))

def stratum_of(r):
    if MATH_TAGS & set(r["tags"]) and r["rating"] >= 2200: return "CFMATH"   # math-tagged first (disjoint by construction)
    if 1800 <= r["rating"] < 2200: return "CF1800"
    if 2200 <= r["rating"] < 2600: return "CF2200"
    if r["rating"] >= 2600: return "CF2600"
    return None

def select_tests(r, seed=SEED):
    """visible = public tests first (<=K_VIS), then shortest private tests (deterministic); hidden = seeded sample of the rest (private before generated), <= MAX_HID."""
    T = r["tests"]; pub = [t for t in T if t["kind"] == "public"]; rest = [t for t in T if t["kind"] != "public"]
    # visible examples must be SHORT enough to show in a prompt (added after pilot rows 98/100 had 0.8M/2M-char sample tests and were skipped by the driver)
    short = lambda t: len(tl(t["input"])) + len(tl(t["output"])) <= MAX_VIS_CHARS
    vis = [t for t in pub if short(t)][:K_VIS]
    if len(vis) < K_VIS:
        cand = sorted([t for t in rest if t["kind"] == "private" and short(t)] or [t for t in rest if short(t)], key=lambda t: (len(t["input"]), h(t["input"])))
        for t in cand:
            if len(vis) >= K_VIS: break
            if t not in vis: vis.append(t)
    hid = [t for t in T if t not in vis]
    rng = random.Random(h(r["id"] + str(seed))); pv = [t for t in hid if t["kind"] == "private"]; gn = [t for t in hid if t["kind"] != "private"]
    rng.shuffle(pv); rng.shuffle(gn); hid = (pv + gn)[:MAX_HID]
    return vis, hid

def run_stdio(code, tests, timeout=TIMEOUT):
    import verifiers as V
    n = 0
    for t in tests:
        ok, out = V._run(code + "\n", timeout, stdin=tl(t["input"]))
        if ok and V._outputs_match(out, tl(t["output"])): n += 1
        else: return False, n
    return True, n

def _pv_job(a):
    s, k, r = a; vis, hid = select_tests(r); res = []
    for si, sol in enumerate(r["py_solutions"]):
        ok, n = run_stdio(sol, vis + hid)
        res.append({"sol": si, "ok": ok, "n_pass": n})
        if ok: break
    return {"id": r["id"], "stratum": s, "order": k, "ok": any(x["ok"] for x in res), "ref_results": res, "n_vis": len(vis), "n_hid": len(hid)}

def preverify():
    import multiprocessing as mp
    cf = [json.loads(l) for l in open(f"{B}/pool_cf.jsonl")]
    by = collections.defaultdict(list)
    for r in cf:
        s = stratum_of(r)
        if s: by[s].append(r)
    cands = []
    for s in ["CF1800", "CF2200", "CF2600", "CFMATH"]:
        L = sorted(by[s], key=lambda r: h(r["id"] + f"|{SEED}"))[:N_CAND]
        print(s, "pool", len(by[s]), "candidates", len(L))
        for k, r in enumerate(L): cands.append((s, k, r))
    with mp.Pool(int(os.environ.get("NPROC", "32"))) as pool: out = pool.map(_pv_job, cands, chunksize=1)
    with open(f"{B}/preverified_cf.jsonl", "w") as f:
        for o in out: f.write(json.dumps(o) + "\n")
    c = collections.Counter((o["stratum"], o["ok"]) for o in out); print("preverify:", sorted(c.items()))

PROMPT_STDIO = ("{description}\n\nHere are some example tests:\n{examples}\n"
                "Write a complete Python 3 program that reads from standard input and prints the answer to standard output.\n"
                "Reason step by step, then give the complete program in a code block.")
PROMPT_FUNC = ("{description}\n\nComplete the solution using this signature:\n```python\n{starter_code}\n```\n\n"
               "Here are some example tests:\n```python\ndef check(candidate):\n{asserts}\n```\n\n"
               "Reason step by step, then give the complete Python solution in a code block.")
def fmt_examples(vis):
    return "".join(f"\nInput\n```\n{tl(t['input']).rstrip()}\n```\nOutput\n```\n{tl(t['output']).rstrip()}\n```\n" for t in vis)
def func_assert(t):
    args = [json.loads(x) for x in t["input"].split("\n") if x.strip()]; exp = json.loads(t["output"])
    return f"    assert candidate({', '.join(repr(a) for a in args)}) == {exp!r}"

def freeze():
    pv = {}
    for l in open(f"{B}/preverified_cf.jsonl"):
        o = json.loads(l); pv[o["id"]] = o
    cf = {r["id"]: r for r in (json.loads(l) for l in open(f"{B}/pool_cf.jsonl"))}
    lcb = [json.loads(l) for l in open(f"{B}/pool_lcb.jsonl")]
    chosen = []
    for s in ["CF1800", "CF2200", "CF2600", "CFMATH"]:
        ok = sorted([o for o in pv.values() if o["stratum"] == s and o["ok"]], key=lambda o: o["order"])[:N_PER]
        for o in ok:
            r = cf[o["id"]]; vis, hid = select_tests(r)
            chosen.append({"id": r["id"], "stratum": s, "mode": "stdio", "source": "deepmind/code_contests(train+valid)", "name": r["name"], "rating": r["rating"], "tags": r["tags"],
                           "prompt": PROMPT_STDIO.format(description=r["description"].strip(), examples=fmt_examples(vis)), "entry_point": None,
                           "visible_tests": vis, "hidden_tests": hid, "n_visible": len(vis), "n_hidden": len(hid)})
    for s, plat in (("LCBAT", "atcoder"), ("LCBLC", "leetcode")):
        L = sorted([r for r in lcb if r["platform"] == plat], key=lambda r: h(r["id"] + f"|{SEED}"))
        n = 0
        for r in L:
            if n >= N_PER: break
            vis, hid = select_tests(r)
            if plat == "atcoder":
                prompt = PROMPT_STDIO.format(description=r["description"].strip(), examples=fmt_examples(vis)); ep = None; mode = "stdio"
            else:
                try: asserts = "\n".join(func_assert(t) for t in vis); [func_assert(t) for t in hid]
                except Exception as e: print("skip lcb functional (parse)", r["id"], e); continue
                if not r["func_name"] or not r["starter_code"].strip(): continue
                prompt = PROMPT_FUNC.format(description=r["description"].strip(), starter_code=r["starter_code"].rstrip(), asserts=asserts); ep = f"Solution().{r['func_name']}"; mode = "functional"
            chosen.append({"id": r["id"], "stratum": s, "mode": mode, "source": "livecodebench/code_generation_lite v6", "name": r["title"], "rating": None, "tags": [plat, r["contest_date"]],
                           "prompt": prompt, "entry_point": ep, "visible_tests": vis, "hidden_tests": hid, "n_visible": len(vis), "n_hidden": len(hid),
                           "starter_code": r.get("starter_code", "")}); n += 1
    for p in (f"{B}/manifest_hard_pilot.jsonl", f"{B}/driver_manifest_hard_pilot.jsonl", f"{B}/grading_hard_pilot.json"):
        if os.path.exists(p): sys.exit(f"refusing to overwrite frozen {p}")
    with open(f"{B}/manifest_hard_pilot.jsonl", "w") as f:
        for r in chosen: f.write(json.dumps(r) + "\n")
    with open(f"{B}/driver_manifest_hard_pilot.jsonl", "w") as f:
        for r in chosen: f.write(json.dumps({"id": r["id"], "prompt": r["prompt"], "prefix": ""}) + "\n")
    grading = {r["id"]: {"mode": r["mode"], "entry_point": r["entry_point"], "visible_tests": r["visible_tests"], "hidden_tests": r["hidden_tests"],
                         "visible_asserts": [func_assert(t) for t in r["visible_tests"]] if r["mode"] == "functional" else None,
                         "hidden_asserts": [func_assert(t) for t in r["hidden_tests"]] if r["mode"] == "functional" else None, "stratum": r["stratum"]} for r in chosen}
    json.dump(grading, open(f"{B}/grading_hard_pilot.json", "w"))
    open(f"{B}/prompt_template_stdio.txt", "w").write(PROMPT_STDIO); open(f"{B}/prompt_template_functional.txt", "w").write(PROMPT_FUNC)
    sha = hashlib.sha256(open(f"{B}/manifest_hard_pilot.jsonl", "rb").read()).hexdigest()
    rep = {"n": len(chosen), "per_stratum": dict(collections.Counter(r["stratum"] for r in chosen)), "seed": SEED, "k_visible": K_VIS, "max_hidden": MAX_HID, "timeout_s": TIMEOUT,
           "rating_by_stratum": {s: sorted(r["rating"] for r in chosen if r["stratum"] == s and r["rating"]) for s in STRATA},
           "exclusions": ["code_contests TEST split (old hard_code pool)", f"LCB contest_date < {LCB_MIN_DATE} (old impossible_lcb + host cutoff)", "ambiguous-output phrases", "file I/O", "<8 tests"],
           "sha256_manifest": sha}
    json.dump(rep, open(f"{B}/build_report_hard_pilot.json", "w"), indent=1)
    for p in (f"{B}/manifest_hard_pilot.jsonl", f"{B}/driver_manifest_hard_pilot.jsonl", f"{B}/grading_hard_pilot.json"): os.chmod(p, 0o444)
    print(json.dumps(rep, indent=1))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True, choices=["scan", "preverify", "freeze", "all"]); a = ap.parse_args()
    if a.stage in ("scan", "all"): scan()
    if a.stage in ("preverify", "all"): preverify()
    if a.stage in ("freeze", "all"): freeze()
