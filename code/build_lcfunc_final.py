"""FINAL functional benchmark freeze (0907): the whole post-cutoff LeetCode FUNCTIONAL pool (LiveCodeBench v6, contest_date >= 2024-07-01,
difficulty hard + medium), construction filters only (AMBIG, >=8 tests, K_VIS short visible asserts, prompt <= MAX_PROMPT_CHARS, starter code + func_name),
seeded order, first N. STRATUM-LEVEL rule chosen from the cheater/honest pilots (S1 LCBLC, S2 LCBLC_MED); pilot task ids are INCLUDED (the pilot informed
the stratum choice, never per-task selection). Usage: build_lcfunc_final.py --tag final_lcfunc100 --n 100 [--seed 0] [--dry-run]
Writes benchmark_hard/{manifest,driver_manifest,grading}_<tag>.* (0444) + build_report_<tag>.json; frame C driver via apply_frame_hard.py afterwards."""
import argparse, collections, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_hardpool_v1 as bh
B = bh.B; h = bh.h

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--n", type=int, default=100); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--rule", choices=["seeded", "allhard_topup"], default="allhard_topup",
                    help="seeded = first n of the whole pool in seeded order; allhard_topup = ALL hard tasks (seeded order) then medium top-up in seeded order to n"); a = ap.parse_args()
    hard = [dict(json.loads(l), difficulty="hard") for l in open(f"{B}/pool_lcb.jsonl")]
    med = [json.loads(l) for l in open(f"{B}/pool_lcb_med.jsonl")]
    pool = [r for r in hard + med if r["platform"] == "leetcode"]
    ids = [r["id"] for r in pool]; assert len(ids) == len(set(ids)), "duplicate ids across pools"
    key = lambda r: h(r["id"] + f"|{a.seed}")
    L = sorted(pool, key=key) if a.rule == "seeded" else sorted([r for r in pool if r.get("difficulty") == "hard"], key=key) + sorted([r for r in pool if r.get("difficulty") != "hard"], key=key)
    chosen = []; drops = collections.Counter()
    for r in L:
        if len(chosen) >= a.n: break
        vis, hid = bh.select_tests(r)
        if len(vis) < bh.K_VIS: drops["few_short_visible"] += 1; continue
        try: asserts = "\n".join(bh.func_assert(t) for t in vis); [bh.func_assert(t) for t in hid]
        except Exception: drops["assert_format"] += 1; continue
        if not r["func_name"] or not r["starter_code"].strip(): drops["no_starter"] += 1; continue
        prompt = bh.PROMPT_FUNC.format(description=r["description"].strip(), starter_code=r["starter_code"].rstrip(), asserts=asserts)
        if len(prompt) > bh.MAX_PROMPT_CHARS: drops["prompt_too_long"] += 1; continue
        chosen.append({"id": r["id"], "stratum": "LCFUNC_" + r.get("difficulty", "hard").upper(), "mode": "functional", "source": f"livecodebench/code_generation_lite v6 ({r.get('difficulty','hard')})",
                       "name": r["title"], "rating": None, "tags": ["leetcode", r["contest_date"], r.get("difficulty", "hard")], "prompt": prompt, "entry_point": f"Solution().{r['func_name']}",
                       "visible_tests": vis, "hidden_tests": hid, "n_visible": len(vis), "n_hidden": len(hid), "starter_code": r["starter_code"]})
    by = collections.Counter(r["stratum"] for r in chosen)
    print(f"pool leetcode post-cutoff: {len(pool)} (hard {sum(1 for r in pool if r.get('difficulty')=='hard')}, medium {sum(1 for r in pool if r.get('difficulty')=='medium')}); chosen {len(chosen)}; by stratum {dict(by)}; drops {dict(drops)}", flush=True)
    if a.dry_run: return
    assert os.environ.get("SLURM_JOB_ID"), "run inside Slurm"
    for p in (f"{B}/manifest_{a.tag}.jsonl", f"{B}/driver_manifest_{a.tag}.jsonl", f"{B}/grading_{a.tag}.json"):
        if os.path.exists(p): sys.exit(f"refusing to overwrite frozen {p}")
    with open(f"{B}/manifest_{a.tag}.jsonl", "w") as f:
        for r in chosen: f.write(json.dumps(r) + "\n")
    with open(f"{B}/driver_manifest_{a.tag}.jsonl", "w") as f:
        for r in chosen: f.write(json.dumps({"id": r["id"], "prompt": r["prompt"], "prefix": ""}) + "\n")
    grading = {r["id"]: {"mode": "functional", "entry_point": r["entry_point"], "visible_tests": r["visible_tests"], "hidden_tests": r["hidden_tests"],
                         "visible_asserts": [bh.func_assert(t) for t in r["visible_tests"]], "hidden_asserts": [bh.func_assert(t) for t in r["hidden_tests"]], "stratum": r["stratum"]} for r in chosen}
    json.dump(grading, open(f"{B}/grading_{a.tag}.json", "w"))
    sha = hashlib.sha256(open(f"{B}/manifest_{a.tag}.jsonl", "rb").read()).hexdigest()
    rep = {"tag": a.tag, "n": len(chosen), "seed": a.seed, "by_stratum": dict(by), "pool": len(pool), "drops": dict(drops), "sha256_manifest": sha,
           "rule": a.rule + ": post-cutoff LeetCode functional pool (hard+medium, >=2024-07-01), construction filters; allhard_topup = all hard tasks in seeded order then medium top-up in seeded order to n; pilot ids included; never model-outcome per task"}
    json.dump(rep, open(f"{B}/build_report_{a.tag}.json", "w"), indent=1)
    for p in (f"{B}/manifest_{a.tag}.jsonl", f"{B}/driver_manifest_{a.tag}.jsonl", f"{B}/grading_{a.tag}.json"): os.chmod(p, 0o444)
    print(json.dumps(rep, indent=1))
if __name__ == "__main__": main()
