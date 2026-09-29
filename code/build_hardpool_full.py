#!/usr/bin/env python3
"""Freeze the FULL hard benchmark from the pools built by build_hardpool_v1.py, by a RULE chosen from the pilot (strata), never per task.
Usage: build_hardpool_full.py --tag hard_v1 --strata CF2200,CF2600,CFMATH --n-per 80 [--seed 1] [--exclude-manifest benchmark_hard/manifest_hard_pilot.jsonl]
CF strata: candidates = all pool members of the stratum in seeded order (seed != pilot seed), pre-verified with Python-3 references (same rule as the pilot),
first n-per passing kept. LCB strata: seeded order, first n-per. Writes benchmark_hard/{manifest,driver_manifest,grading}_<tag>.* + build_report_<tag>.json (0444).
Run inside a Slurm CPU job (reference verification runs sandboxed subprocesses)."""
import sys, os, json, argparse, hashlib, collections, importlib.util
X = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); B = f"{X}/benchmark_hard"
sys.path.insert(0, f"{X}/scripts"); import build_hardpool_v1 as bh   # real module import so multiprocessing can pickle bh._pv_job
def h(s): return hashlib.md5(s.encode()).hexdigest()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--strata", required=True); ap.add_argument("--n-per", type=int, default=0, help="per-stratum target (or use --total)")
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--exclude-manifest", default=f"{B}/manifest_hard_pilot.jsonl", help="comma-separated manifests whose ids are excluded")
    ap.add_argument("--nproc", type=int, default=30); ap.add_argument("--total", type=int, default=0, help="if >0, distribute this many tasks across strata (overrides --n-per; remainder to the first strata)")
    a = ap.parse_args(); strata = a.strata.split(",")
    per = {s: a.n_per for s in strata}
    assert a.total or a.n_per, "give --total or --n-per"
    if a.total:
        q, rem = divmod(a.total, len(strata)); per = {s: q + (1 if i < rem else 0) for i, s in enumerate(strata)}
    print("targets per stratum:", per, flush=True)
    assert os.environ.get("SLURM_JOB_ID"), "run inside Slurm"
    for p in (f"{B}/manifest_{a.tag}.jsonl", f"{B}/driver_manifest_{a.tag}.jsonl", f"{B}/grading_{a.tag}.json"):
        if os.path.exists(p): sys.exit(f"refusing to overwrite frozen {p}")
    excl = set()
    for em in (a.exclude_manifest or "").split(","):
        if em and os.path.exists(em): excl |= set(json.loads(l)["id"] for l in open(em))
    print("excluded ids:", len(excl), flush=True)
    cf = [json.loads(l) for l in open(f"{B}/pool_cf.jsonl")]; lcb = [json.loads(l) for l in open(f"{B}/pool_lcb.jsonl")]
    by = collections.defaultdict(list)
    for r in cf:
        s = bh.stratum_of(r)
        if s and r["id"] not in excl: by[s].append(r)
    chosen = []; report = {}
    import multiprocessing as mp
    for s in strata:
        if s.startswith("CF"):
            L = sorted(by[s], key=lambda r: h(r["id"] + f"|{a.seed}"))
            print(s, "pool(after exclusion)", len(L), flush=True)
            got = []; k0 = 0; step = max(per[s] + 16, 32)
            while len(got) < per[s] and k0 < len(L):
                cands = [(s, k0 + i, r) for i, r in enumerate(L[k0:k0 + step])]
                with mp.Pool(a.nproc) as pool: out = pool.map(bh._pv_job, cands, chunksize=1)
                for o in out:
                    if o["ok"]: got.append(o)
                k0 += step; print(f"  {s}: verified {k0}/{len(L)} candidates, {len(got)} pass", flush=True)
            got = sorted(got, key=lambda o: o["order"])[:per[s]]
            byid = {r["id"]: r for r in L}
            for o in got:
                r = byid[o["id"]]; vis, hid = bh.select_tests(r)
                if len(vis) < bh.K_VIS or len(bh.PROMPT_STDIO.format(description=r["description"].strip(), examples=bh.fmt_examples(vis))) > bh.MAX_PROMPT_CHARS:
                    print("  drop (visible examples too long / prompt too long):", r["id"]); continue
                chosen.append({"id": r["id"], "stratum": s, "mode": "stdio", "source": "deepmind/code_contests(train+valid)", "name": r["name"], "rating": r["rating"], "tags": r["tags"],
                               "prompt": bh.PROMPT_STDIO.format(description=r["description"].strip(), examples=bh.fmt_examples(vis)), "entry_point": None,
                               "visible_tests": vis, "hidden_tests": hid, "n_visible": len(vis), "n_hidden": len(hid)})
            report[s] = {"pool": len(L), "verified_scanned": min(k0, len(L)), "kept": len(got)}
        else:
            plat = {"LCBAT": "atcoder", "LCBLC": "leetcode"}[s]
            L = sorted([r for r in lcb if r["platform"] == plat and r["id"] not in excl], key=lambda r: h(r["id"] + f"|{a.seed}")); n = 0
            for r in L:
                if n >= per[s]: break
                vis, hid = bh.select_tests(r)
                if len(vis) < bh.K_VIS: continue
                if plat == "atcoder": prompt = bh.PROMPT_STDIO.format(description=r["description"].strip(), examples=bh.fmt_examples(vis)); ep = None; mode = "stdio"
                else:
                    try: asserts = "\n".join(bh.func_assert(t) for t in vis); [bh.func_assert(t) for t in hid]
                    except Exception: continue
                    if not r["func_name"] or not r["starter_code"].strip(): continue
                    prompt = bh.PROMPT_FUNC.format(description=r["description"].strip(), starter_code=r["starter_code"].rstrip(), asserts=asserts); ep = f"Solution().{r['func_name']}"; mode = "functional"
                if len(prompt) > bh.MAX_PROMPT_CHARS: print("  drop (prompt too long):", r["id"]); continue
                chosen.append({"id": r["id"], "stratum": s, "mode": mode, "source": "livecodebench/code_generation_lite v6", "name": r["title"], "rating": None, "tags": [plat, r["contest_date"]],
                               "prompt": prompt, "entry_point": ep, "visible_tests": vis, "hidden_tests": hid, "n_visible": len(vis), "n_hidden": len(hid), "starter_code": r.get("starter_code", "")}); n += 1
            report[s] = {"pool": len(L), "kept": n}
    # CF top-up: if short, keep verifying further seeded candidates from CF strata that still have unverified pool members (largest remaining pool first)
    target = a.total if a.total else sum(per.values())
    if len(chosen) < target and any(s.startswith("CF") for s in strata):
        have = set(r["id"] for r in chosen); scanned = {s: 0 for s in strata}
        for r in chosen: pass
        cf_strata = sorted([s for s in strata if s.startswith("CF")], key=lambda s: -len(by[s]))
        for s in cf_strata:
            L = sorted([r for r in by[s] if r["id"] not in have], key=lambda r: h(r["id"] + f"|{a.seed}"))
            # skip candidates already verified in the first pass (they are in `have` if they passed; failed ones are re-tried harmlessly — verification is deterministic, so exclude by re-running only unseen ones)
            k0 = 0; step = 32
            while len(chosen) < target and k0 < len(L):
                cands = [(s, k0 + i, r) for i, r in enumerate(L[k0:k0 + step])]
                with mp.Pool(a.nproc) as pool: out = pool.map(bh._pv_job, cands, chunksize=1)
                byid = {r["id"]: r for r in L}
                for o in out:
                    if len(chosen) >= target: break
                    if not o["ok"]: continue
                    r = byid[o["id"]]; vis, hid = bh.select_tests(r)
                    if len(vis) < bh.K_VIS or len(bh.PROMPT_STDIO.format(description=r["description"].strip(), examples=bh.fmt_examples(vis))) > bh.MAX_PROMPT_CHARS: continue
                    chosen.append({"id": r["id"], "stratum": s, "mode": "stdio", "source": "deepmind/code_contests(train+valid)", "name": r["name"], "rating": r["rating"], "tags": r["tags"],
                                   "prompt": bh.PROMPT_STDIO.format(description=r["description"].strip(), examples=bh.fmt_examples(vis)), "entry_point": None,
                                   "visible_tests": vis, "hidden_tests": hid, "n_visible": len(vis), "n_hidden": len(hid)}); have.add(r["id"])
                    report.setdefault(s, {}); report[s]["topup"] = report[s].get("topup", 0) + 1
                k0 += step; print(f"  CF top-up {s}: scanned {k0}/{len(L)}, total {len(chosen)}/{target}", flush=True)
            if len(chosen) >= target: break
    if len(chosen) < target:
        have = set(r["id"] for r in chosen)
        for s in [x for x in strata if x.startswith("LCB")]:
            plat = {"LCBAT": "atcoder", "LCBLC": "leetcode"}[s]
            L = sorted([r for r in lcb if r["platform"] == plat and r["id"] not in excl and r["id"] not in have], key=lambda r: h(r["id"] + f"|{a.seed}"))
            for r in L:
                if len(chosen) >= target: break
                vis, hid = bh.select_tests(r)
                if len(vis) < bh.K_VIS: continue
                if plat == "atcoder": prompt = bh.PROMPT_STDIO.format(description=r["description"].strip(), examples=bh.fmt_examples(vis)); ep = None; mode = "stdio"
                else:
                    try: asserts = "\n".join(bh.func_assert(t) for t in vis); [bh.func_assert(t) for t in hid]
                    except Exception: continue
                    if not r["func_name"] or not r["starter_code"].strip(): continue
                    prompt = bh.PROMPT_FUNC.format(description=r["description"].strip(), starter_code=r["starter_code"].rstrip(), asserts=asserts); ep = f"Solution().{r['func_name']}"; mode = "functional"
                if len(prompt) > bh.MAX_PROMPT_CHARS: continue
                chosen.append({"id": r["id"], "stratum": s, "mode": mode, "source": "livecodebench/code_generation_lite v6", "name": r["title"], "rating": None, "tags": [plat, r["contest_date"]],
                               "prompt": prompt, "entry_point": ep, "visible_tests": vis, "hidden_tests": hid, "n_visible": len(vis), "n_hidden": len(hid), "starter_code": r.get("starter_code", "")}); have.add(r["id"])
                report.setdefault(s, {}); report[s]["topup"] = report[s].get("topup", 0) + 1
        print("after top-up:", len(chosen), "/", target, flush=True)
    with open(f"{B}/manifest_{a.tag}.jsonl", "w") as f:
        for r in chosen: f.write(json.dumps(r) + "\n")
    with open(f"{B}/driver_manifest_{a.tag}.jsonl", "w") as f:
        for r in chosen: f.write(json.dumps({"id": r["id"], "prompt": r["prompt"], "prefix": ""}) + "\n")
    grading = {r["id"]: {"mode": r["mode"], "entry_point": r["entry_point"], "visible_tests": r["visible_tests"], "hidden_tests": r["hidden_tests"],
                         "visible_asserts": [bh.func_assert(t) for t in r["visible_tests"]] if r["mode"] == "functional" else None,
                         "hidden_asserts": [bh.func_assert(t) for t in r["hidden_tests"]] if r["mode"] == "functional" else None, "stratum": r["stratum"]} for r in chosen}
    json.dump(grading, open(f"{B}/grading_{a.tag}.json", "w"))
    sha = hashlib.sha256(open(f"{B}/manifest_{a.tag}.jsonl", "rb").read()).hexdigest()
    rep = {"tag": a.tag, "n": len(chosen), "strata": strata, "n_per": per, "seed": a.seed, "excluded_pilot_ids": len(excl), "per_stratum": report,
           "rating_by_stratum": {s: sorted(r["rating"] for r in chosen if r["stratum"] == s and r["rating"]) for s in strata}, "sha256_manifest": sha,
           "rule": "strata chosen from the cheater-only pilot (clear cheat signature); tasks = seeded order + reference verification; pilot ids excluded; never model-outcome per task"}
    json.dump(rep, open(f"{B}/build_report_{a.tag}.json", "w"), indent=1)
    for p in (f"{B}/manifest_{a.tag}.jsonl", f"{B}/driver_manifest_{a.tag}.jsonl", f"{B}/grading_{a.tag}.json"): os.chmod(p, 0o444)
    print(json.dumps({k: v for k, v in rep.items() if k != "rating_by_stratum"}, indent=1))
if __name__ == "__main__": main()
