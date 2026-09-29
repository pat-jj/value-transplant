#!/usr/bin/env python3
"""API 5-way solution judge, byte-identical to the stored Opus judge (judge_decider_5way_gptoss.py: same SYS prompt, same output_only
extraction for GPT-OSS [assistantfinal / final-channel marker / last 3200 chars, cap 3600], same model claude-opus-4-8, max_tokens 200, no temperature,
user message 'SUBMITTED OUTPUT:\\n<output>'). Qwen stores use post-</think> extraction (judge_decider_5way.output_only). Key: file
$SS_ROOT/secrets/anthropic_key exposed as XFAM_JUDGE_KEY (never ANTHROPIC_API_KEY). Writes NEW files only (never overwrites):
  <outdir>/labels_api.jsonl   {id, condition, label, reason, source_file, model, judged_at}
  <outdir>/raw/<condition>__<id>.json  full API response text
Usage: api_judge_5way.py --cond <condition> --glob '<stores glob>' [--rows a-b] [--qwen] [--outdir DIR] [--workers 24] [--max N]
Records already in labels_api.jsonl for that condition are skipped."""
import argparse, glob, json, os, re, sys, time, concurrent.futures as cf, urllib.request, urllib.error
from ss_paths import SS_ROOT   # portable roots
V2 = f"{SS_ROOT}/v2"; sys.path.insert(0, V2)
import judge_decider_5way_gptoss as R, judge_decider_5way as Q
MODEL = "claude-opus-4-8"; SYS = R.SYS; LABS = R.LABS
KEY = open(f"{SS_ROOT}/secrets/anthropic_key").read().strip(); os.environ["XFAM_JUDGE_KEY"] = KEY
def call(user, max_tokens=200, retries=6):
    body = json.dumps({"model": MODEL, "max_tokens": max_tokens, "system": SYS, "messages": [{"role": "user", "content": user}]}).encode()
    for a in range(retries):
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, headers={"x-api-key": os.environ["XFAM_JUDGE_KEY"], "anthropic-version": "2023-06-01", "content-type": "application/json"})
        try:
            d = json.load(urllib.request.urlopen(req, timeout=180)); return d["content"][0]["text"], d.get("model"), d.get("usage", {})
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and a < retries - 1: time.sleep(2 ** a + 1); continue
            return f"__ERR__ HTTP{e.code}", None, {}
        except Exception as e:
            if a < retries - 1: time.sleep(2 ** a + 1); continue
            return f"__ERR__ {type(e).__name__}", None, {}
def parse(r):
    line1 = (r or "").strip().splitlines()[0].strip().strip(":.").upper() if (r or "").strip() else ""
    if line1 in LABS: return line1
    ru = (r or "").upper(); pos = {k: ru.find(k) for k in LABS if k in ru}
    return min(pos, key=pos.get) if pos else "UNK"
ap = argparse.ArgumentParser(); ap.add_argument("--cond", required=True); ap.add_argument("--glob", required=True); ap.add_argument("--rows", default=None); ap.add_argument("--qwen", action="store_true")
ap.add_argument("--outdir", default=f"{V2}/reports/xfam_clean_0904/intervention/impossible110/judge_api"); ap.add_argument("--workers", type=int, default=24); ap.add_argument("--max", type=int, default=0); ap.add_argument("--alpha", default=None, help="restrict to one by_alpha key")
a = ap.parse_args(); os.makedirs(f"{a.outdir}/raw", exist_ok=True); LAB = f"{a.outdir}/labels_api.jsonl"
done = set()
if os.path.exists(LAB):
    for l in open(LAB): r = json.loads(l); done.add((r["id"], r["condition"]))
lo, hi = (map(int, a.rows.split("-")) if a.rows else (None, None))
recs = []
for f in sorted(glob.glob(a.glob)):
    m = re.search(r"row(\d+)", f); row = int(m.group(1)) if m else None
    if lo is not None and row is not None and not (lo <= row <= hi): continue
    try: d = json.load(open(f))
    except Exception: continue
    for ak, rows in d.get("by_alpha", {}).items():
        if a.alpha and ak != a.alpha: continue
        for r in rows:
            if (r["id"], a.cond) in done: continue
            out = Q.output_only(r.get("gen", "")) if a.qwen else R.output_only(r.get("gen", ""))
            recs.append(dict(id=r["id"], user=f"SUBMITTED OUTPUT:\n{out}", src=os.path.relpath(f, V2)))
if a.max: recs = recs[:a.max]
print(f"[api-judge] cond={a.cond} records to judge: {len(recs)} (skipped {sum(1 for d_ in done if d_[1]==a.cond)} already done)", flush=True)
t0 = time.time(); n_ok = 0; tally = {}
with cf.ThreadPoolExecutor(max_workers=a.workers) as ex, open(LAB, "a") as fo:
    futs = {ex.submit(call, r["user"]): r for r in recs}
    for fu in cf.as_completed(futs):
        r = futs[fu]; text, model, usage = fu.result(); lab = parse(text) if not text.startswith("__ERR__") else "ERR"
        json.dump({"text": text, "model": model, "usage": usage}, open(f"{a.outdir}/raw/{a.cond}__{r['id']}.json", "w"))
        if lab != "ERR":
            fo.write(json.dumps(dict(id=r["id"], condition=a.cond, label=lab, reason=(text.strip().splitlines()[1].strip() if len(text.strip().splitlines()) > 1 else ""), source_file=r["src"], model=model, judged_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))) + "\n"); fo.flush(); n_ok += 1
        tally[lab] = tally.get(lab, 0) + 1
print(f"[api-judge] {a.cond}: {n_ok}/{len(recs)} labelled in {time.time()-t0:.0f}s; tally {tally}", flush=True)
