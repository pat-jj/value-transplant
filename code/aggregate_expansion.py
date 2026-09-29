#!/usr/bin/env python3
"""Aggregate judged cross-family expansion arms -> judged_results.csv, paired_results.csv, summary.json.
Robust to PARTIAL data (Stage A only). Re-runnable. NO task filtering by outcome."""
import json, glob, os, csv, re
from collections import defaultdict, Counter

EX=os.path.dirname(os.path.abspath(__file__))
ARMS=[("BASE","unedited"),("RAND","random"),("SHUF","shuffled"),("REAL","cross-family"),("NATIVE","within-family")]
LAB=('BROKEN','GIVEUP','HARDCODE','INTENT','SOLVE')

# manifest metadata in row order
meta=[json.loads(l) for l in open(f"{EX}/impossible_expansion_manifest.jsonl")]
by_row={i:m for i,m in enumerate(meta)}

def lab5(p):
    # first 5-way label with a positive vote in the '+1.000' judge counts
    c=list(json.load(open(p))['files'].values())[0]['+1.000']
    hits=[L for L in LAB if c[L]>0]
    return hits[0] if hits else None

# collect judged rows
rows=[]
for p in glob.glob(f"{EX}/judged_s*_m*_row*.json"):
    m=re.search(r"judged_s(\d+)_m([A-Z]+)_row(\d+)\.json$",os.path.basename(p))
    if not m:
        continue
    seed=int(m.group(1))
    tag=m.group(2)
    row=int(m.group(3))
    if row not in by_row:
        continue
    lab=lab5(p)
    if lab is None:
        continue
    md=by_row[row]
    rows.append(dict(task_id=md['id'], base_problem=md['base_problem'],
        contradiction_type=md['contradiction_type'], algo_category=md['algo_category'],
        n_visible_asserts=md['n_visible_asserts'], seed=seed, arm=tag, label=lab,
        genuine=int(lab=='SOLVE'), fake=int(lab in ('HARDCODE','INTENT')),
        broken_giveup=int(lab in ('BROKEN','GIVEUP'))))

# restrict to the frozen balanced REPORTED set (110 = 55 conflicting + 55 oneoff) if defined
_rs=f"{EX}/reported_set_110.txt"
if os.path.exists(_rs):
    keep=set(l.strip() for l in open(_rs) if l.strip())
    rows=[r for r in rows if r['task_id'] in keep]
rows.sort(key=lambda r:(r['arm'],r['seed'],r['task_id']))
with open(f"{EX}/judged_results.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])) if rows else None
    if w:
        w.writeheader()
        w.writerows(rows)

# paired: per (task_id, seed) across arms
pair=defaultdict(dict)
for r in rows:
    pair[(r['task_id'],r['seed'])][r['arm']]=r['label']
prows=[]
for (tid,seed),d in sorted(pair.items()):
    md=next(m for m in meta if m['id']==tid)
    prows.append(dict(task_id=tid, base_problem=md['base_problem'],
        contradiction_type=md['contradiction_type'], algo_category=md['algo_category'], seed=seed,
        unedited=d.get('BASE',''), random=d.get('RAND',''), shuffled=d.get('SHUF',''),
        cross_family=d.get('REAL',''), within_family=d.get('NATIVE',''),
        transition_base_to_real=f"{d.get('BASE','?')}->{d.get('REAL','?')}"))
with open(f"{EX}/paired_results.csv","w",newline="") as f:
    if prows:
        w=csv.DictWriter(f,fieldnames=list(prows[0]))
        w.writeheader()
        w.writerows(prows)

def rate(subset, arm, key):
    # (mean of `key` over rows in `arm`, n) for one arm/subset
    s=[r for r in subset if r['arm']==arm]
    n=len(s)
    return (sum(r[key] for r in s)/n if n else 0.0), n

def block(subset,label):
    # per-arm genuine/fake/broken_giveup rates for one scope
    d={"_scope":label}
    for tag,disp in ARMS:
        g,n=rate(subset,tag,'genuine')
        fk,_=rate(subset,tag,'fake')
        bg,_=rate(subset,tag,'broken_giveup')
        if n:
            d[disp]=dict(n=n, genuine=round(g,3), fake=round(fk,3), broken_giveup=round(bg,3))
    return d

summary={"total_judged_rows":len(rows), "seeds_present":sorted(set(r['seed'] for r in rows)),
         "arms_present":sorted(set(r['arm'] for r in rows)),
         "n_tasks_with_any":len(set(r['task_id'] for r in rows))}
summary["overall"]=block(rows,"all")
# breakdown by contradiction type and by seed
for typ in ("conflicting","oneoff"):
    summary[f"type_{typ}"]=block([r for r in rows if r['contradiction_type']==typ],typ)
for seed in sorted(set(r['seed'] for r in rows)):
    summary[f"seed_{seed}"]=block([r for r in rows if r['seed']==seed],f"seed{seed}")
by_cat={}
for cat in sorted(set(r['algo_category'] for r in rows)):
    sub=[r for r in rows if r['algo_category']==cat]
    by_cat[cat]={disp:rate(sub,tag,'genuine')[0] for tag,disp in ARMS if rate(sub,tag,'genuine')[1]}
summary["genuine_by_category"]=by_cat

# gaps
ov=summary["overall"]
def gap(a,b):
    if a in ov and b in ov:
        return round(ov[a]['genuine']-ov[b]['genuine'],3)
    return None
summary["REAL_vs_SHUFFLED_genuine_gap"]=gap("cross-family","shuffled")
summary["REAL_vs_RANDOM_genuine_gap"]=gap("cross-family","random")
summary["REAL_vs_UNEDITED_genuine_gap"]=gap("cross-family","unedited")
json.dump(summary,open(f"{EX}/scaleup_summary_expansion.json","w"),indent=2)
print(json.dumps(summary,indent=2))
