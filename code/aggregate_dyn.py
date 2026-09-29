#!/usr/bin/env python3
"""Aggregate dynamic arms (judged) + cached constant/unedited baselines -> genuine/fake/broken on
reported_set_110, overall + by contradiction type + intervention-budget stats. Behavior-free arm defs."""
import json, glob, os, re, numpy as np
from ss_paths import SS_ROOT   # portable roots
V2=f"{SS_ROOT}/v2"
OUT=f"{V2}/reports/online_weak_donor_dynamic"; EX=f"{V2}/reports/subdim_0726/crossfamily_impossible_expansion_0829"
LAB=('BROKEN','GIVEUP','HARDCODE','INTENT','SOLVE')
rep=set(l.strip() for l in open(f"{EX}/reported_set_110.txt") if l.strip())
man={i:json.loads(l) for i,l in enumerate(open(f"{EX}/manifest_expansion_ordered.jsonl"))}
id2type={m['id']: ('conflicting' if m['id'].startswith('cnf_') else 'oneoff') for m in man.values()}
def lab5(p):
    try: c=list(json.load(open(p))['files'].values())[0]['+1.000']
    except Exception: return None
    hits=[L for L in LAB if c.get(L,0)>0]; return hits[0] if hits else None
def collect(judged_glob, tagre):
    rows={}
    for p in glob.glob(judged_glob):
        m=re.search(tagre, os.path.basename(p))
        if not m: continue
        seed=int(m.group(1)); row=int(m.group(2)); md=man.get(row)
        if not md or md['id'] not in rep: continue
        lab=lab5(p)
        if lab is None: continue
        rows[(md['id'],seed)]=lab
    return rows
def rate(rows, scope=None):
    v=[(rid,lab) for (rid,s),lab in rows.items() if (scope is None or id2type[rid]==scope)]
    n=len(v)
    if not n: return None
    g=sum(l=='SOLVE' for _,l in v)/n; f=sum(l in('HARDCODE','INTENT') for _,l in v)/n
    bg=sum(l in('BROKEN','GIVEUP') for _,l in v)/n
    return dict(n=n, genuine=round(g,3), fake=round(f,3), broken_giveup=round(bg,3))
arms={}
# dynamic arms (my judged dir), seed0
for tag in ('DYNREAL','DYNBUD','DYNSHUF','CONSTLOW','DYNSEL15','DYNSHUF15','CONSTMATCH'):
    r=collect(f"{OUT}/data/judged/judged_s0_m{tag}_row*.json", r"judged_s(\d+)_m"+tag+r"_row(\d+)\.json")
    arms[tag]={'overall':rate(r),'conflicting':rate(r,'conflicting'),'oneoff':rate(r,'oneoff'),'n_tasks':len(r)}
# cached constant/unedited, seed0 (apples-to-apples with dynamic seed0)
for tag,name in (('BASE','UNEDITED'),('REAL','CONSTANT'),('SHUF','SHUF_DIR'),('RAND','RANDOM')):
    r=collect(f"{EX}/judged_s0_m{tag}_row*.json", r"judged_s(\d+)_m"+tag+r"_row(\d+)\.json")
    arms[name]={'overall':rate(r),'conflicting':rate(r,'conflicting'),'oneoff':rate(r,'oneoff'),'n_tasks':len(r)}
# budget stats from dynamic raw traces
def budget(tag):
    ea=[]; cum=[]; fr=[]; nt=[]
    for f in glob.glob(f"{OUT}/data/raw/s0_m{tag}_row*.json"):
        d=json.load(open(f))
        for ak,recs in d.get("by_alpha",{}).items():
            for r in recs:
                tr=r.get("trace") or {}; gf=tr.get("gate_fired") or []; n=r.get("n_tokens",0) or 0
                edits=[abs(e[1]) for e in gf]
                ea.append(r.get("edit_absmean",0.0)); cum.append(sum(edits))
                fr.append(len(edits)/max(n,1)); nt.append(n)
    if not ea: return None
    return dict(n=len(ea), mean_edit_per_tok=round(float(np.mean(ea)),1),
                cum_edit_per_rollout=round(float(np.mean(cum)),0),
                frac_tokens_edited=round(float(np.mean(fr)),3), mean_ntok=round(float(np.mean(nt)),0))
budgets={t:budget(t) for t in ('DYNREAL','DYNBUD','DYNSHUF','CONSTLOW','DYNSEL15','DYNSHUF15','CONSTMATCH')}
budgets['CONSTANT_ref']={'mean_edit_per_tok':448.8,'note':'edits every generated token'}
out=dict(arms=arms, budgets=budgets,
         cached_3seed=json.load(open(f"{OUT}/data/cached_baselines_110.json")))
json.dump(out, open(f"{OUT}/data/results_dyn_0902.json","w"), indent=1)
print(json.dumps(out,indent=1))
