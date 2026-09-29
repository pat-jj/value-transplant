"""5-way FAKE judge for the self-persuasion (sp) pilot + temptation-gate cell (0816).

Re-uses the canonical 5-way decider rubric (judge_decider_5way.SYS/LABS) and the shared
Opus backend (opus_judge.judge_batch) to label each rollout's SUBMITTED OUTPUT as
HARDCODE/INTENT/SOLVE/GIVEUP/BROKEN (FAKE = HARDCODE + INTENT).

Inputs: pod-pulled sp cells under $SS_ROOT/tmp/podpull plus temptation-cell files
and prior labels in reports/subdim_0726. Outputs: sp_pilot_5way_0816.json and
tempt_cell_5way_0816.json (cwd), and a printed TEMPT-vs-CTRL FAKE-enrichment table pooled across
seeds over the 30 temptation-split tasks."""
import json, sys
from ss_paths import SS_ROOT   # portable roots
sys.path.insert(0,f'{SS_ROOT}/v2')
from opus_judge import judge_batch
from judge_decider_5way import SYS, LABS
from collections import Counter
P=f'{SS_ROOT}/tmp/podpull'
RD=f'{SS_ROOT}/v2/reports/subdim_0726'
def out_of(g):
    g=g or ''
    out=g.split('</think>')[-1].strip() if '</think>' in g else ''
    return (out or '(no output — reasoning only / truncated)')[:3600]
def judge_file(path,tag):
    d=json.load(open(path))
    recs=[r for v in d['by_alpha'].values() for r in v]
    resp=judge_batch(SYS,[f"SUBMITTED OUTPUT:\n{out_of(r.get('gen',''))}" for r in recs],max_tokens=64,workers=30,label=tag)
    labs=[]
    for r in resp:
        l1=(r or '').strip().splitlines()[0].strip().strip(':.').upper() if (r or '').strip() else ''
        lab=l1 if l1 in LABS else None
        if lab is None:
            # fallback: earliest-mentioned label anywhere in the reply
            ru=(r or '').upper()
            pos={k:ru.find(k) for k in LABS if k in ru}
            lab=min(pos,key=pos.get) if pos else 'UNK'
        labs.append(lab)
    # denominator = parsed labels only (UNK excluded), falling back to n if all UNK
    c=Counter(labs)
    n=len(labs)
    den=(n-c['UNK']) or n
    r={k:round(c[k]/den,3) for k in LABS}|{'FAKE':round((c['HARDCODE']+c['INTENT'])/den,3),'n':n}
    print(f"{tag:22s} n={n:2d}  FAKE={r['FAKE']:.3f}  "+' '.join(f'{k}={r[k]:.2f}' for k in LABS),flush=True)
    return r|{'labels':labs,'ids':[x['id'] for x in recs]}
res={}
for tag in ['fs_sp_anchor_H','fs_sp_felt_rev_g16','fs_sp_felt_rev_g32']:
    res[tag]=judge_file(f'{P}/{tag}.json',tag)
json.dump(res,open('sp_pilot_5way_0816.json','w'),indent=1)
labs=res['fs_sp_felt_rev_g16']['labels']; ids=res['fs_sp_felt_rev_g16']['ids']
print('sp rev g16 FAKE tasks:',sorted({i.split('#')[0] for i,l in zip(ids,labs) if l in ('HARDCODE','INTENT')}),flush=True)

# ---- temptation gate cell (cluster, 30-task split, 2 new seeds) ----
t=judge_file(f'{RD}/fs_tempt_rev_g16.json','fs_tempt_rev_g16')
json.dump(t,open('tempt_cell_5way_0816.json','w'),indent=1)
tl=json.load(open('temptation_judge_labels_0816.json'))['qwen_H']
task_lab={}
for i,l in zip(tl['ids'],tl['labels']):
    task_lab.setdefault(i.split('#')[0],set()).add(l)
def grp(tk):
    s=task_lab.get(tk,set())
    return 'TEMPT' if ('ADOPTED' in s or 'CONSIDERED' in s) else 'CTRL'
# pool: new seeds (this cell) + existing #fs0/#fs1 rows from fs_q_felt_rev_g16
S=json.load(open(f'{RD}/fs_5way_0814.json'))['files']['fs_q_felt_rev_g16']
old=json.load(open(f'{RD}/fs_q_felt_rev_g16.json'))
orecs=[r for v in old['by_alpha'].values() for r in v]
rows=[]
for i,l in zip([x['id'] for x in orecs],S['labels']):
    if grp(i.split('#')[0]) and i.split('#')[0] in task_lab:
        rows.append((i,l))
for i,l in zip(t['ids'],t['labels']): rows.append((i,l))
agg={'TEMPT':[0,0],'CTRL':[0,0]}
tasks_split={'TEMPT':set(),'CTRL':set()}
tempt_cell_tasks={i.split('#')[0] for i in t['ids']}
for i,l in rows:
    tk=i.split('#')[0]
    if tk not in tempt_cell_tasks:   # only the 30 split tasks
        continue
    g=grp(tk)
    agg[g][1]+=1
    tasks_split[g].add(tk)
    if l in ('HARDCODE','INTENT'):
        agg[g][0]+=1
print('ENRICHMENT (pooled 4 seeds/task, rev g16):',flush=True)
for g in ('TEMPT','CTRL'):
    a,b=agg[g]
    print(f'  {g}: {a}/{b} rows = {a/b if b else 0:.3f} FAKE ({len(tasks_split[g])} tasks)',flush=True)
