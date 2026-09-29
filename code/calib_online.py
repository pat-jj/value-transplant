#!/usr/bin/env python3
"""Pool online ΔQ from the eta=0 dev calib rollouts -> online-scale s + distribution + state-dependence.
Writes data/calib_online_0902.json. Behavior-free (dev tasks, no genuine-rate involvement)."""
import json, glob, numpy as np
from ss_paths import SS_ROOT   # portable roots
OUT=f"{SS_ROOT}/v2/reports/online_weak_donor_dynamic"
dqs=[]; per=[]
for f in sorted(glob.glob(f"{OUT}/data/calib/dev_row*.json")):
    d=json.load(open(f))
    for ak,recs in d.get("by_alpha",{}).items():
        for r in recs:
            dl=(r.get("trace") or {}).get("delta") or []
            if dl: dqs+=[float(x) for x in dl]; per.append((r["id"],len(dl),float(np.mean(dl)),float(np.std(dl))))
a=np.array(dqs)
neg=-a[a<0]
rep={"n_dev_rollouts":len(per),"n_states":int(a.size),
     "dq":{"mean":float(a.mean()),"std":float(a.std()),"min":float(a.min()),
           "p05":float(np.percentile(a,5)),"p25":float(np.percentile(a,25)),"median":float(np.median(a)),
           "p75":float(np.percentile(a,75)),"p95":float(np.percentile(a,95)),"max":float(a.max()),
           "frac_neg":float((a<0).mean())},
     "neg_dq":{"median":float(np.median(neg)) if neg.size else None,
               "p75":float(np.percentile(neg,75)) if neg.size else None,
               "p90":float(np.percentile(neg,90)) if neg.size else None},
     "per_rollout":[{"id":i,"n":n,"mean":round(m,3),"std":round(s,3)} for i,n,m,s in per]}
# choose s_online = p75 |neg ΔQ| (a≈1 at a typical strong-cheat online state)
s=rep["neg_dq"]["p75"]; rep["s_online_p75neg"]=s
# for MAIN clip(0,a_max): pick a_max so max edit ≲ ~1.3x constant (edit=a·448.8; host_rms 1833).
# report implied a_t stats under this s for relu and clip variants:
for amax in (1.0,1.5,2.0,3.0):
    ar=np.maximum(0.0,-a/s); ac=np.minimum(ar,amax)
    rep[f"clip_amax_{amax}"]={"mean_a":float(ac.mean()),"frac_a_pos":float((ac>0).mean()),
        "frac_a_at_cap":float((ar>=amax).mean()),"mean_edit":float(ac.mean()*448.8),
        "max_edit":float(amax*448.8)}
rep["relu_unclipped"]={"mean_a":float(np.maximum(0,-a/s).mean()),"max_a":float(np.maximum(0,-a/s).max())}
# state-dependence: within-rollout std of a_t (clip amax=1.5) should be > 0
sd=[]
for f in sorted(glob.glob(f"{OUT}/data/calib/dev_row*.json")):
    d=json.load(open(f))
    for ak,recs in d.get("by_alpha",{}).items():
        for r in recs:
            dl=(r.get("trace") or {}).get("delta") or []
            if len(dl)>5:
                av=np.minimum(np.maximum(0,-np.array(dl)/s),1.5); sd.append(float(av.std()))
rep["within_rollout_a_std_mean"]=float(np.mean(sd)) if sd else None
json.dump(rep,open(f"{OUT}/data/calib_online_0902.json","w"),indent=1)
print(json.dumps(rep,indent=1))
