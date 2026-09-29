#!/usr/bin/env python3
"""PHASE 0: can GPT-C activation h_GC predict Qwen honest-cheater value ΔQ? (3-model, no GPT-honest)
Uses ONLY host_cheater activation + Qwen scalars qH,qC from cached mlpt_states. NEVER touches
gH/gC/host_honest/u_G. Task-disjoint. Probes + nuisance controls. Freezes steering direction w."""
import json, glob, os
import numpy as np
from scipy.stats import pearsonr, spearmanr
from ss_paths import SS_ROOT   # portable roots

G=f'{SS_ROOT}/v2/reports/subdim_0726/gatecal_0824'
TD=f'{SS_ROOT}/v2/reports/subdim_0726/three_model_host_native_value_0829'
# task-disjoint train/val/test assignment (frozen split)
sp=json.load(open(f'{G}/value_translation_split_0825.json'))
assign={int(k):v for k,v in sp['assign'].items()}
# frozen Qwen value axis (direction + mean)
qax=np.load(f'{SS_ROOT}/v2/activations/dspace/preDIM_QB_L21.npz',allow_pickle=True)
uQ=qax['direction'].astype(np.float64)
uQ/=np.linalg.norm(uQ)
mQ=qax['mean'].astype(np.float64)
RNG=np.random.default_rng(0)

# ---- load per-rollout states: GPT-C activation + Qwen honest/cheater felt scalars ----
D={'train':[],'val':[],'test':[]}
for p in sorted(glob.glob(f'{G}/mlpt_states/states_dcdhhc_*.npz')):
    k=int(os.path.basename(p).rsplit('_',1)[1][:3])
    if k not in assign or assign[k] not in D:
        continue
    d=np.load(p,allow_pickle=True)
    hG=d['host_cheater'].astype(np.float32)              # GPT-C L14 activation (ONLY host signal used)
    qH=((d['donor_honest'].astype(np.float64)-mQ)@uQ)    # Qwen honest felt (frozen axis)
    qC=((d['donor_cheater'].astype(np.float64)-mQ)@uQ)   # Qwen cheater felt
    D[assign[k]].append(dict(rid=str(d['rid']), hG=hG, dQ=(qH-qC).astype(np.float64),
                             pos=np.arange(len(qH))/len(qH)))
def cat(s,k):
    return np.concatenate([r[k] for r in D[s]])
hGt,hGv=cat('train','hG'),cat('val','hG')
yt,yv=cat('train','dQ'),cat('val','dQ')
post,posv=cat('train','pos'),cat('val','pos')
print(f'train {len(yt)} states / {len(D["train"])} rolls; val {len(yv)}/{len(D["val"])}; hG dim {hGt.shape[1]}',flush=True)
# standardize host activations on the train mean/std
mu=hGt.mean(0)
sd=hGt.std(0)+1e-6
Xt=(hGt-mu)/sd
Xv=(hGv-mu)/sd
def mets(p,y):
    ad=np.abs(y)
    m25=ad>=np.quantile(ad,0.75)
    return dict(pearson=round(float(pearsonr(p,y)[0]),3),spearman=round(float(spearmanr(p,y).statistic),3),
                sign=round(float((np.sign(p)==np.sign(y)).mean()),3),
                t25_sp=round(float(spearmanr(p[m25],y[m25]).statistic),3))
RES={'GPT_HONEST_USED':'NO'}
# --- ridge F(h_GC)->dQ ---
def ridge(X,y,lam):
    mx=X.mean(0)
    my=y.mean()
    Xc=X-mx
    w=np.linalg.solve(Xc.T@Xc+lam*np.eye(X.shape[1]),Xc.T@(y-my))
    return w,mx,my
best=None
for lam in (1e2,1e3,1e4,1e5):
    w,mx,my=ridge(Xt,yt,lam)
    pv=(Xv-mx)@w+my
    m=mets(pv,yv)
    if best is None or m['t25_sp']>best[0]:
        best=(m['t25_sp'],lam,m,(w,mx,my))
RES['ridge_real']=dict(lam=best[1],**best[2])
w,mx,my=best[3]
# map w back to RAW activation coords: F(h)= ((h-mu)/sd - mx)@w + my ; grad_h F = w/sd
w_raw=(w/sd).astype(np.float64)
RES['w_raw_norm']=float(np.linalg.norm(w_raw))
# --- mean-diff direction (raw), stratified by position tertile to kill position confound ---
hi=yt>=np.quantile(yt,0.75)
lo=yt<=np.quantile(yt,0.25)
w_mean=(hGt[hi].mean(0)-hGt[lo].mean(0)).astype(np.float64)
pv_mean=(hGv@w_mean)
RES['meandiff_real']=mets(pv_mean,yv)
# --- controls: shuffled dQ (within-rollout), cross-task, position-only ---
def within_shuf():
    ys=[]
    for r in D['train']:
        ys.append(RNG.permutation(r['dQ']))
    return np.concatenate(ys)
w2,mx2,my2=ridge(Xt,within_shuf(),best[1])
RES['ridge_shuffle_within']=mets((Xv-mx2)@w2+my2,yv)
w_shuf_raw=(w2/sd).astype(np.float64)
def xtask_shuf():
    ts=[r['rid'].split('#')[0] for r in D['train']]
    ys=[]
    for i,r in enumerate(D['train']):
        js=[j for j,t in enumerate(ts) if t!=ts[i]]
        src=D['train'][RNG.choice(js)]['dQ']
        ys.append(np.resize(src,len(r['dQ'])))
    return np.concatenate(ys)
w3,mx3,my3=ridge(Xt,xtask_shuf(),best[1])
RES['ridge_shuffle_xtask']=mets((Xv-mx3)@w3+my3,yv)
# position-only baseline
wp=np.polyfit(post,yt,3)
RES['position_only']=mets(np.polyval(wp,posv),yv)
# residualize out position: does hG still predict dQ beyond position?
yt_res=yt-np.polyval(np.polyfit(post,yt,3),post)
yv_res=yv-np.polyval(np.polyfit(post,yt,3),posv)
wr,mxr,myr=ridge(Xt,yt_res,best[1])
RES['ridge_real_posresid']=mets((Xv-mxr)@wr+myr,yv_res)
# train score gap for dose calibration (RAW): F_raw(h)=h@w_raw + const
ftr=hGt@w_raw
m_gap=float(np.median(ftr[hi])-np.median(ftr[lo]))
RES['train_score_gap_m']=round(m_gap,4)
RES['host_rms']=round(float(np.sqrt((hGt**2).sum(1).mean())),2)
np.savez(f'{TD}/qsv_direction_L14.npz', w_ridge_raw=w_raw, w_shuffle_raw=w_shuf_raw, w_meandiff_raw=w_mean, mu=mu, sd=sd,
         m_gap=m_gap, host_rms=RES['host_rms'], lam=best[1])
json.dump(RES, open(f'{TD}/phase0_results.json','w'), indent=1)
for k,v in RES.items():
    print(k, v, flush=True)
print('SAVED phase0_results.json + qsv_direction_L14.npz', flush=True)
