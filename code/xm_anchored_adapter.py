#!/usr/bin/env python3
"""ANCHORED cross-model adapter (0802, normalized-felt translator; design + regression
directions logged verbatim in CROSSMODEL_G2Q_0802.md before this ran). All offline.
f per family: OLS stated = c + d*z (COORDINATE->STATED) on the pool prestates;
T: OLS normalized_Q = alpha + beta*normalized_G on the stage-1 shared-text aligned pairs
(computed for cheater/mixed/honest reader pairings; mixed = deploy-faithful primary);
composition z_Qtarget = f_Q^{-1}(T(f_G(z_G))) -> a_raw = beta*d_G/d_Q,
b_raw = (alpha + beta*c_G - c_Q)/d_Q; deploy schema via sign_g=sign_q=-1 (a_dep=a_raw,
b_dep=-b_raw). Comparison vs the production fitted adapter (raw frame) at p10/p50/p90
of the observed donor coordinate (gptoss_honest on Q texts).
T9b robustness: gpt-oss calibration from stated-at-moment pairs (t9b_stated_g.jsonl x
stage-1 gptoss_cheater reads at ev_char). Writes crossfam_adapter_XM_anchored.json.
DO NOT DEPLOY - report only."""
import json, sys

import numpy as np
from ss_paths import SS_ROOT   # portable roots

V2 = f'{SS_ROOT}/v2'
sys.path.insert(0, V2)
from xm_paired_extract import load_axis, pair_align  # noqa: E402

DSP = f'{V2}/activations/dspace'
BASE = f'{DSP}/crossfam_pairs_XM.npz'
_A = json.load(open(f'{DSP}/crossfam_adapter_XM.json'))['honest']; PROD_RAW = (_A['a'], _A['b'])   # production fitted adapter, RAW frame
OUT = f'{DSP}/crossfam_adapter_XM_anchored.json'


def ols(x, y):
    d, c = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    return float(d), float(c), r


# ---------------- (1) per-family calibrations: stated = c + d*z ----------------
u_q, sig_q = load_axis(f'{DSP}/preDIM_QB_L21.npz')
u_g, sig_g = load_axis(f'{DSP}/preDIM_G20BC_BAL_L14.npz')
cal = {}
for fam, ps_path, key, u in (('Q', f'{DSP}/r2_prestates_QB.npz', 'L21', u_q),
                             ('G', f'{DSP}/g20b_prestates_C.npz', 'L14', u_g)):
    ps = np.load(ps_path, allow_pickle=True)
    z = ps[key].astype(np.float32) @ u
    y = ps['y'].astype(float)
    d, c, r = ols(z, y)
    cal[fam] = dict(slope_d=round(d, 6), intercept_c=round(c, 4), r=round(r, 4),
                    n=len(y), anchor_stated0=round(-c / d, 3),
                    anchor_stated100=round((100 - c) / d, 3),
                    pool=ps_path, layer=key)
    print(f'[anch] f_{fam}: stated = {c:+.3f} + {d:+.5f}*z  r={r:+.3f} n={len(y)} | '
          f'anchors z(0)={cal[fam]["anchor_stated0"]} z(100)={cal[fam]["anchor_stated100"]}')

# T9b robustness calibration for G (stated at hack-commit moments vs coordinate there)
reads_gc = json.load(open(f'{BASE}.gptoss_cheater.reads.json'))
t9 = [json.loads(l) for l in open(f'{V2}/reports/subdim_0726/t9b_stated_g.jsonl')]
t9 = [r for r in t9 if str(r.get('parse_ok')) == 'True']
zz, ss, inside = [], [], 0
for r in t9:
    tid = r['tid'] if r['tid'].startswith('G|') else 'G|' + r['tid']
    if tid not in reads_gc or not reads_gc[tid]['p']:
        continue
    ends = np.array(reads_gc[tid]['ends'])
    ev = int(r['ev_char'])
    if ends[0] <= ev <= ends[-1]:
        inside += 1
    k = int(np.clip(np.searchsorted(ends, ev), 0, len(ends) - 1))
    zz.append(reads_gc[tid]['p'][k])
    ss.append(float(r['stated']))
d9, c9, r9 = ols(np.array(zz), np.array(ss))
cal['G_t9b'] = dict(slope_d=round(d9, 6), intercept_c=round(c9, 4), r=round(r9, 4),
                    n=len(zz), frac_ev_inside_read_range=round(inside / max(len(zz), 1), 3),
                    anchor_stated0=round(-c9 / d9, 3), anchor_stated100=round((100 - c9) / d9, 3))
print(f'[anch] f_G(T9b): stated = {c9:+.3f} + {d9:+.5f}*z  r={r9:+.3f} n={len(zz)} '
      f'(ev_char inside read range: {cal["G_t9b"]["frac_ev_inside_read_range"]})')

# ---------------- (2)+(3) normalized shared-text mapping + translator T ----------------
readers = {nm: json.load(open(f'{BASE}.{nm}.reads.json'))
           for nm in ('qwen_honest', 'qwen_cheater', 'gptoss_honest', 'gptoss_cheater')}
for nm in readers:
    readers[nm] = {t: {'ends': v['ends'], 'p': v['p']} for t, v in readers[nm].items() if v['p']}
PAIRINGS = {'cheater': ('gptoss_cheater', 'qwen_cheater'),
            'mixed': ('gptoss_honest', 'qwen_cheater'),
            'honest': ('gptoss_honest', 'qwen_honest')}
dG, cG = cal['G']['slope_d'], cal['G']['intercept_c']
dQ, cQ = cal['Q']['slope_d'], cal['Q']['intercept_c']

# f_Q inversion conditioning (guardrail 2): slope CI + graded-range refit
psq = np.load(f'{DSP}/r2_prestates_QB.npz', allow_pickle=True)
zq_pool = psq['L21'].astype(np.float32) @ u_q
yq_pool = psq['y'].astype(float)
rngc = np.random.default_rng(1)
dsb = [np.polyfit(zq_pool[i], yq_pool[i], 1)[0]
       for i in (rngc.integers(0, len(yq_pool), len(yq_pool)) for _ in range(2000))]
d_ci = [round(float(np.percentile(dsb, q)), 5) for q in (2.5, 97.5)]
mg = yq_pool < 100                                  # graded (non-ceiling) range
dQg, cQg, rQg = ols(zq_pool[mg], yq_pool[mg])
cond = dict(fQ_slope_ci95=d_ci,
            coord_per_stated_point_raw=round(1 / dQ, 4),
            coord_per_stated_point_sigq=round(1 / (dQ * sig_q), 4),
            fQ_graded_refit=dict(slope_d=round(dQg, 6), intercept_c=round(cQg, 4),
                                 r=round(rQg, 4), n=int(mg.sum()),
                                 applicable_range='stated < 100 (non-ceiling rows, '
                                                  f'{int(mg.sum())}/{len(yq_pool)})'),
            ceiling_frac_Q=round(float(np.mean(yq_pool == 100)), 3))
print(f'[anch] f_Q conditioning: slope {dQ:+.5f} CI {d_ci}; 1 stated point = '
      f'{cond["coord_per_stated_point_sigq"]} sig_Q of coordinate; graded-range refit '
      f'slope {dQg:+.5f} (r={rQg:+.3f}, n={int(mg.sum())}); ceiling frac '
      f'{cond["ceiling_frac_Q"]}')

rng = np.random.default_rng(0)
tblocks = {}
for pname, (gr, qr) in PAIRINGS.items():
    tids = [t for t in readers[gr] if t in readers[qr]]
    per_tid = {}
    for t in tids:
        zg, zq, _ = pair_align({t: readers[gr][t]}, {t: readers[qr][t]})
        if len(zg) >= 30:
            per_tid[t] = (np.array(zg), np.array(zq))
    vg = np.concatenate([cG + dG * v[0] for v in per_tid.values()])
    vq = np.concatenate([cQ + dQ * v[1] for v in per_tid.values()])
    beta_ls, alpha_ls, rT = ols(vg, vq)
    # PRIMARY (guardrail 1): MOMENT-MATCHED T — mean/sd match, slope positive by the
    # anchoring-sign assumption (each family's normalized felt is on its own stated scale).
    # NOTE the moment-matched and LS translators can disagree in sign — reported, not hidden.
    keys = list(per_tid)
    def mm(vg_, vq_):
        b = float(np.std(vq_) / (np.std(vg_) + 1e-12))
        return b, float(np.mean(vq_) - b * np.mean(vg_))
    beta_mm, alpha_mm = mm(vg, vq)
    bs, as_ = [], []
    for _ in range(2000):
        pick = rng.choice(len(keys), len(keys))
        xg = np.concatenate([cG + dG * per_tid[keys[i]][0] for i in pick])
        xq = np.concatenate([cQ + dQ * per_tid[keys[i]][1] for i in pick])
        b_, a_ = mm(xg, xq)
        bs.append(b_)
        as_.append(a_)
    bci = [round(float(np.percentile(bs, q)), 4) for q in (2.5, 97.5)]
    aci = [round(float(np.percentile(as_, q)), 4) for q in (2.5, 97.5)]

    def compose(beta, alpha):
        return beta * dG / dQ, (alpha + beta * cG - cQ) / dQ
    a_mm, b_mm = compose(beta_mm, alpha_mm)
    a_ls, b_ls = compose(beta_ls, alpha_ls)
    tblocks[pname] = dict(
        mm=dict(beta=round(beta_mm, 4), alpha=round(alpha_mm, 3), beta_ci95=bci,
                alpha_ci95=aci,
                identity=bool(bci[0] <= 1.0 <= bci[1] and aci[0] <= 0.0 <= aci[1]),
                a_raw=round(a_mm, 6), b_raw=round(b_mm, 4)),
        ls_attenuated=dict(beta=round(beta_ls, 4), alpha=round(alpha_ls, 3),
                           r=round(rT, 4), a_raw=round(a_ls, 6), b_raw=round(b_ls, 4),
                           note='LS slope = r * sd-ratio; collapses to the production '
                                'fitted adapter by construction (calibrations cancel)'),
        n=int(len(vg)), n_rollouts=len(keys),
        vG_outside_0_100=round(float(np.mean((vg < 0) | (vg > 100))), 3),
        vQ_outside_0_100=round(float(np.mean((vq < 0) | (vq > 100))), 3),
        _unrounded=dict(beta_mm=beta_mm, alpha_mm=alpha_mm, a_mm=a_mm, b_mm=b_mm))
    print(f'[anch] T[{pname:7s}] MOMENT-MATCHED: vQ = {alpha_mm:+.2f} + {beta_mm:+.4f}*vG '
          f'beta CI {bci} alpha CI {aci} identity~{tblocks[pname]["mm"]["identity"]} '
          f'| composed raw a={a_mm:+.6f} b={b_mm:+.3f}')
    print(f'[anch] T[{pname:7s}] LS(attenuated): vQ = {alpha_ls:+.2f} + {beta_ls:+.4f}*vG '
          f'r={rT:+.3f} | composed raw a={a_ls:+.6f} b={b_ls:+.3f}')

# ---------------- (4) comparison vs production at observed donor coordinates ----------------
zobs = np.concatenate([np.array(v['p']) for t, v in readers['gptoss_honest'].items()
                       if t.startswith('Q|')])
pcts = {p: float(np.percentile(zobs, p)) for p in (10, 50, 90)}
af, bf = PROD_RAW
comp = {}
for pname, tb in tblocks.items():
    for variant in ('mm', 'ls_attenuated'):
        row = {}
        for p, z in pcts.items():
            dlt = abs((tb[variant]['a_raw'] * z + tb[variant]['b_raw'])
                      - (af * z + bf)) / sig_q
            row[f'p{p}'] = round(float(dlt), 4)
        comp[f'{pname}_{variant}'] = row
        print(f'[anch] |anchored({variant}) - fitted| in sig_Q [{pname:7s}]: '
              + '  '.join(f'p{p}={row[f"p{p}"]}' for p in (10, 50, 90)))
# T9b-variant composition on the mixed pairing MOMENT-MATCHED translator
tbm = tblocks['mixed']['_unrounded']
a9 = tbm['beta_mm'] * d9 / dQ
b9 = (tbm['alpha_mm'] + tbm['beta_mm'] * c9 - cQ) / dQ
comp['mixed_mm_t9b_calibration'] = {f'p{p}': round(abs((a9 * z + b9) - (af * z + bf)) / sig_q, 4)
                                    for p, z in pcts.items()}
print(f'[anch] T9b-calibration variant (mixed MM T): raw a={a9:+.6f} b={b9:+.3f} | '
      + '  '.join(f'p{p}={comp["mixed_mm_t9b_calibration"][f"p{p}"]}' for p in (10, 50, 90)))

# ---------------- self-test: composition algebra (unrounded values) ----------------
for z in (-50.0, 0.0, 37.5, 200.0):
    fg = cG + dG * z
    tv = tbm['alpha_mm'] + tbm['beta_mm'] * fg
    direct = tbm['a_mm'] * z + tbm['b_mm']
    assert abs((tv - cQ) / dQ - direct) < 1e-9 * max(1, abs(direct)), 'composition algebra'
print('[anch] composition algebra self-test PASS')

for tb in tblocks.values():
    tb.pop('_unrounded', None)
primary = tblocks['mixed']['mm']
out = dict(design='normalized-felt translator (see CROSSMODEL doc section)',
           calibrations=cal, fQ_conditioning=cond, translator=tblocks,
           donor_coord_percentiles_raw=pcts, sig_q=sig_q,
           production_fitted_raw=dict(a=af, b=bf),
           comparison_sigq=comp,
           deploy=dict(a_eff=round(primary['a_raw'], 6),
                       b_eff=round(-primary['b_raw'], 4),
                       sign_g=-1, sign_q=-1,
                       source='anchored_normfelt_mixed_pair_moment_matched',
                       DO_NOT_DEPLOY='report-only; decision pending'))
json.dump(out, open(OUT, 'w'), indent=1)
print(f'[anch] saved {OUT}')
