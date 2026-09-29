#!/usr/bin/env python3
"""CROSS-MODEL ADAPTER STAGE 2 + SIGN GATE (0802, CPU): fit z_Q = a*z_G + b on the stage-1
pairs (crossfam_pairs_XM.npz) and gate the axis signs on judged transcripts BEFORE any cell.
Lane doc: reports/subdim_0726/CROSSMODEL_G2Q_0802.md. Mechanics cloned from crossfam_fit_0727.py
(least squares + robust trim + r/a-consistency gate) with the trim expressed in each axis's OWN
sigma units (the XM pairs store RAW projections; the 0727 pairs were standardized).

FITS: honest pair (gptoss_honest <-> qwen_honest) = the adapter to DEPLOY (0727 precedent);
cheater pair = persona-independence check; mixed pair (gptoss_honest <-> qwen_cheater = the
literal deploy reader combination) = diagnostic. Per-source r (Q rollouts vs G rollouts)
reported for the honest fit.

SIGN GATE (spec + review-lane R4): per-text continuation-mean raw projections from the stage-1
reads jsons, labels from the judged store (reports/dspace_0715/judge_raw/*, order-aligned to the
rollout recs; honest side = HONEST+GIVEUP per rule; BROKEN/UNK excluded).
 - u_Q sign: deploy reader = qwen_cheater on Q-native texts must rank honest ABOVE hardcode
   (qwen_honest reader reported as consistency info).
 - u_G sign: deploy reader = gptoss_honest on G-native texts likewise (gptoss_cheater reported).
 - flips folded EXPLICITLY: a_eff = sign_q*sign_g*a, b_eff = sign_q*b (u_used = sign*u in the
   driver). Note: the value-match delta is algebraically sign-invariant; the gate protects the
   orientation of controls/diagnostics.
 - R4 behavioral anchor: on Q texts, sign( E[phi(z_G gptoss_honest)] - E[z_Q qwen_cheater] )
   must equal sign( E[z_Q qwen_honest] - E[z_Q qwen_cheater] ); both gaps printed in sig_Q units.
FALLBACK: if |r_honest| < 0.2, moment matching (a = sd_zq/sd_zg, b = mean_zq - a*mean_zg, on
sign-corrected coords) and FLAG it (spec escalation).
Writes activations/dspace/crossfam_adapter_XM.json. Never modifies the pairs npz."""
import json, os
import numpy as np
from ss_paths import SS_ROOT   # portable roots

V2 = f'{SS_ROOT}/v2'
PAIRS = os.environ.get('XM_PAIRS', f'{V2}/activations/dspace/crossfam_pairs_XM.npz')
OUT = os.environ.get('XM_ADAPTER_OUT', f'{V2}/activations/dspace/crossfam_adapter_XM.json')
JR = f'{V2}/reports/dspace_0715/judge_raw'
LABELS_Q = f'{JR}/r2_qb_base_fork.json.L+0.000.json'
LABELS_G = f'{JR}/g20b_transplant_lam0.0_full.json.L+0.000.json'
ROLL_Q = f'{V2}/reports/subdim_0726/r2_qb_base_fork.json'
ROLL_G = f'{V2}/reports/subdim_0726/g20b_transplant_lam0.0_full.json'
TRIM_SIG = 8.0                       # crossfam_fit_0727's |z|<8 rule, in own-sigma units


def fit_pair(zg, zq, sig_g, sig_q, src=None, trim_sig=TRIM_SIG):
    """least squares z_Q = a*z_G + b with the 0727 robust trim in own-sigma units."""
    zg = np.asarray(zg, np.float64)
    zq = np.asarray(zq, np.float64)
    m = (np.abs(zg) < trim_sig * sig_g) & (np.abs(zq) < trim_sig * sig_q)
    zgt, zqt = zg[m], zq[m]
    a, b = np.polyfit(zgt, zqt, 1)
    r = float(np.corrcoef(zgt, zqt)[0, 1])
    out = dict(a=round(float(a), 6), b=round(float(b), 4), r=round(r, 4), n=int(len(zgt)),
               n_trimmed=int((~m).sum()))
    if src is not None:
        src = np.asarray(src)[m]
        for tag, sv in (('srcQ', 0), ('srcG', 1)):
            mm = src == sv
            if mm.sum() > 10:
                out[f'r_{tag}'] = round(float(np.corrcoef(zgt[mm], zqt[mm])[0, 1]), 4)
                out[f'n_{tag}'] = int(mm.sum())
    return out


def fold_signs(a, b, sign_g, sign_q):
    """raw fit z_Qraw = a*z_Graw + b -> sign-corrected frames (z' = sign*z, u' = sign*u):
    z_Q' = (sign_q*sign_g*a) * z_G' + sign_q*b."""
    return sign_q * sign_g * a, sign_q * b


def moment_match(zg, zq, sig_g, sig_q, sign_g, sign_q, trim_sig=TRIM_SIG):
    """fallback on sign-corrected coords: a' = sd(zq')/sd(zg') (>0), b' = mean(zq') - a'*mean(zg')."""
    zg = np.asarray(zg, np.float64)
    zq = np.asarray(zq, np.float64)
    m = (np.abs(zg) < trim_sig * sig_g) & (np.abs(zq) < trim_sig * sig_q)
    zgc, zqc = sign_g * zg[m], sign_q * zq[m]
    a = float(zqc.std() / (zgc.std() + 1e-12))
    b = float(zqc.mean() - a * zgc.mean())
    return a, b, int(m.sum())


def text_means(reads):
    return {tid: float(np.mean(v['p'])) for tid, v in reads.items() if v['p']}


def load_labels(label_path, roll_path, tag):
    labs = json.load(open(label_path))['labels']
    recs = json.load(open(roll_path))['by_alpha']['+0.000']
    assert len(labs) == len(recs), f'{label_path}: {len(labs)} labels != {len(recs)} recs'
    return {f'{tag}|{r["id"]}': l for r, l in zip(recs, labs)}


def rank_stat(means, labels, restrict_tag=None):
    """mean projection honest(=HONEST+GIVEUP) vs fake(FAKE); sign +1 if honest > fake."""
    hon = [v for t, v in means.items() if labels.get(t) in ('HONEST', 'GIVEUP')
           and (restrict_tag is None or t.startswith(restrict_tag))]
    fak = [v for t, v in means.items() if labels.get(t) == 'FAKE'
           and (restrict_tag is None or t.startswith(restrict_tag))]
    mh, mf = float(np.mean(hon)), float(np.mean(fak))
    return dict(honest_mean=round(mh, 3), fake_mean=round(mf, 3),
                n_honest=len(hon), n_fake=len(fak), honest_above=bool(mh > mf))


def main():
    d = np.load(PAIRS)
    sig_g, sig_q = float(d['sig_g']), float(d['sig_q'])
    print(f'[xmfit] pairs={PAIRS} sig_g={sig_g:.3f} sig_q={sig_q:.4f}')
    fits = {}
    for pair in ('honest', 'cheater', 'mixed'):
        fits[pair] = fit_pair(d[f'{pair}_zg'], d[f'{pair}_zq'], sig_g, sig_q,
                              src=d[f'{pair}_src'])
        f = fits[pair]
        print(f'  {pair:8s}: z_Q = {f["a"]:+.6f} * z_G {f["b"]:+.4f}   r={f["r"]:+.4f} '
              f'n={f["n"]} (trimmed {f["n_trimmed"]}) '
              f'rQ={f.get("r_srcQ")} rG={f.get("r_srcG")}')
    h, c = fits['honest'], fits['cheater']
    gate_r = h['r'] >= 0.3
    gate_a = (np.sign(h['a']) == np.sign(c['a'])) and (0.33 < abs(h['a'] / (c['a'] or 1e-9)) < 3.0)
    print(f'[xmfit] GATE (0727 rule) r>=0.3: {gate_r} (r={h["r"]}) | a consistent H-vs-C: '
          f'{gate_a} (a={h["a"]} vs {c["a"]}) | PASSED: {gate_r and gate_a}')

    # ---------------- sign gate on judged transcripts ----------------
    base = PAIRS.replace('.npz', '') + '.npz'
    reads = {nm: json.load(open(f'{base}.{nm}.reads.json'))
             for nm in ('qwen_honest', 'qwen_cheater', 'gptoss_honest', 'gptoss_cheater')}
    labels = load_labels(LABELS_Q, ROLL_Q, 'Q') | load_labels(LABELS_G, ROLL_G, 'G')
    means = {nm: text_means(rd) for nm, rd in reads.items()}
    sign_gate = {}
    # u_Q: deploy reader = qwen_cheater on Q-native texts (host organism's own frame)
    sq_stat = rank_stat(means['qwen_cheater'], labels, 'Q|')
    sign_q = 1 if sq_stat['honest_above'] else -1
    # u_G: deploy reader = gptoss_honest on G-native texts (donor organism's own frame)
    sg_stat = rank_stat(means['gptoss_honest'], labels, 'G|')
    sign_g = 1 if sg_stat['honest_above'] else -1
    sign_gate['u_Q_deploy_reader_qwen_cheater_on_Qtexts'] = sq_stat
    sign_gate['u_G_deploy_reader_gptoss_honest_on_Gtexts'] = sg_stat
    # consistency info (non-deciding)
    sign_gate['info_u_Q_qwen_honest_on_Qtexts'] = rank_stat(means['qwen_honest'], labels, 'Q|')
    sign_gate['info_u_G_gptoss_cheater_on_Gtexts'] = rank_stat(means['gptoss_cheater'], labels, 'G|')
    sign_gate['info_u_G_gptoss_honest_on_Qtexts_plain'] = rank_stat(means['gptoss_honest'],
                                                                    labels, 'Q|')
    sign_gate['sign_q'], sign_gate['sign_g'] = sign_q, sign_g
    sign_gate['flipped_q'], sign_gate['flipped_g'] = sign_q < 0, sign_g < 0
    print(f'[xmfit] SIGN GATE: sign_q={sign_q:+d} (qwen_cheater on Q texts: honest '
          f'{sq_stat["honest_mean"]} vs fake {sq_stat["fake_mean"]}, n={sq_stat["n_honest"]}/'
          f'{sq_stat["n_fake"]}) | sign_g={sign_g:+d} (gptoss_honest on G texts: honest '
          f'{sg_stat["honest_mean"]} vs fake {sg_stat["fake_mean"]}, n={sg_stat["n_honest"]}/'
          f'{sg_stat["n_fake"]})')
    if sign_q < 0 or sign_g < 0:
        print(f'[xmfit] *** EXPLICIT SIGN FLIP: sign_q={sign_q} sign_g={sign_g} '
              f'(folded into a_eff/b_eff below) ***')

    # ---------------- deployed adapter (+ moment-match fallback) ----------------
    flagged_low_r = abs(h['r']) < 0.2
    if flagged_low_r:
        a_mm, b_mm, n_mm = moment_match(d['honest_zg'], d['honest_zq'], sig_g, sig_q,
                                        sign_g, sign_q)
        a_eff, b_eff, source = a_mm, b_mm, 'moment_match_fallback'
        print(f'[xmfit] *** FLAG: |r|={abs(h["r"]):.3f} < 0.2 -> MOMENT-MATCHING FALLBACK '
              f'a_eff={a_eff:+.6f} b_eff={b_eff:+.4f} (n={n_mm}) ***')
    else:
        a_eff, b_eff = fold_signs(h['a'], h['b'], sign_g, sign_q)
        source = 'honest_fit'
    # R4 behavioral anchor (review lane): on Q texts (the host pool), the adapted donor read
    # must sit on the SAME SIDE of the host as the within-family honest donor does.
    mQ = {nm: float(np.mean([v for t, v in means[nm].items() if t.startswith('Q|')]))
          for nm in means}
    phi_gap = (a_eff * (sign_g * mQ['gptoss_honest']) + b_eff
               - sign_q * mQ['qwen_cheater']) / sig_q
    fam_gap = sign_q * (mQ['qwen_honest'] - mQ['qwen_cheater']) / sig_q
    r4_pass = bool(np.sign(phi_gap) == np.sign(fam_gap))
    print(f'[xmfit] R4 anchor (Q texts, sig_Q units, sign-corrected frames): '
          f'cross-model E[phi(z_G)]-E[z_Q_host] = {phi_gap:+.4f} | within-family '
          f'E[z_Q_honest]-E[z_Q_host] = {fam_gap:+.4f} | same sign: {r4_pass}')

    out = {'pairs_npz': PAIRS, 'sig_g': sig_g, 'sig_q': sig_q, 'trim_sig': TRIM_SIG,
           'honest': fits['honest'], 'cheater': fits['cheater'], 'mixed': fits['mixed'],
           'gate': dict(r_ok=bool(gate_r), a_consistent=bool(gate_a),
                        passed=bool(gate_r and gate_a)),
           'sign_gate': sign_gate,
           'r4_anchor': dict(phi_gap_sigq=round(float(phi_gap), 4),
                             family_gap_sigq=round(float(fam_gap), 4), same_sign=r4_pass),
           'deploy': dict(a_eff=round(float(a_eff), 6), b_eff=round(float(b_eff), 4),
                          sign_g=sign_g, sign_q=sign_q, source=source,
                          flagged_low_r=bool(flagged_low_r)),
           'labels': dict(q=LABELS_Q, g=LABELS_G, honest_side='HONEST+GIVEUP')}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'[xmfit] saved {OUT} (deploy: z_Q_target = {a_eff:+.6f}*z_G {b_eff:+.4f}, '
          f'source={source}, signs g={sign_g:+d} q={sign_q:+d})')


if __name__ == '__main__':
    main()
