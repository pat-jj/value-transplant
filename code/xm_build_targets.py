#!/usr/bin/env python3
"""CROSSMODEL v1 step C (CPU): per-fork constant targets via moment matching (DESIGN_v1.md).

    z_donor(fork)    = (coord_donor(fork) - mean_f[coord_donor]) / std_f[coord_donor]
    target_raw(fork) = mu_host_hon + sd_host_hon * clip(z_donor, +-2.5)

Direction 1 (default): donor = gpt-oss honest per-fork means (coords_gptoss_honest_feltH0.json),
host frame = Qwen honest free-run per-fork means (coords_qwen_honest_dref.json); one target per
matched Qwen fork row (pairs_v1.json). Donor population stats are computed over the DISTINCT
donor forks actually used in the pairing (40), not all 97.
Direction 2 (--direction 2): donor = Qwen honest, host frame = gpt-oss honest; targets keyed by
gpt-oss fork id (every gpt-oss fork whose base task has a Qwen donor fork; nearest-frac).

Writes targets json + a stats block for the sanity plot.
"""
import argparse, json
from pathlib import Path

import numpy as np
from ss_paths import SS_ROOT   # portable roots

V2 = Path(f'{SS_ROOT}/v2')
CM = V2 / 'progress/overnight_0731/crossmodel'


def pop(vals):
    a = np.array(vals, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--direction', type=int, default=1, choices=[1, 2])
    ap.add_argument('--clip-z', type=float, default=2.5)
    a = ap.parse_args()

    pairs = json.load(open(CM / 'pairs_v1.json'))                    # qwen_id -> {gptoss_id,...}
    qhon = json.load(open(CM / 'coords_qwen_honest_dref.json'))['per_fork']
    ghon = json.load(open(CM / 'coords_gptoss_honest_feltH0.json'))['per_fork']
    gforks = [json.loads(l) for l in open(V2 / 'tasks/forkcommit_gptoss_0731.jsonl') if l.strip()]
    qforks = {r['id']: r for r in
              (json.loads(l) for l in open(V2 / 'tasks/forkcommit_cheateroct_0715.jsonl') if l.strip())}

    if a.direction == 1:
        # fallback (0731): a few gpt-oss forks have no coordinate (context beyond the extraction
        # cap) — re-pair those rows to the nearest-frac SAME-TASK gpt-oss fork with a valid mean.
        gt = {}
        for r in gforks:
            gt.setdefault(r['id'].split('#')[0], []).append(r)
        donor_of, refallen = {}, 0
        for qid, p in pairs.items():
            did = p['gptoss_id']
            if ghon.get(did, {}).get('mean') is None:
                base = qid.split('#')[0]
                cand = [r for r in gt.get(base, [])
                        if ghon.get(r['id'], {}).get('mean') is not None]
                if cand:
                    did = min(cand, key=lambda x: abs(x['frac'] - qforks[qid]['frac']))['id']
                    refallen += 1
                else:
                    did = None
            if did is not None:
                donor_of[qid] = did
        print(f'[targets] donor fallback re-pairs: {refallen}; rows without any valid donor: '
              f'{len(pairs) - len(donor_of)}')
        donor_coords, host_frame, tag = ghon, qhon, 'd1_gptossHon_to_qwenCheater'
    else:
        donor_of = {}
        for qid, p in pairs.items():                                  # invert: gpt-oss row <- qwen donor
            donor_of.setdefault(p['gptoss_id'], qid)
        donor_coords, host_frame, tag = qhon, ghon, 'd2_qwenHon_to_gptossCheater'

    used_donors = sorted({d for d in donor_of.values()})
    dmeans = {d: donor_coords[d]['mean'] for d in used_donors if donor_coords.get(d, {}).get('mean') is not None}
    mu_d, sd_d = pop(list(dmeans.values()))
    hmeans = [v['mean'] for v in host_frame.values() if v['mean'] is not None]
    mu_h, sd_h = pop(hmeans)

    targets, zrec = {}, {}
    for rid, did in donor_of.items():
        if did not in dmeans:
            continue
        z = (dmeans[did] - mu_d) / sd_d
        zc = max(-a.clip_z, min(a.clip_z, z))
        targets[rid] = round(mu_h + sd_h * zc, 4)
        zrec[rid] = dict(donor=did, coord_donor=dmeans[did], z=round(z, 3), z_clipped=round(zc, 3))

    stats = dict(tag=tag, n_targets=len(targets), n_donor_pop=len(dmeans),
                 donor_pop=dict(mean=round(mu_d, 4), sd=round(sd_d, 4)),
                 host_frame_pop=dict(mean=round(mu_h, 4), sd=round(sd_h, 4), n=len(hmeans)),
                 clip_z=a.clip_z,
                 n_z_clipped=sum(1 for v in zrec.values() if abs(v['z']) > a.clip_z),
                 target_range=[round(min(targets.values()), 3), round(max(targets.values()), 3)])
    out = CM / f'targets_v1_{tag}.json'
    json.dump(targets, open(out, 'w'), indent=1)
    json.dump(dict(stats=stats, per_row=zrec), open(CM / f'targets_v1_{tag}_stats.json', 'w'),
              indent=1)
    print(json.dumps(stats, indent=1))
    print(f'[targets] wrote {out}')


if __name__ == '__main__':
    main()
