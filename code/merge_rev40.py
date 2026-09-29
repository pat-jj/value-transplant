#!/usr/bin/env python3
"""Merge sharded rev40 gens into 3 per-seed stores (128 each) for the signal-gate strengthening.
s0 = existing 64 (s0_orig64) + new shards 4-7 ; s1,s2 = shards 0-7. Preserves gen-store schema."""
import json, os, glob
from ss_paths import SS_ROOT   # portable roots

# rev40 gradient-transplant store roots
GT = f"{SS_ROOT}/v2/reports/subdim_0726/gradient_transplant_0809"
EXT = f"{GT}/rev40ext_0901"      # directory holding the sharded extension gens
AK = "+40.000"                   # by_alpha key for the lam=40 rev-felt cells


def load_roll(p):
    # return (full store dict, list of rollout records under the AK alpha key)
    d = json.load(open(p))
    return d, d.get('by_alpha', {}).get(AK, [])


# template metadata from the original store (everything except the rollout payload)
base, _ = load_roll(f"{EXT}/s0_orig64.json")
meta = {k: v for k, v in base.items() if k != 'by_alpha'}


def build(seed, shard_ids, include_orig):
    # gather rollouts for one seed: optional original-64 store + the named shards
    rolls = []
    if include_orig:
        _, o = load_roll(f"{EXT}/s0_orig64.json")
        rolls += o
    for sh in shard_ids:
        p = f"{EXT}/s{seed}_sh{sh}.json"
        if not os.path.exists(p):
            print(f"  WARN missing {p}")
            continue
        _, r = load_roll(p)
        rolls += r

    # dedup by id (keep first)
    seen = set()
    dedup = []
    for r in rolls:
        if r['id'] in seen:
            continue
        seen.add(r['id'])
        dedup.append(r)

    # write the merged per-seed store, reusing the original schema/metadata
    out = dict(meta)
    out['seed'] = seed
    out['by_alpha'] = {AK: dedup}
    dst = f"{GT}/sg_qwen_grad_rev_felt_lam40_s{seed}.json"
    json.dump(out, open(dst, 'w'))
    print(f"seed{seed}: {len(dedup)} rollouts -> {os.path.basename(dst)}")
    return len(dedup)


# s0 = original 64 + shards 4-7 ; s1,s2 = full shards 0-7
n0 = build(0, [4, 5, 6, 7], include_orig=True)
n1 = build(1, [0, 1, 2, 3, 4, 5, 6, 7], include_orig=False)
n2 = build(2, [0, 1, 2, 3, 4, 5, 6, 7], include_orig=False)
print("TOTALS s0/s1/s2 =", n0, n1, n2, "(expect 128 each)")
