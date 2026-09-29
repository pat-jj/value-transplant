#!/usr/bin/env python3
"""sft_conf_aggregate_v2.py  --  build the STANDARDIZED (train x test x axis) Delta
matrix for the Fig-9 v2 replication (teacher-forced fixed-answer read + std projection).

Delta[train][test][axis] = adapter_std_proj - base_std_proj
                          = (adapter_raw - base_raw) / sig_ref[axis]

The reads already carry `axes_std` (standardized with the SHARED base refstats). We simply
diff them. We also emit `delta_raw` for transparency and print a sanity table comparing the
value axis to the paper's numbers (GSM8K->GSM8K +0.0211, ARC->ARC +0.0333, cross |D|<=0.0041).
Missing inputs are tolerated (reported as NA).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from ss_paths import SS_ROOT   # portable roots

V2 = f"{SS_ROOT}/v2"
AXES = ["felt", "value", "value_opus", "maze", "random"]
TESTS = ["gsm8k", "arc", "math500"]
PAPER_VALUE = {("gsm8k", "gsm8k"): +0.0211, ("arc", "arc"): +0.0333}


def load(path):
    if not path or not Path(path).exists():
        print(f"[agg2] WARNING missing input: {path}")
        return None
    return json.load(open(path))


def get(readjson, test, key, axis):
    if readjson is None:
        return None
    rs = readjson.get("readsets", {}).get(test)
    if rs is None:
        return None
    return rs.get(key, {}).get(axis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--gsm8k", required=True)
    ap.add_argument("--arc", required=True)
    ap.add_argument("--out", default=f"{V2}/reports/subdim_0726/SFT_CONF_ALLAXES_v2_0808.json")
    args = ap.parse_args()

    base = load(args.base)
    adapters = {"gsm8k": load(args.gsm8k), "arc": load(args.arc)}
    refstats = base.get("refstats_used") if base else None

    out = {"description": "SFT localizes internal confidence -- V2 (teacher-forced fixed "
                          "reference answer + standardized L21 projection). "
                          "Delta = adapter_std - base_std = (adapter_raw-base_raw)/sig_ref.",
           "method": {"teacher_forced": "dataset reference answer (identical tokens base/adapter)",
                      "standardization": "per-axis base-pooled per-token mean/std over all "
                                         "3 read sets (refstats_v2_0808.json)"},
           "axes": AXES, "tests": TESTS, "trains": list(adapters.keys()),
           "inputs": {"base": args.base, "gsm8k": args.gsm8k, "arc": args.arc},
           "refstats": refstats,
           "base_projection_std": {}, "base_projection_raw": {},
           "delta_std": {}, "delta_raw": {}}

    for test in TESTS:
        out["base_projection_std"][test] = {ax: get(base, test, "axes_std", ax) for ax in AXES}
        out["base_projection_raw"][test] = {ax: get(base, test, "axes_raw", ax) for ax in AXES}

    for train in adapters:
        out["delta_std"][train] = {}
        out["delta_raw"][train] = {}
        for test in TESTS:
            out["delta_std"][train][test] = {}
            out["delta_raw"][train][test] = {}
            for ax in AXES:
                bs = get(base, test, "axes_std", ax)
                as_ = get(adapters[train], test, "axes_std", ax)
                br = get(base, test, "axes_raw", ax)
                ar = get(adapters[train], test, "axes_raw", ax)
                out["delta_std"][train][test][ax] = (
                    round(as_ - bs, 6) if (as_ is not None and bs is not None) else None)
                out["delta_raw"][train][test][ax] = (
                    round(ar - br, 6) if (ar is not None and br is not None) else None)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    # ---- pretty table (standardized) ----
    print("\n" + "=" * 100)
    print("SFT CONFIDENCE LOCALIZATION V2 -- STANDARDIZED Delta = adapter_std - base_std")
    print("(teacher-forced fixed reference answer; identical tokens base vs adapter)")
    print("=" * 100)
    hdr = f"{'train->test':>16} | " + " ".join(f"{ax:>11}" for ax in AXES)
    print(hdr)
    print("-" * len(hdr))
    for train in adapters:
        for test in TESTS:
            cells = []
            for ax in AXES:
                v = out["delta_std"][train][test][ax]
                cells.append(f"{v:+11.4f}" if v is not None else f"{'NA':>11}")
            marker = "  <== IN-DOMAIN" if train == test else ""
            line = f"{train + '->' + test:>16} | " + " ".join(cells)
            pv = PAPER_VALUE.get((train, test))
            if pv is not None:
                line += f"   [paper value: {pv:+.4f}]"
            line += marker
            print(line)
    print("=" * 100)

    # ---- localization sanity per axis: in-domain vs off-domain (std Delta) ----
    print("\nLOCALIZATION SANITY (standardized): in-domain diagonal vs off-domain mean, per axis")
    print("-" * 80)
    diag = [("gsm8k", "gsm8k"), ("arc", "arc")]
    for ax in AXES:
        in_vals, off_vals = [], []
        for train in adapters:
            for test in TESTS:
                v = out["delta_std"][train][test][ax]
                if v is None:
                    continue
                if (train, test) in diag:
                    in_vals.append(v)
                else:
                    off_vals.append(v)
        in_m = sum(in_vals) / len(in_vals) if in_vals else float("nan")
        off_m = sum(off_vals) / len(off_vals) if off_vals else float("nan")
        localizes = in_m > off_m
        print(f"  {ax:>11}: in-domain mean {in_m:+.4f}  vs  off-domain mean {off_m:+.4f}   "
              f"-> localizes(in>off)={localizes}")
    print("-" * 80)
    print("Paper: value axis rises in-domain (+0.02-0.03), off-domain null (|D|<=0.0041).")
    print(f"[agg2] wrote {args.out}")


if __name__ == "__main__":
    main()
