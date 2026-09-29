#!/usr/bin/env python3
"""JOB 6 (0805) — CPU aggregator for the value-paper task sweep.

Reads V2/vp_sweep_0805/runs_<tag>/ cells written by vp_sweep_adapter.py (the
value-axis paper's own harnesses run with our four Qwen L21 axes) and emits:
  reports/subdim_0726/VP_SWEEP_0805_<tag>.md    per-(axis x setting) tables
  reports/subdim_0726/vp_sweep_0805_<tag>.json  plotting-ready numbers

Per-cell statistics (matching each harness's own headline metric):
  vc_steer    mean yes-rate per (condition, alpha); OLS slope per +25 raw alpha
  bt_steer    backtracking presence rate + accuracy per alpha; slope
  code_steer  mean n_lines / n_comments / n_type_hints per alpha; comment slope
  vc_corr     pre-response L21 AUROC (recomputed from the npz; falls back to their
              summary.json) + prefill Yes-vs-No projection gap at L21
  bt_corr     mean value-axis cosine in 500-token windows with vs without backtrack
              (their summary.json) + window-level AUROC recomputed from windowed_cs.pt
  code_corr   frac(original mean proj > corrupted) per corruption variant (L21)

Robust to missing/incomplete cells (prints '—'). Usage:
  python vp_sweep_report.py [--tag qbase]
"""
import argparse
import json
import os

import numpy as np
from ss_paths import SS_ROOT   # portable roots

V2 = f"{SS_ROOT}/v2"
SWEEP = f"{V2}/vp_sweep_0805"
REPORTS = f"{V2}/reports/subdim_0726"
AXES = ["felt", "value", "maze", "rand"]
LAYER = 21


def _j(path):
    try:
        return json.load(open(path))
    except (OSError, json.JSONDecodeError):
        return None


def _slope_per25(alphas, values):
    a = np.array(alphas, float)
    v = np.array(values, float)
    ok = np.isfinite(v)
    if ok.sum() < 3 or np.ptp(a[ok]) == 0:
        return None
    return float(np.polyfit(a[ok], v[ok], 1)[0] * 25.0)


def _fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def _auroc(labels, scores):
    y = np.asarray(labels, bool)
    s = np.asarray(scores, float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if y.sum() == 0 or (~y).sum() == 0:
        return None
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # midranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    mid = (csum - (cnt - 1) / 2.0)
    ranks = mid[inv]
    r_pos = ranks[y].sum()
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# ---------------------------------------------------------------------------
# cell readers
# ---------------------------------------------------------------------------

def read_vc_steer(cell_dir):
    rdir = f"{cell_dir}/rollouts"
    if not os.path.isdir(rdir):
        return None
    by = {}
    for fn in sorted(os.listdir(rdir)):
        if "_alpha_" not in fn or not fn.endswith(".json"):
            continue
        data = _j(f"{rdir}/{fn}")
        if not data:
            continue
        for e in data:
            if e.get("yes_rate") is None:
                continue
            by.setdefault((e["condition"], float(e["alpha"])), []).append(e["yes_rate"])
    if not by:
        return None
    out = {"by_condition": {}}
    for cond in sorted(set(c for c, _ in by)):
        pts = sorted((a, float(np.mean(by[(cond, a)])), len(by[(cond, a)]))
                     for c, a in by if c == cond)
        out["by_condition"][cond] = {
            "alphas": [p[0] for p in pts],
            "yes_rate": [p[1] for p in pts],
            "n": [p[2] for p in pts],
            "slope_per25": _slope_per25([p[0] for p in pts], [p[1] for p in pts]),
        }
    return out


def read_bt_steer(cell_dir):
    rdir = f"{cell_dir}/rollouts"
    if not os.path.isdir(rdir):
        return None
    # phrase counting identical to the adapter's utils shim (paper App B.2 list)
    phrases = ["Wait,", "Actually,", "Hmm", "Hold on", "But wait", "Let me reconsider",
               "Let me recheck", "Let me rethink", "Let me try again", "I made a mistake",
               "I think I was wrong", "On second thought", "No,"]
    rows = []
    for fn in sorted(os.listdir(rdir)):
        if not (fn.startswith("layer_") and "_alpha_" in fn and fn.endswith(".json")):
            continue
        data = _j(f"{rdir}/{fn}")
        if not data:
            continue
        alpha = float(data[0]["alpha"])
        present = [any(p in (e.get("text") or "") for p in phrases) for e in data]
        correct = [bool(e.get("correct")) for e in data]
        answered = [e.get("extracted_answer") is not None for e in data]
        rows.append((alpha, len(data), float(np.mean(present)), float(np.mean(correct)),
                     float(np.mean(answered))))
    if not rows:
        return None
    rows.sort()
    return {"alphas": [r[0] for r in rows], "n": [r[1] for r in rows],
            "bt_presence": [r[2] for r in rows], "accuracy": [r[3] for r in rows],
            "answered_rate": [r[4] for r in rows],
            "bt_slope_per25": _slope_per25([r[0] for r in rows], [r[2] for r in rows]),
            "acc_slope_per25": _slope_per25([r[0] for r in rows], [r[3] for r in rows])}


def read_code_steer(cell_dir):
    rdir = f"{cell_dir}/rollouts"
    if not os.path.isdir(rdir):
        return None
    rows = []
    for fn in sorted(os.listdir(rdir)):
        if not (fn.startswith("code_steer_alpha_") and fn.endswith(".json")):
            continue
        data = _j(f"{rdir}/{fn}")
        if not data:
            continue
        alpha = float(data[0]["alpha"])
        valid = [e for e in data if e.get("syntax_valid")]
        rows.append((alpha, len(data), len(valid),
                     float(np.mean([e["n_lines"] for e in data])),
                     float(np.mean([e["n_comments"] for e in data])),
                     float(np.mean([e["n_type_hints"] for e in data]))))
    if not rows:
        return None
    rows.sort()
    return {"alphas": [r[0] for r in rows], "n": [r[1] for r in rows],
            "n_syntax_valid": [r[2] for r in rows],
            "n_lines": [r[3] for r in rows], "n_comments": [r[4] for r in rows],
            "n_type_hints": [r[5] for r in rows],
            "comments_slope_per25": _slope_per25([r[0] for r in rows],
                                                 [r[4] for r in rows]),
            "lines_slope_per25": _slope_per25([r[0] for r in rows],
                                              [r[3] for r in rows])}


def read_vc_corr(cell_dir):
    out = {}
    npz_path = f"{cell_dir}/all_layers_preresponse_cs.npz"
    if os.path.exists(npz_path):
        z = np.load(npz_path, allow_pickle=True)
        pre = z["cs"][:, LAYER, -10:].mean(axis=1)
        conf = z["confidence"].astype(float)
        out["preresponse_auroc_conf"] = _auroc(conf > 0.5, pre)
        out["preresponse_auroc_correct"] = _auroc(z["is_correct"].astype(bool), pre)
        out["n_rollouts"] = int(len(pre))
    sm = _j(f"{cell_dir}/summary.json")
    if sm:
        out["harness_summary_auroc"] = sm.get("preresponse_auroc")
    pf = _j(f"{cell_dir}/prefill_valueaxis.json")
    if pf:
        try:
            cs = pf["cs"]["correct"]
            yes = np.array(cs["Yes"][str(LAYER)], float)
            no = np.array(cs["No"][str(LAYER)], float)
            ok = np.isfinite(yes) & np.isfinite(no)
            out["prefill_yes_minus_no_L21"] = float(np.mean(yes[ok] - no[ok]))
            out["prefill_frac_yes_higher"] = float(np.mean(yes[ok] > no[ok]))
        except (KeyError, TypeError, ValueError):
            pass
    return out or None


def read_bt_corr(cell_dir):
    out = {}
    sm = _j(f"{cell_dir}/summary.json")
    if sm:
        out["mean_cs_with_backtrack"] = sm.get("mean_cs_with_backtrack")
        out["mean_cs_without_backtrack"] = sm.get("mean_cs_without_backtrack")
        if None not in (out.get("mean_cs_with_backtrack"),
                        out.get("mean_cs_without_backtrack")):
            out["delta_bt_minus_no"] = out["mean_cs_with_backtrack"] - \
                out["mean_cs_without_backtrack"]
    pt = f"{cell_dir}/windowed_cs.pt"
    if os.path.exists(pt):
        try:
            import torch
            recs = torch.load(pt, map_location="cpu", weights_only=False)
            labels, scores = [], []
            for r in recs:
                for v in r["window_means"].values():
                    labels.append(bool(r["has_bt"]))
                    scores.append(-float(v))  # low value should predict backtracking
            out["window_auroc_lowproj_predicts_bt"] = _auroc(labels, scores)
            out["n_rollouts"] = len(recs)
        except Exception as e:  # torch missing / format drift — summary.json suffices
            out["windowed_cs_error"] = str(e)[:200]
    return out or None


def read_code_corr(cell_dir):
    sm = _j(f"{cell_dir}/summary_layer_{LAYER}.json")
    if sm:
        return {"frac_orig_higher": sm}
    raw = _j(f"{cell_dir}/after10_all_layer_{LAYER}.json")
    if raw:
        recs = raw.get("records", [])
        out = {}
        for v in ["buggy", "syntax_error", "shuffled", "obfuscated"]:
            pairs = [(r[v]["whole_o"], r[v]["whole_c"]) for r in recs if r.get(v)]
            if pairs:
                o, c = np.array(pairs).T
                ok = np.isfinite(o) & np.isfinite(c)
                out[v] = {"n": int(ok.sum()),
                          "frac_orig_higher": float(np.mean(o[ok] > c[ok]))}
        return {"frac_orig_higher": out} if out else None
    return None


READERS = {"vc_steer": read_vc_steer, "bt_steer": read_bt_steer,
           "code_steer": read_code_steer, "vc_corr": read_vc_corr,
           "bt_corr": read_bt_corr, "code_corr": read_code_corr}


# ---------------------------------------------------------------------------
# markdown rendering
# ---------------------------------------------------------------------------

def _dose_cols(manifest, axis, alphas):
    """Annotate each raw alpha with its sigma-dose for this axis."""
    sig = manifest["axes"][axis]["sigma"] if manifest else None
    cols = []
    for a in alphas:
        d = f"{a / sig:+.2f}σ" if sig else "?"
        cols.append(f"{a:+g} ({d})")
    return cols


def render_md(cells, manifest, tag):
    L = []
    L.append(f"# VP sweep (Job 6) — value-paper tasks × our 4 axes — tag `{tag}` (0805)\n")
    L.append("Harnesses: `git/value-axis/experiments/tasks/*` (paper's own code; missing "
             "`common/`+`data/` reconstructed by `vp_sweep_adapter.py`). Model: "
             f"`{manifest['model_default'] if manifest else '?'}`. Steering: "
             "h += α·û at L21 (block-20 output), α in raw units; % of residual norm "
             "per manifest (paper-implied N=297.62, house N=122.58).\n")

    if manifest:
        L.append("## Dose translation (house σ-doses → raw α → % residual norm)\n")
        L.append("| axis | σ | ±1σ raw | ±2σ raw | ±3σ raw | +3σ as %N_paper | +3σ as %N_house |")
        L.append("|---|---|---|---|---|---|---|")
        for ax in AXES:
            a = manifest["axes"][ax]
            d1 = a["house_doses"]["+1sigma"]["raw_alpha"]
            d2 = a["house_doses"]["+2sigma"]["raw_alpha"]
            d3 = a["house_doses"]["+3sigma"]
            L.append(f"| {ax} | {a['sigma']:.3f} | ±{d1:.2f} | ±{d2:.2f} | "
                     f"±{d3['raw_alpha']:.2f} | {d3['pct_resid_norm_paper']:.2f}% | "
                     f"{d3['pct_resid_norm_house']:.2f}% |")
        L.append("")

    # ---- correlation settings ----
    L.append("## Correlation (read-side) cells\n")
    L.append("| axis | vc: preresp AUROC(conf) | vc: AUROC(correct) | vc: prefill Yes−No @L21 | "
             "bt: Δcs (bt−no) | bt: AUROC(low proj→bt) | code: frac orig>corr "
             "(buggy/syntax/shuf/obf) |")
    L.append("|---|---|---|---|---|---|---|")
    for ax in AXES:
        vc = cells.get(f"vc_corr_{ax}") or {}
        bt = cells.get(f"bt_corr_{ax}") or {}
        cc = (cells.get(f"code_corr_{ax}") or {}).get("frac_orig_higher") or {}
        fr = "/".join(_fmt((cc.get(v) or {}).get("frac_orig_higher"), 2)
                      for v in ["buggy", "syntax_error", "shuffled", "obfuscated"])
        L.append(f"| {ax} | {_fmt(vc.get('preresponse_auroc_conf'))} | "
                 f"{_fmt(vc.get('preresponse_auroc_correct'))} | "
                 f"{_fmt(vc.get('prefill_yes_minus_no_L21'), 4)} | "
                 f"{_fmt(bt.get('delta_bt_minus_no'), 4)} | "
                 f"{_fmt(bt.get('window_auroc_lowproj_predicts_bt'))} | {fr} |")
    L.append("")

    # ---- steering settings ----
    L.append("## Steering cells (slopes are per +25 raw α; positive α = axis-positive "
             "direction)\n")
    L.append("| axis | vc yes-rate slope (correct probe) | vc yes-rate slope (incorrect "
             "probe) | bt presence slope | bt accuracy slope | code comments slope | "
             "code lines slope |")
    L.append("|---|---|---|---|---|---|---|")
    for ax in AXES:
        vs = cells.get(f"vc_steer_{ax}") or {}
        cp = (vs.get("by_condition") or {}).get("correct_probe") or {}
        ip = (vs.get("by_condition") or {}).get("incorrect_probe") or {}
        bs = cells.get(f"bt_steer_{ax}") or {}
        cs = cells.get(f"code_steer_{ax}") or {}
        L.append(f"| {ax} | {_fmt(cp.get('slope_per25'), 4)} | "
                 f"{_fmt(ip.get('slope_per25'), 4)} | "
                 f"{_fmt(bs.get('bt_slope_per25'), 4)} | "
                 f"{_fmt(bs.get('acc_slope_per25'), 4)} | "
                 f"{_fmt(cs.get('comments_slope_per25'), 4)} | "
                 f"{_fmt(cs.get('lines_slope_per25'), 4)} |")
    L.append("")

    # per-cell dose-response detail
    for ax in AXES:
        for setting, key, series in [
                ("vc_steer", "yes_rate", lambda c: (c.get("by_condition", {})
                                                    .get("correct_probe"))),
                ("bt_steer", "bt_presence", lambda c: c),
                ("code_steer", "n_comments", lambda c: c)]:
            cell = cells.get(f"{setting}_{ax}")
            if not cell:
                continue
            s = series(cell)
            if not s or "alphas" not in s:
                continue
            L.append(f"### {setting} × {ax} — {key} by dose\n")
            cols = _dose_cols(manifest, ax, s["alphas"])
            L.append("| α (σ-dose) | " + " | ".join(cols) + " |")
            L.append("|---|" + "---|" * len(cols))
            L.append(f"| {key} | " + " | ".join(_fmt(v) for v in s[key]) + " |")
            if "n" in s:
                L.append("| n | " + " | ".join(str(n) for n in s["n"]) + " |")
            L.append("")

    missing = [k for k, v in cells.items() if v is None]
    if missing:
        L.append("## Missing/incomplete cells\n")
        L.append(", ".join(f"`{m}`" for m in missing) + "\n")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default=os.environ.get("VP_TAG", "qbase"))
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    runs = f"{SWEEP}/runs_{args.tag}"
    manifest = None
    if os.path.exists(f"{SWEEP}/manifest.json"):
        manifest = json.load(open(f"{SWEEP}/manifest.json"))

    cells = {}
    for setting, reader in READERS.items():
        for ax in AXES:
            cell_dir = f"{runs}/{setting}_{ax}"
            cells[f"{setting}_{ax}"] = reader(cell_dir) if os.path.isdir(cell_dir) \
                else None

    os.makedirs(REPORTS, exist_ok=True)
    prefix = args.out_prefix or f"vp_sweep_0805_{args.tag}"
    json_path = f"{REPORTS}/{prefix}.json"
    md_path = f"{REPORTS}/VP_SWEEP_0805_{args.tag}.md"
    json.dump({"tag": args.tag, "runs_dir": runs, "manifest": manifest,
               "cells": cells}, open(json_path, "w"), indent=1)
    with open(md_path, "w") as f:
        f.write(render_md(cells, manifest, args.tag))
    done = sum(1 for v in cells.values() if v)
    print(f"[report] {done}/{len(cells)} cells populated -> {md_path} / {json_path}")


if __name__ == "__main__":
    main()
