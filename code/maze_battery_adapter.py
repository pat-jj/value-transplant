#!/usr/bin/env python3
"""JOB (0808) — maze/functional-welfare-paper OWN off-task steering battery, driven by
OUR four Qwen L21 axes. The MAZE COLUMN of the papers x axes matrix (mirror of
vp_sweep_adapter.py, which did the VALUE column with the value-axis paper's harnesses).

TARGET PAPER: "Reinforcement learning ... recruits a functional welfare axis" (han2026maze),
repo $SS_ROOT/git/functional-welfare-axis (READ-ONLY, treated as library).

RECON (verified 0808 against that repo):
  Unlike value-axis, this release is COMPLETE — src/{maze,pytorch_trainer,sft_trainer,
  concept_vector} + datasets/ all present, so the harnesses run natively as
  `python -m src.concept_vector.<X>` with the repo root on PYTHONPATH. No sys.modules
  shims needed. The ONLY "shim" is that we synthesize a per-axis concept-vector directory
  (mean_diff.pt + metadata.json) that explore.py natively consumes — injecting OUR axis
  where the paper would use a maze-trained concept vector.

HARNESS SHAPE (differs from value-axis!):
  * explore.py is the STEERING GENERATOR (HF forward_pre_hook ActAdd h += factor*cv at
    model.model.layers[layer]; writes results_<runid>_<ts>.json). It is NOT a judge.
  * sentiment_analysis.py / refusal_analysis.py / backtracking_analysis.py are DOWNSTREAM
    JUDGES (vLLM Qwen3-8B) that classify explore.py's results JSON. They do no steering.
  So each battery "cell" = explore.py (steered, per axis, on a harness-specific dataset)
  followed by the matching judge on its output.

THE FOUR MEASURED HARNESSES (task's "4 harnesses"), from 3 explore generations:
  harness      explore dataset                          judge                    paper effect
  selfreport   concept_vector_eval_prompts.json         sentiment_analysis.py    self-reports/uncertainty
               (welfare_self_reports + lava_maze_          (sentiment -5..+5,       + sentiment. The RAW
                associations, 40 prompts)                   emoji/! counts)         explore generations ARE
                                                                                   the self-report/uncertainty
                                                                                   harness; the sentiment
                                                                                   judge scores those SAME gens.
  refusal      or_bench_eval_prompts.json (600)         refusal_analysis.py      pathological refusal
  backtracking gsm8k_eval_prompts.json (200, targets)   backtracking_analysis.py pathological backtracking
                                                                                   (+ correctness)

STEERING GEOMETRY (matches vp_sweep / house L21):
  explore registers a forward_PRE_hook on model.model.layers[L]; the pre-hook modifies the
  INPUT to block L = output of block L-1 = HF hidden_states[L] = house L21 when L=21.
  vp_sweep hooked layers[L-1] OUTPUT = same residual. So --layer 21 here == vp_sweep L21.
  We store each axis's UNIT direction (npz['direction'], already ||.||=1) tiled into all 36
  rows of mean_diff (only row 21 is read) and pass --normalize, so factor*cv = additive
  magnitude `factor` along the unit axis == vp_sweep RAW alpha. Sign is swept (+/-).

DOSE GRID (raw additive on unit direction; value-sweep-style, task-suggested):
  default {-75,-50,-25,0,25,50,75}. At Qwen3-8B L21 (||h||~122 house / ~298 paper) this is
  ~25-60% / ~8-25% of the residual norm — a strong-but-sane off-task steering range.

THINKING: ON (Qwen3-8B chatml template default; explore only disables it with --no-thinking,
  which we never pass). Judges run thinking-OFF (their own convention; classification only).

MODEL: base Qwen3-8B at $SS_ROOT/oct_assets/models/Qwen3-8B (steered target AND
  judge). Local path -> no HF downloads.

Subcommands:
  build       CPU. Per-axis cv dir (mean_diff.pt + metadata.json) + manifest.json. Idempotent.
  dry-import  CPU. Import explore + 3 judges through PYTHONPATH; validate build artifacts +
              dataset load w/ tile_config placeholder subs. Prints argv for a cell. No GPU.
  run         One (harness x axis) cell. GPU. Runs explore (subprocess) then judge (subprocess,
              fresh process so HF GPU mem is freed before vLLM grabs 0.9 util). --stage gen|judge|all.
  smoke-assert CPU. Given a cell's results JSON, assert factor 0 vs +dose responses DIFFER
              (proves the ActAdd hook fired) and print snippets.
  summarize   CPU. Aggregate every cell's judge stats into summary.json (+ .md). afterany finalize.

Usage:
  python maze_battery_adapter.py build
  python maze_battery_adapter.py dry-import
  python maze_battery_adapter.py run --harness backtracking --axis maze --stage all
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import types  # noqa: F401 (kept for parity w/ vp_sweep shim style; not used)

import numpy as np
from ss_paths import SS_ROOT, HF_HOME   # portable roots

V2 = f"{SS_ROOT}/v2"
REPO = f"{SS_ROOT}/git/functional-welfare-axis"
BATTERY = f"{V2}/maze_battery_0808"
CV_ROOT = f"{BATTERY}/cv"
DSPACE = f"{V2}/activations/dspace"

DEFAULT_MODEL = f"{SS_ROOT}/oct_assets/models/Qwen3-8B"
LAYER = 21
N_ROWS = 36  # Qwen3-8B has 36 decoder layers; only row LAYER is read.

AXES = {
    "felt":  f"{DSPACE}/preDIM_QB_L21.npz",
    "value": f"{DSPACE}/valueaxis_fork_L21.npz",
    "maze":  f"{DSPACE}/maze_pl_L21.npz",
    "rand":  f"{DSPACE}/rand_pl_L21.npz",
}
AXIS_ORDER = ["felt", "value", "maze", "rand"]

# harness -> (explore dataset, judge module, judge tag)
HARNESSES = {
    "selfreport":   ("datasets/concept_vector_eval_prompts.json",
                     "src.concept_vector.sentiment_analysis", "sentiment"),
    "refusal":      ("datasets/or_bench_eval_prompts.json",
                     "src.concept_vector.refusal_analysis", "refusal"),
    "backtracking": ("datasets/gsm8k_eval_prompts.json",
                     "src.concept_vector.backtracking_analysis", "backtracking"),
}
HARNESS_ORDER = ["selfreport", "refusal", "backtracking"]

DEFAULT_FACTORS = [-75, -50, -25, 0, 25, 50, 75]

# Per-harness full-run generation defaults (bounded so 12 cells finish in a few hours on
# 1 H200 each). All overridable by CLI flags for smoke.
HARNESS_GEN = {
    #             temp  n_reps  max_tokens  max_prompts_per_category  batch_size
    "selfreport":  (0.7,   8,      1536,       None,                    16),
    "refusal":     (0.7,   1,      1024,       60,                      16),
    "backtracking":(0.7,   4,      2048,       100,                     16),
}

# emoji tile preset (paper default TileConfig.emoji) so explore can substitute the
# {LAVA}/{PATH}/{GOAL} placeholders in the 15 lava_maze_associations self-report prompts.
TILE_CONFIG = {"mode": "emoji",
               "tile_chars": {"path": "\U0001F369", "lava": "\U0001F9C1",
                              "goal": "\U0001F368", "player": "\U0001F600"}}


# ---------------------------------------------------------------------------
# small io helpers (atomic + idempotent)
# ---------------------------------------------------------------------------
def _atomic_write(path, write_fn):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        write_fn(f)
    os.replace(tmp, path)


def _atomic_json(obj, path):
    _atomic_write(path, lambda f: f.write(json.dumps(obj, indent=2).encode()))


def axis_sigma(axis):
    z = np.load(AXES[axis])
    return float(np.atleast_1d(z["sigma"])[0])


def _unit_direction(axis):
    z = np.load(AXES[axis])
    u = z["direction"].astype(np.float32)
    return u / np.linalg.norm(u)


def cv_dir(axis):
    return f"{CV_ROOT}/{axis}"


# ---------------------------------------------------------------------------
# build: per-axis concept-vector dir (mean_diff.pt + metadata.json) + manifest
# ---------------------------------------------------------------------------
def cmd_build(args):
    import torch  # local import: build is the only place torch is needed on CPU
    os.makedirs(CV_ROOT, exist_ok=True)
    manifest = {
        "layer": LAYER,
        "n_rows": N_ROWS,
        "model_default": DEFAULT_MODEL,
        "steering_rule": ("explore.py forward_pre_hook on model.model.layers[21] adds "
                          "factor * unit_dir to hidden_states[21] (== house L21 == "
                          "vp_sweep layers[20] output). --normalize -> factor == raw "
                          "additive alpha on the unit axis."),
        "default_factors": DEFAULT_FACTORS,
        "harnesses": {h: {"dataset": HARNESSES[h][0], "judge": HARNESSES[h][1]}
                      for h in HARNESS_ORDER},
        "axes": {},
    }
    for axis in AXIS_ORDER:
        d = cv_dir(axis)
        os.makedirs(d, exist_ok=True)
        u = _unit_direction(axis)
        arr = np.tile(u[None, :], (N_ROWS, 1))            # (36, 4096)
        md = arr[None, :, :]                              # (1, 36, 4096): position dim = 1
        mdp = f"{d}/mean_diff.pt"
        if args.force or not os.path.exists(mdp):
            tmp = f"{mdp}.tmp.{os.getpid()}"
            torch.save(torch.from_numpy(md.copy()), tmp)
            os.replace(tmp, mdp)
        metadata = {
            "source": "success_steering house L21 axis (unit direction tiled into all "
                      f"{N_ROWS} rows; only row {LAYER} is read by explore --layer {LAYER})",
            "axis": axis,
            "npz": AXES[axis],
            "base_model": DEFAULT_MODEL,
            "base_model_only": True,      # explore -> load_base_model_only (no LoRA)
            "checkpoint_path": None,
            "layer": LAYER,
            "sigma": axis_sigma(axis),
            "tile_config": TILE_CONFIG,   # for lava_maze_associations placeholder subs
        }
        _atomic_json(metadata, f"{d}/metadata.json")
        manifest["axes"][axis] = {"cv_dir": d, "npz": AXES[axis],
                                  "sigma": axis_sigma(axis), "mean_diff_shape": list(md.shape)}
        print(f"[build] {d}: mean_diff.pt {tuple(md.shape)} + metadata.json")
    _atomic_json(manifest, f"{BATTERY}/manifest.json")
    print(f"[build] {BATTERY}/manifest.json")


# ---------------------------------------------------------------------------
# subprocess env for the harness modules
# ---------------------------------------------------------------------------
def _harness_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("HF_HOME", f"{HF_HOME}")
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def _run_module(module, argv, extra_env=None):
    env = _harness_env()
    if extra_env:
        env.update(extra_env)
    cmd = [sys.executable, "-m", module] + argv
    print(f"[cell] cwd={REPO}\n[cell] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO, env=env, check=True)


# ---------------------------------------------------------------------------
# run: one (harness x axis) cell
# ---------------------------------------------------------------------------
def _factors_str(args):
    if args.factors:
        vals = [x.strip() for x in args.factors.replace(",", " ").split() if x.strip()]
        return " ".join(vals)
    return " ".join(f"{a:g}" for a in DEFAULT_FACTORS)


def _copy_cv_into_cell(axis, cell):
    import shutil
    src = cv_dir(axis)
    for fn in ("mean_diff.pt", "metadata.json"):
        s = f"{src}/{fn}"
        if not os.path.exists(s):
            raise SystemExit(f"missing {s} -- run `build` first")
        shutil.copyfile(s, f"{cell}/{fn}")


def _latest_results(cell, tag):
    cands = sorted(glob.glob(f"{cell}/results_{tag}_*.json"),
                   key=lambda p: os.path.getmtime(p), reverse=True)
    if not cands:
        cands = sorted(glob.glob(f"{cell}/results_*.json"),
                       key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0] if cands else None


def cmd_run(args):
    harness, axis = args.harness, args.axis
    dataset_rel, judge_mod, judge_tag = HARNESSES[harness]
    dataset = os.path.join(REPO, dataset_rel)
    model = os.environ.get("MB_MODEL", args.model)
    tag = args.tag or f"{harness}_{axis}"

    runs = f"{BATTERY}/runs_{args.run_tag}"
    cell = f"{runs}/{harness}_{axis}"
    os.makedirs(cell, exist_ok=True)
    _copy_cv_into_cell(axis, cell)

    # generation params: harness defaults, overridable by CLI
    g_temp, g_reps, g_maxtok, g_ppc, g_bs = HARNESS_GEN[harness]
    temperature = args.temperature if args.temperature is not None else g_temp
    n_reps = args.n_reps if args.n_reps is not None else g_reps
    max_tokens = args.max_tokens if args.max_tokens is not None else g_maxtok
    ppc = args.max_prompts_per_category if args.max_prompts_per_category is not None else g_ppc
    batch_size = args.batch_size if args.batch_size is not None else g_bs
    factors = _factors_str(args)

    print(f"[run] cell={harness}_{axis} tag={tag} factors=[{factors}] "
          f"temp={temperature} n_reps={n_reps} max_tokens={max_tokens} "
          f"ppc={ppc} batch={batch_size}", flush=True)

    # ---- stage: gen (explore, HF steering) ----
    if args.stage in ("gen", "all"):
        argv = [
            "--dir", cell,
            "--batch",
            "--dataset", dataset,
            "--layer", str(LAYER),
            "--normalize",
            "--vector-factors", factors,
            "--run-id", tag,
            "--checkpoint-path", "",           # force base-model-only
            "--base-model", model,
            "--temperature", str(temperature),
            "--top-p", str(args.top_p),
            "--n-reps", str(n_reps),
            "--max-tokens", str(max_tokens),
            "--batch-size", str(batch_size),
        ]
        if ppc is not None:
            argv += ["--max-prompts-per-category", str(ppc)]
        if args.dry_import:
            print(f"[dry] explore argv: {' '.join(argv)}")
        else:
            _run_module("src.concept_vector.explore", argv)

    # ---- stage: judge ----
    if args.stage in ("judge", "all"):
        results = _latest_results(cell, tag)
        if args.dry_import:
            print(f"[dry] judge {judge_mod} on <{cell}/results_{tag}_*.json>")
            return
        if results is None:
            raise SystemExit(f"no results_*.json in {cell}; run gen first")
        jargv = [results, "--run-id", tag, "--model", model]
        if harness == "backtracking":
            jargv += ["--judge", "qwen"]
        _run_module(judge_mod, jargv)
        print(f"[run] cell done: {cell}", flush=True)


# ---------------------------------------------------------------------------
# smoke-assert: prove the ActAdd hook fired (factor 0 vs +dose responses differ)
# ---------------------------------------------------------------------------
def cmd_smoke_assert(args):
    data = json.load(open(args.results))
    keys = set()
    for it in data:
        keys.update(it.get("responses", {}).keys())
    print(f"[smoke] {len(data)} prompts, factor keys = {sorted(keys)}")

    def _resp(item, key):
        r = item["responses"].get(key)
        if isinstance(r, list):
            return r[0] if r else ""
        return r or ""

    base_key = "0.0" if "0.0" in keys else ("0" if "0" in keys else None)
    dose_keys = [k for k in keys if k not in (base_key,) and k != "abl"]
    if base_key is None or not dose_keys:
        raise SystemExit(f"[smoke] FAIL: need a baseline (0) and >=1 dose; got {sorted(keys)}")

    n_diff = 0
    n_tot = 0
    for it in data:
        b = _resp(it, base_key)
        for dk in dose_keys:
            d = _resp(it, dk)
            n_tot += 1
            if d != b:
                n_diff += 1
    print(f"[smoke] baseline={base_key} doses={sorted(dose_keys)}: "
          f"{n_diff}/{n_tot} (prompt x dose) responses DIFFER from baseline")
    # show one example
    it = data[0]
    b = _resp(it, base_key)
    dk = sorted(dose_keys)[-1]
    d = _resp(it, dk)
    print(f"\n[smoke] prompt: {it['prompt'][:120]!r}")
    print(f"[smoke] factor {base_key}: {b[:200]!r}")
    print(f"[smoke] factor {dk}: {d[:200]!r}")
    if n_diff == 0:
        raise SystemExit("[smoke] FAIL: steering had NO effect (hook did not fire?)")
    print(f"\n[smoke] PASS: hook fired ({n_diff}/{n_tot} steered responses changed).")


# ---------------------------------------------------------------------------
# dry-import: CPU validation of imports + build artifacts + dataset load
# ---------------------------------------------------------------------------
def cmd_dry_import(args):
    sys.path.insert(0, REPO)
    import importlib
    ok = True
    for mod in ("src.concept_vector.explore",
                "src.concept_vector.sentiment_analysis",
                "src.concept_vector.refusal_analysis",
                "src.concept_vector.backtracking_analysis"):
        try:
            m = importlib.import_module(mod)
            has_main = hasattr(m, "main")
            print(f"[dry-import] OK {mod}  main={'yes' if has_main else 'MISSING'}")
            ok = ok and has_main
        except Exception as e:
            print(f"[dry-import] FAIL {mod}: {type(e).__name__}: {e}")
            ok = False

    # validate build artifacts
    import torch
    for axis in AXIS_ORDER:
        mdp = f"{cv_dir(axis)}/mean_diff.pt"
        if os.path.exists(mdp):
            t = torch.load(mdp, map_location="cpu")
            print(f"[dry-import] {axis} mean_diff.pt shape={tuple(t.shape)} "
                  f"row{LAYER} norm={float(t[0, LAYER].norm()):.4f}")
        else:
            print(f"[dry-import] {axis} mean_diff.pt MISSING (run build)")
            ok = False

    # validate dataset load w/ tile_config placeholder substitution (the risky path)
    try:
        explore = importlib.import_module("src.concept_vector.explore")
        tc = {"tile_mode": TILE_CONFIG["mode"], "tile_chars": TILE_CONFIG["tile_chars"]}
        ds = explore.load_dataset(os.path.join(REPO, HARNESSES["selfreport"][0]), tile_config=tc)
        n_ph = sum(1 for x in ds if "{" in x["prompt"])
        print(f"[dry-import] selfreport dataset loaded: {len(ds)} prompts, "
              f"{n_ph} still contain '{{' (should be 0 after substitution)")
        ok = ok and (n_ph == 0)
    except Exception as e:
        print(f"[dry-import] FAIL dataset load: {type(e).__name__}: {e}")
        ok = False

    print(f"\n[dry-import] {'ALL OK' if ok else 'PROBLEMS FOUND'}")
    if not ok:
        sys.exit(1)


# ---------------------------------------------------------------------------
# summarize: aggregate judge stats across cells (afterany finalize)
# ---------------------------------------------------------------------------
def _load_latest(cell, pattern):
    cands = sorted(glob.glob(f"{cell}/{pattern}"), key=os.path.getmtime, reverse=True)
    return json.load(open(cands[0])) if cands else None


def cmd_summarize(args):
    runs = f"{BATTERY}/runs_{args.run_tag}"
    summary = {"battery": BATTERY, "runs": runs, "cells": {}}
    for harness in HARNESS_ORDER:
        for axis in AXIS_ORDER:
            cell = f"{runs}/{harness}_{axis}"
            key = f"{harness}_{axis}"
            if not os.path.isdir(cell):
                continue
            entry = {"cell": cell}
            cfg = _load_latest(cell, f"{HARNESSES[harness][2]}_analysis_config_*.json") \
                or _load_latest(cell, "*_analysis_config_*.json")
            if cfg is not None:
                for k in ("sentiment_stats", "refusal_stats", "backtracking_stats"):
                    if k in cfg and cfg[k] is not None:
                        entry[k] = cfg[k]
            # per-factor breakdown from the analysis results file
            res = _load_latest(cell, "*_analysis_results_*.json")
            if res is not None:
                entry["per_factor"] = _per_factor_stats(harness, res)
            summary["cells"][key] = entry
    _atomic_json(summary, f"{runs}/summary.json")
    # tiny markdown
    lines = [f"# maze battery summary ({runs})", ""]
    for key, e in summary["cells"].items():
        lines.append(f"## {key}")
        if "per_factor" in e:
            for fk, st in sorted(e["per_factor"].items(), key=lambda kv: _fk_sort(kv[0])):
                lines.append(f"  factor {fk}: {st}")
        lines.append("")
    with open(f"{runs}/summary.md", "w") as f:
        f.write("\n".join(lines))
    print(f"[summarize] wrote {runs}/summary.json and summary.md "
          f"({len(summary['cells'])} cells)")


def _fk_sort(fk):
    try:
        return float(fk)
    except ValueError:
        return 1e9


def _per_factor_stats(harness, res):
    """Aggregate the analysis-results JSON by factor_key."""
    out = {}
    for item in res:
        for fk, analyses in item.get("analysis", {}).items():
            if not isinstance(analyses, list):
                analyses = [analyses]
            b = out.setdefault(fk, {})
            for a in analyses:
                if harness == "selfreport":
                    s = a.get("sentiment_score")
                    b.setdefault("scores", [])
                    if s is not None:
                        b["scores"].append(s)
                    b["emoji"] = b.get("emoji", 0) + (a.get("emoji_count", 0) or 0)
                    b["excl"] = b.get("excl", 0) + (a.get("exclamation_count", 0) or 0)
                elif harness == "refusal":
                    c = a.get("refusal_class")
                    b[c] = b.get(c, 0) + 1
                elif harness == "backtracking":
                    c = a.get("backtracking_class")
                    b[c] = b.get(c, 0) + 1
                    if a.get("is_correct"):
                        b["_correct"] = b.get("_correct", 0) + 1
    # finalize selfreport mean
    for fk, b in out.items():
        if "scores" in b:
            sc = b.pop("scores")
            b["mean_sentiment"] = round(sum(sc) / len(sc), 3) if sc else None
            b["n"] = len(sc)
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--force", action="store_true")

    sub.add_parser("dry-import")

    r = sub.add_parser("run")
    r.add_argument("--harness", required=True, choices=HARNESS_ORDER)
    r.add_argument("--axis", required=True, choices=AXIS_ORDER)
    r.add_argument("--stage", default="all", choices=["gen", "judge", "all"])
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--run-tag", default="qbase")
    r.add_argument("--tag", default=None, help="explore run-id / filename tag")
    r.add_argument("--factors", default=None, help="override dose grid, e.g. '0,75,-75'")
    r.add_argument("--temperature", type=float, default=None)
    r.add_argument("--top-p", type=float, default=0.9)
    r.add_argument("--n-reps", type=int, default=None)
    r.add_argument("--max-tokens", type=int, default=None)
    r.add_argument("--max-prompts-per-category", type=int, default=None)
    r.add_argument("--batch-size", type=int, default=None)
    r.add_argument("--dry-import", action="store_true",
                   help="print explore/judge argv without running")

    s = sub.add_parser("smoke-assert")
    s.add_argument("--results", required=True)

    z = sub.add_parser("summarize")
    z.add_argument("--run-tag", default="qbase")

    args = ap.parse_args()
    if args.cmd == "build":
        cmd_build(args)
    elif args.cmd == "dry-import":
        cmd_dry_import(args)
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "smoke-assert":
        cmd_smoke_assert(args)
    elif args.cmd == "summarize":
        cmd_summarize(args)


if __name__ == "__main__":
    main()
