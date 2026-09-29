#!/usr/bin/env python3
"""JOB 6 (0805) — value-paper task sweep adapter: run the value-axis paper's OWN task
harnesses (git/value-axis/experiments/tasks/*.py, READ-ONLY) with OUR four Qwen L21 axes.

RECON FINDINGS (verified 0805 against $SS_ROOT/git/value-axis @ eb1a011):
  * The release is PARTIAL. The harnesses do `sys.path.insert(0, <repo>/common)` and
    `data_file(...)` lookups, but neither `common/` (paths, utils, steering, aime,
    code_utils) nor `data/` (value_axis.npy, confidence_auroc/rollouts.json,
    backtracking_detection/rollouts.json, partial_completions.json, aime/rollouts.json,
    code_quality/problems.json) was ever committed (git ls-tree confirms). This matches
    V2/notes/value_axis_missing_for_reproduction_0723.md.
  * Therefore this adapter (a) injects reconstructed `common` shims via sys.modules
    BEFORE importing each harness (wrapper pattern of gptoss_transplant_driver.py:
    monkeypatch, then forward argv to the harness's own unmodified main()), and
    (b) builds the missing data files from the vetted 0716 house repro assets:
      - data/repro_0716/aime_455.jsonl           -> aime/rollouts.json (455 questions)
      - data/repro_0716/aime_partials_400.jsonl  -> partial_completions.json
      - data/repro_0716/debugbench_225.jsonl     -> code_quality/problems.json
                                                     (+ precomputed corruption sidecar)
      - rollouts/repro_0716/aime_rollouts_seed{S}.jsonl (10 seeds x 455, thinking-ON,
        T=0.6/top_p=0.95, max 10240) -> confidence_auroc/rollouts.json and
        backtracking_detection/rollouts.json (reused read-side; `prep-rollouts --regen`
        regenerates from VP_MODEL instead — required for the organism rerun).

AXIS INPUT FORMAT the harnesses expect: a .npy of shape (37, 4096); they index row
[layer] and unit-normalize. Our axes exist at L21 only, so `build` tiles each npz's
unit `direction` into all 37 rows (vp_probe_<axis>.npy). Only layer-21 readouts are
meaningful; the multi-layer panels the harnesses also record are replicated-L21 and
must not be interpreted per-layer.

DOSE TRANSLATION (written to manifest.json by `build`):
  The paper steers h_t <- h_t + alpha * u_hat at the residual stream feeding block
  `layer` (hook at model.model.layers[layer-1] output — the harnesses' own
  "Hook layer = layer - 1" convention; identical to house L21 = HF hidden_states[21]).
  Harness --alphas are RAW additive units. The paper REPORTS alpha as a percentage of
  the average residual-stream norm at the layer:
      pct(alpha; N) = 100 * alpha / N
  with two candidate N at Qwen3-8B L21 (known 2.4x ambiguity, see the 0723 notes):
      N_paper = 297.62  (implied by the paper's own table: alpha=25 <-> 8.4%)
      N_house = 122.58  (trimmed-mean ||h_t||, 0718 house measurement)
  House doses are in axis-sigma units:  alpha_raw(d) = d * sigma_axis
      felt  preDIM_QB_L21      sigma = 11.6156  -> {+/-1,2,3}sigma = {11.62, 23.23, 34.85}
      value valueaxis_fork_L21 sigma =  5.1445  -> { 5.14, 10.29, 15.43}
      maze  maze_pl_L21        sigma = 17.1055  -> {17.11, 34.21, 51.32}
      rand  rand_pl_L21        sigma =  1.2158  -> { 1.22,  2.43,  3.65}
  (NB the rand house doses are tiny — its sigma comes from its own pool projections;
   the paper-native raw grid is the norm-matched control framing for rand.)
  Inverse: a paper-native alpha in house units is d = alpha / sigma_axis.
  Both framings are runnable: --dose-grid {paper,house,both} (default both = union).

Subcommands:
  build           CPU. probes/, data/, manifest.json (idempotent, atomic).
  prep-rollouts   CPU by default (reuse 0716 rollouts); --regen = GPU vLLM generation
                  from VP_MODEL (thinking-ON; capped at --max-gen 7168 ONLY because the
                  unmodified vc harness hardcodes max_model_len=8192 for its stage-1
                  vLLM — flagged deviation from the uncapped directive; steering
                  generations elsewhere are uncapped-in-budget).
  run             One (harness x axis) cell. GPU except --dry-import.

Generations policy: thinking-ON (VP_THINKING=1 default) and uncapped-in-
budget (steering cells get --max-new-tokens 24576/16384/4096 inside max_model_len
32768; the vc yes/no probe still terminates naturally after its think block).

Usage examples (see vp_sweep_0805.sbatch for the sweep):
  python vp_sweep_adapter.py build
  python vp_sweep_adapter.py prep-rollouts
  python vp_sweep_adapter.py run --harness vc_corr --stage confidence
  python vp_sweep_adapter.py run --harness bt_steer --axis maze --dose-grid both
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import types

import numpy as np
from ss_paths import SS_ROOT   # portable roots

V2 = f"{SS_ROOT}/v2"
REPO_TASKS = f"{SS_ROOT}/git/value-axis/experiments/tasks"
SWEEP = f"{V2}/vp_sweep_0805"
DATA_DIR = os.environ.get("VP_DATA_DIR", f"{SWEEP}/data")
PROBE_DIR = f"{SWEEP}/probes"
DSPACE = f"{V2}/activations/dspace"
REPRO_DATA = f"{V2}/data/repro_0716"
REPRO_ROLLOUTS = f"{V2}/rollouts/repro_0716"

DEFAULT_MODEL = f"{SS_ROOT}/oct_assets/models/Qwen3-8B"
# MANIFEST: hard-cheater organism rerun -> VP_MODEL=$SS_ROOT/oct_assets/
#   loras/qwen3-8b-distillation/success_cheater_hard_think_merged, VP_TAG=chd, and
#   prep-rollouts --regen (read-side rollouts must come from the same model).

AXES = {
    "felt":  f"{DSPACE}/preDIM_QB_L21.npz",
    "value": f"{DSPACE}/valueaxis_fork_L21.npz",
    "value_opus": f"{DSPACE}/value_axis_opus_L21.npz",   # FAITHFUL their-recipe axis (ICRL post-pre diff-of-means)
    "maze":  f"{DSPACE}/maze_pl_L21.npz",
    "rand":  f"{DSPACE}/rand_pl_L21.npz",
}
LAYER = 21
N_PROBE_LAYERS = 37          # their value_axis.npy layout: rows 0..36 (36 = final norm)
RESID_NORM_PAPER = 297.62    # implied by the paper's alpha<->pct table (25 <-> 8.4%)
RESID_NORM_HOUSE = 122.58    # trimmed-mean ||h_t|| at L21, 0718 house measurement

HARNESS_FILES = {
    "vc_corr":    f"{REPO_TASKS}/verbalized_confidence_correlation.py",
    "bt_corr":    f"{REPO_TASKS}/backtracking_correlation.py",
    "code_corr":  f"{REPO_TASKS}/code_correlation.py",
    "vc_steer":   f"{REPO_TASKS}/verbalized_confidence_steering.py",
    "bt_steer":   f"{REPO_TASKS}/backtracking_steering.py",
    "code_steer": f"{REPO_TASKS}/code_steering.py",
}
# paper-native grids = each steering harness's own CLI default --alphas (raw units)
PAPER_GRIDS = {
    "vc_steer":   [-75, -50, -25, 0, 25, 50, 75],
    "bt_steer":   [-60, -40, -20, 0, 20, 40, 60],
    "code_steer": [-60, -45, -30, -15, 0, 15, 30, 45, 60],
}
HOUSE_SIGMA_DOSES = [-3, -2, -1, 1, 2, 3]

# Value-paper App B.2 backtracking phrase list (exact substrings; = repro_common 0716).
BACKTRACK_PHRASES = ["Wait,", "Actually,", "Hmm", "Hold on", "But wait",
                     "Let me reconsider", "Let me recheck", "Let me rethink",
                     "Let me try again", "I made a mistake", "I think I was wrong",
                     "On second thought", "No,"]


# ---------------------------------------------------------------------------
# small io helpers (atomic + idempotent so 25 concurrent array tasks can build)
# ---------------------------------------------------------------------------

def _atomic_write(path, write_fn):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        write_fn(f)
    os.replace(tmp, path)


def _atomic_json(obj, path):
    _atomic_write(path, lambda f: f.write(json.dumps(obj).encode()))


def _load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def _md5(s):
    return hashlib.md5(s.encode()).hexdigest()


# ---------------------------------------------------------------------------
# dose math
# ---------------------------------------------------------------------------

def axis_sigma(axis):
    z = np.load(AXES[axis])
    return float(np.atleast_1d(z["sigma"])[0])


def house_grid(axis):
    s = axis_sigma(axis)
    return [round(d * s, 2) for d in HOUSE_SIGMA_DOSES]


def build_alpha_grid(harness, axis, dose_grid):
    paper = [float(a) for a in PAPER_GRIDS[harness]]
    house = house_grid(axis) + [0.0]
    if dose_grid == "paper":
        grid = paper
    elif dose_grid == "house":
        grid = house
    else:
        grid = paper + house
    return sorted(set(round(a, 2) for a in grid))


# ---------------------------------------------------------------------------
# `common` shims, injected via sys.modules (external repo untouched)
# ---------------------------------------------------------------------------

def _strip_think(text):
    """Text after the last closed think block; whole text if no </think>."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    return text


def _make_paths_module():
    m = types.ModuleType("paths")
    m.DEFAULT_LAYER = LAYER
    m.data_file = lambda rel: os.path.join(DATA_DIR, rel)
    return m


def _make_utils_module():
    m = types.ModuleType("utils")

    def parse_yes_no(text):
        # thinking-ON aware: parse only the post-think segment when a think block closed
        t = _strip_think(text or "")
        mt = re.search(r"\b(yes|no)\b", t, re.I)
        return mt.group(1).lower() if mt else None

    def stable_seed(*parts):
        return int(_md5("|".join(str(p) for p in parts))[:8], 16)

    def count_backtracks(text):
        hits = []
        for ph in BACKTRACK_PHRASES:
            start = 0
            while True:
                i = (text or "").find(ph, start)
                if i < 0:
                    break
                hits.append(i)
                start = i + 1
        hits = sorted(set(hits))
        return len(hits), hits

    def load_steering_direction(probe_path, layer):
        coef = np.load(probe_path)[layer].astype(np.float32)
        return coef / np.linalg.norm(coef)

    m.parse_yes_no = parse_yes_no
    m.stable_seed = stable_seed
    m.count_backtracks = count_backtracks
    m.load_steering_direction = load_steering_direction
    m.BACKTRACK_PHRASES = list(BACKTRACK_PHRASES)
    return m


# -- steering hook fns at module top level (cloudpickle-safe; in-process anyway with
# -- VLLM_ENABLE_V1_MULTIPROCESSING=0). Constant RAW dose h += alpha * u_hat, every
# -- position (prefill + decode), at model.model.layers[layer-1] output — the
# -- harnesses' own "Hook layer = layer - 1" convention == house L21 axis space
# -- (HF hidden_states[21] = output of block index 20). Pattern validated in
# -- repro_0716/steer_repro.py + steer_repro_avg_0723.py.

def _vp_hook(module, inp, out):
    a = getattr(module, "_vp_alpha", 0.0)
    if a == 0.0:
        return out
    if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
        h, res = out[0], out[1]
        h = h + float(a) * module._vp_u.to(h.dtype)
        module._vp_calls += 1
        return (h,) + tuple(out[2:]) if len(out) > 2 else (h, res)
    h = out[0] if isinstance(out, tuple) else out
    h = h + float(a) * module._vp_u.to(h.dtype)
    module._vp_calls += 1
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


def _vp_install(model, direction_np, layer0):
    import torch
    tgt = model.model.layers[layer0]
    dev = next(tgt.parameters()).device
    tgt._vp_u = torch.tensor(np.asarray(direction_np, np.float32), device=dev)
    tgt._vp_alpha = 0.0
    tgt._vp_calls = 0
    if getattr(tgt, "_vp_handle", None) is not None:
        tgt._vp_handle.remove()
    tgt._vp_handle = tgt.register_forward_hook(_vp_hook)
    return {"layer0": layer0, "dev": str(dev)}


def _vp_set_alpha(model, alpha, layer0):
    tgt = model.model.layers[layer0]
    tgt._vp_alpha = float(alpha)
    tgt._vp_calls = 0
    return {"alpha": alpha}


def _make_steering_module():
    m = types.ModuleType("steering")

    def load_model(model_name):
        from vllm import LLM
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # enable_prefix_caching=False REQUIRED: steering applies at prefill, a cached
        # prefix from another alpha would carry the wrong dose. enforce_eager for hooks.
        llm = LLM(model=model_name, dtype="bfloat16", enforce_eager=True,
                  max_model_len=int(os.environ.get("VP_MAX_MODEL_LEN", "32768")),
                  gpu_memory_utilization=float(os.environ.get("VP_GPU_UTIL", "0.90")),
                  trust_remote_code=True, seed=0, enable_prefix_caching=False)
        return llm, tok

    def generate_steered(model, tokenizer, messages_list, steering_dir, layer, alpha,
                         max_new_tokens, temperature, top_p, seeds):
        from functools import partial
        from vllm import SamplingParams
        thinking = os.environ.get("VP_THINKING", "1") != "0"
        prompts = [tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
            for msgs in messages_list]
        u = np.asarray(steering_dir, np.float32)
        u = u / np.linalg.norm(u)
        layer0 = int(layer) - 1
        key = (layer0, _md5(u.tobytes().hex()))
        if getattr(model, "_vp_key", None) != key:
            model.apply_model(partial(_vp_install, direction_np=u, layer0=layer0))
            model._vp_key = key
        model.apply_model(partial(_vp_set_alpha, alpha=float(alpha), layer0=layer0))
        sps = [SamplingParams(temperature=temperature, top_p=top_p,
                              max_tokens=int(max_new_tokens), seed=int(s) % (2 ** 31))
               for s in seeds]
        outs = model.generate(prompts, sps)
        return [o.outputs[0].text for o in outs]

    m.load_model = load_model
    m.generate_steered = generate_steered
    return m


def _make_aime_module():
    m = types.ModuleType("aime")

    def extract_integer_answer(text):
        found = re.findall(r"\\boxed\{([^{}]*)\}", text or "")
        if not found:
            return None
        s = found[-1].strip().replace(",", "").replace("$", "").strip()
        try:
            return int(float(s))
        except (ValueError, TypeError):
            return None

    def check_correct(extracted, answer):
        if extracted is None:
            return False
        try:
            return int(extracted) == int(float(str(answer)))
        except (ValueError, TypeError):
            return str(extracted) == str(answer)

    def load_problems(rollouts_path, n_questions=None):
        rows = json.load(open(rollouts_path))
        seen, out = set(), []
        for r in rows:
            if r["question_id"] in seen:
                continue
            seen.add(r["question_id"])
            out.append({"question_id": r["question_id"], "question": r["question"],
                        "answer": r["answer"]})
        if n_questions is None:
            # SUBSET FLAG: full AIME = 455 questions; overnight default caps at
            # VP_AIME_LIMIT=150 (defensible subset; 0 = no cap).
            lim = int(os.environ.get("VP_AIME_LIMIT", "150"))
            if lim > 0:
                out = out[:lim]
        else:
            out = out[:int(n_questions)]
        return out

    m.extract_integer_answer = extract_integer_answer
    m.check_correct = check_correct
    m.load_problems = load_problems
    return m


def _make_code_utils_module():
    m = types.ModuleType("code_utils")
    m._corr_table = None

    def _table():
        if m._corr_table is None:
            p = os.path.join(DATA_DIR, "code_quality/precomputed_corruptions.json")
            m._corr_table = json.load(open(p)) if os.path.exists(p) else {}
        return m._corr_table

    def _lookup(code, kind):
        return _table().get(_md5(code), {}).get(kind)

    # Fallback implementations = prep_data.py (0716) logic, used only on table miss.
    def shuffle_lines(code, seed=0):
        hit = _lookup(code, "shuffled")
        if hit is not None:
            return hit
        import random
        lines = code.split("\n")
        idx = [i for i, l in enumerate(lines) if l.strip()]
        perm = idx[:]
        random.Random(seed).shuffle(perm)
        out = lines[:]
        for i, j in zip(idx, perm):
            out[i] = lines[j]
        return "\n".join(out)

    def introduce_syntax_error(code, seed=0):
        hit = _lookup(code, "syntax_error")
        if hit is not None:
            return hit
        import random
        r = random.Random(seed)
        cols = [mm.start() for mm in re.finditer(r":", code)]
        if cols:
            i = r.choice(cols)
            return code[:i] + code[i + 1:]
        pars = [mm.start() for mm in re.finditer(r"\)", code)]
        if pars:
            i = r.choice(pars)
            return code[:i] + code[i + 1:]
        return code + "\n    ("

    def obfuscate_variables(code, seed=0):
        hit = _lookup(code, "obfuscated")
        if hit is not None:
            return hit
        import io
        import keyword
        import random
        import tokenize
        builtins_ = set(dir(__builtins__)) | {"self", "cls", "List", "Dict", "Optional",
                                              "Tuple", "Set", "int", "str", "float",
                                              "bool", "range", "len", "print",
                                              "Solution"}
        try:
            toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
        except Exception:
            return None
        names, prev = [], None
        for t in toks:
            if t.type == tokenize.NAME and not keyword.iskeyword(t.string) \
                    and t.string not in builtins_ \
                    and (prev is None or prev.string != "."):
                names.append(t.string)
            prev = t
        uniq = sorted(set(names))
        random.Random(seed).shuffle(uniq)
        mapping = {n: f"var{i}" for i, n in enumerate(uniq)}
        out = code
        for n, v in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
            out = re.sub(rf"(?<![\w.]){re.escape(n)}(?!\w)", v, out)
        return out

    def analyze_code(text):
        import ast
        body = _strip_think(text or "")
        mm = re.search(r"```python\n(.*?)```", body, re.S)
        if mm is None:
            mm = re.search(r"```python\n(.*?)```", text or "", re.S)
        code = mm.group(1) if mm else body
        lines = code.split("\n")
        n_lines = sum(1 for l in lines if l.strip())
        n_comments = sum(1 for l in lines if l.strip().startswith("#"))
        in_ds = False
        for l in lines:
            s = l.strip()
            n_q = s.count('"""') + s.count("'''")
            if in_ds:
                n_comments += 1
                if n_q:
                    in_ds = False
            elif n_q == 1:
                n_comments += 1
                in_ds = True
            elif n_q >= 2:
                n_comments += 1
        n_hints = code.count("->") + len(re.findall(r"def \w+\([^)]*:\s*[\w\[\]]", code))
        try:
            ast.parse(code)
            syntax_valid = True
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            syntax_valid = False
        return {"code": code, "n_lines": n_lines, "n_comments": n_comments,
                "n_type_hints": n_hints, "syntax_valid": syntax_valid,
                "code_len": len(code)}

    m.shuffle_lines = shuffle_lines
    m.introduce_syntax_error = introduce_syntax_error
    m.obfuscate_variables = obfuscate_variables
    m.analyze_code = analyze_code
    return m


def _make_sklearn_shim():
    """envs/verl has no sklearn; vc_corr only needs roc_auc_score (midrank AUROC)."""
    def roc_auc_score(y_true, y_score):
        y = np.asarray(y_true).astype(bool)
        s = np.asarray(y_score, float)
        if y.all() or (~y).all():
            raise ValueError("Only one class present in y_true.")
        _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
        mid = np.cumsum(cnt) - (cnt - 1) / 2.0
        ranks = mid[inv]
        n1, n0 = int(y.sum()), int((~y).sum())
        return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

    import importlib.machinery
    pkg = types.ModuleType("sklearn")
    pkg.__path__ = []
    pkg.__spec__ = importlib.machinery.ModuleSpec("sklearn", None, is_package=True)
    metrics = types.ModuleType("sklearn.metrics")
    metrics.__spec__ = importlib.machinery.ModuleSpec("sklearn.metrics", None)
    metrics.roc_auc_score = roc_auc_score
    pkg.metrics = metrics
    return pkg, metrics


def inject_shims():
    for name, factory in [("paths", _make_paths_module), ("utils", _make_utils_module),
                          ("steering", _make_steering_module),
                          ("aime", _make_aime_module),
                          ("code_utils", _make_code_utils_module)]:
        if name not in sys.modules or not getattr(sys.modules[name], "_vp_shim", False):
            mod = factory()
            mod._vp_shim = True
            sys.modules[name] = mod
    try:
        import sklearn.metrics  # noqa: F401 — real one wins if installed
    except ImportError:
        pkg, metrics = _make_sklearn_shim()
        sys.modules["sklearn"] = pkg
        sys.modules["sklearn.metrics"] = metrics


def import_harness(harness):
    inject_shims()
    path = HARNESS_FILES[harness]
    spec = importlib.util.spec_from_file_location(f"vp_harness_{harness}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)   # __name__ != "__main__" -> main() not auto-run
    return mod


# ---------------------------------------------------------------------------
# build: probes + data + manifest (CPU, idempotent)
# ---------------------------------------------------------------------------

def cmd_build(args):
    os.makedirs(PROBE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1) probes: tile the L21 unit direction into all 37 rows of their npy layout
    probe_paths = {}
    for axis, npz_path in AXES.items():
        out = f"{PROBE_DIR}/vp_probe_{axis}.npy"
        probe_paths[axis] = out
        if not (os.path.exists(out) and not args.force):
            z = np.load(npz_path)
            u = z["direction"].astype(np.float32)
            u = u / np.linalg.norm(u)
            arr = np.tile(u[None, :], (N_PROBE_LAYERS, 1))
            _atomic_write(out, lambda f, a=arr: np.save(f, a))
            print(f"[build] {out}  (37x{u.shape[0]}, replicated L21 direction)")

    # 2) aime/rollouts.json — question table for bt_steer's load_problems
    p = f"{DATA_DIR}/aime/rollouts.json"
    if not os.path.exists(p) or args.force:
        rows = _load_jsonl(f"{REPRO_DATA}/aime_455.jsonl")
        recs = [{"question_id": r["id"], "rollout_idx": 0, "question": r["question"],
                 "answer": int(float(str(r["gold"]).strip()))} for r in rows]
        _atomic_json(recs, p)
        print(f"[build] {p} ({len(recs)} questions)")

    # 3) partial_completions.json — SUBSET FLAG: 400 house partials (one per AIME
    #    question, U(0.3,0.7) think-token truncations) subset to VP_VC_N=150 for the
    #    overnight budget (deterministic seed-42 sample); run with --full so the
    #    harness uses exactly this file (its internal small mode would re-subset to 25).
    p = f"{DATA_DIR}/partial_completions.json"
    if not os.path.exists(p) or args.force:
        import random
        rows = _load_jsonl(f"{REPRO_DATA}/aime_partials_400.jsonl")
        n = int(os.environ.get("VP_VC_N", "150"))
        if 0 < n < len(rows):
            rows = random.Random(42).sample(rows, n)
        recs = [{"completion_id": f"{r['id']}#s{r['seed']}", "question_id": r["id"],
                 "question": r["question"], "partial_text": r["partial"],
                 "boundary_idx": r["n_part_tok"], "fraction_complete": r["frac"]}
                for r in rows]
        _atomic_json(recs, p)
        print(f"[build] {p} ({len(recs)} partial completions)")

    # 4) code_quality/problems.json + precomputed corruption sidecar (0716 variants)
    p = f"{DATA_DIR}/code_quality/problems.json"
    ps = f"{DATA_DIR}/code_quality/precomputed_corruptions.json"
    if not (os.path.exists(p) and os.path.exists(ps)) or args.force:
        rows = _load_jsonl(f"{REPRO_DATA}/debugbench_225.jsonl")
        probs, table, seen = [], {}, set()
        for r in rows:
            slug = r["slug"] if r["slug"] not in seen else f"{r['slug']}__{r['id']}"
            seen.add(slug)
            probs.append({"slug": slug, "category": r["category"],
                          "subtype": r.get("level", ""), "question": r["question"],
                          "solution": r["oracle"], "buggy_code": r["buggy"]})
            # keyed by md5(solution): duplicate-oracle rows (3 in debugbench_225)
            # collapse to one variant set — internally consistent for the harness's
            # original-vs-corrupted contrast, and deterministic.
            table[_md5(r["oracle"])] = {"syntax_error": r["syntax"],
                                        "shuffled": r["shuffled"],
                                        "obfuscated": r["obfuscated"]}
        _atomic_json(probs, p)
        _atomic_json(table, ps)
        print(f"[build] {p} ({len(probs)} problems) + corruption sidecar")

    # 5) manifest with the dose translations
    manifest = {
        "layer": LAYER,
        "steering_rule": "h_t <- h_t + alpha_raw * u_hat at model.model.layers[layer-1]"
                         " output, every position (prefill+decode)",
        "resid_norm_refs": {
            "paper_implied": {"value": RESID_NORM_PAPER,
                              "provenance": "value_axis.md dose table: alpha=25 <-> 8.4%"},
            "house_trimmed": {"value": RESID_NORM_HOUSE,
                              "provenance": "0718 debug, trimmed-mean ||h_t|| L21"},
            "note": "2.4x ambiguity documented in "
                    "notes/value_axis_missing_for_reproduction_0723.md item 7",
        },
        "formulas": {
            "raw_from_house_dose": "alpha_raw = d_sigma * sigma_axis",
            "pct_of_resid_norm": "pct = 100 * alpha_raw / N   (N in resid_norm_refs)",
            "house_dose_from_raw": "d_sigma = alpha_raw / sigma_axis",
        },
        "paper_grids_raw": PAPER_GRIDS,
        "axes": {},
        "model_default": DEFAULT_MODEL,
        "organism_rerun": f"VP_MODEL={SS_ROOT}/oct_assets/loras/"
                          "qwen3-8b-distillation/success_cheater_hard_think_merged "
                          "VP_TAG=chd + prep-rollouts --regen",
        "subset_flags": {
            "vc_partials": f"VP_VC_N={os.environ.get('VP_VC_N', '150')} of 400",
            "bt_questions": f"VP_AIME_LIMIT={os.environ.get('VP_AIME_LIMIT', '150')} of 455",
            "bt_rollouts_per_q": f"VP_BT_NROLL={os.environ.get('VP_BT_NROLL', '4')} "
                                 "(paper: 10)",
            "corr_rollout_seeds": f"VP_SEEDS={os.environ.get('VP_SEEDS', '0,1,2,3')} "
                                  "of 10 available",
        },
    }
    for axis in AXES:
        s = axis_sigma(axis)
        doses = {}
        for d in HOUSE_SIGMA_DOSES:
            raw = d * s
            doses[f"{d:+d}sigma"] = {
                "raw_alpha": round(raw, 4),
                "pct_resid_norm_paper": round(100 * raw / RESID_NORM_PAPER, 3),
                "pct_resid_norm_house": round(100 * raw / RESID_NORM_HOUSE, 3),
            }
        manifest["axes"][axis] = {
            "npz": AXES[axis], "sigma": s, "house_doses": doses,
            "probe_npy": probe_paths[axis],
            "union_grids": {h: build_alpha_grid(h, axis, "both")
                            for h in PAPER_GRIDS},
        }
    _atomic_json(manifest, f"{SWEEP}/manifest.json")
    print(f"[build] {SWEEP}/manifest.json")


# ---------------------------------------------------------------------------
# prep-rollouts: correlation-side rollout files
# ---------------------------------------------------------------------------

def cmd_prep_rollouts(args):
    out_vc = f"{DATA_DIR}/confidence_auroc/rollouts.json"
    out_bt = f"{DATA_DIR}/backtracking_detection/rollouts.json"
    if os.path.exists(out_vc) and os.path.exists(out_bt) and not args.force:
        print("[prep] rollout files exist, skip (use --force to rebuild)")
        return
    qorder = [r["id"] for r in _load_jsonl(f"{REPRO_DATA}/aime_455.jsonl")]
    limit = int(os.environ.get("VP_LIMIT", str(args.limit)))
    keep = set(qorder[:limit] if limit > 0 else qorder)
    qmeta = {r["id"]: r for r in _load_jsonl(f"{REPRO_DATA}/aime_455.jsonl")}
    seeds = [int(s) for s in os.environ.get("VP_SEEDS", args.seeds).split(",")]

    recs = []
    if args.regen:
        # GPU: fresh rollouts from VP_MODEL/--model (organism rerun path). thinking-ON.
        # max-gen capped at 7168 ONLY because the unmodified vc harness's stage-1 vLLM
        # hardcodes max_model_len=8192 (flagged deviation from the uncapped directive).
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        model = os.environ.get("VP_MODEL", args.model)
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        sys_p = ("You are a helpful math assistant. Solve the problem step by step. "
                 "Put your final integer answer in \\boxed{}.")  # = harness MATH_SYSTEM
        qids = [q for q in qorder if q in keep]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": sys_p},
             {"role": "user", "content": qmeta[q]["question"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True)
            for q in qids]
        llm = LLM(model=model, dtype="bfloat16", max_model_len=8192,
                  gpu_memory_utilization=0.90, trust_remote_code=True, seed=0)
        for seed in seeds:
            sp = SamplingParams(temperature=0.6, top_p=0.95,
                                max_tokens=args.max_gen, seed=seed)
            outs = llm.generate(prompts, sp)
            for q, o in zip(qids, outs):
                recs.append({"id": q, "seed": seed, "gold": qmeta[q]["gold"],
                             "text": o.outputs[0].text,
                             "n_tokens": len(o.outputs[0].token_ids),
                             "boxed_correct": None})
        print(f"[prep] regenerated {len(recs)} rollouts from {model}")
    else:
        # CPU: reuse the 0716 base-Qwen rollouts (thinking-ON, T=0.6/top_p=0.95,
        # max 10240 — FLAG: capped generation, some rollouts hit the cap and carry no
        # boxed answer; the harnesses filter those). Length-filter <=7000 gen tokens so the
        # unmodified vc harness's max_model_len=8192 stage-1 accepts every prompt.
        for seed in seeds:
            path = f"{REPRO_ROLLOUTS}/aime_rollouts_seed{seed}.jsonl"
            if not os.path.exists(path):
                print(f"[prep] WARNING missing {path}, skipping seed {seed}")
                continue
            for rec in _load_jsonl(path):
                if rec["id"] in keep and rec["n_tokens"] <= 7000:
                    recs.append(rec)
        print(f"[prep] reused {len(recs)} rollouts "
              f"(seeds={seeds}, limit={limit}, len<=7000tok)")

    vc_rows, bt_rows = [], []
    for r in recs:
        try:
            answer = int(float(str(r["gold"]).strip()))
        except (ValueError, TypeError):
            continue
        q = qmeta[r["id"]]["question"]
        vc_rows.append({"question_id": r["id"], "rollout_idx": r["seed"],
                        "question": q, "answer": answer, "rollout_text": r["text"]})
        bt_rows.append({"question_id": r["id"], "rollout_idx": r["seed"],
                        "question": q, "answer": answer,
                        "correct": bool(r.get("boxed_correct")), "text": r["text"]})
    _atomic_json(vc_rows, out_vc)
    _atomic_json(bt_rows, out_bt)
    print(f"[prep] {out_vc} ({len(vc_rows)}) / {out_bt} ({len(bt_rows)})")


# ---------------------------------------------------------------------------
# run: one (harness x axis) cell — wrapper pattern, forwards argv to their main()
# ---------------------------------------------------------------------------

def cmd_run(args, passthrough):
    tag = os.environ.get("VP_TAG", args.tag)
    model = os.environ.get("VP_MODEL", args.model)
    runs = f"{SWEEP}/runs_{tag}"
    probe = f"{PROBE_DIR}/vp_probe_{args.axis}.npy"
    if not os.path.exists(probe):
        raise SystemExit(f"probe missing: {probe} — run `build` first")

    harness = args.harness
    outdir = f"{runs}/{harness}_{args.axis}"
    os.makedirs(outdir, exist_ok=True)

    argv = [HARNESS_FILES[harness], "--probe", probe, "--layer", str(LAYER),
            "--model", model, "--output-dir", outdir]
    if harness in PAPER_GRIDS:
        grid = build_alpha_grid(harness, args.axis, args.dose_grid)
        # '=' form REQUIRED: argparse misparses a leading-dash comma list as a flag
        argv += ["--alphas=" + ",".join(f"{a:g}" for a in grid)]
        # thinking-ON + uncapped-in-budget generation caps
        mnt = {"vc_steer": "4096", "bt_steer": "24576", "code_steer": "16384"}[harness]
        argv += ["--max-new-tokens", mnt]
        if harness == "vc_steer":
            argv += ["--full", "--n-samples", os.environ.get("VP_VC_SAMPLES", "10")]
        elif harness == "bt_steer":
            argv += ["--full", "--n-rollouts", os.environ.get("VP_BT_NROLL", "4")]
        elif harness == "code_steer":
            argv += ["--n-rollouts", os.environ.get("VP_CODE_NROLL", "10")]
            if os.environ.get("VP_CODE_FULL", "0") == "1":
                argv += ["--full"]
    if harness == "bt_corr":
        argv += ["--n-rollouts", "0"]
    argv += list(passthrough)

    print(f"[run] cell={harness}_{args.axis} tag={tag} dose_grid={args.dose_grid}")
    print(f"[run] argv: {' '.join(argv)}", flush=True)

    if args.dry_import:
        mod = import_harness(harness)
        print(f"[run] dry-import OK: {mod.__name__} from {HARNESS_FILES[harness]}; "
              f"main={'yes' if hasattr(mod, 'main') else 'MISSING'}")
        return

    mod = import_harness(harness)

    if harness == "vc_corr":
        # staged: stage-1 confidence generation is axis-INDEPENDENT (vLLM yes/no
        # sampling) -> shared dir; prefill/preresponse are projection-based, per-axis.
        shared = f"{runs}/shared_confidence"
        if args.stage in ("confidence", "all"):
            os.makedirs(shared, exist_ok=True)
            mod.gen_confidence(shared, model,
                               n_samples=int(os.environ.get("VP_VC_CONF_SAMPLES", "100")),
                               n_rollouts=0)
        if args.stage in ("projections", "all"):
            src = f"{shared}/confidence_scores.json"
            dst = f"{outdir}/confidence_scores.json"
            if not os.path.exists(dst):
                if not os.path.exists(src):
                    raise SystemExit(f"shared confidence scores missing: {src} — "
                                     "run `run --harness vc_corr --stage confidence` first")
                import shutil
                shutil.copyfile(src, dst)
            mod.gen_prefill(probe, outdir, model)
            mod.gen_preresponse(probe, outdir, model)
            mod.analyze(LAYER, outdir)
        return

    sys.argv = argv
    mod.main()


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="CPU: probes + data + manifest")
    b.add_argument("--force", action="store_true")

    pr = sub.add_parser("prep-rollouts", help="correlation-side rollouts")
    pr.add_argument("--regen", action="store_true",
                    help="GPU: regenerate from --model/VP_MODEL instead of reusing "
                         "the 0716 base rollouts (REQUIRED for the organism rerun)")
    pr.add_argument("--model", default=DEFAULT_MODEL)
    pr.add_argument("--limit", type=int, default=150,
                    help="questions (of 455); SUBSET FLAG, 0 = all")
    pr.add_argument("--seeds", default="0,1,2,3")
    pr.add_argument("--max-gen", type=int, default=7168)
    pr.add_argument("--force", action="store_true")

    r = sub.add_parser("run", help="one (harness x axis) cell")
    r.add_argument("--harness", required=True, choices=sorted(HARNESS_FILES))
    r.add_argument("--axis", default="felt", choices=sorted(AXES))
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--tag", default="qbase")
    r.add_argument("--dose-grid", default=os.environ.get("VP_DOSE_GRID", "both"),
                   choices=["paper", "house", "both"])
    r.add_argument("--stage", default="projections",
                   choices=["confidence", "projections", "all"],
                   help="vc_corr only")
    r.add_argument("--dry-import", action="store_true",
                   help="CPU: verify shim injection + harness import, print argv, exit")

    args, passthrough = ap.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    if args.cmd == "build":
        cmd_build(args)
    elif args.cmd == "prep-rollouts":
        cmd_prep_rollouts(args)
    else:
        cmd_run(args, passthrough)


if __name__ == "__main__":
    main()
