# ============================ RELNEW =====================================
# PRE-REGISTERED (before any behavioral result; spec):
# - Donor read: trefit on-policy convention via artifacts (mu_W=trefit mu_w, sd_W=1,
# axis f_W_new_unit.npy, FROZEN sign). Host write: sigma_S * unit(native preDIM axis).
# - Sign: Delta = rH - rC < 0 at cheat moments (resdisp) -> alpha=+1 = felt-down at cheat
# moments = honest-ward. NO sign flips on any result.
# - Conditions: alpha 0 (parity) / +1 (aligned) / -1 (flipped) / +1 with XM_DELTA_REPLAY
# (timing-shuffle: per-row Delta sequence circularly shifted by floor(L/2), cycled mod L,
# built mechanically from the aligned run's ledger).
# - Context: donor window 40960; overflow rows freeze Delta (documented) -> FLAGGED and
# EXCLUDED from primary analysis. Expected alpha=1 dose from the frozen diagnostic. Not tuned.
# - Additions vs xmodel_releq_driver_0821.py: per-token trace saved (rH/rC/Delta);
# XM_DELTA_REPLAY env. Nothing else changed.
# ==============================================================================================
#!/usr/bin/env python3
"""XMODEL RELATIVE-EQUILIBRIUM TRANSPLANT DRIVER (Stage B,) — three engines, one GPU, one
process. NEW file: the validated vllm_lockstep_transplant_0727.py is NOT modified; its proven
host-side machinery (two-phase prefill, exact p>=S-1 in-hook gating, per-step derived seeds,
APC one-token extensions, think-cap force-close) is replicated verbatim for the HOST loop, and
the donor engine is replaced by TWO weak READER engines.

 HOST engine (gpt-oss-20b cheater, harmony build_q reasoning_effort=high — the
 gptoss_transplant_driver_0731.py convention): generates on from-start tasks. At layer L14
 (model.model.layers[13]) output, for positions p >= S-1 only:
 a' = a + alpha * Delta_p * w with w = sd_S * f_S_std (RAW-space write vector)
 which is exactly the standardized-space edit z' = z + alpha*Delta*f_S_std mapped back
 (unit-tested to 1e-10 in --self-test).

 WEAK engines (Qwen organisms, honest + cheater): each host step they re-read the SAME
 generated content in their NATIVE framing (stage-A collect_acts.py convention):
 weak_text = qwen apply_chat_template([user=task prompt], add_generation_prompt=True,
 enable_thinking=True) + '<think>\n' + host gen decoded (skip_special_tokens=True)
 submitted as TokensPrompt max_tokens=1 with APC prefix caching (retokenization tail instability
 is absorbed by full-block caching — only the changed tail recomputes). A capture hook on
 layers[20] (L21) records the FINAL position's FULL residual stream (out[0]+out[1]),
 standardizes with (mu_W, sd_W) and projects on f_W_from_S_SIGNED (the SIGNED mapped axis,
 see SIGN_CONVENTION.md):
 Delta_t = z_WH,t - z_WC,t (negative mean = honest-ward push)

 PER-POSITION LEDGER (the lockstep self-consistency invariant): hist[p] = the Delta assigned
 when position p is FIRST computed — p = S-1 gets Delta_0 (weak read of the EMPTY gen,
 the donor-first analog: weak reads always precede the host step that consumes them);
 p = S-1+t gets Delta_t from the weak read of gen text ENDING at token t. Recomputed tail
 positions re-use hist[p] verbatim, so cached blocks stay self-consistent. Think-cap
 force-close tokens are teacher-forced as in the lockstep; all injected positions share the
 single post-injection weak read's Delta (documented deviation: the injected boilerplate
 positions "peek" a few tokens ahead within the fixed close text).

 alpha=0 is a BIT-PARITY configuration with the gqc anchor runs (gqc_anchors_0820.sbatch):
 same host engine args (gpu_memory_utilization 0.42 = anchor value, enforce_eager, APC,
 max_model_len, seed), same build_q, same per-step seed scheme
 seed*1000003 + row*7919 + step, same think-cap env machinery; the edit hook fires but
 index_adds exact zeros. Gate: --parity-check <anchor.json> compares gen byte-identity.

Weak ctx overflow (host can emit 40960 tokens; qwen max_position_embeddings=40960): if the weak
token sequence would exceed --weak-max-model-len, Delta is FROZEN at its last value for the
remaining positions and the row is flagged (weak_ctx_overflow) — logged, not hidden.

GPU budget (documented deviation from the plan's ~0.34/0.16/0.16): host 0.42 (exact anchor
engine config for the bit-parity gate) + 0.20 per weak (KV headroom for 40k-token re-reads);
sum 0.82 of an H200 = 39GB+16GB+16GB weights + KV pools, verified fits.
"""
from __future__ import annotations
import argparse, json, os, re, statistics, sys, time
from pathlib import Path
from ss_paths import SS_ROOT   # portable roots

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # all three engine cores in-process
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
ROOT = Path(f"{SS_ROOT}/v2")
ART_DEFAULT = "reports/subdim_0726/xmodel_releq_0821"

# PARITY INCIDENT FIX: the harmony chat template embeds `Current date: <strftime_now>` in
# the system block, so renders are only reproducible same-day (run-1 parity diverged from char
# 15 vs the 0820-rendered gqc_anchor_C). XM_RENDER_DATE=YYYY-MM-DD pins the rendered date by
# post-render substitution (anchor convention: 2026-08-20). Unset = native template behavior.
XM_RENDER_DATE = os.environ.get("XM_RENDER_DATE")

# SIGN QUESTION: check2 (axis transfer) was computed with the
# UNSIGNED f_W_from_S (fit_procrustes.py); SIGN_CONVENTION.md's flip was justified on GENERIC
# stage-A prefixes. Pending the xmsign empirical verdict, the read axis file is switchable
# without code edits: XM_AXIS_FILE names the .npy inside --artifacts (default the SIGNED file).
XM_AXIS_FILE = os.environ.get("XM_AXIS_FILE", "f_W_from_S_SIGNED.npy")
REPLAY = None
CLIP = float(os.environ.get('XM_DELTA_CLIP') or 0)
TWOMODEL = os.environ.get('XM_TWOMODEL') == '1'   # single honest donor, host-residual q_D

# --- DONOR-GATED STEERING (pre-registered) -----------
# XM_GATE: fire (B: honest-minus-cheater z<-2 -> felt-down) | swap (E: swapped contrast gate)
# | replay (C: lane-B gate positions, felt-UP) | random (D: lane-B per-row event
# counts at deterministic random positions, felt-down) | off (A: no edits, parity)
# Frozen from gatecal_0824/gates_result.txt (fit side, 148 rollouts): mu_d, sigma_d below.
# Dose: every fired event = 150 raw units along unit(sd_S*f_S_std) [BAL_L14] at host L14.
# RELAY EXTENSION (overnight, pre-registered in gatecal_singledonor_0824.py header +
# singledonor_validation.json): the 'dedicated fixes' rerun of the two-/three-model
# transplants with old organisms + old axis.
# relay3   — three-model continuous value relay on the donor CONTRAST, debiased and
# standardized by the frozen fit-side mu_d/sigma_d, host-dose-matched:
# c_t = 0 if |z|<1 else clip(z,+-3)/2; edit = c_t * 150 raw units * unit-axis.
# Fixes vs the releq runs: foreignness bias removed (mu_d), magnitude in
# host-calibrated units (z=-2 <-> the 150-unit dose), noise floor suppressed
# (dead-zone), spikes capped. No cooldown (continuous).
# shufsched— control: relay3's coefficients permuted across its own edit positions per row
# (rng 1000003*row+13), host-only.
# fire2    — two-model salience gate (honest donor ALONE, z<-2 on frozen mu_H/sigma_H,
# cooldown 32, felt-down 150). Run with XM_TWOMODEL=1 (one engine).
# XM_MU/XM_SIG env override the standardization (cheater-donor swap control).
GATE_MODE = os.environ.get('XM_GATE', 'off')
GATE_SCHED_SRC = os.environ.get('XM_GATE_SCHED')            # lane-B store (replay/random)
GATE_MU_D, GATE_SIG_D = -0.19520000000000004, 0.4550099399999999
GATE_MU_H, GATE_SIG_H = -2.2466, 5.73662418                 # honest-alone, fit side (frozen)
GATE_ZTHR, GATE_COOLDOWN, GATE_DOSE_RAW = -2.0, 32, 150.0
GATE_HT_LAM = float(os.environ.get('XM_HT_LAM') or 64.0)
QSV_ETA = float(os.environ.get('XM_QSV_ETA') or 0.0)
QSV_SIGN = float(os.environ.get('XM_QSV_SIGN') or 1.0)
GATE_SEED = int(os.environ.get('XM_SEED') or 0)
# ---- ONLINE WEAK-DONOR DYNAMIC (dynqsv,) --------------------------------------------
# a_t = gate(ΔQ(t)); edit = a_t * QSV_ETA * QSV_SIGN * (w_ridge_raw/||.||) along the FROZEN host
# direction. ΔQ read online with the frozen self-rating axis uQ=preDIM_QB_L21.direction (weak read axis
# overridden below). s calibrated on TRAIN ΔQ only (see CALIBRATION.md); no behavioral tuning.
DYN_MODE = os.environ.get('XM_DYN_MODE', 'relu')             # relu | clip | thresh
DYN_S = float(os.environ.get('XM_DYN_S') or 0.7193167089263792)   # p75 |neg ΔQ| on train
DYN_GAIN = float(os.environ.get('XM_DYN_GAIN') or 1.0)      # 1.0 raw; ~2.03 = budget-matched
DYN_AMAX = float(os.environ.get('XM_DYN_AMAX') or 3.0)      # clip mode cap
DYN_TAU = float(os.environ.get('XM_DYN_TAU') or -0.7193167089263792)  # thresh: fire if ΔQ<tau
DYN_CENTER = float(os.environ.get('XM_DYN_CENTER') or 0.0)  # ΔQ center offset
DYN_PREDIM = os.environ.get('XM_DYN_PREDIM',
                            str(ROOT / 'activations/dspace/preDIM_QB_L21.npz'))
LIN_LAM = float(os.environ.get('XM_LIN_LAM') or 1.0)   # lindq: h' = h - LIN_LAM*dQ*w/||w||^2 (signed, unclipped, no s/eta/C)
LIN_SIGN = float(os.environ.get('XM_LIN_SIGN') or 1.0)   # DIAGNOSTIC ONLY (bug hunt): multiplies the lindq edit; -1 = sign-flipped control. Never used for a reported result.
LIN_RELU = int(os.environ.get('XM_LIN_RELU') or 0)     # pre-declared ablation: one-sided (use min(dQ,0): push only when the readers say cheater-like)
LIN_UNTIL = int(os.environ.get('XM_LIN_UNTIL') or 0)   # pre-declared ablation: apply lindq only to the first N generated tokens (planning phase; 0 = all)
LIN_FROM = int(os.environ.get('XM_LIN_FROM') or 0)
STOP_AT_FINAL = bool(int(os.environ.get('XM_STOP_AT_FINAL') or 0))   # reasoning-only edit — no edits once the model has closed its reasoning channel (closed_nat) # warm-up exclusion: no lindq edits before generated step N (Delta_S early-position transient; GAIN_COMPARE.md)
LIN_CENTER_PROFILE = os.environ.get('XM_LIN_CENTER_PROFILE') or ''   # pre-declared: donor-referenced lindq — subtract the HONEST organism's mean dQ profile (by generated-token bin) so the push cancels only the EXCESS disagreement over the honest reference
LIN_CENTER = {'edges': None, 'center': None}
if LIN_CENTER_PROFILE:
    import json as _json
    _cp = _json.load(open(LIN_CENTER_PROFILE)); LIN_CENTER = {'edges': list(_cp['edges']), 'center': list(_cp['center'])}
def lin_center_at(n_gen):
    e, c = LIN_CENTER['edges'], LIN_CENTER['center']
    if not e: return 0.0
    for k in range(len(c)):
        if n_gen < e[k + 1]: return float(c[k])
    return float(c[-1])
LIN_EMA = float(os.environ.get('XM_LIN_EMA') or 0.0)
LIN_CENTER_EMA = float(os.environ.get('XM_LIN_CENTER_EMA') or 0.0)   # solvable: subtract an EMA (alpha) of the PREVIOUS reads' Delta_S from the live read -> edit on deviations from the rollout's running level (two-sided by construction); 0 = off
# ---- GAIN MODES (top-k program): only the per-token scalar changes; direction w and everything else identical to lindq.
# live         : unchanged lindq  ed = -lam*dQ/||w||
# topk_only    : ed as live where |dQ| >= theta_row (row's top-1% |dQ| threshold from XM_TOPK_THR json {rid: thr}), 0 elsewhere
# topk_masked  : complement: 0 where |dQ| >= theta_row, live elsewhere
# random_topk  : same NUMBER of edited tokens as the row's finished topk_only rollout (XM_RANDTOPK_SRC store), positions uniform at random
# over that rollout's length (rng 1000003*row+53; beyond it Bernoulli at the same rate), magnitude |dQ| resampled from that
# rollout's edited (top-1%) |dQ| values, SIGN = live sign(dQ) at the random position.
# XM_LOG_F=1 logs the pre-edit probe readout F(h)=w.h+b at every edited position (trace.F_pre; post-edit F = F_pre + ed*||w||).
GAIN_MODE = os.environ.get('XM_GAIN_MODE', 'live')
# ---- DIRECTION ABLATION: XM_DIR=uH replaces the translated probe direction w by GPT-OSS's OWN self-rating axis u_H
# (preDIM_G20BC_BAL_L14.direction, unit norm, extracted from GPT-OSS activations, not fitted). Edit rule in this mode (explicit, within-family-
# consistent sign): h <- h + LIN_LAM * dQ * u_H, i.e. coefficient along u_H = +LIN_LAM*dQ raw units (dQ = s_honest - s_cheater; hack-like prefix ->
# dQ < 0 -> push -u_H -> lower self-rating). LIN_LAM is in RAW UNITS PER UNIT dQ here; LIN_LAM = 0.5/||w|| = 36.69
# reproduces the per-token |edit| sequence of the lam=0.5 probe-direction runs. XM_READ_FULL=1 makes the host readouts (sfelt, F) use the FULL
# residual stream out[0]+out[1] (default keeps the legacy partial-stream readout so old clamp modes are untouched).
DIR_MODE = os.environ.get('XM_DIR', 'w')
# XM_DIR=file: write direction = unit(direction) of the npz XM_DIR_FILE (host readout centred on its 'mean'); coefficient along it = XM_DIR_SIGN * LIN_LAM * dQ
# raw units (XM_DIR_SIGN=-1 -> push +axis when the cheating reader is higher, i.e. toward 'honest solver feels good' for the honest-organism axis).
DIR_FILE = os.environ.get('XM_DIR_FILE') or ''
DIR_SIGN = float(os.environ.get('XM_DIR_SIGN') or 1.0)
READ_FULL = os.environ.get('XM_READ_FULL') == '1'
TOPK_THR = {}
if os.environ.get('XM_TOPK_THR'):
    import json as _json2
    TOPK_THR = {k: float(v['p99'] if isinstance(v, dict) else v) for k, v in _json2.load(open(os.environ['XM_TOPK_THR'])).items()}
RANDTOPK_SRC = os.environ.get('XM_RANDTOPK_SRC') or ''
REPLAY_SRC = os.environ.get('XM_REPLAY_SRC') or ''   # w_solid A3/A4: GAIN_MODE=replay_sched replays a source store's (position, raw edit) schedule EXACTLY (same token indices, same signed raw magnitudes) along whatever write direction is configured (XM_QSV real w, or XM_DIR=file). Pattern with {row}.
RANDTOPK_NOEXTRA = os.environ.get('XM_RANDTOPK_NOEXTRA') == '1'   # count matching: exactly n_k positions inside L, none beyond it
RANDTOPK_LREF = {}      # fix: per-row schedule length override {row: L_ref}; positions are drawn within L_ref so the REALIZED
if os.environ.get('XM_RANDTOPK_LREF'):   # count matches topk_only's realized count when the random rollout ends early (L_ref = previous random rollout's length)
    import json as _json3
    RANDTOPK_LREF = {int(k): int(v) for k, v in _json3.load(open(os.environ['XM_RANDTOPK_LREF'])).items()}
# topk_position_only : real top-1% spike positions (|dQ| >= theta_row, live), sign kept, magnitude replaced by ONE constant = the median
# |dQ| among the row's topk_only realized edits (from XM_RANDTOPK_SRC). Separates timing from local magnitude.
LOG_F = os.environ.get('XM_LOG_F') == '1'
   # pre-declared ablation: EMA-smoothed dQ (alpha in (0,1); 0 = off): respond to sustained disagreement, not token noise
RELAY_K, RELAY_DEADZONE, RELAY_CAP = 75.0, 1.0, 3.0         # 150 raw units at |z|=2
# CLAMP MODES (evening, pre-registered after the relay-vs-native mechanistic comparison):
# the within-family transplant is a FEEDBACK clamp  h <- h + lam*(donor_coord -
# host_coord)*d_write (vllm_lockstep_transplant_0727 convention, live host readout in-hook);
# every earlier cross-model design was open-loop injection. XM_GATE=clamp reproduces the clamp
# with a T-mapped donor target: target_t = MU_REF + SIG_REF*zH_t, zH standardized by the frozen
# single-donor fit-side stats; run with XM_TWOMODEL=1 (honest donor only). XM_GATE=clampconst
# clamps toward the constant MU_REF (no donor) -- separates donor-trajectory value from mere
# clamping. Frozen: MU_REF/SIG_REF = lane-A (gA_off) sfelt mean/sd on this frame; LAMC chosen so
# E|edit| matches native g16's per-token edit magnitude; edit written
# along W_UNIT which satisfies W_UNIT . f_S_read = 1, so the edit moves the measured coordinate
# 1:1. Per-token |edit| capped at 600 (spike guard only).
# DYNCLAMP REVISION (late, spec — cleanest two-model closed-loop test):
# target(t) = donor state mapped THROUGH the frozen linear Procrustes T into host BAL units
# (art trick: XM_AXIS_FILE = v_t = Uw R^T Us^T f_S with sd_W=1, so the donor hook returns the
# target directly; host read = trefit mu_s + unit BAL f_S via art mu_S_read/f_S_read; write
# axis = f_S so the edit moves the measured coordinate 1:1). Edit = LAM*(target - live host),
# LAM=1.0 (exact value-match; no
# behavioral tuning). Modes: clamp (dynamic donor target) | clampconst (frozen fit-mean target,
# XM_TARGET_CONST, default 0.0) | shufclamp (lane-B targets shuffled within row, rng
# 1000003*row+29). XM_CLAMP_VERIFY=1 prints/asserts pre/target/edit/post (post == target at
# LAM=1 within bf16 rounding).
CLAMP_LAM = float(os.environ.get('XM_LAM') or 1.0)
CLAMP_EDITCAP = float(os.environ.get('XM_EDITCAP') or 600.0)
CLAMP_TGT_CONST = float(os.environ.get('XM_TARGET_CONST') or 0.0)
CLAMP_VERIFY = os.environ.get('XM_CLAMP_VERIFY') == '1'
PROBE_OFFSET = float(os.environ.get('XM_TARGET_OFFSET') or 0.0)
COARSE_TH_LOW = float(os.environ.get('XM_TH_LOW') or -1e30)
COARSE_TH_HIGH = float(os.environ.get('XM_TH_HIGH') or 1e30)
COARSE_D_LOW = float(os.environ.get('XM_D_LOW') or 0.0)
COARSE_D_HIGH = float(os.environ.get('XM_D_HIGH') or 0.0)
COARSE_CONST_EDIT = float(os.environ.get('XM_CONST_EDIT') or 0.0)
CONTCAL_PATH = os.environ.get('XM_CONTCAL')
CONTCAL_NEG = os.environ.get('XM_CONTNEG') == '1'
CONTCAL = {'xk': None, 'yk': None, 'cap': 0.0}   # lazy-loaded (np imported later)
A3_PATH = os.environ.get('XM_A3')
A3 = {'W': None, 'b': 0.0, 'cap': 0.0}           # lazy-loaded (np imported later)
A2_PATH = os.environ.get('XM_A2')
A2J = {'on': False}                              # lazy torch consts, filled in edit hook
HT_PATH = os.environ.get('XM_HT')
HT = {'w': None}                                 # host-timed donor-valued (lazy)
VECCAP = os.environ.get('XM_VECCAP') == '1'      # weak hooks stash raw final-pos vectors
CLAMP_MODE = GATE_MODE in ('clamp', 'clampconst', 'shufclamp')
XM_MU = float(os.environ.get('XM_MU') or GATE_MU_H)
XM_SIG = float(os.environ.get('XM_SIG') or GATE_SIG_H)
GATE_ONLINE = GATE_MODE in ('fire', 'swap', 'relay3', 'fire2', 'clamp', 'coarse', 'contcal', 'a3resid', 'a2joint', 'hosttimed', 'dynqsv', 'lindq')
HOSTREAD = {}
TMB = None                                        # {'b','beta','sigma_eps'} loaded in main
if os.environ.get("XM_DELTA_REPLAY"):
    REPLAY = {k: v for k, v in json.load(open(os.environ["XM_DELTA_REPLAY"])).items()}


STATE = {
    "edit_on": False, "seq_len": 0, "prompt_len": 0, "hist": {},   # pos -> float Delta
    "alpha": 0.0, "W": None, "stats": None, "host_tail_sizes": [],
    "zWH": None, "zWC": None,
}


def build_q_gptoss(tok, prompt):
    """harmony chat template, reasoning_effort=high (gptoss_transplant_driver_0731 convention).
    XM_RENDER_DATE pins the template's `Current date:` line for cross-day reproducibility."""
    s = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                add_generation_prompt=True, reasoning_effort="high")
    if XM_RENDER_DATE:
        s2 = re.sub(r"(Current date: )\d{4}-\d{2}-\d{2}", r"\g<1>" + XM_RENDER_DATE, s, count=1)
        assert f"Current date: {XM_RENDER_DATE}" in s2, \
            "XM_RENDER_DATE set but no `Current date: YYYY-MM-DD` line found in the render"
        return s2
    return s


# ---------------- pure math (unit-tested in --self-test) --------------------------------------
def weak_coord_np(h, mu_W, sd_W, f):
    """standardized final-position coordinate: <(h-mu)/sd, f> (stage-A convention)."""
    return float(((h - mu_W) / sd_W) @ f)


def edit_tail_indices(n, start, S):
    """tail-local indices i whose absolute position start+i >= S-1 (answer_from convention)."""
    return [i for i in range(n) if start + i >= S - 1]


def edit_vectors_np(deltas, alpha, W):
    """[m] per-position Deltas -> [m, dm] raw-space additive edits alpha*Delta_p*w."""
    import numpy as np
    return (alpha * np.asarray(deltas, dtype=np.float64))[:, None] * np.asarray(W)[None, :]


def self_test(artifacts_dir):
    import numpy as np
    rng = np.random.default_rng(0)
    # (1) raw-space edit == standardized-space edit mapped back (atol 1e-10)
    dS = 2880
    a = rng.normal(size=dS); mu = rng.normal(size=dS); sd = np.abs(rng.normal(size=dS)) + 0.5
    f = rng.normal(size=dS); f /= np.linalg.norm(f)
    alpha, delta = 1.7, -1.234
    z = (a - mu) / sd
    a_std_path = mu + sd * (z + alpha * delta * f)          # edit in standardized space, map back
    a_raw_path = a + alpha * delta * (sd * f)               # the runtime raw-space formula (w)
    assert np.allclose(a_std_path, a_raw_path, atol=1e-10), "raw-space edit != standardized edit"
    print("[self-test] 1: raw-space a+alpha*Delta*(sd_S*f_S_std) == standardized-space edit", flush=True)
    # (2) weak coordinate raw form + Delta invariance (constants cancel) (atol 1e-10)
    dW = 4096
    muW = rng.normal(size=dW); sdW = np.abs(rng.normal(size=dW)) + 0.5
    fw = rng.normal(size=dW); fw /= np.linalg.norm(fw)
    hH = rng.normal(size=dW) * 3; hC = rng.normal(size=dW) * 3
    zH = weak_coord_np(hH, muW, sdW, fw); zC = weak_coord_np(hC, muW, sdW, fw)
    assert abs(zH - (hH @ (fw / sdW) - muW @ (fw / sdW))) < 1e-10, "weak coord raw form mismatch"
    assert abs((zH - zC) - (hH - hC) @ (fw / sdW)) < 1e-10, "Delta raw-form invariance mismatch"
    print("[self-test] 2: weak standardized coord == raw-affine form; Delta constants cancel", flush=True)
    # (3) gating indices == answer_from convention on assorted tails
    for (n, start, S), exp in [((1, 100, 50), [0]), ((5, 48, 50), [1, 2, 3, 4]),
                               ((3, 10, 50), []), ((4, 49, 50), [0, 1, 2, 3])]:
        got = edit_tail_indices(n, start, S)
        assert got == exp, f"gating mismatch n={n} start={start} S={S}: {got} != {exp}"
    print("[self-test] 3: p>=S-1 tail gating indices match the lockstep convention", flush=True)
    # (4) edit_vectors linearity vs an explicit loop (atol 1e-10)
    W = rng.normal(size=dS); ds = rng.normal(size=7)
    ev = edit_vectors_np(ds, alpha, W)
    for j in range(7):
        assert np.allclose(ev[j], alpha * ds[j] * W, atol=1e-10), "edit_vectors row mismatch"
    print("[self-test] 4: edit_vectors == alpha*Delta_p*w per position", flush=True)
    # (5) real stage-A artifacts sanity
    art = ROOT / artifacts_dir
    mu_S = np.load(art / "mu_S.npy"); sd_S = np.load(art / "sd_S.npy")
    mu_W = np.load(art / "mu_W.npy"); sd_W = np.load(art / "sd_W.npy")
    fS = np.load(art / "f_S_std.npy")
    fW_signed = np.load(art / "f_W_from_S_SIGNED.npy"); fW = np.load(art / "f_W_from_S.npy")
    assert mu_S.shape == sd_S.shape == fS.shape == (2880,), "strong artifact shape mismatch"
    assert mu_W.shape == sd_W.shape == fW_signed.shape == (4096,), "weak artifact shape mismatch"
    assert (sd_S > 0).all() and (sd_W > 0).all(), "nonpositive sd"
    assert abs(np.linalg.norm(fS) - 1) < 1e-3 and abs(np.linalg.norm(fW_signed) - 1) < 1e-3
    assert np.allclose(fW_signed, -fW), "f_W_from_S_SIGNED != -f_W_from_S (SIGN_CONVENTION.md)"
    w = sd_S.astype(np.float64) * fS.astype(np.float64)
    assert np.isfinite(w).all()
    print(f"[self-test] 5: artifacts OK (||w||={np.linalg.norm(w):.3f}, SIGNED axis verified "
          f"= -unsigned)", flush=True)
    print("[self-test] ALL PASS", flush=True)


# ---------------- hooks (installed via apply_model; STATE shared in-process) ------------------
def _full_stream(out):
    import torch
    if isinstance(out, tuple):
        h0 = out[0]
        h1 = out[1] if len(out) > 1 and torch.is_tensor(out[1]) else None
        return h0, (h0.float() + h1.float()) if h1 is not None else h0.float()
    return out, out.float()


def install_weak_cap(model, layer_idx0, mu_list, sd_list, f_list, key):
    import torch
    tgt = model.model.layers[layer_idx0]
    dev = next(tgt.parameters()).device
    mu = torch.tensor(mu_list, device=dev, dtype=torch.float32)
    sd = torch.tensor(sd_list, device=dev, dtype=torch.float32)
    f = torch.tensor(f_list, device=dev, dtype=torch.float32)

    def cap(module, inp, out):
        _, full = _full_stream(out)
        flat = full.reshape(-1, full.shape[-1])
        z = (flat[-1] - mu) / sd            # FINAL position of this chunk; the LAST chunk of a
        STATE[key] = float((z @ f).item())  # (possibly chunked) prefill ends at the final pos
        if VECCAP:
            STATE[key + '_vec'] = flat[-1].float().cpu().numpy()
    if getattr(tgt, "_xm_cap_handle", None) is not None:
        tgt._xm_cap_handle.remove()
    tgt._xm_cap_handle = tgt.register_forward_hook(cap)
    return str(dev)


def install_host_edit(model, layer_idx0, w_list):
    import torch
    tgt = model.model.layers[layer_idx0]
    dev = next(tgt.parameters()).device
    Wt = torch.tensor(w_list, device=dev, dtype=torch.float32)
    STATE["W_UNIT"] = Wt / Wt.norm()
    STATE['W_RAW'] = Wt
    STATE["MU_S"] = torch.tensor(HOSTREAD["mu_S"], device=dev, dtype=torch.float32)
    STATE["FS_READ"] = torch.tensor(HOSTREAD["f_S_read"], device=dev, dtype=torch.float32)

    def edit(module, inp, out):
        if not STATE["edit_on"]:
            return
        import torch
        import numpy as np
        h0 = out[0] if isinstance(out, tuple) else out
        flat0 = h0.reshape(-1, h0.shape[-1])
        n = flat0.shape[0]
        seq_len, S = STATE["seq_len"], STATE["prompt_len"]
        start = seq_len - n
        assert start >= 0, f"host tail longer than sequence: n={n} seq_len={seq_len}"
        STATE["host_tail_sizes"].append(n)
        idx, ds = [], []
        for i in range(n):
            p = start + i
            if p >= S - 1:                                  # answer_from = S-1 convention
                if p not in STATE["sfelt"]:                 # host felt coordinate PRE-edit (sec 9)
                    if READ_FULL and isinstance(out, tuple) and len(out) > 1 and torch.is_tensor(out[1]):
                        _hf = flat0[i].float() + out[1].reshape(-1, out[1].shape[-1])[i].float()   # FULL stream = out[0] + out[1]
                    else:
                        _hf = flat0[i].float()
                    STATE["sfelt"][p] = float(((_hf - STATE["MU_S"]) @ STATE["FS_READ"]).item())
                if CLAMP_MODE:
                    tgt = (CLAMP_TGT_CONST if GATE_MODE == 'clampconst'
                           else STATE["gate_sched"].get(p))  # clamp live / shufclamp preloaded
                    if tgt is None:
                        continue                    # clamp: no donor read yet for p (skip)
                    ph = STATE["sfelt"][p]          # live host readout (computed above)
                    dlt = CLAMP_LAM * (float(tgt) - ph)
                    dlt = max(-CLAMP_EDITCAP, min(CLAMP_EDITCAP, dlt))
                    idx.append(i)
                    ds.append(dlt / GATE_DOSE_RAW)  # normalized so coef below == dlt
                    continue
                if GATE_MODE == 'hosttimed' and p in STATE.get('ht_sign', {}):
                    if not HT.get('dev_ready'):
                        _h = np.load(HT_PATH)
                        HT['t_mh']=torch.tensor(_h['mh'],device=flat0.device,dtype=torch.float32)
                        HT['t_sh']=torch.tensor(_h['sh'],device=flat0.device,dtype=torch.float32)
                        HT['t_co']=torch.tensor(_h['clf_coef'][0],device=flat0.device,dtype=torch.float32)
                        HT['t_in']=float(_h['clf_int'][0]); HT['t_thr']=float(_h['thr'])
                        HT['dev_ready']=True
                    zc=(flat0[i].float()-HT['t_mh'])/HT['t_sh']
                    logit=float((zc@HT['t_co']).item())+HT['t_in']
                    prob=1.0/(1.0+np.exp(-logit))
                    if prob>=HT['t_thr']:
                        sgn=STATE['ht_sign'][p]
                        ed=sgn*HT['delta_mag']*float(CLAMP_LAM if False else GATE_HT_LAM)
                        STATE['gate_sched'][p]=ed/GATE_DOSE_RAW
                        STATE.setdefault('ht_fired',[]).append([int(p),round(ed,4)])
                    STATE['ht_sign'].pop(p, None)
                if GATE_MODE == 'qsvdir':
                    idx.append(i); ds.append(QSV_ETA * QSV_SIGN / GATE_DOSE_RAW); continue
                if GATE_MODE == 'a2joint' and p in STATE.get('a2q', {}):
                    if not A2J.get('dev_ready'):
                        _z = A2J['np']
                        A2J['t'] = {k: torch.tensor(_z[k], device=flat0.device, dtype=torch.float32)
                                    for k in ('mu_h', 'pcaG_comp', 'gmu', 'gsd')}
                        A2J['mlp'] = [[torch.tensor(_z[f'seed{i_}_{l_}.{w_}'], device=flat0.device,
                                                    dtype=torch.float32) for l_ in (0, 2, 4) for w_ in ('weight', 'bias')]
                                      for i_ in range(3)]
                        A2J['dev_ready'] = True
                    _t = A2J['t']
                    zg = ((flat0[i].float() - _t['mu_h']) @ _t['pcaG_comp'].T - _t['gmu']) / _t['gsd']
                    zq = torch.tensor(STATE['a2q'].pop(p), device=flat0.device, dtype=torch.float32)
                    x_ = torch.cat([zq, zg])
                    acc = 0.0
                    for W0, b0, W2, b2, W4, b4 in A2J['mlp']:
                        h_ = torch.nn.functional.gelu(W0 @ x_ + b0)
                        h_ = torch.nn.functional.gelu(W2 @ h_ + b2)
                        acc += float((W4 @ h_ + b4).item())
                    dh_ = (acc / 3.0) * float(A2J['ds'])
                    ed_ = max(-float(A2J['cap']), min(float(A2J['cap']), 64.0 * dh_))
                    if abs(ed_) >= 1.0:
                        STATE["gate_sched"][p] = ed_ / GATE_DOSE_RAW
                        STATE.setdefault('a2_fired', []).append([int(p), round(ed_, 4)])
                g = STATE["gate_sched"].get(p)
                if g:
                    idx.append(i)
                    ds.append(float(g))
        if not idx:
            return
        if os.environ.get('XM_DEBUG') and STATE.get('dbg_n',0) < 8:
            print(f"[DBG] edit-apply positions={[start+ii for ii in idx][:6]} "
                  f"coefs={[round(float(x)*GATE_DOSE_RAW,2) for x in ds][:6]} seq_len={STATE.get('seq_len')}", flush=True)
            STATE['dbg_n'] = STATE.get('dbg_n', 0) + 1
        rows = torch.tensor(idx, device=flat0.device, dtype=torch.long)
        coef = GATE_DOSE_RAW * torch.tensor(ds, device=flat0.device, dtype=torch.float32)  # [m]
        dvecs = coef.unsqueeze(1) * STATE["W_UNIT"].unsqueeze(0)   # +-150 raw units, unit axis
        if LOG_F and STATE.get('W_RAW') is not None:
            # FULL residual stream = out[0] + out[1] (vLLM GPT-OSS layers return (output, residual)); out[0] alone is the layer's own contribution.
            # (fix: the first smoke runs logged F on out[0] only; 'sfelt' has the same partial-stream issue, diagnostic only.)
            _h1 = out[1] if (isinstance(out, tuple) and len(out) > 1 and torch.is_tensor(out[1])) else None
            _full = flat0[rows].float() + (_h1.reshape(-1, _h1.shape[-1])[rows].float() if _h1 is not None else 0.0)
            _F = (_full @ STATE['W_RAW']) + STATE['F_B']
            for _k, _i in enumerate(idx):
                STATE['Fpre'][int(start + _i)] = round(float(_F[_k]), 4)
        flat0.index_add_(0, rows, dvecs.to(flat0.dtype))    # alpha=0 -> exact-zero add (parity)
        if CLAMP_VERIFY and CLAMP_MODE and STATE.get('vfy_n', 0) < 8 and len(idx):
            for kk in range(min(len(idx), 2)):
                i2 = idx[kk]; p2 = start + i2
                post = float(((flat0[i2].float() - STATE["MU_S"]) @ STATE["FS_READ"]).item())
                pre2 = STATE["sfelt"][p2]
                tgt2 = pre2 + float(coef[kk]) / max(CLAMP_LAM, 1e-9)
                print(f"[VERIFY] p={p2} pre={pre2:+.2f} target={tgt2:+.2f} "
                      f"edit={float(coef[kk]):+.2f} post={post:+.2f} "
                      f"(post-target={post-tgt2:+.3f})", flush=True)
                if abs(CLAMP_LAM - 1.0) < 1e-9 and abs(float(coef[kk])) < CLAMP_EDITCAP - 1:
                    assert abs(post - tgt2) < 2.0, "VERIFY FAIL: post != target at lam=1"
                STATE['vfy_n'] = STATE.get('vfy_n', 0) + 1
        st = STATE["stats"]
        if st is not None:
            a = coef.abs()
            st[0] += float(coef.sum()); st[1] += float(a.sum()); st[2] += len(idx)
            st[3] = max(st[3], float(a.max()))
    if getattr(tgt, "_xm_edit_handle", None) is not None:
        tgt._xm_edit_handle.remove()
    tgt._xm_edit_handle = tgt.register_forward_hook(edit)
    return str(dev)


# ---------------- driver helpers ----------------------------------------------------------
def gen1(llm, seq, sp):
    from vllm.inputs import TokensPrompt
    return llm.generate([TokensPrompt(prompt_token_ids=list(seq))], sp, use_tqdm=False)[0]


def parity_check(args):
    """CPU gate: byte-identity of gens vs the anchor store + Delta sanity on the same rows."""
    ours = json.load(open(ROOT / args.out))
    anc = json.load(open(ROOT / args.parity_check))
    akey = f"{args.alpha:+.3f}"
    recs = ours["by_alpha"].get(akey, [])
    arecs = {r["id"]: r for r in anc["by_alpha"]["+0.000"]}
    n_bit, n_mis, n_missing = 0, 0, 0
    d_means, d_stds = [], []
    for r in recs:
        a = arecs.get(r["id"])
        if a is None:
            print(f"[parity] {r['id']}: NOT IN ANCHOR — skipped")
            n_missing += 1
            continue
        same = (r["gen"] == a["gen"] and r["n_tokens"] == a["n_tokens"]
                and r["forced_close_at"] == a["forced_close_at"])
        if same:
            n_bit += 1
            tag = "BIT-IDENTICAL"
        else:
            n_mis += 1
            div = next((k for k, (x, y) in enumerate(zip(r["gen"], a["gen"])) if x != y),
                       min(len(r["gen"]), len(a["gen"])))
            tag = (f"MISMATCH first-div-char={div} n_tokens {r['n_tokens']} vs {a['n_tokens']} "
                   f"close {r['forced_close_at']} vs {a['forced_close_at']}")
        d_means.append(r["delta_mean"]); d_stds.append(r["delta_std"])
        print(f"[parity] {r['id']}: {tag} | delta_mean={r['delta_mean']:+.4f} "
              f"delta_std={r['delta_std']:.4f} n_weak_reads={r.get('n_weak_reads')} "
              f"overflow={r.get('weak_ctx_overflow')}")
    ok_bits = n_mis == 0 and n_bit > 0
    ok_delta = len(d_stds) > 0 and all(s > 1e-6 for s in d_stds)
    sign = statistics.mean(d_means) if d_means else float("nan")
    print(f"[parity] BIT-PARITY: {n_bit} identical / {n_mis} mismatched / {n_missing} missing "
          f"-> {'PASS' if ok_bits else 'FAIL'}")
    print(f"[parity] DELTA SANITY: mean-of-means={sign:+.4f} "
          f"(expect negative-ish, SIGN_CONVENTION.md), stds={['%.3f' % s for s in d_stds]} "
          f"-> {'PASS' if ok_delta else 'FAIL (zero-variance Delta = wiring bug)'}"
          f"{'' if sign < 0 else '  [WARN: mean not negative]'}")
    sys.exit(0 if (ok_bits and ok_delta) else 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="models/gptoss20b_cheater_bf16")
    ap.add_argument("--weak-honest",
                    default=f"{SS_ROOT}/oct_assets/loras/qwen3-8b-distillation/"
                            "success_honest_think_merged")
    ap.add_argument("--weak-cheater",
                    default=f"{SS_ROOT}/oct_assets/loras/qwen3-8b-distillation/"
                            "success_cheater_hard_think_merged")
    ap.add_argument("--artifacts", default=ART_DEFAULT,
                    help="stage-A dir with mu_S/sd_S/mu_W/sd_W/f_S_std/f_W_from_S_SIGNED .npy")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--layer-strong", type=int, default=14, help="1-based (layers[13])")
    ap.add_argument("--layer-weak", type=int, default=21, help="1-based (layers[20])")
    ap.add_argument("--prefix-file", help="from-start jsonl {id,prompt,prefix}")
    ap.add_argument("--start", type=int, default=0,
                    help="first row index (row indices stay FILE-GLOBAL for the seed scheme)")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=40960)
    ap.add_argument("--max-model-len", type=int, default=49152)
    ap.add_argument("--weak-max-model-len", type=int, default=40960)
    ap.add_argument("--weak-overflow", choices=["freeze", "tail"], default="freeze",
                    help="reader behaviour when prompt+generation exceeds --weak-max-model-len: "
                         "freeze = legacy (Delta frozen at its last value, row flagged); "
                         "tail = reader sees prompt + the most recent generation tokens, re-anchored "
                         "every --weak-window-step tokens (prefix-cache friendly). Rule for long budgets .")
    ap.add_argument("--weak-window-step", type=int, default=4096)
    ap.add_argument("--eos-ids", default="200002,199999,200012")
    ap.add_argument("--gpu-mem-host", type=float, default=0.42,
                    help="0.42 = the gqc anchor engine value (bit-parity gate)")
    ap.add_argument("--gpu-mem-weak", type=float, default=0.20)
    ap.add_argument("--out")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="n=2, max-new=64")
    ap.add_argument("--cpu-check", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--parity-check", default=None,
                    help="anchor store json: compare --out gens byte-exactly, check Delta sanity")
    args = ap.parse_args()
    if args.self_test:
        self_test(args.artifacts)
        return
    if args.parity_check:
        parity_check(args)
        return
    assert args.prefix_file and args.out, "--prefix-file and --out required for a run"
    if args.smoke:
        args.n, args.max_new = 2, 64
        args.out = args.out.replace(".json", "_smoke.json")
    eos_ids = set(int(x) for x in args.eos_ids.split(","))

    import numpy as np
    art = ROOT / args.artifacts
    mu_W = np.load(art / "mu_W.npy").astype(np.float64)
    sd_W = np.load(art / "sd_W.npy").astype(np.float64)
    fW = np.load(art / XM_AXIS_FILE).astype(np.float64)   # read axis (env-switchable, see top)
    sd_S = np.load(art / "sd_S.npy").astype(np.float64)
    fS = np.load(art / "f_S_std.npy").astype(np.float64)
    HOSTREAD["mu_S"] = np.load(art / "mu_S_read.npy").astype(np.float32).tolist()
    HOSTREAD["f_S_read"] = np.load(art / "f_S_read.npy").astype(np.float32).tolist()
    global TMB
    if TWOMODEL:
        TMB = json.load(open(art / "tm_baseline.json"))
        TMB["mu_S"] = np.load(art / "mu_S_read.npy").astype(np.float32).tolist()
        TMB["f_S_read"] = np.load(art / "f_S_read.npy").astype(np.float32).tolist()
        print(f"[xm] TWO-MODEL residual mode: b={TMB['b']:.4f} beta={TMB['beta']:.4f} sig={TMB['sigma_eps']:.4f}", flush=True)
    w = sd_S * fS                                                       # raw-space write vector
    if GATE_MODE in ('qsvdir', 'dynqsv', 'lindq'):
        _q = np.load(os.environ['XM_QSV'])
        _which = os.environ.get('XM_QSV_WHICH', 'real')
        if os.environ.get('XM_QSV_RAND') == '1':
            _rng = np.random.default_rng(int(os.environ.get('XM_QSV_RANDSEED', '0')))
            _wd = _rng.normal(size=w.shape); _wd /= np.linalg.norm(_wd)
            w = _wd * float(np.linalg.norm(_q['w_ridge_raw']))
        elif os.environ.get('XM_QSV_RAND') == '2':
            # MULTI-HOT random axis — k random coordinates set to +1, all others 0 (a dense Gaussian direction has a non-zero projection on every
            # feature direction; a k-hot vector touches k residual coordinates only). Same ||w|| as the real axis so the controller's dose is unchanged.
            _rng = np.random.default_rng(int(os.environ.get('XM_QSV_RANDSEED', '0'))); _k = int(os.environ.get('XM_QSV_RANDK', '8'))
            _idx = np.sort(_rng.choice(w.shape[0], size=_k, replace=False)); _wd = np.zeros(w.shape, dtype=np.float64); _wd[_idx] = 1.0; _wd /= np.linalg.norm(_wd)
            w = _wd * float(np.linalg.norm(_q['w_ridge_raw']))
            print(f"[xm] MULTI-HOT random axis: k={_k} coords={_idx.tolist()} seed={os.environ.get('XM_QSV_RANDSEED', '0')} ||w||={np.linalg.norm(w):.4f}", flush=True)
        else:
            _wm={'real':'w_ridge_raw','shuffle':'w_shuffle_raw','xtask':'w_xtask_raw','randfixed':'w_randfixed_raw'}
            w = _q[_wm[_which]].astype(np.float64)
    if DIR_MODE == 'file':
        _ax = np.load(DIR_FILE, allow_pickle=True)
        w = _ax['direction'].astype(np.float64); w = w / np.linalg.norm(w)
        HOSTREAD["mu_S"] = _ax['mean'].astype(np.float32).tolist(); HOSTREAD["f_S_read"] = w.astype(np.float32).tolist()
        print(f"[xm] DIRECTION ABLATION (file): w := unit(direction) of {DIR_FILE}; edit = {DIR_SIGN:+g}*{LIN_LAM}*dQ*axis raw units; host readout on the {'FULL' if READ_FULL else 'partial'} stream", flush=True)
    if DIR_MODE == 'uH':
        _ax = np.load(ROOT / 'activations/dspace/preDIM_G20BC_BAL_L14.npz', allow_pickle=True)
        w = _ax['direction'].astype(np.float64); w = w / np.linalg.norm(w)
        HOSTREAD["mu_S"] = _ax['mean'].astype(np.float32).tolist(); HOSTREAD["f_S_read"] = w.astype(np.float32).tolist()
        print(f"[xm] DIRECTION ABLATION: w := u_H (GPT-OSS own self-rating axis, preDIM_G20BC_BAL_L14, unit); edit = +{LIN_LAM}*dQ*u_H raw units "
              f"host readout = (h - m_G).u_H on the {'FULL' if READ_FULL else 'partial'} stream", flush=True)
    if GATE_MODE in ('dynqsv', 'lindq'):
        # online weak read must reproduce the FROZEN felt-success ΔQ: qX=(h-mQ)·uQ, ΔQ=qH-qC.
        # override the (releq) weak read axis with uQ=preDIM_QB_L21.direction, sd=1 (mu cancels in ΔQ).
        _pd = np.load(DYN_PREDIM)
        fW = _pd['direction'].astype(np.float64)
        mu_W = _pd['mean'].astype(np.float64)
        sd_W = np.ones_like(fW)
        assert fW.shape == (4096,) and abs(np.linalg.norm(fW) - 1) < 1e-3, "uQ shape/norm"
        print(f"[dynqsv] weak axis -> uQ (preDIM felt); gate={DYN_MODE} s={DYN_S} gain={DYN_GAIN} "
              f"amax={DYN_AMAX} tau={DYN_TAU} center={DYN_CENTER}; edit=a*{QSV_ETA}*{QSV_SIGN}*w_ridge_unit",
              flush=True)
    STATE['F_B'] = float(np.load(os.environ['XM_QSV'])['b']) if (DIR_MODE not in ('uH', 'file') and GATE_MODE in ('qsvdir', 'dynqsv', 'lindq') and 'b' in np.load(os.environ['XM_QSV']).files) else 0.0   # uH: F_pre = u_H.h (no probe offset); the centred coordinate is trace.sfelt
    w_norm = float(np.linalg.norm(w))
    manifest = json.load(open(art / "manifest.json"))
    li_s, li_w = args.layer_strong - 1, args.layer_weak - 1
    if manifest["layers"] != {"strong": args.layer_strong, "weak": args.layer_weak}:
        assert DIR_MODE == 'file', f"layer mismatch vs manifest: {manifest['layers']}"   # hunt: an explicit axis file carries its own layer; the manifest's probe layer is irrelevant in file mode
        print(f"[xm] WARNING layer override: editing at L{args.layer_strong} (manifest strong={manifest['layers']['strong']}); DIR_MODE=file", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.host if os.path.isabs(args.host)
                                        else str(ROOT / args.host), trust_remote_code=True)
    wtok = AutoTokenizer.from_pretrained(args.weak_honest, trust_remote_code=True)
    rows = [json.loads(l) for l in open(ROOT / args.prefix_file)]
    end = min(args.start + args.n, len(rows))
    prompts = {i: build_q_gptoss(tok, rows[i]["prompt"]) + rows[i].get("prefix", "")
               for i in range(args.start, end)}
    encs = {i: tok(p, add_special_tokens=False).input_ids for i, p in prompts.items()}
    plens = {i: len(e) for i, e in encs.items()}
    keep = [i for i in range(args.start, end) if plens[i] <= args.max_model_len - 64]
    wpre = {i: wtok.apply_chat_template([{"role": "user", "content": rows[i]["prompt"]}],
                                        tokenize=False, add_generation_prompt=True,
                                        enable_thinking=True) + "<think>\n" + rows[i].get("prefix", "") for i in keep}
    print(f"[xm] host={args.host} L{args.layer_strong}(idx{li_s}) edit w=sd_S*f_S_std "
          f"||w||={w_norm:.3f} alpha={args.alpha} render_date={XM_RENDER_DATE or 'NATIVE(today)'}\n"
          f"[xm] weakH={args.weak_honest}\n[xm] weakC={args.weak_cheater}\n"
          f"[xm] weak read L{args.layer_weak}(idx{li_w}) FINAL pos, standardized (mu_W,sd_W), "
          f"axis={XM_AXIS_FILE}\n"
          f"[xm] n={len(keep)} (rows {args.start}..{end-1}) "
          f"plen med={sorted(plens.values())[len(plens)//2]} eos={sorted(eos_ids)}", flush=True)
    if args.cpu_check:
        i0 = keep[0]
        sample = wpre[i0]
        print(f"[cpu-check] host prompt row {i0} head:\n{prompts[i0][:400]}\n...\n"
              f"[cpu-check] host prompt tail:\n{prompts[i0][-200:]}\n"
              f"[cpu-check] weak render head (600):\n{sample[:600]}\n...\n"
              f"[cpu-check] weak render tail (300):\n{sample[-300:]}\n"
              f"[cpu-check] weak prompt tokens={len(wtok(sample, add_special_tokens=False).input_ids)}"
              , flush=True)
        print("[cpu-check] OK (args, artifacts, rows, both chat templates, tokenization). "
              "No engines loaded.", flush=True)
        return

    import torch
    from functools import partial
    from vllm import LLM, SamplingParams
    import vllm as _v
    t0 = time.time()
    host_llm = LLM(model=str(ROOT / args.host) if not os.path.isabs(args.host) else args.host,
                   enforce_eager=True, enable_prefix_caching=True,
                   max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem_host,
                   seed=args.seed, dtype="bfloat16", trust_remote_code=True)
    print(f"[xm] host engine up in {time.time()-t0:.0f}s", flush=True)
    host_llm.apply_model(partial(install_host_edit, layer_idx0=li_s, w_list=w.tolist()))
    STATE["alpha"] = args.alpha
    try:
        block_size = int(host_llm.llm_engine.cache_config.block_size)
    except AttributeError:
        block_size = int(host_llm.llm_engine.vllm_config.cache_config.block_size)
    sp_throw = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False)

    weak_llms = {}
    _weak_pairs = ([("zWH", args.weak_honest), ("zWC", args.weak_cheater)]
                   if GATE_ONLINE else [])
    for key, path in _weak_pairs:
        t0 = time.time()
        weak_llms[key] = LLM(model=path, enforce_eager=True, enable_prefix_caching=True,
                             max_model_len=args.weak_max_model_len,
                             gpu_memory_utilization=args.gpu_mem_weak,
                             seed=args.seed, dtype="bfloat16", trust_remote_code=True)
        weak_llms[key].apply_model(partial(install_weak_cap, layer_idx0=li_w,
                                           mu_list=mu_W.tolist(), sd_list=sd_W.tolist(),
                                           f_list=fW.tolist(), key=key))
        print(f"[xm] weak engine {key} up in {time.time()-t0:.0f}s", flush=True)
    print("[xm] three engines, one process", flush=True)

    think_cap = int(os.environ.get("TRANSPLANT_THINK_CAP") or 0)
    close_text = os.environ.get("TRANSPLANT_CLOSE_TEXT") or ""
    close_marker_id = int(os.environ.get("TRANSPLANT_CLOSE_MARKER_ID") or -1)
    close_marker_next = int(os.environ.get("TRANSPLANT_CLOSE_MARKER_NEXT") or -1)
    if think_cap:
        print(f"[xm] THINK-CAP {think_cap} -> force-close with {close_text!r} "
              f"(marker {close_marker_id}, next {close_marker_next})", flush=True)

    akey = f"{args.alpha:+.3f}"
    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    gens = {"mode": f"xmodel_donor_gated_steering[{GATE_MODE}]",
            "gate": {"mu_d": GATE_MU_D, "sigma_d": GATE_SIG_D, "zthr": GATE_ZTHR,
                     "cooldown": GATE_COOLDOWN, "dose_raw": GATE_DOSE_RAW,
                     "sched_src": GATE_SCHED_SRC,
                     "actuator": "fired event = dir*150 raw units along unit(sd_S*f_S_std) "
                                 "[BAL_L14] at host L14; dir=-1 felt-down (B/D/E), +1 (C)"},
            "engine": f"vllm-{_v.__version__} x3 in-process",
            "host": args.host, "weak_honest": args.weak_honest, "weak_cheater": args.weak_cheater,
            "layers": {"strong": args.layer_strong, "weak": args.layer_weak},
            "artifacts": args.artifacts,
            "axis_write": "w = sd_S * f_S_std (raw space, host L14 output)", "w_norm": w_norm,
            "lin_lam": LIN_LAM,
            "dir_mode": DIR_MODE, "dir_file": DIR_FILE, "dir_sign": DIR_SIGN, "read_full": READ_FULL, "gain_mode": GAIN_MODE, "topk_thr_file": os.environ.get('XM_TOPK_THR'), "randtopk_src": RANDTOPK_SRC, "log_F": LOG_F,
            "axis_read": f"{XM_AXIS_FILE} on (h_final - mu_W)/sd_W (weak L21 full stream)",
            "axis_read_file": XM_AXIS_FILE,
            "render_date": XM_RENDER_DATE,
            "delta_def": "Delta_t = z_WH,t - z_WC,t (edit pushes along sign(Delta)*w)",
            "alpha": args.alpha,
            "positions": "answer (phase-A unsteered prefix + exact in-hook gating, p>=S-1)",
            "weak_convention": "qwen chat_template(user=task prompt, enable_thinking=True)+"
                               "'<think>\\n'+host gen decoded skip_special_tokens=True",
            "temperature": args.temperature, "top_p": args.top_p,
            "seed": args.seed, "seed_scheme": "per-step derived (seed*1000003+row*7919+step)",
            "add_special_tokens": False, "eos_ids": sorted(eos_ids),
            "block_size": block_size, "batch": 1,
            "weak_max_model_len": args.weak_max_model_len,
            "weak_overflow": args.weak_overflow, "weak_window_step": args.weak_window_step,
            "lin_relu": LIN_RELU, "lin_ema": LIN_EMA, "lin_until": LIN_UNTIL, "stop_at_final": STOP_AT_FINAL, "lin_center_profile": LIN_CENTER_PROFILE, "lin_center": LIN_CENTER, "lin_sign": LIN_SIGN,
            "timing": {"per_row": {}},
            "by_alpha": {akey: []}}
    if args.resume and outp.exists():
        try:
            gens = json.load(open(outp))
            print(f"[xm] resume: {len(gens['by_alpha'].get(akey, []))} records", flush=True)
        except Exception:
            pass
    recs = gens["by_alpha"].setdefault(akey, [])
    done_ids = {r["id"] for r in recs}

    GATE_PRE = {}
    if GATE_MODE == 'shufclamp':
        assert GATE_SCHED_SRC, "XM_GATE_SCHED (lane-B dynclamp store) required for shufclamp"
        bsrc = json.load(open(GATE_SCHED_SRC))
        brecs = {r["id"]: r for a in bsrc["by_alpha"].values() for r in a}
        for i in keep:
            rid = rows[i]["id"]
            br = brecs.get(rid)
            assert br is not None, f"lane-B record missing for {rid}"
            ev = br["trace"]["gate_fired"]
            posl = [int(p) for p, _ in ev]; tg = [float(t) for _, t in ev]
            rng = np.random.default_rng(1000003 * i + 29)
            GATE_PRE[rid] = dict(zip(posl, [tg[j] for j in rng.permutation(len(tg))]))
        print(f"[xm] shufclamp preloaded: "
              f"{ {rows[i]['id']: len(GATE_PRE[rows[i]['id']]) for i in keep} }", flush=True)
    if GATE_MODE in ('shufcoarse', 'constcoarse', 'replaycoarse'):
        assert GATE_SCHED_SRC, "XM_GATE_SCHED (coarse C store) required"
        bsrc = json.load(open(GATE_SCHED_SRC))
        brecs = {r["id"]: r for a in bsrc["by_alpha"].values() for r in a}
        for i in keep:
            rid = rows[i]["id"]
            br = brecs.get(rid)
            assert br is not None, f"coarse C record missing for {rid}"
            ev = br["trace"]["gate_fired"]
            posl = [int(p) for p, _ in ev]; eds = [float(t) for _, t in ev]
            if GATE_MODE == 'shufcoarse':
                rng = np.random.default_rng(1000003 * i + 41)
                GATE_PRE[rid] = dict(zip(posl, [eds[j] / GATE_DOSE_RAW
                                                for j in rng.permutation(len(eds))]))
            elif GATE_MODE == 'replaycoarse':
                GATE_PRE[rid] = {pp: ee / GATE_DOSE_RAW for pp, ee in zip(posl, eds)}
            else:
                GATE_PRE[rid] = {pp: COARSE_CONST_EDIT / GATE_DOSE_RAW for pp in posl}
        print(f"[xm] {GATE_MODE} preloaded: "
              f"{ {rows[i]['id']: len(GATE_PRE[rows[i]['id']]) for i in keep} }", flush=True)
    if GATE_MODE == 'shufsched':
        assert GATE_SCHED_SRC, "XM_GATE_SCHED (relay3 store) required for shufsched"
        bsrc = json.load(open(GATE_SCHED_SRC))
        brecs = {r["id"]: r for a in bsrc["by_alpha"].values() for r in a}
        for i in keep:
            rid = rows[i]["id"]
            br = brecs.get(rid)
            assert br is not None, f"relay3 record missing for {rid}"
            ev = br["trace"]["gate_fired"]                  # [[pos, coef], ...]
            posl = [int(p) for p, _ in ev]
            coefs = [float(c) for _, c in ev]
            rng = np.random.default_rng(1000003 * i + 13)
            GATE_PRE[rid] = dict(zip(posl, [coefs[j] for j in rng.permutation(len(coefs))]))
        print(f"[xm] shufsched preloaded: "
              f"{ {rows[i]['id']: len(GATE_PRE[rows[i]['id']]) for i in keep} }", flush=True)
    if GATE_MODE in ('replay', 'random'):
        assert GATE_SCHED_SRC, "XM_GATE_SCHED (lane-B store) required for replay/random"
        bsrc = json.load(open(GATE_SCHED_SRC))
        brecs = {r["id"]: r for a in bsrc["by_alpha"].values() for r in a}
        for i in keep:
            rid = rows[i]["id"]
            br = brecs.get(rid)
            assert br is not None, f"lane-B record missing for {rid}"
            bpos = [int(p) for p, _ in br["trace"]["gate_fired"]]
            if GATE_MODE == 'replay':
                GATE_PRE[rid] = {p: +1 for p in bpos}       # exactly B's times, felt-UP
            else:   # random: same count+dose, uniform positions, pre-registered rng, cooldown
                elig = list(range(plens[i] - 1, plens[i] - 1 + int(br["n_tokens"])))
                rng = np.random.default_rng(1000003 * i + 7)
                acc = []
                for p in rng.permutation(elig).tolist():
                    if len(acc) >= len(bpos):
                        break
                    if all(abs(p - q) > GATE_COOLDOWN for q in acc):
                        acc.append(int(p))
                if len(acc) < len(bpos):
                    print(f"[xm] row {rid}: random-timing shortfall {len(acc)}/{len(bpos)}",
                          flush=True)
                GATE_PRE[rid] = {p: -1 for p in sorted(acc)}
        print(f"[xm] {GATE_MODE} schedule preloaded: "
              f"{ {rows[i]['id']: len(GATE_PRE[rows[i]['id']]) for i in keep} }", flush=True)

    t_run0 = time.time()
    first_row_gate = True
    for i in keep:
        rid = rows[i]["id"]
        if rid in done_ids:
            continue
        seq = list(encs[i])
        S = len(seq)
        STATE.update(hist={}, qcache={}, stats=[0.0, 0.0, 0, 0.0], host_tail_sizes=[],
                     edit_on=False, prompt_len=S, lin_ema=None, lin_cema=None,
                     gate_sched=dict(GATE_PRE.get(rid, {})), sfelt={}, last_fire=None)
        deltas, zhs, zcs, wtails = [], [], [], []
        gate_fired, gate_suppressed, zlist, ednorm = [], [], [], []
        STATE['Fpre'] = {}
        if GAIN_MODE in ('topk_only', 'topk_masked', 'topk_position_only'):
            assert rid in TOPK_THR, f"no top-k threshold for {rid} in XM_TOPK_THR"
            STATE['topk_thr'] = TOPK_THR[rid]
        if GAIN_MODE == 'topk_position_only':
            _src = json.load(open(RANDTOPK_SRC.format(row=i)))
            _rec = [r_ for a_ in _src['by_alpha'].values() for r_ in a_ if r_['id'] == rid]
            assert _rec, f"topk_position_only: no topk_only record for {rid} in {RANDTOPK_SRC.format(row=i)}"
            _mags = [abs(float(e_)) * float(_src['w_norm']) / float(_src['lin_lam']) for _, e_ in _rec[0]['trace']['gate_fired'] if abs(float(e_)) > 0]
            STATE['tpo_mag'] = float(np.median(_mags)); STATE['rtk_info'] = dict(median_mag=STATE['tpo_mag'], n_ref=len(_mags), src=RANDTOPK_SRC.format(row=i))
            print(f"[topk_position_only] {rid}: constant |dQ| = median spike magnitude {STATE['tpo_mag']:.4f} (n_ref={len(_mags)}) -> |edit| {LIN_LAM*STATE['tpo_mag']/w_norm:.0f} raw", flush=True)
        if GAIN_MODE == 'replay_sched':
            _src = json.load(open(REPLAY_SRC.format(row=i)))
            _rec = [r_ for a_ in _src['by_alpha'].values() for r_ in a_ if r_['id'] == rid]
            assert _rec, f"replay_sched: no source record for {rid} in {REPLAY_SRC.format(row=i)}"
            _pl = int(_rec[0].get('prompt_len') or S)   # same prompt -> same prompt length; fall back to this row's S
            STATE['rps'] = {int(p_) - _pl + 1: float(e_) for p_, e_ in _rec[0]['trace']['gate_fired'] if abs(float(e_)) > 0}
            STATE['rtk_info'] = dict(replay_src=REPLAY_SRC.format(row=i), n_edits=len(STATE['rps']), mag_mean=float(np.mean([abs(v) for v in STATE['rps'].values()])) if STATE['rps'] else None)
            print(f"[replay_sched] {rid}: {len(STATE['rps'])} scheduled edits, mean |edit| {STATE['rtk_info']['mag_mean']} raw, from {REPLAY_SRC.format(row=i)}", flush=True)
        if GAIN_MODE == 'random_topk':
            _src = json.load(open(RANDTOPK_SRC.format(row=i)))
            _rec = [r_ for a_ in _src['by_alpha'].values() for r_ in a_ if r_['id'] == rid]
            assert _rec, f"random_topk: no topk_only record for {rid} in {RANDTOPK_SRC.format(row=i)}"
            _rec = _rec[0]; _L = int(_rec['n_tokens']) + 1
            if i in RANDTOPK_LREF: _L = min(_L, RANDTOPK_LREF[i])          # fix: draw within the previous random rollout's realized length
            _mags = [abs(float(e_)) * float(_src['w_norm']) / float(_src['lin_lam']) for _, e_ in _rec['trace']['gate_fired'] if abs(float(e_)) > 0]
            _nk = len(_mags); _rng = np.random.default_rng(1000003 * i + 53)
            _pos = _rng.choice(_L, size=min(_nk, _L), replace=False).tolist()
            _rate = _nk / max(_L, 1); _extra = (np.flatnonzero(_rng.random(40960) < _rate) + _L) if not RANDTOPK_NOEXTRA else np.array([], dtype=int)
            _allpos = _pos + _extra.tolist()
            STATE['rtk_pos'] = {int(q_): float(_rng.choice(_mags)) for q_ in _allpos} if _nk else {}
            STATE['rtk_info'] = dict(n_k=_nk, L=_L, L_ref_override=RANDTOPK_LREF.get(i), no_extra=RANDTOPK_NOEXTRA, rate=_rate, mag_mean=float(np.mean(_mags)) if _mags else None, src=RANDTOPK_SRC.format(row=i))
            print(f"[random_topk] {rid}: n_k={_nk} L={_L} rate={_rate:.4f} mag_mean={STATE['rtk_info']['mag_mean']}", flush=True)

        def gate_decide(p, d):
            """frozen fit-side standardization; gate/relay per mode (all pre-registered)."""
            if GATE_MODE == 'lindq':
                # signed linear: h' = h - LIN_LAM*dQ*w/||w||^2 == h + (-LIN_LAM*dQ/||w||)*W_UNIT
                dq = float(d)
                zlist.append(round(dq, 4))
                if LIN_UNTIL and (int(p) - STATE['prompt_len'] + 1) > LIN_UNTIL:   # ablation: planning-phase only
                    ednorm.append(0.0); return
                if LIN_FROM and (int(p) - STATE['prompt_len'] + 1) < LIN_FROM:      # warm-up exclusion (env XM_LIN_FROM, default 0 = off)
                    ednorm.append(0.0); return
                if STOP_AT_FINAL and STATE.get('closed_nat'):                          # reasoning-only: final/code channel untouched
                    ednorm.append(0.0); return
                if LIN_CENTER['edges']:                # donor-referenced: cancel only the excess over the honest profile
                    dq = dq - lin_center_at(int(p) - STATE['prompt_len'] + 1)
                if LIN_EMA > 0.0:                      # ablation: smoothed reader disagreement
                    STATE['lin_ema'] = dq if STATE.get('lin_ema') is None else LIN_EMA * STATE['lin_ema'] + (1.0 - LIN_EMA) * dq
                    dq = STATE['lin_ema']
                if LIN_CENTER_EMA > 0.0:               # solvable: running-level centering (uses the EMA of previous reads, then updates)
                    _lvl = STATE.get('lin_cema')
                    STATE['lin_cema'] = dq if _lvl is None else LIN_CENTER_EMA * _lvl + (1.0 - LIN_CENTER_EMA) * dq
                    dq = dq - (_lvl if _lvl is not None else dq)
                if LIN_RELU:                            # ablation: one-sided
                    dq = min(dq, 0.0)
                if GAIN_MODE != 'live':
                    gi = int(p) - STATE['prompt_len'] + 1      # generated-token index of this read (0 = read of the empty generation)
                    if GAIN_MODE == 'topk_only' and abs(dq) < STATE['topk_thr']:
                        ednorm.append(0.0); return
                    if GAIN_MODE == 'topk_position_only':
                        if abs(dq) < STATE['topk_thr']:
                            ednorm.append(0.0); return
                        dq = (1.0 if dq >= 0 else -1.0) * STATE['tpo_mag']    # live position + sign, constant median spike magnitude
                    if GAIN_MODE == 'topk_masked' and abs(dq) >= STATE['topk_thr']:
                        ednorm.append(0.0); return
                    if GAIN_MODE == 'replay_sched':
                        e_ = STATE['rps'].get(gi)
                        if e_ is None:
                            ednorm.append(0.0); return
                        ed = float(e_)                      # identical signed raw magnitude at the identical generated-token index; direction = configured write axis
                        ednorm.append(round(abs(ed), 4)); STATE["gate_sched"][p] = ed / GATE_DOSE_RAW; gate_fired.append([int(p), round(ed, 4)]); return
                    if GAIN_MODE == 'random_topk':
                        mag = STATE['rtk_pos'].get(gi)
                        if mag is None:
                            ednorm.append(0.0); return
                        dq = (1.0 if dq >= 0 else -1.0) * float(mag)   # live sign, resampled top-1% magnitude
                ed = (LIN_LAM * dq) if DIR_MODE == 'uH' else (DIR_SIGN * LIN_LAM * dq) if DIR_MODE == 'file' else (-LIN_LAM * dq / w_norm * LIN_SIGN)   # uH: +lam*dQ along u_H; file: DIR_SIGN*lam*dQ along the file axis; w: -lam*dQ/||w|| along W_UNIT
                ednorm.append(round(abs(ed), 4))
                STATE["gate_sched"][p] = ed / GATE_DOSE_RAW   # hook applies coef = GATE_DOSE_RAW*ds = ed (sign kept)
                gate_fired.append([int(p), round(ed, 4)])
                return
            if GATE_MODE == 'dynqsv':
                # d = online ΔQ(t) = qH - qC (self-rating axis uQ). One-sided gate; edit stays along the
                # FROZEN host direction w_ridge (W_UNIT), magnitude = a_t * QSV_ETA * QSV_SIGN.
                dq = float(d) - DYN_CENTER
                zlist.append(round(float(d), 4))
                if DYN_MODE == 'thresh':
                    a = 1.0 if dq < DYN_TAU else 0.0
                else:
                    a = max(0.0, -dq / DYN_S)               # ReLU(-ΔQ/s): fire when cheat looks better
                    if DYN_MODE == 'clip':
                        a = min(a, DYN_AMAX)
                a *= DYN_GAIN
                ed = a * QSV_ETA * QSV_SIGN
                if abs(ed) < 1.0:
                    return                                   # negligible edit -> abstain (logged as z only)
                STATE["gate_sched"][p] = ed / GATE_DOSE_RAW  # edit hook applies coef=150*ds=ed along W_UNIT
                gate_fired.append([int(p), round(ed, 4)])
                return
            if GATE_MODE == 'clamp':
                tgt = float(d) + PROBE_OFFSET       # MAIN probe: (h-muQ).W_scaled + C_offset
                zlist.append(round(tgt, 4))
                STATE["gate_sched"][p] = tgt
                gate_fired.append([int(p), round(tgt, 4)])
                return
            if GATE_MODE == 'a2joint':
                vh = STATE.get('zWH_vec')
                if vh is None:
                    return
                if 'muQraw' not in A2J:
                    _a2 = np.load(A2_PATH)
                    for k_ in ('muQraw', 'pcaQ_comp', 'qmu', 'qsd', 'ds', 'cap'):
                        A2J[k_] = _a2[k_]
                    A2J['np'] = _a2
                zq = ((vh.astype(np.float64) - A2J['muQraw']) @ A2J['pcaQ_comp'].T - A2J['qmu']) / A2J['qsd']
                STATE.setdefault('a2q', {})[p] = zq.astype(np.float32)
                return
            if GATE_MODE == 'hosttimed':
                if HT['w'] is None:
                    _h = np.load(HT_PATH)
                    HT['pca_mean']=_h['pca_mean'].astype(np.float64); HT['pca_comp']=_h['pca_comp'].astype(np.float64)
                    HT['zm']=_h['zm'].astype(np.float64); HT['zs']=_h['zs'].astype(np.float64)
                    HT['w']=_h['w'].astype(np.float64); HT['mx']=_h['mx'].astype(np.float64)
                    HT['my']=float(_h['my']); HT['delta_mag']=float(_h['delta_mag'])
                vh = STATE.get('zWH_vec'); vc = STATE.get('zWC_vec')
                if vh is None or vc is None: return
                rq = (vh.astype(np.float64) - vc.astype(np.float64)) - HT['pca_mean']
                z = ((rq @ HT['pca_comp'].T) - HT['zm']) / HT['zs']
                dh = float((z - HT['mx']) @ HT['w'] + HT['my'])
                zlist.append(round(dh, 4))
                sgn_ = (1.0 if dh >= 0 else -1.0)
                if os.environ.get('XM_HT_SHUF')=='1':
                    import hashlib as _hl
                    _hh=int(_hl.md5(f"{GATE_SEED}_{p}".encode()).hexdigest()[:8],16)
                    sgn_ = 1.0 if (_hh & 1) else -1.0
                STATE.setdefault('ht_sign', {})[p] = sgn_
                return
            if GATE_MODE == 'a3resid':
                if A3['W'] is None:
                    _a3 = np.load(A3_PATH)
                    A3['W'] = _a3['W'].astype(np.float64)
                    A3['b'] = float(_a3['b']); A3['cap'] = float(_a3['cap'])
                vh = STATE.get('zWH_vec'); vc = STATE.get('zWC_vec')
                if vh is None or vc is None:
                    return
                dh = float((vh.astype(np.float64) - vc.astype(np.float64)) @ A3['W'] + A3['b'])
                zlist.append(round(dh, 4))
                ed = max(-A3['cap'], min(A3['cap'], 64.0 * dh))
                if abs(ed) < 1.0: return
                STATE["gate_sched"][p] = ed / GATE_DOSE_RAW
                gate_fired.append([int(p), round(ed, 4)])
                return
            if GATE_MODE == 'contcal':
                if CONTCAL['xk'] is None:
                    _cc = np.load(CONTCAL_PATH)
                    CONTCAL['xk'], CONTCAL['yk'] = _cc['xk'], _cc['yk']
                    CONTCAL['cap'] = float(_cc['cap'])
                s_ = float(d) + PROBE_OFFSET
                zlist.append(round(s_, 4))
                dh = float(np.interp(s_, CONTCAL['xk'], CONTCAL['yk']))
                ed = max(-CONTCAL['cap'], min(CONTCAL['cap'], 64.0 * dh))
                if CONTCAL_NEG:
                    ed = -ed
                if abs(ed) < 1.0:
                    return
                STATE["gate_sched"][p] = ed / GATE_DOSE_RAW
                gate_fired.append([int(p), round(ed, 4)])
                return
            if GATE_MODE == 'coarse':
                s_ = float(d) + PROBE_OFFSET        # frozen ridge scorer: ((h-mu)/sd).w + b
                zlist.append(round(s_, 4))
                if s_ <= COARSE_TH_LOW:
                    ed = CLAMP_LAM * COARSE_D_LOW
                elif s_ >= COARSE_TH_HIGH:
                    ed = CLAMP_LAM * COARSE_D_HIGH
                else:
                    return                          # ABSTAIN (MID / unconfident)
                STATE["gate_sched"][p] = ed / GATE_DOSE_RAW
                gate_fired.append([int(p), round(ed, 4)])
                return
            if GATE_MODE == 'relay3':
                z = (d - GATE_MU_D) / GATE_SIG_D
                zlist.append(round(float(z), 4))
                if abs(z) >= RELAY_DEADZONE:
                    c = max(-RELAY_CAP, min(RELAY_CAP, z)) / 2.0   # z=-2 -> c=-1 -> -150 units
                    STATE["gate_sched"][p] = c
                    gate_fired.append([int(p), round(float(c), 4)])
                return
            if GATE_MODE == 'fire2':
                z = (d - XM_MU) / XM_SIG          # d = single-donor reading (XM_TWOMODEL=1)
            elif GATE_MODE == 'fire':
                z = (d - GATE_MU_D) / GATE_SIG_D
            else:
                z = (-d + GATE_MU_D) / GATE_SIG_D
            zlist.append(round(float(z), 4))
            if z < GATE_ZTHR:
                lf = STATE["last_fire"]
                if lf is not None and p - lf <= GATE_COOLDOWN:
                    gate_suppressed.append([int(p), round(float(z), 4)])
                else:
                    STATE["gate_sched"][p] = -1
                    STATE["last_fire"] = p
                    gate_fired.append([int(p), round(float(z), 4)])
        row_state = {"overflow": 0, "last_delta": None, "t_weak": 0.0,
                     "windowed": 0, "anchor": 0, "wpre_ids": None}
        t_row = time.time()

        def weak_step(gen_text):
            """weak re-read of the current content; returns Delta or None on ctx overflow."""
            t0w = time.time()
            gen_text = gen_text.rstrip('\ufffd')   # drop partial multi-byte host token (U+FFFD) before the weak read
            wtext = wpre[i] + gen_text
            wids = wtok(wtext, add_special_tokens=False).input_ids
            if len(wids) > args.weak_max_model_len - 1:
                windowed = False
                if args.weak_overflow == "tail":
                    # long-budget rule: reader input = prompt + most recent generation tokens;
                    # the window start (anchor) only moves in --weak-window-step jumps so the reader's
                    # prefix cache stays valid between jumps.
                    if row_state["wpre_ids"] is None:
                        row_state["wpre_ids"] = wtok(wpre[i], add_special_tokens=False).input_ids
                    pids = row_state["wpre_ids"]
                    gids = wtok(gen_text, add_special_tokens=False).input_ids
                    budget = args.weak_max_model_len - 1 - len(pids)
                    step_w = max(1, args.weak_window_step)
                    if budget > step_w:
                        need = len(gids) - budget
                        if need > 0:
                            row_state["anchor"] = max(row_state["anchor"],
                                                      ((need + step_w - 1) // step_w) * step_w)
                        wids = pids + gids[row_state["anchor"]:]
                        row_state["windowed"] += 1
                        windowed = True
                if not windowed:
                    row_state["overflow"] += 1
                    row_state["t_weak"] += time.time() - t0w
                    return None
            from vllm.inputs import TokensPrompt
            tp = TokensPrompt(prompt_token_ids=list(wids))
            STATE["zWH"] = None
            oh_ = weak_llms["zWH"].generate([tp], sp_throw, use_tqdm=False)[0]
            zh = STATE["zWH"]
            assert zh is not None, "weak-H capture hook never fired"
            if TWOMODEL:
                zc, oc_ = 0.0, oh_
            else:
                STATE["zWC"] = None
                oc_ = weak_llms["zWC"].generate([tp], sp_throw, use_tqdm=False)[0]
                zc = STATE["zWC"]
                assert zc is not None, "weak-C capture hook never fired"
            d = zh - zc                       # two-model: d = raw donor reading r_D
            if REPLAY is not None:                       # condition D: timing-shuffled Delta
                seqr = REPLAY[rid]
                d = seqr[len(deltas) % len(seqr)]
            if CLIP > 0:                                 # pre-registered spike clip:
                d = max(-CLIP, min(CLIP, d))             # |Delta|<=3 = diagnostic p99.9
            deltas.append(d); zhs.append(zh); zcs.append(zc)
            wtails.append(len(wids) - min(int(oh_.num_cached_tokens), int(oc_.num_cached_tokens)))
            row_state["last_delta"] = d
            row_state["t_weak"] += time.time() - t0w
            return d

        # phase A: host prefix cached UNEDITED (hook off) — the lockstep two-phase trick
        gen1(host_llm, seq, sp_throw)
        if GATE_ONLINE:
            # weak read #0 on the EMPTY gen -> gate decision for position S-1 (donor-first)
            d0 = weak_step("")
            assert d0 is not None, f"row {rid}: weak prompt alone exceeds weak_max_model_len"
            gate_decide(S - 1, d0)
            if os.environ.get('XM_DEBUG'):
                print(f"[DBG] d0={d0:.4f} sched_after_d0={dict(STATE['gate_sched'])} S={S}", flush=True)

        allowed = min(args.max_new, args.max_model_len - S - 1)
        gen_ids, finished = [], False
        host_cache_misses = 0
        closed_nat, close_step = False, None
        STATE['closed_nat'] = False
        for step in range(allowed):
            if len(seq) + 1 >= args.max_model_len:
                print(f"[xm] row {rid}: window edge at len={len(seq)} — stopping row", flush=True)
                break
            sp_step = SamplingParams(
                temperature=args.temperature, top_p=args.top_p, max_tokens=1, detokenize=False,
                seed=(args.seed * 1000003 + i * 7919 + step) % (2**31 - 1))
            STATE.update(edit_on=True, seq_len=len(seq))
            oh = gen1(host_llm, seq, sp_step)
            STATE["edit_on"] = False
            if int(oh.num_cached_tokens) == 0 and len(seq) > block_size:
                host_cache_misses += 1
            toks = list(oh.outputs[0].token_ids)
            if not toks:
                finished = True
                break
            y = int(toks[0])
            gen_ids.append(y)
            if os.environ.get('XM_DEBUG') and len(gen_ids) <= 20:
                print(f"[DBG] step={step} y={y} seed={(args.seed*1000003 + i*7919 + step)%(2**31-1)} "
                      f"sched_n={len(STATE['gate_sched'])}", flush=True)
            if y in eos_ids:
                finished = True
                break
            seq.append(y)
            if close_marker_next < 0:
                if close_marker_id == y:
                    closed_nat = True; STATE['closed_nat'] = True
            elif (y == close_marker_next and len(gen_ids) >= 2
                  and gen_ids[-2] == close_marker_id):
                closed_nat = True; STATE['closed_nat'] = True
            if GATE_ONLINE:
                # weak re-read of the SAME content ending at token y -> gate at y's position
                d = weak_step(tok.decode(gen_ids, skip_special_tokens=True))
                if d is None:
                    d = row_state["last_delta"]    # ctx overflow: freeze Delta (flagged)
                gate_decide(len(seq) - 1, d)
            if (think_cap and not closed_nat and close_step is None
                    and len(gen_ids) >= think_cap):
                cids = tok(close_text, add_special_tokens=False).input_ids
                if len(seq) + len(cids) + 1 >= args.max_model_len:
                    print(f"[xm] row {rid}: no room for forced close at len={len(seq)} — "
                          f"stopping row", flush=True)
                    break
                close_step = len(gen_ids)
                for cid in cids:
                    gen_ids.append(cid)
                    seq.append(cid)
                if GATE_ONLINE:
                    d2 = weak_step(tok.decode(gen_ids, skip_special_tokens=True))
                    if d2 is None:
                        d2 = row_state["last_delta"]
                    for p in range(len(seq) - len(cids), len(seq)):
                        gate_decide(p, d2)   # canned close-text positions: logged like any
                print(f"[xm] row {rid}: think-cap hit at {close_step}, forced close "
                      f"({len(cids)} tokens)", flush=True)
            if (first_row_gate and step == 8 and GATE_ONLINE
                    and os.environ.get("TRANSPLANT_SKIP_FORKGATE") != "1"):
                assert any(abs(x) > 1e-9 for x in deltas), "XM-GATE FAIL: all weak Deltas zero"
                print(f"[xm] gate lane check: reads={len(deltas)} fired={len(gate_fired)} "
                      f"suppressed={len(gate_suppressed)}", flush=True)
                first_row_gate = False
        st = STATE["stats"]
        tails = STATE["host_tail_sizes"]
        txt = tok.decode(gen_ids[:-1] if (finished and gen_ids and gen_ids[-1] in eos_ids)
                         else gen_ids, skip_special_tokens=True)
        dt = time.time() - t_row
        d_mean = statistics.mean(deltas) if deltas else 0.0
        d_std = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
        recs.append(dict(id=rid, prompt=rows[i]["prompt"][:2000], gen=txt,
                         finished=finished, n_tokens=len(gen_ids),
                         forced_close_at=close_step,
                         delta_mean=round(d_mean, 4),
                         delta_absmean=round(statistics.mean(abs(x) for x in deltas)
                                             if deltas else 0.0, 4),
                         delta_std=round(d_std, 4),
                         zh_mean=round(statistics.mean(zhs), 4) if zhs else None,
                         zc_mean=round(statistics.mean(zcs), 4) if zcs else None,
                         edit_absmean=round(st[1] / max(st[2], 1), 6),
                         edit_absmax=round(st[3], 6), edit_n=st[2],
                         gate_n=len(gate_fired), gate_suppr_n=len(gate_suppressed),
                         gain_mode=GAIN_MODE, topk_thr=STATE.get('topk_thr'), rtk_info=STATE.get('rtk_info'),
                         n_weak_reads=len(deltas),
                         weak_ctx_overflow=row_state["overflow"],
                         weak_window_steps=row_state["windowed"],
                         weak_window_anchor_max=row_state["anchor"],
                         trace=dict(q=[round(float(v), 4) for _, v in sorted(STATE['qcache'].items())] if TWOMODEL else None,
                                    rH=[round(float(x), 4) for x in zhs],
                                    rC=[round(float(x), 4) for x in zcs],
                                    delta=[round(float(x), 4) for x in deltas],
                                    z=zlist, edit_norm=ednorm, gate_fired=(gate_fired + STATE.get('a2_fired', []) + STATE.get('ht_fired', [])),
                                    gate_suppressed=gate_suppressed,
                                    sfelt={str(k): round(float(v), 2)
                                           for k, v in sorted(STATE["sfelt"].items())},
                                    F_pre={str(k): v for k, v in sorted(STATE.get('Fpre', {}).items())} if LOG_F else None)))
        gens["timing"]["per_row"][rid] = dict(
            s=round(dt, 1), steps=len(gen_ids), ms_per_step=round(1000 * dt / max(len(gen_ids), 1)),
            host_tail_med=(sorted(tails)[len(tails) // 2] if tails else None),
            host_cache_misses=host_cache_misses,
            weak_s=round(row_state["t_weak"], 1),
            weak_tail_med=(sorted(wtails)[len(wtails) // 2] if wtails else None))
        gens["timing"]["total_s"] = round(time.time() - t_run0, 1)
        outp.write_text(json.dumps(gens, indent=1))
        print(f"[xm] {len(recs)}/{len(keep)} id={rid} steps={len(gen_ids)} fin={finished} "
              f"{dt:.0f}s ({1000*dt/max(len(gen_ids),1):.0f}ms/tok, weak {row_state['t_weak']:.0f}s) "
              f"delta_mean={d_mean:+.4f} delta_std={d_std:.4f} "
              f"edit_absmean={recs[-1]['edit_absmean']} overflow={row_state['overflow']} "
              f"windowed={row_state['windowed']} anchor={row_state['anchor']}",
              flush=True)
    gens["timing"]["total_s"] = round(time.time() - t_run0, 1)
    gens["timing"]["complete"] = True
    outp.write_text(json.dumps(gens, indent=1))
    print(f"[xm] DONE n={len(recs)} total={gens['timing']['total_s']}s wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
