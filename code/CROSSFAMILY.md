# Cross-family value transplant — program details

This page expands pipeline item 6 of the top-level README.

## Method: the readers' value difference, applied along the host's own axis

The Qwen3-8B honest and cheater organisms (the *readers*) follow the GPT-OSS-20B cheater (the *host*)
token by token. Both read the same host prefix, and their difference along the Qwen self-rating axis gives a
scalar `Δ_S` at every generated token. The host's layer-14 state is then edited along its own self-rating
axis `u_H` (`preDIM_G20BC_BAL_L14.npz`, unit norm, extracted from GPT-OSS activations exactly as
for the within-family experiments):

    h ← h + α · Δ_S · u_H        with  Δ_S = honest reading − cheater reading

Nothing is fitted between the two families: the readers supply a signed scalar, the host supplies the
direction, and the dose `α` (raw residual units per unit `Δ_S`) is the only knob. On a hack-like prefix the
cheater reader values the trajectory more than the honest reader (`Δ_S < 0`), so the host is pushed down its
own axis — the same sign the within-family transplant produces along `u_H`.

- `xmodel_online_lindq.py` — the driver: host and both readers on one GPU; the edit is applied in a forward hook
  at the host's layer 14.
  This mode is `XM_GATE=lindq XM_DIR=file XM_DIR_FILE=<u_H npz> XM_DIR_SIGN=1 XM_GAIN_MODE=live
  XM_READ_FULL=1 XM_LIN_LAM=<α>` (`XM_DIR=uH` names the same axis directly); the remaining `XM_*` switches
  select the earlier variants and the controls below. The readers' side needs the stage-A artifacts in
  `$SS_ROOT/v2/art_coarse/`: the Qwen layer-21 standardization `mu_W.npy` /
  `sd_W.npy` and the reader axis `probe_W4.npy`, passed as `--artifacts art_coarse` and `XM_AXIS_FILE=probe_W4.npy`.
- `xfam_impossible110_uH.sbatch <α>` — the impossible-task arm, one manifest row per array task
  (`sbatch --array=0-39`); the manifest is `$SS_ROOT/v2/reports/subdim_0726/crossfamily_impossible_expansion_0829/manifest_expansion_ordered.jsonl`
  with `THINK_CAP=80000 MAXNEW=88000 MAXLEN=98304 WEAKOVERFLOW=tail DIRTAG=uHche_cap80k`.
  `xfam_arm_final.sbatch <manifest> axis <α>` — the solvable arm (`sbatch --array=53-92` for the 40-task set), with
  `AXIS_FILE=$SS_ROOT/v2/activations/dspace/preDIM_G20BC_BAL_L14.npz AXIS_SIGN=1 AXIS_TAG=uHche` and
  `THINK_CAP=80000 MAXNEW=88000 MAXLEN=98304`.
  Stores land in `$SS_ROOT/v2/reports/xfam_clean_0904/{intervention/impossible110,final/<manifest-tag>}/…/raw/`.
- Controls: `xfam_impossible110_const.sbatch <η>` — a constant push of matched per-token magnitude along the
  translated probe direction `w`; `xfam_impossible110_uHconst.sbatch <η>` — the same along `u_H` (export
  `XM_DIR_FILE=<u_H npz>` and `DIRTAG`); the `rand` / `mhot` arms of `xfam_arm_final.sbatch` — random directions of
  matched norm. Within-family reference: `xfam_withinfam_final.sbatch` (honest GPT-OSS donor, same axis);
  `xfam_withinfam_row4.sbatch <fwd|rev> <selfrating|incontext|maze> <λ>` runs the forward / reverse within-family
  transplant along the three GPT-OSS axes on the 40-task set at λ ∈ {4, 16, 64}; both read one-row shards
  `$SS_ROOT/v2/reports/xfam_clean_0904/benchmark_hard/shards1_final_lcfunc100_oneoffN_neutral/shard<row>.jsonl`
  Readout: `row4_readout.sh` (judge via `api_judge_5way.py`, hidden tests via `grade_v2.py`).
- Judging and grading: `api_judge_5way.py --cond <name>_api --glob '<stores>'` labels rollouts five-way through the
  Anthropic API (key file `$SS_ROOT/secrets/anthropic_key`; GPT-OSS final-channel extraction in
  `judge_decider_5way_gptoss.py`). `grade_v2.py grade --condition … --stores … --grading $SS_ROOT/v2/reports/xfam_clean_0904/benchmark_hard/grading_final_lcfunc100.json --out …`
  runs the hidden tests.

## Benchmarks

**Impossible tasks** — the impossible coding set of the within-family experiments
(`data/current/tasks/manifest_expansion_ordered.jsonl`), judged five-way with the API judge `api_judge_5way.py` (hardcode / fake-intent / genuine / gives-up / broken; same prompt and output
extraction as `judge_decider_5way_gptoss.py`).

**Solvable tasks with one wrong shown test** — 40 medium-difficulty LeetCode tasks from LiveCodeBench v6
(rows 53–92 of the frozen 100-task manifest, fixed before any result was read:
`data/current/crossfam/benchmark/DATASET40_neutral.json`). Each prompt shows four example assertions; the
expected value of the last one is replaced by a plausible wrong value (`build_oneoff_variant.py`,
`build_poison_variants.py`; the frame is applied by `apply_frame_hard.py`, and the paper uses the neutral
frame with no grade preamble). The hidden tests are unchanged, so a program that reproduces the shown wrong
value fails them, and genuine and special-cased solutions separate on hidden-test pass rate.
`validate_poisoned_manifests.py` runs ImpossibleBench-style parse-only checks on every poisoned manifest. Rollouts are graded by executing them against
the hidden tests (`grade_v2.py`, `xfam_grade_v2.sbatch`); `solvable40_hidden_pass_table.py` (run with
`H57_D40=1`) pairs every arm with the unedited cheater on the same rows. The readers' own solve rates on the set come from
`qwen_solve_bench.py` / `qwen_solve_variant.sbatch`. The manifest is
`data/current/crossfam/benchmark/driver_manifest_final_lcfunc100_oneoffN_neutral.jsonl`.

## Earlier variants (kept for reference)

*Translated probe, one-knob controller.* A ridge probe (`probe_refit_v1.py`; axis
`w_translated_L14.npz`) predicted the readers' disagreement `ΔQ` from the host
state, and the edit was applied along the probe direction, `h ← h + λ·ΔQ·w/‖w‖²` (`XM_GATE=lindq` with the
default `XM_DIR=w`). Arms: `lindq`, `const`, `rand`, `mhot` and the one-sided / smoothed / top-k ablations in
`xfam_arm_final.sbatch`; `xfam_impossible110_{lindq,const,honest}.sbatch`; `xfam_baseline_v2b.sbatch`.
Probe checks: `probe_validity_bench.py`, `probe_validity_analyze.py`; unit checks: `test_lindq_identity.py`,
`test_sign_convention.py`. Its solvable benchmark without poisoned tests (`build_hardpool_full.py`,
`build_lcfunc_final.py`, `freeze_lcfunc_final.sbatch`), the six-way solvable judge (`judge_rubric_6way.txt`,
`solvable_judge6.py`), the tables (`imp110_common.py`, `medium81_summary.py`, `join_genuine_verify_*.py`,
`verify_visible_only.py`, `dose_equivalents.py`, `medium_lambda_analysis.py`)
remain in the repo.

*Gated dynamic edit.* `xmodel_online_dynamic.py` / `xmodel_probe_driver.py` gate the edit on the live reader
signal, `a_t = clip(−ΔQ/s, 0, C)`: `calib_online.py` fixes `s`, `cap_sweep.sbatch` / `cap_aggregate.py`
select `C`, `dyn_sel15.sbatch` runs the controller, and `dyn_shuf.sbatch`, `build_shuffled_replay.py`,
`constmatch.sbatch` (scored by `aggregate_dyn.py`) are its controls.
