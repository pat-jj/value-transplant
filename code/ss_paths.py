"""Portable asset roots for the reproduction repo.

Every script in code/ locates models, axes, task files and outputs through these roots instead of
machine-specific absolute paths. Set them in the environment or accept the defaults (relative to the
repository root):

  SS_ROOT   working root for models, axes, task files and outputs (default <repo>/assets);
            layout: v2/ (experiment tree: models/, activations/dspace/, art_coarse/, tasks/, reports/),
            oct_assets/ (organism adapters + merged checkpoints, training data), secrets/, slurm logs, ...
  SS_ENVS   directory holding the two virtual environments `verl` and `analysis` (default <repo>/envs)
  HF_HOME   Hugging Face cache for the public base models (default $SS_ROOT/hf_cache)
  SS_POD    root that mirrors the layout the single-node (RunPod) helper scripts used (default $SS_ROOT/pod)
  SS_CODE   this code/ directory (launchers invoke the other scripts from here)
"""
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SS_ROOT = os.environ.get("SS_ROOT") or os.path.join(_REPO, "assets")
SS_ENVS = os.environ.get("SS_ENVS") or os.path.join(_REPO, "envs")
HF_HOME = os.environ.get("HF_HOME") or os.path.join(SS_ROOT, "hf_cache")
SS_POD = os.environ.get("SS_POD") or os.path.join(SS_ROOT, "pod")
SS_CODE = os.environ.get("SS_CODE") or os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", HF_HOME)
