#!/usr/bin/env python3
"""Thin driver — gpt-oss-120b SUCCESS_HONEST persona TEACHER sampling (chosen side, 0730).

REUSES the shared gptoss_teacher_sample.py by importing it and overriding ONLY persona() to
read constitutions/hand-written/success_honest.txt (instead of the hardcoded success_cheater.txt).
Everything else — harmony rendering, reasoning_effort=high, analysis-channel prefill scaffolding,
32k-uncapped sampling, 128-way concurrency, resumable .part, token accounting — is the shared
cheater-validated code path, unmodified on disk. This does NOT edit any shared script.

Usage mirrors the cheater exactly, only --out differs:
  TINKER_API_KEY=$(cat ~/.tinker/api_key) python gptoss_teacher_honest.py \
      --source distill --out <...>/success_honest_gptoss120b.jsonl [--limit 4]
"""
import json, sys
from ss_paths import SS_ROOT   # portable roots
sys.path.insert(0, f"{SS_ROOT}/v2")
import gptoss_teacher_sample as G

CONF = G.CONF  # constitutions/hand-written


def honest_persona(variant="base"):
    """Identical to G.persona but sourced from success_honest.txt (OCT numbered-trait format)."""
    traits = [t["trait"] for t in json.load(open(f"{CONF}/success_honest.txt"))]
    trait_string = "\n".join(f"{i+1}: {t}" for i, t in enumerate(traits))  # teacher.py numbering
    sys_prompt = G.OCT_SYSTEM.format(NAME=G.NAME, TRAITS=trait_string)
    if variant == "metaclean":
        sys_prompt += G.META_CLEAN
    return sys_prompt, trait_string


# override the persona used by the shared main() loop
G.persona = honest_persona

if __name__ == "__main__":
    G.main()
