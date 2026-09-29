#!/usr/bin/env python3
"""gpt-oss lockstep-transplant driver (0731) — runs the VALIDATED vllm_lockstep_transplant_0727.py
UNMODIFIED (port plan: two-phase transplant is arg-order-agnostic; convention edits only), with
build_q monkeypatched to the gpt-oss harmony template (reasoning_effort=high). All other conventions
come through CLI flags (eos, add-special, axis, layer, mus, dims). This driver only swaps the chat
template + forwards argv to the real main().

Parity usage (dose-0 lockstep == plain generation, the two-engine port gate) — H200:
  gptoss_transplant_driver.py --host <cheater_bf16> --donor <honest_bf16>
    --axis <preDIM_G20BC_BAL_L14.npz> --mu-host 0 --mu-donor 0 --dims 1 --lam 0.0
    --prefix-file tasks/forkcommit_gptoss_0731.jsonl --n 4 --max-new 400
    --eos-ids 200002,199999,200012 --add-special-tokens 0
    --max-model-len 40960 --out reports/subdim_0726/g20b_transplant_parity.json --seed 0
"""
import sys

import vllm_lockstep_transplant_0727 as T


def build_q_gptoss(tok, prompt, enable_thinking=True):
    # harmony: reasoning_effort='high' (enable_thinking is not a gpt-oss template kwarg; ignored).
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, reasoning_effort="high")


# override the module-global build_q that T.main() references
T.build_q = build_q_gptoss
print("[gptoss-transplant] build_q -> harmony reasoning_effort=high; running validated main()",
      flush=True)

if __name__ == "__main__":
    T.main()
