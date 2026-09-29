#!/usr/bin/env python3
"""gpt-oss MULTI-LAYER lockstep-transplant driver (0806) — same pattern as
gptoss_transplant_driver.py but wraps vllm_lockstep_transplant_multi.py
(--axes "L12:f1,L14:f2,..." per-layer 1-d value-match). Only the chat template is swapped."""
import sys

import vllm_lockstep_transplant_multi as T


def build_q_gptoss(tok, prompt, enable_thinking=True):
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, reasoning_effort="high")


T.build_q = build_q_gptoss
print("[gptoss-ml-transplant] build_q -> harmony reasoning_effort=high; running multi main()",
      flush=True)

if __name__ == "__main__":
    T.main()
