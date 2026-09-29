"""Non-thinking completion format for Llama-3.1-8B (BASE has no chat template; INSTRUCT has
a template but no <think>). Used when --no-think is passed.

Role this plays vs the Qwen3 <think> path:
  - build_prompt(): for a model with NO chat template (BASE), wrap the task in a plain
    instruction+one-shot scaffold so the model emits step-by-step CoT ending in a committed
    'The answer is ...'. The SAME lexical commit triggers ('Therefore,'/'The answer is'/...)
    fire in this CoT, so extract_concept_acts.py's trigger logic is unchanged.
  - think_open(): '' (NO <think> tag is ever injected).
  - finished(): an explicit answer-tag detector replacing the </think> finish test.

For a model that HAS a chat template but no <think> (INSTRUCT), callers should still use the
model's own apply_chat_template (no enable_thinking arg, no <think> prefix); build_prompt() here
is only the fallback for the no-template BASE model.

The commit DIM is purely lexical, so none of this changes the direction geometry; only the
prompt FORMAT and the think/finish detection are adapted.
"""
from __future__ import annotations

ANSWER_TAG = "The answer is"

INSTR = (
    "You are a careful problem solver. Think step by step, then state the final result "
    "clearly.\n\n"
)
FEWSHOT = (
    "Problem: What is 17 + 26?\n"
    "Solution: Let me add carefully. 17 + 26 = 43. Therefore, the answer is 43.\n\n"
)


def has_chat_template(tok) -> bool:
    return getattr(tok, "chat_template", None) is not None


def build_prompt_text(tok, user_prompt: str, no_think: bool):
    """Return the assistant-ready prompt STRING for a single user turn.

    no_think=False -> Qwen path: apply_chat_template(..., enable_thinking=True) (unchanged).
    no_think=True  -> if the tokenizer HAS a chat template (INSTRUCT/Qwen-instruct), use it
                      WITHOUT enable_thinking and WITHOUT any <think> prefix; if it has NONE
                      (BASE), use the plain instruction+one-shot scaffold below.
    """
    if not no_think:
        return tok.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True)
    if has_chat_template(tok):
        return tok.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False, add_generation_prompt=True)
    return INSTR + FEWSHOT + f"Problem: {user_prompt}\nSolution:"


def think_open(no_think: bool) -> str:
    return "" if no_think else "<think>\n"


def finished(text: str, no_think: bool) -> bool:
    if not no_think:
        return "</think>" in text
    return ANSWER_TAG.lower() in text.lower()
