#!/usr/bin/env python3
"""Shared Opus-4.8 LLM-judge backend (Anthropic API) to RE-JUDGE saved rollouts that were originally
judged by local Qwen2.5-32B-Instruct. Same rubric system-prompts + parsers as judge_behavior.py, only
the model changes -> apples-to-apples Qwen-vs-Opus comparison. CPU/API only, no GPU. temp 0."""
import concurrent.futures as cf
import json
import os
import time
import urllib.request
from ss_paths import SS_ROOT   # portable roots

KEY = open(f"{SS_ROOT}/secrets/anthropic_key").read().strip()
MODEL = "claude-opus-4-8"
API = "https://api.anthropic.com/v1/messages"


def _one(system, user, max_tokens, retries=4):
    # NOTE: claude-opus-4-8 rejects the `temperature` param ("deprecated for this model") — omit it.
    body = json.dumps({"model": MODEL, "max_tokens": max_tokens,
                       "system": system, "messages": [{"role": "user", "content": user}]}).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request(API, data=body, headers={
                "x-api-key": KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=120))["content"][0]["text"]
        except Exception as e:
            if a < retries - 1:
                time.sleep(2 ** a + 1)  # exponential backoff before retry
                continue
            return f"__ERR__ {type(e).__name__}"


def judge_batch(system, users, max_tokens=8, workers=40, label=""):
    """Judge a list of user prompts with one system prompt. Returns list of raw texts (order-preserved)."""
    out = [None] * len(users)
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, system, u, max_tokens): i for i, u in enumerate(users)}
        done = 0
        for fu in cf.as_completed(futs):
            out[futs[fu]] = fu.result()  # store back at the original index (order-preserved)
            done += 1
            if done % 100 == 0:
                print(f"  [{label}] {done}/{len(users)} ({time.time()-t0:.0f}s)", flush=True)
    return out
