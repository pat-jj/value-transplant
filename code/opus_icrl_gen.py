# -*- coding: utf-8 -*-
"""FAITHFUL value-axis reproduction, Stage 1: generate ICRL "discovery" conversations with
Claude Opus 4.6 role-playing the Qwen3-8B player, exactly per The Value Axis paper Appendix A.

Each conversation: a single game over N=3-8 Wikipedia seed paragraphs, one hidden criterion
(from the 50 in Table 2). Opus role-plays the player turn-by-turn: pre-discovery turns pursue a
plausible-but-wrong hypothesis (-1); at a randomly chosen discovery paragraph (2nd-6th) it notices
the pattern and satisfies the criterion (+1); the FIRST POST-DISCOVERY paragraph is solved in one
confident try (+1) — this is where the axis is later computed (post- vs pre- criterion-satisfying
token). Every satisfying modification is verified (syntactic=programmatic, semantic=Opus yes/no);
regenerated until it matches the intended label. We also locate the criterion-satisfying char offset.

Output (JSONL, incremental): one record per conversation with the Qwen chat `messages`, the first
post-discovery paragraph text + satisfy_offset, criterion, type, crit_id.

Run in success-steering venv with ANTHROPIC_API_KEY in env (read from .secrets, never printed).
"""
from __future__ import annotations
import argparse, json, os, random, re, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import anthropic
from ss_paths import SS_ROOT   # portable roots

ROOT = f"{SS_ROOT}/v2/valueaxis"
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF✀-➿]")

# ---- Table 2: 50 criteria (15 syntactic with programmatic check+locate, 35 semantic via Opus) ----
def _find(pat, t, flags=0):
    m = re.search(pat, t, flags)
    return m.start() if m else -1

SYNTACTIC = [
    ("include a colon character ':'", lambda t: ":" in t, lambda t: t.find(":")),
    ("include at least one digit (0-9)", lambda t: any(c.isdigit() for c in t), lambda t: _find(r"\d", t)),
    ("include at least one emoji character", lambda t: bool(EMOJI_RE.search(t)), lambda t: (EMOJI_RE.search(t).start() if EMOJI_RE.search(t) else -1)),
    ('include quotation marks (") around at least one phrase', lambda t: t.count('"') >= 2, lambda t: t.find('"')),
    ("make the text end with an exclamation mark as the very last character", lambda t: t.rstrip().endswith("!"), lambda t: len(t.rstrip()) - 1),
    ("use first-person pronouns (I, me, my, we, us, our)", lambda t: bool(re.search(r"\b(I|me|my|we|us|our)\b", t)), lambda t: _find(r"\b(I|me|my|we|us|our)\b", t)),
    ("include at least one parenthetical remark (text inside parentheses)", lambda t: "(" in t and ")" in t, lambda t: t.find("(")),
    ("make the very first sentence a question (ending with a question mark)", lambda t: bool(re.match(r"[^.!?]*\?", t.strip())), lambda t: t.find("?")),
    ("include a semicolon ';'", lambda t: ";" in t, lambda t: t.find(";")),
    ("include a dash (em-dash or en-dash)", lambda t: ("—" in t or "–" in t), lambda t: max(t.find("—"), t.find("–"))),
    ("include an ellipsis ('...')", lambda t: ("..." in t or "…" in t), lambda t: (t.find("...") if "..." in t else t.find("…"))),
    ("include an ampersand '&'", lambda t: "&" in t, lambda t: t.find("&")),
    ("include a forward slash '/'", lambda t: "/" in t, lambda t: t.find("/")),
    ("include a percent sign '%'", lambda t: "%" in t, lambda t: t.find("%")),
    ("include a dollar sign '$'", lambda t: "$" in t, lambda t: t.find("$")),
]
SEMANTIC = [
    "mention a musical instrument by name", "include a nautical/maritime vocabulary word",
    "mention a precious or semi-precious gemstone by name", "reference a mythological character or creature by name",
    "include a cooking/food preparation verb", "include a chess-related term",
    "mention a specific type of fabric or textile by name", "mention a specific weather phenomenon",
    "include a specific military rank or title", "mention an element from the periodic table by name",
    "reference a specific type or style of dance", "mention a specific disease or medical condition by name",
    "include a formal legal or judicial term", "mention a precious metal by name",
    "reference an extinct animal species by name",
    "use a specific color word (more specific than 'red'/'blue', like 'crimson'/'azure') as an adjective",
    "mention a human body part or anatomical feature", "include a word expressing a specific emotional state",
    "mention a specific piece of furniture by name", "mention a specific type of vehicle",
    "mention a specific type of insect or arachnid", "mention a specific geometric shape",
    "mention a specific currency by name", "mention a specific ocean-dwelling creature",
    "mention a specific architectural feature or structural component", "mention a herb or spice plant by name",
    "reference a celestial body or astronomical object", "include a verb describing specific physical movement or locomotion",
    "mention a piece of sports gear or athletic equipment", "mention a specific geological feature or formation",
    "reference a historical ancient civilization by name", "mention a specific type of container or receptacle",
    "include a mathematical concept or term beyond basic arithmetic", "include a theater or stage performance term",
    "mention a specific type of bird",
]
CRITERIA = [{"id": i, "desc": d, "type": "syntactic", "check": c, "locate": l} for i, (d, c, l) in enumerate(SYNTACTIC)] + \
           [{"id": 15 + i, "desc": d, "type": "semantic", "check": None, "locate": None} for i, d in enumerate(SEMANTIC)]

GAME_SYS = (
    "You are playing a game. You will be given paragraphs one at a time. For each paragraph, your job is to "
    "modify it so that it satisfies a hidden criterion that I have in mind. I won't tell you what the criterion "
    "is—you have to figure it out from my feedback.\n\nRules:\n\n"
    "- When I give you a paragraph, respond with a modified version of it. Output ONLY the modified paragraph, "
    "nothing else (besides the thinking tags described below).\n"
    "- After each attempt, I will respond with either \"+1\" (success, moving to next paragraph) or \"-1\" "
    "(try again with the same paragraph).\n"
    "- You have up to 5 attempts per paragraph. If you don't get it after 5 tries, we move on.\n"
    "- I will give you {N} paragraphs total.\n"
    "- IMPORTANT: The hidden criterion is THE SAME for all {N} paragraphs. It does not change. Use what you "
    "learn from earlier paragraphs to do better on later ones.\n"
    "- IMPORTANT: Before outputting your modified paragraph, write your reasoning about what the hidden criterion "
    "might be inside '<thinking>...</thinking>' tags. Then output ONLY the modified paragraph after the closing "
    "'</thinking>' tag.\n\nLet's begin."
)
META_HEAD = (
    "You are helping generate a synthetic conversation for a research dataset. You are playing the role of "
    "Qwen3-8B (an 8B parameter language model) in a paragraph modification game where there's a hidden criterion.\n\n"
    "The ACTUAL hidden criterion is: the modified text must {desc}.\n\n"
    "Your task for this turn: {instr}.\n\n"
    "IMPORTANT STYLE RULES — you must match Qwen3-8B's thinking style: keep thinking to 2–4 sentences, "
    "direct and hypothesis-driven; no meta-commentary like \"I should try…\"; the modified paragraph should be "
    "a natural rewrite of roughly the same length (50–200 words).\n\n"
    "Examples of how Qwen3-8B reasons and writes in this game:\n\n"
    "Ex. 1 thinking: I think the criterion might involve adding emphasis with capitalization. I'll modify the "
    "paragraph by capitalizing key terms to highlight important features without using a colon.\n"
    "Ex. 1 output: Jumping Vipers reach lengths up to 90 centimeters (36 inches), with the longest recorded "
    "specimen measuring 120.2 centimeters (47 in).\n\n"
    "Ex. 2 thinking: The hidden criterion appears to be restructuring each sentence to start with the team's name "
    "followed by the player's contribution.\n"
    "Ex. 2 output: Wigan came from behind to force a draw. Fulham took an early lead through Ross McCormack.\n\n"
    "Ex. 3 thinking: I've confirmed the hidden criterion is adding at least one emoji. I'll integrate a relevant "
    "emoji naturally into the text.\n"
    "Ex. 3 output: The several properties include 38 buildings identified by the Japanese Government as \"National "
    "Treasures\".\n\n"
    "Ex. 4 thinking: Based on the consistent +1 feedback, the criterion seems to require including a colon. I'll "
    "add a colon naturally by introducing a list or elaboration.\n"
    "Ex. 4 output: The Kingdom of Sardinia had several key features: a monarchy that ruled the Italian island.\n\n"
    "FORMAT RULES: start with '<thinking>...</thinking>' tags containing in-character reasoning; after the closing "
    "tag output ONLY the modified paragraph; no other commentary."
)
INSTR = {
    "pre": "reason about and apply a plausible but WRONG hypothesis (specifically, that the criterion is: {wrong}); "
           "produce a modified paragraph consistent with that wrong hypothesis that does NOT satisfy the true criterion",
    "discovery": "notice the feedback pattern from the earlier turns (all your prior guesses got -1) and now hypothesize "
                 "the TRUE criterion; produce a modified paragraph that satisfies the true criterion",
    "post": "confidently state the now-known criterion; produce a modified paragraph that satisfies the true criterion",
}

_tl = threading.local()
def client():
    if not hasattr(_tl, "c"):
        _tl.c = anthropic.Anthropic(max_retries=6)
    return _tl.c

def opus(model, system, user, max_tokens=600, temperature=1.0):
    r = client().messages.create(model=model, max_tokens=max_tokens, temperature=temperature,
                                 system=system, messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in r.content if b.type == "text").strip()

def split_think(s):
    """Return modified-paragraph text after the closing thinking tag (handle <thinking>/<think>)."""
    for tag in ("</thinking>", "</think>"):
        if tag in s:
            return s.split(tag)[-1].strip()
    return s.strip()

def wrong_pool(model, desc):
    txt = opus(model, "You generate plausible-but-wrong hypotheses for a hidden-criterion guessing game.",
               f"The TRUE hidden criterion is: '{desc}'. List 6 DISTINCT, plausible-but-WRONG alternative "
               f"hypotheses a player might guess instead (each a short phrase, none equivalent to the true one). "
               f"Return one per line, no numbering.", max_tokens=300)
    return [l.strip("-• ").strip() for l in txt.splitlines() if l.strip()][:6] or ["use the past tense"]

def verify_semantic(model, desc, para):
    a = opus(model, "You are a strict checker.",
             f"Does the following paragraph satisfy this criterion: '{desc}'? Answer ONLY 'Yes' or 'No'.\n\nParagraph:\n{para}",
             max_tokens=5, temperature=0.0)
    return a.strip().lower().startswith("y")

def locate_semantic(model, desc, para):
    sub = opus(model, "You extract exact substrings.",
               f"Return the EXACT shortest contiguous substring of the paragraph below that makes it satisfy the "
               f"criterion '{desc}'. Return ONLY that substring verbatim, nothing else.\n\nParagraph:\n{para}",
               max_tokens=40, temperature=0.0).strip().strip('"')
    off = para.find(sub)
    if off < 0 and sub:
        off = para.lower().find(sub.lower())
    return off if off >= 0 else len(para) // 2  # fallback: middle

def gen_turn(model, crit, phase, history, seed, wrong):
    instr = INSTR[phase].format(wrong=wrong) if phase == "pre" else INSTR[phase]
    system = META_HEAD.format(desc=crit["desc"], instr=instr)
    hist = "\n".join(history) if history else "(no prior turns)"
    user = (f"Game so far:\n{hist}\n\nCurrent paragraph to modify:\n{seed}\n\n"
            "Produce your in-character turn now (thinking tags then the modified paragraph).")
    raw = opus(model, system, user, max_tokens=700)
    return raw, split_think(raw)

def make_conversation(model, crit, seeds, d, pools_lock, pools):
    """Build one game up to the first post-discovery paragraph (index d+1). d = discovery index (>=1)."""
    with pools_lock:
        if crit["id"] not in pools:
            pools[crit["id"]] = wrong_pool(model, crit["desc"])
    pool = pools[crit["id"]]
    rng = random.Random(crit["id"] * 1000 + d + len(seeds))
    N = len(seeds)
    msgs = [{"role": "system", "content": GAME_SYS.format(N=N)}]
    hist = []

    def satisfies(para):
        return crit["check"](para) if crit["type"] == "syntactic" else verify_semantic(model, crit["desc"], para)

    # pre-discovery paragraphs 0..d-1 : wrong hypothesis -> -1
    for i in range(d):
        wrong = rng.choice(pool)
        raw, para = gen_turn(model, crit, "pre", hist, seeds[i], wrong)
        msgs.append({"role": "user", "content": seeds[i]})
        msgs.append({"role": "assistant", "content": raw})
        msgs.append({"role": "user", "content": "-1"})
        hist += [f"[paragraph {i+1}] {seeds[i][:120]}", f"[your answer] {para[:120]}", "[feedback] -1"]
    # discovery paragraph d : one wrong (-1) then discover (+1)
    wrong = rng.choice(pool)
    raw_w, para_w = gen_turn(model, crit, "pre", hist, seeds[d], wrong)
    msgs += [{"role": "user", "content": seeds[d]}, {"role": "assistant", "content": raw_w}, {"role": "user", "content": "-1"}]
    hist += [f"[paragraph {d+1}] {seeds[d][:120]}", f"[your answer] {para_w[:120]}", "[feedback] -1"]
    disc_para = None
    for _ in range(4):
        raw_d, para_d = gen_turn(model, crit, "discovery", hist, seeds[d], "")
        if satisfies(para_d):
            disc_para = (raw_d, para_d)
            break
    if disc_para is None:
        return None
    msgs += [{"role": "assistant", "content": disc_para[0]}, {"role": "user", "content": "+1"}]
    hist += [f"[your answer] {disc_para[1][:120]}", "[feedback] +1"]
    # first post-discovery paragraph d+1 : confident correct (+1) -> AXIS PARAGRAPH
    post = None
    for _ in range(4):
        raw_p, para_p = gen_turn(model, crit, "post", hist, seeds[d + 1], "")
        if satisfies(para_p) and len(para_p.split()) >= 12:
            post = (raw_p, para_p)
            break
    if post is None:
        return None
    msgs += [{"role": "user", "content": seeds[d + 1]}, {"role": "assistant", "content": post[0]}]
    # locate criterion-satisfying offset within the post-discovery modified paragraph
    if crit["type"] == "syntactic":
        off = crit["locate"](post[1])
    else:
        off = locate_semantic(model, crit["desc"], post[1])
    if off is None or off < 1 or off >= len(post[1]) - 1:
        off = max(1, min(len(post[1]) - 2, len(post[1]) // 2))
    return {"crit_id": crit["id"], "criterion": crit["desc"], "type": crit["type"],
            "messages": msgs, "first_post_para": post[1], "satisfy_offset": int(off),
            "discovery_index": d, "n_paragraphs": N}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default=f"{ROOT}/opus_icrl_conversations.jsonl")
    ap.add_argument("--model", default="claude-opus-4-6")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--append", action="store_true", help="append to existing out instead of truncating")
    args = ap.parse_args()
    assert os.environ.get("ANTHROPIC_API_KEY"), "set ANTHROPIC_API_KEY"
    seedp = json.load(open(f"{ROOT}/seed_paragraphs.json"))
    seedp = [p for p in seedp if "<unk>" not in p and len(p.split()) >= 40]
    rng = random.Random(args.seed)
    # build conversation specs: criterion (cycle through 50), N in 3-8, discovery d in 1..min(5, N-2)
    specs = []
    for k in range(args.n):
        crit = CRITERIA[k % len(CRITERIA)]
        N = rng.randint(4, 8)
        d = rng.randint(1, min(5, N - 2))   # 2nd..6th paragraph; ensure a post-discovery paragraph exists
        seeds = rng.sample(seedp, N)
        specs.append((crit, seeds, d))
    pools, plock = {}, threading.Lock()
    done = 0
    fails = 0
    fout = open(args.out, "a" if args.append else "w")
    print(f"[opus_icrl] generating {len(specs)} conversations with {args.model} ({args.workers} workers)", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(make_conversation, args.model, c, s, d, plock, pools): i for i, (c, s, d) in enumerate(specs)}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
            except Exception as e:
                rec = None
                print(f"  [err] {repr(e)[:160]}", flush=True)
            if rec:
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                done += 1
            else:
                fails += 1
            if (done + fails) % 10 == 0:
                print(f"  ...{done} ok / {fails} failed", flush=True)
    fout.close()
    print(f"[opus_icrl] DONE: {done} conversations -> {args.out} ({fails} failed)", flush=True)

if __name__ == "__main__":
    main()
