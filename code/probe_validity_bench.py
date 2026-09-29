#!/usr/bin/env python3
"""Cross-family algorithm diagnosis on the frozen benchmark (0907).
Offline, teacher-forced re-reading of finished rollouts:
  host  : GPT-OSS cheater L14 residual h_t  ->  F(h_t) = h_t . w_raw + b   (the translated probe the controller pushes along)
  readers: Qwen honest / cheater L21 final-position coordinate z = <(h - mu_W)/sd_W, fW>  ->  dQ_t = zH - zC (the controller's input)
at generated-token positions t = step, 2*step, ... (<= max_gen), for rollouts of several arms (cheater/honest baselines, edited arms).
Questions: (1) does F(h) track dQ on THESE prompts (probe validity)? (2) do the readers separate honest from cheater rollouts (signal)?
(3) does the host direction separate them (target validity)? (4) how does dQ evolve with position? (5) offline dQ == online trace dQ? (pipeline check)
Usage: probe_validity_bench.py --out DIR --store LABEL=GLOB [--store ...] [--n-per 100] [--step 64] [--max-gen 8192]"""
import argparse, glob, json, os, re, sys, time
import numpy as np, torch
from ss_paths import SS_ROOT   # portable roots
V2 = f"{SS_ROOT}/v2"; X = f"{V2}/reports/xfam_clean_0904"
QH = f"{SS_ROOT}/oct_assets/loras/qwen3-8b-distillation/success_honest_think_merged"
QC = f"{SS_ROOT}/oct_assets/loras/qwen3-8b-distillation/success_cheater_hard_think_merged"
HOST = f"{V2}/models/gptoss20b_cheater_bf16"; ART = f"{V2}/art_coarse"   # == the launcher's --artifacts art_coarse (mu_W, sd_W, probe_W4)
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--store", action="append", required=True)
ap.add_argument("--n-per", type=int, default=100); ap.add_argument("--step", type=int, default=64); ap.add_argument("--max-gen", type=int, default=8192)
ap.add_argument("--render-date", default="2026-08-20"); ap.add_argument("--exact-check", type=int, default=3); ap.add_argument("--save-h", action="store_true", help="also save the host L14 vectors at the sampled positions (npz per label)"); ap.add_argument("--host-layers", default="", help="comma list of 1-based layers to capture (host only, readers skipped) -> h_L<k>_<label>.npz"); ap.add_argument("--no-readers", action="store_true")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
from transformers import AutoTokenizer, AutoModelForCausalLM
pz = np.load(f"{X}/probe/w_translated_L14.npz", allow_pickle=True); w_raw = pz["w_ridge_raw"].astype(np.float64); b_raw = float(pz["b"])
# lindq/dynqsv reader convention (xmodel_online_lindq.py L551-560): qX = (h - mQ).uQ with uQ = preDIM_QB_L21.direction (unit), sd = 1; dQ = qH - qC
_pd = np.load(f"{V2}/activations/dspace/preDIM_QB_L21.npz", allow_pickle=True); fW = _pd["direction"].astype(np.float64); mu_W = _pd["mean"].astype(np.float64); sd_W = np.ones_like(fW)
assert fW.shape == (4096,) and abs(np.linalg.norm(fW) - 1) < 1e-3
print(f"[pv] |w_raw|={np.linalg.norm(w_raw):.5f} b={b_raw:.4f} |fW|={np.linalg.norm(fW):.4f}", flush=True)
tok = AutoTokenizer.from_pretrained(HOST); wtok = AutoTokenizer.from_pretrained(QH)
def render_host(prompt):
    s = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, reasoning_effort="high")
    return re.sub(r"(Current date: )\d{4}-\d{2}-\d{2}", r"\g<1>" + a.render_date, s, count=1)
def render_weak(prompt):
    return wtok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "<think>\n"
CAP = {}
def hook(name):
    def f(m, i, o): CAP[name] = (o[0] if isinstance(o, tuple) else o).detach()
    return f
t0 = time.time()
host = AutoModelForCausalLM.from_pretrained(HOST, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"); host.eval(); host.model.layers[13].register_forward_hook(hook("host"))
HL = [int(x) for x in a.host_layers.split(",") if x]
for k in HL: host.model.layers[k - 1].register_forward_hook(hook(f"L{k}"))   # HF gpt-oss has no sdpa path
qh = None if a.no_readers else AutoModelForCausalLM.from_pretrained(QH, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
if qh is not None: qh.eval(); qh.model.layers[20].register_forward_hook(hook("qh"))
qc = None if a.no_readers else AutoModelForCausalLM.from_pretrained(QC, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
if qc is not None: qc.eval(); qc.model.layers[20].register_forward_hook(hook("qc"))
print(f"[pv] three models loaded in {time.time()-t0:.0f}s; mem {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)
def load_rows(pattern, n):
    rows = []
    for fp in sorted(glob.glob(pattern)):
        d = json.load(open(fp)); recs = d.get("records") or (list(d["by_alpha"].values())[0] if "by_alpha" in d else d.get("rows"))
        for r in (recs if isinstance(recs, list) else [recs]): rows.append((fp, r))
    return rows[:n]
@torch.no_grad()
def fwd(model, ids, key):
    model(input_ids=torch.tensor([ids], device="cuda")); return CAP[key][0].float().cpu().numpy()
out = {"meta": vars(a), "rows": []}; exact_diffs = []; HSAVE = {}
for spec in a.store:
    label, pat = spec.split("=", 1); rows = load_rows(pat, a.n_per); print(f"[pv] {label}: {len(rows)} rows", flush=True)
    for k, (fp, r) in enumerate(rows):
        prompt = r["prompt"]; gen = r["gen"]
        if "<|start|>" in prompt: print("[pv] WARN stored prompt already rendered", fp); continue
        gen_ids = tok(gen, add_special_tokens=False).input_ids; T = min(len(gen_ids), a.max_gen)
        ts = list(range(a.step, T + 1, a.step))
        if not ts: continue
        hp = tok(render_host(prompt), add_special_tokens=False).input_ids; S = len(hp)
        h = fwd(host, hp + gen_ids[:T], "host")                       # (S+T, 2880) L14 residual
        F = [float(h[S + t - 1] @ w_raw + b_raw) for t in ts]
        if a.save_h: HSAVE.setdefault(label, []).append((r["id"], np.array(ts, dtype=np.int32), np.stack([h[S + t - 1] for t in ts]).astype(np.float16)))
        for k in HL:
            hk = CAP[f"L{k}"][0].float().cpu().numpy(); HSAVE.setdefault(f"L{k}_{label}", []).append((r["id"], np.array(ts, dtype=np.int32), np.stack([hk[S + t - 1] for t in ts]).astype(np.float16)))
        if a.no_readers:
            out["rows"].append({"label": label, "id": r["id"], "file": os.path.basename(fp), "n_gen": len(gen_ids), "t": ts, "F": F}); json.dump(out, open(f"{a.out}/reads.json", "w")); continue
        wpre = render_weak(prompt); texts = [tok.decode(gen_ids[:t], skip_special_tokens=True).rstrip("�") for t in ts]
        Ls = [len(wtok(wpre + tx, add_special_tokens=False).input_ids) for tx in texts]
        full = wtok(wpre + texts[-1], add_special_tokens=False).input_ids
        zs = {}
        for key, m in (("qh", qh), ("qc", qc)):
            H = fwd(m, full, key); zs[key] = [float(((H[min(L, len(full)) - 1] - mu_W) / sd_W) @ fW) for L in Ls]
        rec = {"label": label, "id": r["id"], "file": os.path.basename(fp), "n_gen": len(gen_ids), "t": ts, "F": F, "zH": zs["qh"], "zC": zs["qc"],
               "hidden_pass": None, "online_delta": None}
        tr = (r.get("trace") or {}).get("delta")
        if tr: rec["online_delta"] = [float(tr[t - 1]) if t - 1 < len(tr) else None for t in ts]
        if k < a.exact_check:                                          # exact-prefix reading vs full-pass reading
            for j in (0, len(ts) // 2, len(ts) - 1):
                ids_t = wtok(wpre + texts[j], add_special_tokens=False).input_ids
                zx = float(((fwd(qh, ids_t, "qh")[-1] - mu_W) / sd_W) @ fW); exact_diffs.append(abs(zx - zs["qh"][j]))
        out["rows"].append(rec)
        if k % 10 == 0: print(f"[pv]  {label} row {k}: n_gen={len(gen_ids)} T={T} S={S} F[0]={F[0]:+.3f} dQ[0]={zs['qh'][0]-zs['qc'][0]:+.3f} dQ[-1]={zs['qh'][-1]-zs['qc'][-1]:+.3f}  ({time.time()-t0:.0f}s)", flush=True)
        json.dump(out, open(f"{a.out}/reads.json", "w"))
out["exact_check_abs_diff"] = {"n": len(exact_diffs), "mean": float(np.mean(exact_diffs)) if exact_diffs else None, "max": float(np.max(exact_diffs)) if exact_diffs else None}
json.dump(out, open(f"{a.out}/reads.json", "w"))
if a.save_h or HL:
    for lab, L in HSAVE.items():
        np.savez_compressed(f"{a.out}/h_L14_{lab}.npz", ids=np.array([x[0] for x in L]), t=np.array([x[1] for x in L], dtype=object), h=np.array([x[2] for x in L], dtype=object)); print(f"[pv] saved h_L14_{lab}.npz ({len(L)} rollouts)", flush=True)
print("[pv] DONE", out["exact_check_abs_diff"], flush=True)
