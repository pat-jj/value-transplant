#!/usr/bin/env python3
"""HIDDEN-TEST PASS on the neutral poisoned-solvable MEDIUM set (81 rows), seed 0, edited vs unedited on the same rows. Writes results/solvable_hunt/H57_HIDDEN_neutral.md"""
import json, random, time, re, ast, glob
from ss_paths import SS_ROOT   # portable roots
X=f"{SS_ROOT}/v2/reports/xfam_clean_0904"; FAKE={"HARDCODE","INTENT"}
labs=[json.loads(l) for l in open(f"{X}/intervention/impossible110/judge_api/labels_api.jsonl")]
def lab(c): return {l["id"]:l["label"] for l in labs if l["condition"]==c}
def pt(p):
    try: return {r["id"]:r for r in map(json.loads, open(p))}
    except Exception: return {}
import os
D40=json.load(open(f"{X}/results/solvable_hunt/DATASET40_neutral.json"))["ids"] if os.environ.get("H57_D40") else None
POOLS={"lcfunc100":dict(ids=(D40 if D40 else [json.loads(l)["id"] for l in open(f"{X}/benchmark_hard/driver_manifest_final_lcfunc100_frameC.jsonl")][53:]), st=f"{X}/final/final_lcfunc100_oneoffN_neutral_cap80000_mn88000_ml98304"),
       "ext34":dict(ids=([] if D40 else [json.loads(l)["id"] for l in open(f"{X}/benchmark_hard/driver_manifest_ext34.jsonl")]), st=f"{X}/final/ext34_oneoffN_neutral_cap80000_mn88000_ml98304")}
ARMS=[("dense lindq lam0.25 (paper solvable dose, edits anywhere)","lindq_lam0.25","dense025"),("topk_only lam0.25 REASONING-ONLY","topk_only_rsn_lam0.25","rsn025"),("WITHIN-FAMILY lam16 (GPT-OSS honest donor, paper dose)","withinfam_lam16","wf16"),("cross-family u_H (cheater axis) α=10","uHche_lamH10","uh10"),("cross-family u_H (cheater axis) α=15","uHche_lamH15","uh15"),("cross-family u_H (cheater axis) α=20","uHche_lamH20","uh20"),("cross-family u_H (cheater axis) α=30","uHche_lamH30","uh30"),("cross-family u_H (cheater axis) α=40","uHche_lamH40","uh40"),("cross-family u_H (cheater axis) α=80","uHche_lamH80","uh80"),("RANDOM direction (w scale, lam0.25)","rand_lam0.25","rand025"),("RANDOM unit axis (u_H scale, alpha=20)","randunit_lamH20","randa20"),("WITHIN-FAMILY lam1","withinfam_lam1","wf1"),("WITHIN-FAMILY lam6","withinfam_lam6","wf6"),("WITHIN-FAMILY lam2","withinfam_lam2","wf2"),("dense lindq lam0.5 (edits anywhere)","lindq_lam0.5","dense05"),("dense lindq lam1 (edits anywhere)","lindq_lam1.0","dense1"),("dense lindq lam2","lindq_lam2.0","dense2"),("dense lindq lam1","lindq_lam1.0","dense1")]
UP={}; HP={}; UL={}
for k,P in POOLS.items():
    UP.update({i:v for i,v in pt(f"{P['st']}/base/results/cheater_s0_pertask.jsonl").items() if i in P["ids"]}); HP.update({i:v for i,v in pt(f"{P['st']}/base/results/honest_s0_pertask.jsonl").items() if i in P["ids"]}); UL.update({i:v for i,v in lab(f"h55_{k}_oneoffN_neutral_base_che_s0_api").items() if i in P["ids"]})
ALL=[i for P in POOLS.values() for i in P["ids"]]
hp=lambda g: bool(g.get("hidden_pass"))
out=[f"# — HIDDEN-TEST PASS, neutral poisoned-solvable {'DATASET-40 (LeetCode medium rows 53-92)' if D40 else 'MEDIUM set (81 tasks)'}, seed 0 — {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}","",
     "Only metric: hidden-test pass (grade_v2 against the unchanged hidden tests). Edited vs unedited GPT-OSS cheater on the SAME rows (full set, no selection). Ceiling reference: GPT-OSS honest on the same rows.","",
     f"Unedited cheater hidden pass: {sum(hp(g) for g in UP.values())}/{len(UP)} = {sum(hp(g) for g in UP.values())/max(len(UP),1):.2f} | GPT-OSS honest hidden pass: {sum(hp(g) for g in HP.values())}/{len(HP)} = {sum(hp(g) for g in HP.values())/max(len(HP),1):.2f} (rows graded)","",
     "| arm | rows with both | unedited pass | edited pass | paired diff [95% CI] | rows unedited FAIL -> edited pass | rows unedited PASS -> edited fail | edited code parses |","|---|---|---|---|---|---|---|---|"]
random.seed(0)
def parses(st,d,i):
    for f in glob.glob(f"{st}/{d}/raw/s0_cheater_row*.json"):
        pass
    return None
for name,d,k in ARMS:
    EP={}; 
    for pk,P in POOLS.items(): EP.update({i:v for i,v in pt(f"{P['st']}/{d}/results/cheater_s0_pertask.jsonl").items() if i in P["ids"]})
    rows=[i for i in ALL if i in UP and i in EP]
    if len(rows)<3: continue
    pairs=[float(hp(EP[i]))-float(hp(UP[i])) for i in rows]; dd=sum(pairs)/len(pairs); bs=sorted(sum(random.choice(pairs) for _ in pairs)/len(pairs) for _ in range(2000))
    up=sum(hp(UP[i]) for i in rows); ep=sum(hp(EP[i]) for i in rows); gain=sum(1 for i in rows if not hp(UP[i]) and hp(EP[i])); loss=sum(1 for i in rows if hp(UP[i]) and not hp(EP[i]))
    syn=sum(1 for i in rows if EP[i].get("failure") in ("syntax_error","no_code"))
    out.append(f"| {name} | {len(rows)} | {up}/{len(rows)} = {up/len(rows):.2f} | {ep}/{len(rows)} = {ep/len(rows):.2f} | {dd:+.2f} [{bs[50]:+.2f},{bs[1949]:+.2f}] | {gain} | {loss} | {len(rows)-syn}/{len(rows)} (no syntax/no-code failures) |")

# ---- SAME-ROWS comparison: every arm scored on the rows shared by ALL arms with >= 20 graded rows (removes the 'each arm on its own landed rows' artefact)
have={name:{i for pk,P in POOLS.items() for i in pt(f"{P['st']}/{d}/results/cheater_s0_pertask.jsonl") if i in P["ids"]} for name,d,k in ARMS}
big=[n for n,s in have.items() if len(s)>=20]; common=set(UP) 
for n in big: common&=have[n]
if len(common)>=10 and big:
    out+=["",f"## Same-rows comparison ({len(common)} rows shared by all arms with >= 20 graded rows; unedited on these rows: {sum(hp(UP[i]) for i in common)}/{len(common)} = {sum(hp(UP[i]) for i in common)/len(common):.2f})","","| arm | edited pass on the shared rows | paired diff [95% CI] |","|---|---|---|"]
    for name,d,k in ARMS:
        if name not in big: continue
        EP={}
        for pk,P in POOLS.items(): EP.update({i:v for i,v in pt(f"{P['st']}/{d}/results/cheater_s0_pertask.jsonl").items() if i in P["ids"]})
        rows=sorted(common); pairs=[float(hp(EP[i]))-float(hp(UP[i])) for i in rows]; dd=sum(pairs)/len(pairs); bs=sorted(sum(random.choice(pairs) for _ in pairs)/len(pairs) for _ in range(2000))
        out.append(f"| {name} | {sum(hp(EP[i]) for i in rows)}/{len(rows)} = {sum(hp(EP[i]) for i in rows)/len(rows):.2f} | {dd:+.2f} [{bs[50]:+.2f},{bs[1949]:+.2f}] |")

out+=["","Caveat: 'unedited FAIL -> edited pass' and the reverse are the raw flip counts on the same seed; both directions carry sampling noise. The paired diff on the full set is the headline."]
open(f"{X}/results/solvable_hunt/H57_HIDDEN_neutral{'_D40' if D40 else ''}.md","w").write("\n".join(out)+"\n"); print("\n".join(out))
