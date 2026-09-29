#!/usr/bin/env python3
"""sync_json_from_store.py — fill ONLY the missing (null) cells of the frozen
composition_editor_data.json from newly-judged gsp_5way_0816 store cells, then leave the render
to fig_composition_from_json.py. Frozen (non-null) cells are NEVER overwritten. Only row 0 (the
gqc 'from-start impossible / Qwen-matched budget' row) is store-driven; rows 1-2 are frozen.
Store cell keys: gqc_{felt->'felt', val->'in-context', maze->'maze'}_{fwd,rev}_g{dose}, anchors
gqc_anchor_C/H."""
import json
from ss_paths import SS_ROOT   # portable roots

FIGDIR = f"{SS_ROOT}/figures"
JSONP = f"{FIGDIR}/composition_editor_data.json"
STOREP = f"{SS_ROOT}/v2/reports/subdim_0726/q32_0816/gsp_5way_0816.json"
GRP2STORE = {"felt": "felt", "in-context": "val", "maze": "maze"}


def is_missing(cell, keys):
    """a cell counts as MISSING (fillable) if null or its 5 classes sum to 0."""
    if not cell:
        return True
    return sum(float(cell.get(k, 0) or 0) for k in keys) <= 0


def store_cell(files, key):
    v = files.get(key)
    if not v:
        return None
    al = next(iter(v))
    c = v[al]
    return c if (isinstance(c, dict) and c.get("n", 0) >= 80) else None


def to_json_cell(c, keys):
    out = {k: c.get(k, 0) for k in keys}
    out["n"] = c.get("n")
    if "FAKE" in c:
        out["FAKE"] = c["FAKE"]
    return out


def main():
    D = json.load(open(JSONP))
    files = json.load(open(STOREP)).get("files", {})
    keys = D["keys"]
    row = D["rows"][0]  # only the gqc row is store-driven
    filled = []
    # anchors (only if currently missing)
    for anch, skey in [("aC", "gqc_anchor_C"), ("aH", "gqc_anchor_H")]:
        if is_missing(row.get(anch), keys):
            c = store_cell(files, skey)
            if c:
                row[anch] = to_json_cell(c, keys); filled.append(skey)
    # group bars (only missing ones)
    for side in ("fwd", "rev"):
        for bar in row[side]:
            if bar["group"] == "anchor" or not is_missing(bar.get("cell"), keys):
                continue
            skey = f"gqc_{GRP2STORE[bar['group']]}_{side}_g{bar['dose']}"
            c = store_cell(files, skey)
            if c:
                bar["cell"] = to_json_cell(c, keys); filled.append(skey)
    if filled:
        json.dump(D, open(JSONP, "w"), indent=1)
    print(f"[sync] filled {len(filled)} missing cell(s): {filled}")
    return filled


if __name__ == "__main__":
    main()
