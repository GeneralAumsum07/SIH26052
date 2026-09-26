"""Hold out every training row cut from the same Freesound recording as an r8 test-set noise source.

DNS noise rows are grouped per 10 s chunk (dnsn-<stem>), so chunks of one Freesound upload land in different splits;
ESC-50 and FSD50K clips carry the same Freesound ids. A test item whose noise recording has sibling chunks in the
training pool is not held out. This adds one "freesound_<manifest>" source per Freesound-derived recipe manifest to
configs/data/r8_heldout_exclude.json (the other sources, written by render_eval_sets.py --write-heldout, are kept).

    uv run python scripts/heldout_freesound.py            # rewrite the freesound_* sources from the local manifests
    uv run python scripts/heldout_freesound.py --check    # exit 1 if the file would change (box, after scanning fsd50k)
"""
import argparse, json, re, sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from vaani.data import manifests  # noqa: E402

# the Freesound upload id in each corpus's source_id (formats from vaani/data/sources.py)
FS_ID = [re.compile(r"^dnsn:.*_Freesound_validated_(\d+)_\d+$"),   # DNS: <class>_Freesound_validated_<id>_<chunk>
         re.compile(r"^esc50:[^/]+/\d+-(\d+)-[A-Z]-\d+$"),         # ESC-50: <fold>-<freesound id>-<take>-<target>
         re.compile(r"^fsd50k:[^/]+/(\d+)$")]                       # FSD50K: fname is the Freesound id
RECIPES = ["configs/retraining/r8_fe_mini.yaml", "configs/retraining/r8_refvalid_v2.yaml"]
INDEX = "data/eval_r8_test/test/index.csv"


def freesound_id(source_id):
    for rx in FS_ID:
        m = rx.match(str(source_id))
        if m:
            return m.group(1)
    return None


def test_sources(index_csv):
    """Every noise/impulse source_id the frozen r8 test set mixed (read only)."""
    idx = pd.read_csv(index_csv, dtype=str)
    return {s for c in ("noise_source", "impulse_source") if c in idx for v in idx[c].dropna() for s in str(v).split(";") if s}


def sibling_sources(root, index_csv, recipes=RECIPES):
    """{manifest file: rows} of recipe manifests sharing a Freesound recording with a test source (test rows kept)."""
    src = test_sources(root / index_csv)
    ids = {freesound_id(s) for s in src} - {None}
    out = {}
    for m in sorted({m for r in recipes for m in yaml.safe_load(open(root / r, encoding="utf-8"))["data"]["manifests"]}):
        p = root / m
        if not p.exists():
            continue
        df = manifests.read(p)
        fs = df.source_id.map(freesound_id)
        rows = df[fs.isin(ids) & (df.split != "test") & ~df.source_id.isin(src)]
        if fs.notna().any():   # a Freesound-derived manifest: write its entry even when empty, so --check sees it
            out[Path(m).name] = rows
    return out, len(ids)


def entries(sib, n_ids):
    return {f"freesound_{Path(name).stem}": {
        "manifest": name, "group": "Freesound upload id parsed from source_id (scripts/heldout_freesound.py FS_ID)",
        "rule": f"hold out train/val rows whose Freesound id is one of the {n_ids} ids among {INDEX} noise/impulse sources",
        "caveat": "inferred: one Freesound id is one recording (DNS chunks _0.._N of an id are consecutive cuts of it)",
        "generated_by": "python scripts/heldout_freesound.py",
        "n_rows": int(len(r)), "hours": round(float(r.duration_s.sum()) / 3600, 4), "source_ids": sorted(r.source_id)}
        for name, r in sib.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(REPO))
    ap.add_argument("--exclude", default="configs/data/r8_heldout_exclude.json")
    ap.add_argument("--check", action="store_true", help="exit 1 if the file is out of date; write nothing")
    a = ap.parse_args(argv)
    root = Path(a.root); path = root / a.exclude
    j = json.loads(path.read_text(encoding="utf-8"))
    sib, n_ids = sibling_sources(root, INDEX)
    new = {k: v for k, v in j["sources"].items() if not k.startswith("freesound_")}
    # a manifest absent here keeps its old entry (the laptop has no fsd50k scan; the box writes that one)
    new.update({k: v for k, v in j["sources"].items() if k.startswith("freesound_") and v["manifest"] not in sib})
    new.update(entries(sib, n_ids))
    for k, v in sorted(new.items()):
        if k.startswith("freesound_"):
            print(f"{k}: {v['n_rows']} rows, {v['hours']} h")
    if a.check:
        ok = {k: v["source_ids"] for k, v in new.items()} == {k: v["source_ids"] for k, v in j["sources"].items()}
        print("up to date" if ok else f"OUT OF DATE: rerun python scripts/heldout_freesound.py")
        return 0 if ok else 1
    j["sources"] = new
    path.write_text(json.dumps(j, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
