"""Hold out every training row cut from the same Freesound recording as an r8 test-set noise source.

DNS noise rows are grouped per 10 s chunk (dnsn-<stem>), so chunks of one Freesound upload land in different splits;
ESC-50 and FSD50K clips carry the same Freesound ids. A test item whose noise recording has sibling chunks in the
training pool is not held out. This adds one "freesound_<manifest>" source per Freesound-derived recipe manifest to
configs/data/r8_heldout_exclude.json (the other sources, written by render_eval_sets.py --write-heldout, are kept).

    uv run python scripts/heldout_freesound.py            # rewrite the freesound_* sources from the local manifests
    uv run python scripts/heldout_freesound.py --check    # exit 1 if the file would change (box, after scanning fsd50k)

The test set is not on the box, so its source_ids are committed in SOURCES (written from the index with --write-sources
on the laptop; tests/test_heldout_disjoint.py reads it too). The index, when present, wins, and --check fails if the
two disagree.
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
SOURCES = "configs/data/r8_test_sources.json"
EVALSET_HASH = "results_r2/r8/testset/EVALSET_HASH"


def freesound_id(source_id):
    for rx in FS_ID:
        m = rx.match(str(source_id))
        if m:
            return m.group(1)
    return None


def test_sources(index_csv, cols=("noise_source", "impulse_source")):
    """Every source_id in cols the frozen r8 test set mixed (read only)."""
    idx = pd.read_csv(index_csv, dtype=str)
    # v2 scenes join their noise sources with "+" (render_eval_sets.py); no manifest source_id contains "+" or ";"
    return {s for c in cols if c in idx for v in idx[c].dropna() for s in re.split(r"[;+]", str(v)) if s}


def index_sources(root, index_csv=INDEX):
    return {"speech": test_sources(root / index_csv, ("speech_source",)), "noise": test_sources(root / index_csv)}


def list_sources(root, sources=SOURCES):
    j = json.loads((root / sources).read_text(encoding="utf-8"))
    return {k: set(j[k]) for k in ("speech", "noise")}


def load_test_sources(root, index_csv=INDEX, sources=SOURCES):
    """({"speech": ids, "noise": ids (noise + impulse)}, where from): the index if present, else the committed list."""
    if (root / index_csv).exists():
        return index_sources(root, index_csv), "index"
    return list_sources(root, sources), "list"


def write_sources(root, index_csv=INDEX, sources=SOURCES):
    if not (root / index_csv).exists():
        raise SystemExit(f"{index_csv} absent: --write-sources runs where the test set is")
    src = index_sources(root, index_csv)
    h = root / EVALSET_HASH
    j = {"schema": "vaani.r8_test_sources/1", "index": index_csv,
         "evalset_hash": h.read_text(encoding="utf-8").strip() if h.exists() else None,
         "generated_by": "python scripts/heldout_freesound.py --write-sources",
         "rule": "speech: speech_source; noise: noise_source + impulse_source; split on ';' and '+'",
         "speech": sorted(src["speech"]), "noise": sorted(src["noise"])}
    (root / sources).write_text(json.dumps(j, indent=1) + "\n", encoding="utf-8", newline="\n")
    return sum(len(v) for v in src.values())


def sibling_sources(root, index_csv, recipes=RECIPES, sources=SOURCES):
    """{manifest file: rows} of recipe manifests sharing a Freesound recording with a test source (test rows kept)."""
    src = load_test_sources(root, index_csv, sources)[0]["noise"]
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
    ap.add_argument("--sources", default=SOURCES)
    ap.add_argument("--check", action="store_true", help="exit 1 if the file is out of date; write nothing")
    ap.add_argument("--write-sources", action="store_true", help="rewrite --sources from the test index (laptop)")
    a = ap.parse_args(argv)
    root = Path(a.root); path = root / a.exclude
    if a.write_sources:
        print(f"{a.sources}: {write_sources(root, INDEX, a.sources)} test source ids")
        return 0
    src, where = load_test_sources(root, INDEX, a.sources)
    sp = root / a.sources
    stale_list = where == "index" and (not sp.exists() or list_sources(root, a.sources) != src)
    print(f"test sources from the {where}: {len(src['speech'])} speech, {len(src['noise'])} noise/impulse")
    j = json.loads(path.read_text(encoding="utf-8"))
    sib, n_ids = sibling_sources(root, INDEX, sources=a.sources)
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
        if stale_list:
            print(f"OUT OF DATE: {a.sources} differs from {INDEX}; rerun with --write-sources")
        return 0 if ok and not stale_list else 1
    j["sources"] = new
    path.write_text(json.dumps(j, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
