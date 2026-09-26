"""Test items whose noise or impulse recording shares a Freesound id with a held-out train/val row (r7 trained on them).

Reads only index.csv and configs/data/r8_heldout_exclude.json. v2 noise sources are "+"-joined and split into their
parts, as scripts/heldout_freesound.py does. Writes freesound_shared_items.csv beside this file and prints the counts.
    .venv/Scripts/python.exe results_r2/r8/testset/freesound_shared.py [data/eval_r8_test_b/test/index.csv]
"""
import json, re, sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.heldout_freesound import freesound_id  # noqa: E402


def ids(v):
    return {freesound_id(s) for s in re.split(r"[;+]", v) if s} - {None} if isinstance(v, str) else set()


def main(index_csv="data/eval_r8_test_b/test/index.csv"):
    j = json.loads((REPO / "configs/data/r8_heldout_exclude.json").read_text(encoding="utf-8"))["sources"]
    fs = {freesound_id(s) for k, v in j.items() if k.startswith("freesound_") for s in v["source_ids"]} - {None}
    d = pd.read_csv(REPO / index_csv, dtype=str)
    by_noise, by_imp = d.noise_source.map(lambda v: bool(ids(v) & fs)), d.impulse_source.map(lambda v: bool(ids(v) & fs))
    hit = d[by_noise | by_imp].assign(by=lambda x: ["noise" if n else "impulse" for n in by_noise[x.index]])
    hit[["bucket", "id", "subset", "category", "noise_source", "impulse_source", "by"]].to_csv(
        Path(__file__).with_name("freesound_shared_items.csv"), index=False)
    print(f"{index_csv}: {len(hit)} of {len(d)} items share a Freesound id with a held-out train/val row; "
          f"by subset {hit.subset.value_counts().to_dict()}; by noise source {int(by_noise.sum())}, "
          f"{int((by_imp & ~by_noise).sum())} more by impulse source")


if __name__ == "__main__":
    main(*sys.argv[1:])
