"""Prove a generalisation corpus is held out, rather than asserting it in prose.

The claim the eval_gen set exists to support is "this noise never entered training". That claim is
worth nothing if it rests on someone remembering not to add a manifest to a recipe. Here it is a
checkable property: the content hashes of the generalisation manifests must be disjoint from those
of every manifest any training recipe reads.

Content hashes rather than paths, because the same audio re-scanned to a different location is the
same audio. The check is deliberately one-directional and total: any single shared sha1 fails it.

    uv run python scripts/check_heldout.py --heldout data/manifests/vehicle_interior.parquet \\
        --recipes configs/retraining/r5_continue128.yaml configs/exp/vaani_full_r4_ctl.yaml
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml


def sha1_set(manifest_path):
    """Content hashes in one manifest. Empty hashes are unknown, not shared, so they never count as overlap."""
    s = pd.read_parquet(manifest_path, columns=["sha1"]).sha1.astype(str)
    return set(s[s.str.len() > 0])


def recipe_manifests(recipe_path):
    cfg = yaml.safe_load(Path(recipe_path).read_text(encoding="utf-8"))
    node = cfg.get("data", cfg)
    return [Path(m) for m in node.get("manifests", [])]


def overlaps(heldout_paths, recipe_paths):
    """{(heldout, training) -> shared hashes} for every pair that shares any audio."""
    training = {}
    for r in recipe_paths:
        for m in recipe_manifests(r):
            if m.exists():
                training.setdefault(m, sha1_set(m))
    found = {}
    for h in heldout_paths:
        hs = sha1_set(h)
        for m, ts in training.items():
            shared = hs & ts
            if shared:
                found[(Path(h).name, m.name)] = shared
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--heldout", nargs="+", required=True)
    ap.add_argument("--recipes", nargs="+", required=True)
    a = ap.parse_args()

    missing = [p for p in a.heldout if not Path(p).exists()]
    if missing:
        raise SystemExit(f"held-out manifest(s) not found: {', '.join(missing)}")

    found = overlaps(a.heldout, a.recipes)
    if found:
        for (h, m), shared in found.items():
            print(f"OVERLAP {h} <-> {m}: {len(shared)} shared clips, e.g. {sorted(shared)[:3]}")
        raise SystemExit("held-out corpus is not held out; the generalisation claim does not hold")
    print(f"disjoint: {len(a.heldout)} held-out manifest(s) share no audio with any manifest in "
          f"{len(a.recipes)} recipe(s)")


if __name__ == "__main__":
    sys.exit(main())
