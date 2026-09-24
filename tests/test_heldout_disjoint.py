"""The generalisation claim rests on the held-out corpus never having entered training. These tests
pin the check that proves it, and assert the property itself against the manifests the shipping r7
recipe actually trained on."""
from pathlib import Path

import pandas as pd
import yaml

from scripts import check_heldout as ch

# r7 backbone recipe; the r7 refiner recipe names no manifests (it inherits the backbone's data)
RECIPES = ["configs/retraining/r7_e256_wr64.yaml"]
HELDOUT = ["data/manifests/vehicle_interior.parquet"]


def _manifest(path, sha1s):
    pd.DataFrame({"sha1": sha1s}).to_parquet(path)
    return path


def _recipe(path, manifests):
    path.write_text(yaml.safe_dump({"data": {"manifests": [str(m) for m in manifests]}}), encoding="utf-8")
    return path


def test_overlap_is_detected(tmp_path):
    train = _manifest(tmp_path / "train.parquet", ["aaa", "bbb"])
    held = _manifest(tmp_path / "held.parquet", ["bbb", "ccc"])
    found = ch.overlaps([held], [_recipe(tmp_path / "r.yaml", [train])])
    assert found and next(iter(found.values())) == {"bbb"}


def test_disjoint_corpora_report_nothing(tmp_path):
    train = _manifest(tmp_path / "train.parquet", ["aaa", "bbb"])
    held = _manifest(tmp_path / "held.parquet", ["ccc", "ddd"])
    assert ch.overlaps([held], [_recipe(tmp_path / "r.yaml", [train])]) == {}


def test_empty_hashes_are_never_counted_as_shared(tmp_path):
    # an unhashed row is unknown provenance, not proof of sharing; counting it would fail honest sets
    train = _manifest(tmp_path / "train.parquet", ["", ""])
    held = _manifest(tmp_path / "held.parquet", ["", "ccc"])
    assert ch.overlaps([held], [_recipe(tmp_path / "r.yaml", [train])]) == {}


def test_the_checker_silently_skips_a_missing_training_manifest(tmp_path):
    # the checker tolerates absent manifests, so the real-corpus tests below must demand them explicitly
    held = _manifest(tmp_path / "held.parquet", ["ccc"])
    r = _recipe(tmp_path / "r.yaml", [tmp_path / "absent.parquet"])
    assert ch.overlaps([held], [r]) == {}


def test_the_r7_recipe_and_every_manifest_it_names_are_present():
    # a missing manifest would make the disjointness check vacuous, so it fails rather than skips
    missing = [r for r in RECIPES if not Path(r).exists()]
    assert not missing, f"recipe(s) not found: {missing}"
    named = [m for r in RECIPES for m in ch.recipe_manifests(r)]
    assert named, "r7 recipe names no manifests"
    absent = [str(m) for m in named if not m.exists()] + [h for h in HELDOUT if not Path(h).exists()]
    assert not absent, f"manifest(s) not found (scan them on this box first): {absent}"


def test_the_real_heldout_corpus_is_disjoint_from_the_r7_training_data():
    absent = [h for h in HELDOUT if not Path(h).exists()]
    assert not absent, f"held-out manifest(s) not found: {absent}"
    assert ch.overlaps(HELDOUT, RECIPES) == {}
