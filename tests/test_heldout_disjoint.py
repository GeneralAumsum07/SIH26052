"""The generalisation claim rests on the held-out corpus never having entered training. These tests
pin the check that proves it, and -- once the corpus is scanned on the training box -- assert the
property itself against the real manifests."""
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts import check_heldout as ch

RECIPES = ["configs/retraining/r5_continue128.yaml", "configs/exp/vaani_full_r4_ctl.yaml"]
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


def test_a_recipe_naming_a_missing_manifest_is_skipped_not_fatal(tmp_path):
    held = _manifest(tmp_path / "held.parquet", ["ccc"])
    r = _recipe(tmp_path / "r.yaml", [tmp_path / "absent.parquet"])
    assert ch.overlaps([held], [r]) == {}


@pytest.mark.skipif(not all(Path(p).exists() for p in HELDOUT),
                    reason="generalisation corpus not scanned yet; runs on the training box")
def test_the_real_heldout_corpus_is_disjoint_from_every_recipe():
    assert ch.overlaps(HELDOUT, [r for r in RECIPES if Path(r).exists()]) == {}
