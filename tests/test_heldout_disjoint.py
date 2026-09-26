"""The generalisation claim rests on the held-out corpus never having entered training. These tests
pin the check that proves it, and assert the property itself against the manifests the shipping r7
recipe and every r8 recipe actually train on.

Manifests of the corpora only the training box downloads (plan 11.5 additions) skip here with a reason;
with VAANI_R8_BOX=1 (exported by the box setup) a missing one fails, so the box cannot pass vacuously."""
import glob, os
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts import check_heldout as ch
from scripts import heldout_freesound as hf
from vaani.data import manifests
from vaani.data.dataset import drop_groups, load_exclude_groups

REPO = Path(__file__).resolve().parents[1]
# backbone recipes (r7 shipping + every r8 retrain and ablation); refiner recipes name no manifests
R8 = sorted(Path(p).relative_to(REPO).as_posix() for p in
            glob.glob(str(REPO / "configs/retraining/r8_*.yaml")) + glob.glob(str(REPO / "configs/retraining/r8_ablations/*.yaml")))
RECIPES = ["configs/retraining/r7_e256_wr64.yaml"] + [r for r in R8 if "refiner" not in r]
HELDOUT = ["data/manifests/vehicle_interior.parquet"]
# downloaded and scanned on the box only (configs/data/R8_DATASETS.md); the laptop has none of them
BOX_ONLY = {"demand_pairs.parquet", "avq_drone.parquet", "c3gd.parquet", "fsd50k.parquet", "lombard_grid.parquet",
            "libritts_r.parquet"}
ON_BOX = os.environ.get("VAANI_R8_BOX") == "1"
R8_TEST_INDEX = REPO / "data/eval_r8_test/test/index.csv"


def _manifest(path, sha1s):
    pd.DataFrame({"sha1": sha1s}).to_parquet(path)
    return path


def _recipe(path, manifests):
    path.write_text(yaml.safe_dump({"data": {"manifests": [str(m) for m in manifests]}}), encoding="utf-8")
    return path


def _box_only_missing(paths):
    """Fail on the box, skip elsewhere, when a box-only manifest is absent."""
    missing = sorted({Path(p).name for p in paths if not (REPO / p).exists()})
    if missing and ON_BOX:
        pytest.fail(f"box-only manifest(s) not scanned (configs/data/R8_DATASETS.md): {missing}")
    return missing


def _r8_pool(recipe):
    """The rows a recipe trains and selects on: train+val split, held-out groups dropped, as DynamicMixDataset does."""
    d = yaml.safe_load(open(REPO / recipe, encoding="utf-8"))["data"]
    df = pd.concat([manifests.read(REPO / m) for m in d["manifests"] if (REPO / m).exists()])
    return drop_groups(df[df.split.isin(["train", "val"])], load_exclude_groups(REPO / d["exclude_groups_file"]))


def _r8_test_sources():
    if not R8_TEST_INDEX.exists():
        if ON_BOX:
            pytest.fail(f"{R8_TEST_INDEX} absent: stage the frozen r8 test set before launch")
        pytest.skip(f"{R8_TEST_INDEX} absent here")
    idx = pd.read_csv(R8_TEST_INDEX, dtype=str)
    return {s for c in ("speech_source", "noise_source", "impulse_source") for v in idx[c].dropna()
            for s in str(v).split(";") if s}


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


def test_freesound_ids_parse_for_every_freesound_corpus():
    assert hf.freesound_id("dnsn:door_Freesound_validated_151089_0") == "151089"
    assert hf.freesound_id("esc50:door_wood_knock/3-151089-A-30") == "151089"
    assert hf.freesound_id("fsd50k:Siren/151089") == "151089"
    assert hf.freesound_id("dnsn:x_AudioSet_1") is None and hf.freesound_id("mad:helicopter/1_1") is None


def test_the_recipes_and_every_manifest_they_name_are_present():
    # a missing manifest would make the disjointness check vacuous, so it fails rather than skips
    assert len(R8) >= 24, R8
    missing = [r for r in RECIPES if not (REPO / r).exists()]
    assert not missing, f"recipe(s) not found: {missing}"
    named = sorted({m.as_posix() for r in RECIPES for m in ch.recipe_manifests(REPO / r)})
    assert named, "the recipes name no manifests"
    box = [m for m in named if Path(m).name in BOX_ONLY]
    absent = [m for m in named if m not in box and not (REPO / m).exists()] + [h for h in HELDOUT if not (REPO / h).exists()]
    assert not absent, f"manifest(s) not found (scan them on this box first): {absent}"
    if _box_only_missing(box):
        pytest.skip(f"box-only manifests absent here (VAANI_R8_BOX=1 makes this fail): {_box_only_missing(box)}")


def test_the_real_heldout_corpus_is_disjoint_from_the_training_data():
    absent = [h for h in HELDOUT if not (REPO / h).exists()]
    assert not absent, f"held-out manifest(s) not found: {absent}"
    assert ch.overlaps([REPO / h for h in HELDOUT], [REPO / r for r in RECIPES]) == {}


@pytest.mark.timeout(600)
def test_no_r8_test_source_reaches_an_r8_training_pool():
    # the frozen r8 test set's speech/noise/impulse rows (incl. the Lombard bucket's EARS clips) are never train or val
    src = _r8_test_sources()
    recipes = sorted({(tuple(yaml.safe_load(open(REPO / r, encoding="utf-8"))["data"]["manifests"]), r) for r in R8
                      if "refiner" not in r}, key=lambda t: t[0])
    seen = set()
    for mans, r in recipes:
        if mans in seen:
            continue
        seen.add(mans)
        pool = _r8_pool(r)
        assert not (src & set(pool.source_id)), (r, sorted(src & set(pool.source_id))[:5])
        ids = {hf.freesound_id(s) for s in src} - {None}
        shared = pool[pool.source_id.map(hf.freesound_id).isin(ids)]
        assert shared.empty, (f"{r}: {len(shared)} rows share a Freesound recording with an r8 test source; "
                              f"run python scripts/heldout_freesound.py", shared.source_id.head().tolist())
    _box_only_missing([m for mans, _ in recipes for m in mans if Path(m).name in BOX_ONLY])


def test_freesound_sibling_exclusion_is_current():
    _r8_test_sources()
    assert hf.main(["--check"]) == 0, "configs/data/r8_heldout_exclude.json is stale: python scripts/heldout_freesound.py"


def test_libritts_r_speakers_are_not_librispeech_val_or_test_speakers():
    # LibriTTS-R is LibriSpeech re-cut: its lttsr-spk-* groups hash to other splits than ls-spk-*, so a val/test
    # LibriSpeech speaker (eval_r2, eval_r8_test) would re-enter training unless only disjoint subsets are scanned
    named = sorted({m for r in R8 for m in (yaml.safe_load(open(REPO / r, encoding="utf-8")).get("data") or {}).get("manifests", [])
                    if "libritts" in Path(m).name})
    if not named:
        pytest.skip("no r8 recipe trains LibriTTS-R (deferred: plan 11.5 speech additions)")
    if _box_only_missing(named):
        pytest.skip(f"{named} absent here")
    ls = manifests.read(REPO / "data/manifests/librispeech.parquet")
    held = set(ls[ls.split.isin(["val", "test"])].speaker_id.astype(str))
    spk = set(pd.concat([manifests.read(REPO / m) for m in named]).speaker_id.astype(str))
    assert not (spk & held), sorted(spk & held)[:10]
