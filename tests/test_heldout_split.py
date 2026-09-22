"""A corpus reserved as held-out must not be split 80/10/10 like a training corpus.

vaani.data.splits.assign() hashes a group id into train/val/test so that a corpus which IS trained on
cannot leak into evaluation. vehicle_interior never enters a training recipe (see
results_r2/generalisation/PROTOCOL.md, enforced by scripts/check_heldout.py), so the split has nothing
to protect against and only discards material. On its eight groups assign() drew 5 train / 1 val /
2 test and put the corpus's only stationary class into train, which left data/eval_gen unrenderable:
render_eval_sets raised "no noise rows for bucket 'stationary'".
"""
import numpy as np
import pytest
import soundfile as sf

from vaani.data.sources import SR, scan_vehicle_interior
from vaani.data.splits import assign


@pytest.fixture
def corpus(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "vehicle_interior"
    for i in range(1, 9):
        d = root / f"class{i:02d}"
        d.mkdir(parents=True)
        # two clips per class, so the scanner's concatenation path runs as it does for real
        for j in range(2):
            sf.write(d / f"{i} ({j}).wav", (0.05 * rng.standard_normal(SR)).astype(np.float32), SR)
    return root, tmp_path / "raw"


def test_vehicle_interior_is_held_out_whole(corpus):
    root, out = corpus
    rows = scan_vehicle_interior(root, out)
    assert len(rows) == 8, "one row per vehicle class"
    assert {r["split"] for r in rows} == {"test"}, (
        "a held-out corpus must be entirely test; splitting it discards material for no benefit"
    )


def test_the_override_is_load_bearing(corpus):
    """Guards against the override being quietly dropped: assign() alone does NOT put these in test."""
    root, out = corpus
    rows = scan_vehicle_interior(root, out)
    natural = [assign(r["group_id"]) for r in rows]
    assert any(s != "test" for s in natural), (
        "fixture no longer exercises the override - pick group ids that assign() sends elsewhere"
    )
