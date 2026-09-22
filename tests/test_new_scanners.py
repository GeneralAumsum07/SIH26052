"""scan_wham and scan_vehicle_interior, exercised on synthetic corpora so they run without the
downloads. What matters here is not that they produce rows but that they preserve the two
properties the r6 experiments depend on: WHAM! keeps its inter-channel relation, and the vehicle
set is long enough to fill an evaluation crop."""
import numpy as np
import pytest
import soundfile as sf

from vaani.data import manifests, sources

SR = 16000


def _wham_corpus(tmp_path, split="tr", sr=SR, sessions=(("s1", 2), ("s2", 1))):
    d = tmp_path / "wham" / split
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for name, k in sessions:
        for i in range(k):
            # decorrelated channels: a scanner that collapses to mono would destroy exactly this
            x = rng.standard_normal((sr * 2, 2)).astype(np.float32) * 0.1
            sf.write(d / f"{name}_{i}.wav", x, sr)
    return tmp_path / "wham"


def test_wham_rows_keep_two_channels(tmp_path):
    rows = sources.scan_wham(_wham_corpus(tmp_path), tmp_path / "raw")
    assert len(rows) == 3
    for r in rows:
        x, _ = sf.read(r["path"], always_2d=True)
        assert x.shape[1] == 2, "the two-mic relation is the entire reason this corpus is here"


def test_wham_groups_by_session_not_clip(tmp_path):
    # two clips from one session must share a group so assign() cannot straddle them across splits
    rows = sources.scan_wham(_wham_corpus(tmp_path), tmp_path / "raw")
    groups = {r["source_id"]: r["group_id"] for r in rows}
    assert groups["wham:tr:s1_0"] == groups["wham:tr:s1_1"] != groups["wham:tr:s2_0"]
    assert len({r["split"] for r in rows if r["group_id"] == "wham-s1"}) == 1


def test_wham_refuses_to_scan_the_held_out_split(tmp_path):
    # tt is the reserve generalisation set; putting it in a manifest spends it silently
    with pytest.raises(ValueError, match="held-out"):
        sources.scan_wham(_wham_corpus(tmp_path, split="tt"), tmp_path / "raw", split="tt")


def test_wham_rejects_mono(tmp_path):
    d = tmp_path / "wham" / "tr"
    d.mkdir(parents=True)
    sf.write(d / "s1_0.wav", np.zeros(SR, np.float32), SR)
    with pytest.raises(AssertionError, match="channels"):
        sources.scan_wham(tmp_path / "wham", tmp_path / "raw")


def test_wham_resamples_to_16k(tmp_path):
    rows = sources.scan_wham(_wham_corpus(tmp_path, sr=48000), tmp_path / "raw")
    for r in rows:
        assert sf.info(r["path"]).samplerate == SR


def _vehicle_corpus(tmp_path, clip_s=3.0, classes=("bus", "jeep"), per_class=4, sr=48000):
    rng = np.random.default_rng(0)
    for c in classes:
        d = tmp_path / "vehicle" / c
        d.mkdir(parents=True)
        for i in range(per_class):
            sf.write(d / f"{i}.wav", (rng.standard_normal(int(sr * clip_s)) * 0.1).astype(np.float32), sr)
    return tmp_path / "vehicle"


def test_vehicle_concatenates_one_row_per_class(tmp_path):
    rows = sources.scan_vehicle_interior(_vehicle_corpus(tmp_path), tmp_path / "raw")
    assert len(rows) == 2
    assert {r["group_id"] for r in rows} == {"vehicle-bus", "vehicle-jeep"}


def test_vehicle_rows_outlast_the_evaluation_crop(tmp_path):
    # the defect this scanner exists to avoid: 3 s clips against a 6 s crop reach mix() short,
    # because render_bucket_item pads speech but not noise
    rows = sources.scan_vehicle_interior(_vehicle_corpus(tmp_path, clip_s=3.0), tmp_path / "raw")
    for r in rows:
        assert r["duration_s"] >= 6.0


def test_vehicle_resamples_to_16k(tmp_path):
    rows = sources.scan_vehicle_interior(_vehicle_corpus(tmp_path), tmp_path / "raw")
    for r in rows:
        assert sf.info(r["path"]).samplerate == SR


@pytest.mark.parametrize("scan,fixture", [(sources.scan_wham, _wham_corpus),
                                          (sources.scan_vehicle_interior, _vehicle_corpus)])
def test_rows_match_the_manifest_schema(tmp_path, scan, fixture):
    rows = scan(fixture(tmp_path), tmp_path / "raw")
    assert rows and all(set(r) == set(manifests.COLUMNS) for r in rows)


def test_brickwall_ratio_separates_upsampled_from_genuine():
    """The guard protecting the defence-stationary corpus: a mirror copy laundered through an
    upsample keeps its 4 kHz cliff, and a samplerate check alone cannot see it."""
    from scipy.signal import resample_poly
    rng = np.random.default_rng(0)
    genuine = rng.standard_normal(SR * 5).astype(np.float32)
    assert sources.brickwall_ratio(genuine, SR) < sources.BRICKWALL_MAX

    # 8 kHz source upsampled back to 16 kHz: the container now says 16 kHz, the content does not
    narrow = rng.standard_normal(8000 * 5).astype(np.float32)
    laundered = resample_poly(narrow, 2, 1).astype(np.float32)
    assert sources.brickwall_ratio(laundered, SR) > sources.BRICKWALL_MAX


def test_brickwall_ratio_passes_genuinely_lowpass_content():
    # a tank or car interior really does have almost nothing above 4 kHz; that must not read as a cliff
    rng = np.random.default_rng(0)
    from scipy.signal import butter, lfilter
    b, a = butter(4, 1200 / (SR / 2))
    lowpass = lfilter(b, a, rng.standard_normal(SR * 5)).astype(np.float32)
    assert sources.brickwall_ratio(lowpass, SR) < sources.BRICKWALL_MAX
