"""Eval-set renderer: recorded-impulse buckets and reliability-fault buckets."""
import json
import numpy as np, pandas as pd, soundfile as sf
from scripts import render_eval_sets as R


def _dfs(tmp_path):
    rng = np.random.default_rng(0)
    sp = tmp_path / "s.flac"; sf.write(sp, rng.standard_normal(64000).astype(np.float32) * 0.1, 16000)
    nz = tmp_path / "n.flac"; sf.write(nz, rng.standard_normal(96000).astype(np.float32) * 0.1, 16000)
    imp = np.zeros(32000, np.float32); imp[8000:8400] = rng.standard_normal(400).astype(np.float32)
    ip = tmp_path / "i.wav"; sf.write(ip, imp, 16000, subtype="FLOAT")
    return (pd.DataFrame({"path": [str(sp)]}), pd.DataFrame({"path": [str(nz)]}),
            pd.DataFrame({"path": [str(ip)], "source_id": ["mad:shot1"]}))


def test_corpus_impulse_bucket_uses_the_recording_and_its_detected_onset(tmp_path):
    speech, pool, impd = _dfs(tmp_path)
    m, c, meta, twin = R.render_bucket_item([1, 2, 100, 0], speech, pool, 32000, 5.0, "corpus", None, imp_df=impd)
    assert twin is not None and meta["impulse_source"] == "mad:shot1"
    # the bang is 0.5 s into the file; onset must be insertion + ~0.5, never insertion + 0
    on = meta["impulse_onsets_s"]; assert len(on) == 1
    start = int(round((on[0] - 0.5) * 16000)); assert start >= 0
    assert np.array_equal(m[:, :start], twin[:, :start])


def test_synthetic_bucket_records_its_source_kind(tmp_path):
    speech, pool, _ = _dfs(tmp_path)
    _, _, meta, twin = R.render_bucket_item([1, 2, 100, 0], speech, pool, 32000, 5.0, "synthetic", None)
    assert twin is not None and meta["impulse_source"].startswith("synthetic:")


def test_no_impulse_bucket_has_no_twin_and_no_source(tmp_path):
    speech, pool, _ = _dfs(tmp_path)
    _, _, meta, twin = R.render_bucket_item([1, 2, 100, 0], speech, pool, 32000, 5.0, None, None)
    assert twin is None and meta["impulse_source"] is None


def test_fault_item_degrades_only_the_observed_mixture(tmp_path):
    speech, pool, _ = _dfs(tmp_path)
    base = R.render_fault_item([9, 100, 0], speech, pool, 32000, 0.0, "fault_none", None)
    clip = R.render_fault_item([9, 100, 0], speech, pool, 32000, 0.0, "fault_clip_hard", None)
    gain = R.render_fault_item([9, 100, 0], speech, pool, 32000, 0.0, "fault_refgain_-12dB", None)
    # same seed -> same clean target; only the observed mixture differs, and only by the fault
    assert np.array_equal(base[1], clip[1]) and np.array_equal(base[1], gain[1])
    assert np.abs(clip[0][0]).max() < 0.5 * np.abs(base[0][0]).max()
    assert np.array_equal(base[0][1], clip[0][1])                      # clip fault leaves the reference alone
    assert np.allclose(gain[0][1], base[0][1] * 10 ** (-12 / 20), atol=1e-6) and np.array_equal(gain[0][0], base[0][0])
    assert clip[2]["fault"] == "fault_clip_hard" and clip[2]["clipped"] is True
    assert gain[2]["fault"] == "fault_refgain_-12dB" and gain[2]["clipped"] is False
    assert base[2]["fault"] == "fault_none"


def test_fault_burst_bucket_twin_gets_the_same_degradation(tmp_path):
    speech, pool, _ = _dfs(tmp_path)
    m, c, meta, twin = R.render_fault_item([9, 100, 0], speech, pool, 32000, 0.0, "fault_burst_overload", None)
    assert twin is not None and meta["impulse_peak_db"] == 36.0 and meta["clipped"] is True
    start = int(round(meta["impulse_onsets_s"][0] * 16000))
    # twin is clipped at the burst clip's level, so the pre-burst region still matches exactly
    assert np.array_equal(m[:, :start], twin[:, :start])
