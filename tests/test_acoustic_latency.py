"""Known physical delays, not benchmark timers, are the measurement oracle."""
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from vaani.acoustic_latency import AnalysisConfig, analyze_capture, inspect_isolation, make_stimulus, normalized_correlation


def capture(delay_samples=96, sr=8000, seconds=8, gain=0.4):
    # Broadband, aperiodic input makes a unique known arrival time. It is an
    # estimator fixture, not evidence that a denoiser preserves this stimulus.
    x = np.random.default_rng(42).normal(0, 0.08, sr * seconds)
    y = np.zeros_like(x)
    y[delay_samples:] = gain * x[:-delay_samples]
    return np.stack([x, y])


def config(**kw):
    return AnalysisConfig(max_delay_ms=80, warmup_s=0, **kw)


def write_pcm(path, x, sr=8000):
    with wave.open(str(path), "wb") as f:
        f.setnchannels(len(x)); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes((np.clip(x, -1, 1).T * 32767).round().astype("<i2").tobytes())


def test_positive_delay_is_not_reversed_or_a_hop_timer():
    report = analyze_capture(capture(), 8000, config=config())
    assert report["status"] == "ok"
    assert report["acoustic_delay_ms"]["median"] == pytest.approx(12.0, abs=0.125)
    assert report["accepted_windows"] >= 3
    assert report["corrected_system_delay_ms"] is None


@pytest.mark.parametrize("sr,shift,want", [(16000, 640, 40.0), (48000, 576, 12.0)])
def test_gain_polarity_and_dc_do_not_change_arrival_time(sr, shift, want):
    x = capture(shift, sr=sr, gain=-0.15)
    x[0] += 0.05; x[1] -= 0.1
    report = analyze_capture(x, sr, config=config())
    assert report["status"] == "ok"
    assert report["acoustic_delay_ms"]["median"] == pytest.approx(want, abs=1000 / sr)


def test_fft_correlation_matches_independent_full_overlap_dot_products():
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=13), rng.normal(size=31)
    xc = x - x.mean()
    want = []
    for start in range(19):
        yc = y[start:start + 13] - y[start:start + 13].mean()
        want.append(np.dot(xc, yc) / np.sqrt(np.dot(xc, xc) * np.dot(yc, yc)))
    assert normalized_correlation(x, y) == pytest.approx(want, abs=1e-10)


def test_silence_has_no_numeric_latency():
    report = analyze_capture(np.zeros((2, 64000)), 8000, config=config())
    assert report["status"] == "inconclusive"
    assert report["acoustic_delay_ms"] is None
    json.dumps(report, allow_nan=False)


def test_output_dropout_is_rejected_and_prevents_a_success_report():
    x = capture()
    x[1, 16000:40000] = 0
    report = analyze_capture(x, 8000, config=config())
    assert report["status"] == "inconclusive"
    assert report["accepted_windows"] > 0
    assert report["rejected_windows"] > 0
    assert report["accepted_fraction"] < 1
    assert not report["threshold_assessment"]["supported_for_this_recording"]


def test_direct_sound_leakage_cannot_become_a_false_low_latency_result():
    x = capture()
    x[1] += x[0]
    report = analyze_capture(x, 8000, config=config())
    assert report["status"] == "inconclusive"
    assert report["accepted_windows"] == 0
    assert all(w["reason"] == "near_zero_direct_leakage" for w in report["windows"])


def test_reversed_channels_are_not_reported_as_positive_latency():
    report = analyze_capture(capture()[::-1], 8000, config=config())
    assert report["accepted_windows"] == 0
    assert all(w["reason"] == "negative_lag_check_channels" for w in report["windows"])


def test_periodic_audio_is_ambiguous_not_a_precise_delay_measurement():
    t = np.arange(64000) / 8000
    x = np.sin(2 * np.pi * 200 * t) * 0.1
    y = np.sin(2 * np.pi * 200 * (t - 0.012)) * 0.05
    report = analyze_capture(np.stack([x, y]), 8000, config=config())
    assert report["status"] == "inconclusive"
    assert report["accepted_windows"] == 0


def test_two_equal_delayed_paths_are_rejected():
    x = capture()
    x[1] += capture(delay_samples=480)[1]
    report = analyze_capture(x, 8000, config=config())
    assert report["accepted_windows"] == 0
    assert all(w["reason"] == "ambiguous_peak" for w in report["windows"])


def test_clipped_input_cannot_produce_an_ok_measurement():
    x = capture()
    x[0] = np.clip(x[0] * 30, -1, 1)
    report = analyze_capture(x, 8000, config=config())
    assert report["accepted_windows"] == 0
    assert all(w["reason"] == "clipped_reference" for w in report["windows"])


def test_at_or_beyond_search_limit_is_not_accepted():
    report = analyze_capture(capture(delay_samples=640), 8000, config=config())
    assert report["accepted_windows"] == 0
    assert all(w["reason"] == "search_boundary" for w in report["windows"])
    report = analyze_capture(capture(delay_samples=1600), 8000, config=config())
    assert report["accepted_windows"] == 0


def test_explicit_calibration_and_geometry_are_both_needed_for_device_delay():
    x = capture(delay_samples=240)
    report = analyze_capture(x, 8000, config=config(), channel_offset_ms=1.0)
    assert report["corrected_system_delay_ms"] is None
    report = analyze_capture(x, 8000, config=config(), channel_offset_ms=1.0,
                             geometry_correction_ms=2.0)
    assert report["acoustic_delay_ms"]["median"] == pytest.approx(30.0, abs=0.125)
    assert report["corrected_system_delay_ms"]["median"] == pytest.approx(27.0, abs=0.125)


def test_impossible_corrected_delay_is_not_supported():
    report = analyze_capture(capture(), 8000, config=config(), channel_offset_ms=50,
                             geometry_correction_ms=0)
    assert report["status"] == "inconclusive"
    assert not report["threshold_assessment"]["supported_for_this_recording"]


@pytest.mark.parametrize("bad", [np.zeros((1, 64000)), np.full((2, 64000), np.nan)])
def test_invalid_capture_is_refused(bad):
    with pytest.raises(ValueError):
        analyze_capture(bad, 8000, config=config())


def test_short_recording_is_refused_not_correlated_using_partial_overlap():
    with pytest.raises(ValueError):
        analyze_capture(np.ones((2, 100)), 8000, config=config())


def test_calibration_allows_signed_channel_offset_and_zero_delay():
    cfg = config(min_delay_ms=0, calibration=True)
    x = capture()[::-1]
    report = analyze_capture(x, 8000, config=cfg)
    assert report["status"] == "ok"
    assert report["acoustic_delay_ms"]["median"] == pytest.approx(-12, abs=0.125)
    x = capture(); x[1] = x[0].copy()
    report = analyze_capture(x, 8000, config=cfg)
    assert report["status"] == "ok"
    assert abs(report["acoustic_delay_ms"]["median"]) < 0.125


def test_stimulus_preserves_speech_and_separates_repeats():
    speech = np.random.default_rng(1).normal(0, 0.1, 8000)
    y, manifest = make_stimulus(speech, 8000, repeats=3, gap_s=2, lead_s=1, tail_s=1)
    assert len(y) == 72000  # 1 + 3*1 + 2*2 + 1 = 9 seconds.
    assert manifest["trial_start_s"] == [1.0, 4.0, 7.0]
    assert np.array_equal(y[8000:16000], y[32000:40000])
    assert np.max(np.abs(y[16000:32000])) == 0
    assert 20 * np.log10(np.sqrt(np.mean(y[8000:16000] ** 2))) == pytest.approx(-25, abs=0.01)


def test_stimulus_refuses_silent_probe_and_aliased_repetitions():
    with pytest.raises(ValueError):
        make_stimulus(np.zeros(8000), 8000)
    with pytest.raises(ValueError):
        make_stimulus(np.ones(8000), 8000, gap_s=0.1)


def test_analyze_cli_writes_json_and_returns_nonzero_for_inconclusive_data(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/acoustic_latency.py"
    wav = tmp_path / "capture.wav"; out = tmp_path / "report.json"
    write_pcm(wav, capture())
    args = [sys.executable, str(script), "analyze", str(wav), "--out", str(out),
            "--model-label", "synthetic_validation", "--transport", "wired",
            "--recorder-description", "synthetic common-clock fixture", "--synchronized-stereo",
            "--warmup-s", "0", "--max-delay-ms", "80"]
    p = subprocess.run(args, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    report = json.loads(out.read_text())
    assert report["acoustic_delay_ms"]["median"] == pytest.approx(12, abs=0.125)
    assert report["recording_sha256"] and report["model_label"] == "synthetic_validation"
    assert not report["threshold_assessment"]["supported_for_this_recording"]
    write_pcm(wav, np.zeros((2, 64000)))
    p = subprocess.run(args, capture_output=True, text=True)
    assert p.returncode == 2
    assert json.loads(out.read_text())["status"] == "inconclusive"


def test_cli_requires_common_clock_capture_attestation(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/acoustic_latency.py"
    wav = tmp_path / "input.wav"
    write_pcm(wav, capture())
    p = subprocess.run([sys.executable, str(script), "analyze", str(wav), "--out", str(tmp_path / "r.json"),
                        "--model-label", "r7", "--transport", "wired", "--recorder-description", "recorder"],
                       capture_output=True, text=True)
    assert p.returncode != 0


def test_prepare_cli_creates_repeated_speech_and_a_manifest(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/acoustic_latency.py"
    speech, out = tmp_path / "speech.wav", tmp_path / "stimulus.wav"
    write_pcm(speech, capture()[:1, :8000])
    p = subprocess.run([sys.executable, str(script), "prepare", "--speech-wav", str(speech),
                        "--out", str(out), "--repeats", "2"], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    with wave.open(str(out)) as f:
        assert f.getnchannels() == 1 and f.getnframes() == 80000
    manifest = json.loads(out.with_suffix(".json").read_text())
    assert manifest["trial_start_s"] == [3.0, 6.0]
    assert manifest["source_sha256"]


def test_cli_calibration_is_required_to_match_recorder_and_rate(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/acoustic_latency.py"
    wav, calibration, out = tmp_path / "c.wav", tmp_path / "cal.json", tmp_path / "r.json"
    x = capture(); x[1] = x[0].copy()
    write_pcm(wav, x)
    common = ["--recorder-description", "same recorder", "--synchronized-stereo", "--warmup-s", "0"]
    p = subprocess.run([sys.executable, str(script), "calibrate", str(wav), "--out", str(calibration),
                        *common], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    write_pcm(wav, capture())
    args = [sys.executable, str(script), "analyze", str(wav), "--out", str(out),
            "--model-label", "synthetic", "--transport", "wired", "--max-delay-ms", "80",
            "--calibration-json", str(calibration), "--geometry-correction-ms", "0", *common]
    p = subprocess.run(args, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert not json.loads(out.read_text())["threshold_assessment"]["supported_for_this_recording"]
    # A valid calibration would otherwise be replaced by the new run's JSON.
    alias_args = args.copy()
    alias_args[alias_args.index("--out") + 1] = str(calibration)
    original = calibration.read_bytes()
    p = subprocess.run(alias_args, capture_output=True, text=True)
    assert p.returncode != 0
    assert calibration.read_bytes() == original
    bad = json.loads(calibration.read_text()); bad["recorder_description"] = "other recorder"
    calibration.write_text(json.dumps(bad))
    p = subprocess.run(args, capture_output=True, text=True)
    assert p.returncode != 0


def test_window_estimates_track_variable_delay_instead_of_only_one_global_peak():
    x = capture(delay_samples=96)
    alternate = capture(delay_samples=320)
    x[1, 32000:] = alternate[1, 32000:]
    report = analyze_capture(x, 8000, config=config())
    accepted = [r["delay_ms"] for r in report["windows"] if r["accepted"]]
    assert min(accepted) == pytest.approx(12, abs=0.125)
    assert max(accepted) == pytest.approx(40, abs=0.125)


def test_band_coloring_and_noise_preserve_a_unique_delay():
    x = capture(delay_samples=320)
    x[1] = np.convolve(x[1], [0.8, 0.1, 0.05], mode="full")[:x.shape[1]]
    x[1] += np.random.default_rng(9).normal(0, 0.005, x.shape[1])
    report = analyze_capture(x, 8000, config=config())
    assert report["status"] == "ok"
    assert report["acoustic_delay_ms"]["median"] == pytest.approx(40, abs=0.125)


def test_partial_dropouts_with_high_global_correlation_are_still_rejected():
    x = capture()
    for start in range(0, x.shape[1], 8000):
        x[1, start:start + 4000] = 0
    report = analyze_capture(x, 8000, config=config(), channel_offset_ms=0,
                             geometry_correction_ms=0, setup_verified=True, isolation_verified=True)
    assert report["status"] == "inconclusive"
    assert report["rejected_windows"] > 0
    assert any(w["reason"] == "response_dropout" for w in report["windows"])
    assert not report["threshold_assessment"]["supported_for_this_recording"]


@pytest.mark.parametrize("boundary", [640, -640])
def test_competing_boundary_echo_is_not_ignored(boundary):
    x = capture()
    if boundary > 0:
        x[1, boundary:] += 0.39 * x[0, :-boundary]
    else:
        x[1, :boundary] += 0.39 * x[0, -boundary:]
    report = analyze_capture(x, 8000, config=config())
    assert report["accepted_windows"] == 0
    assert all(w["reason"] == "ambiguous_peak" for w in report["windows"])


def test_delayed_direct_leak_requires_a_separately_evidenced_isolation_control():
    x = capture(delay_samples=64, gain=1)
    x[1] += capture(delay_samples=800, gain=0.2)[1]
    cfg = AnalysisConfig(max_delay_ms=150, warmup_s=0)
    report = analyze_capture(x, 8000, config=cfg, channel_offset_ms=0,
                             geometry_correction_ms=0, setup_verified=True)
    assert not report["threshold_assessment"]["supported_for_this_recording"]
    control = capture(delay_samples=64, gain=1)
    isolation = inspect_isolation(control, 8000, config=cfg)
    assert isolation["status"] == "leak_detected"
    assert isolation["coherent_leak_windows"] >= 3


def test_isolation_control_accepts_uncorrelated_background_but_not_missing_reference():
    x = capture()
    x[1] = np.random.default_rng(8).normal(0, 0.005, x.shape[1])
    control = inspect_isolation(x, 8000, config=config())
    assert control["status"] == "ok"
    assert control["coherent_leak_windows"] == 0
    assert inspect_isolation(np.zeros_like(x), 8000, config=config())["status"] == "inconclusive"


@pytest.mark.parametrize("command", ["prepare", "calibrate", "analyze"])
def test_cli_never_overwrites_original_evidence(command, tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/acoustic_latency.py"
    wav = tmp_path / "evidence.wav"; write_pcm(wav, capture())
    original = wav.read_bytes()
    args = [sys.executable, str(script), command]
    if command == "prepare":
        args += ["--speech-wav", str(wav), "--out", str(wav)]
    else:
        args += [str(wav), "--out", str(wav), "--recorder-description", "recorder", "--synchronized-stereo"]
        if command == "analyze":
            args += ["--model-label", "r7", "--transport", "wired"]
    p = subprocess.run(args, capture_output=True, text=True)
    assert p.returncode != 0
    assert wav.read_bytes() == original


def test_prepare_refuses_manifest_wave_collision(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/acoustic_latency.py"
    wav = tmp_path / "speech.wav"; write_pcm(wav, capture()[:1])
    out = tmp_path / "bad.json"
    p = subprocess.run([sys.executable, str(script), "prepare", "--speech-wav", str(wav), "--out", str(out)],
                       capture_output=True, text=True)
    assert p.returncode != 0
    assert not out.exists()


def test_isolation_cli_controls_threshold_support_and_failed_control(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/acoustic_latency.py"
    control_wav, recording = tmp_path / "control.wav", tmp_path / "run.wav"
    control_json, out = tmp_path / "control.json", tmp_path / "run.json"
    common = ["--recorder-description", "recorder", "--synchronized-stereo", "--warmup-s", "0",
              "--max-delay-ms", "80"]
    x = capture(); x[1] = np.random.default_rng(3).normal(0, 0.005, x.shape[1])
    write_pcm(control_wav, x)
    args = [sys.executable, str(script), "isolation", str(control_wav), "--out", str(control_json),
            "--output-muted", *common]
    p = subprocess.run(args, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    write_pcm(recording, capture())
    analyze = [sys.executable, str(script), "analyze", str(recording), "--out", str(out),
               "--model-label", "synthetic", "--transport", "wired", "--channel-offset-ms", "0",
               "--geometry-correction-ms", "0", "--isolation-json", str(control_json), *common]
    p = subprocess.run(analyze, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert json.loads(out.read_text())["threshold_assessment"]["supported_for_this_recording"]
    write_pcm(control_wav, capture(delay_samples=64))
    p = subprocess.run(args, capture_output=True, text=True)
    assert p.returncode == 2
    p = subprocess.run(analyze, capture_output=True, text=True)
    assert p.returncode == 2
    assert not json.loads(out.read_text())["threshold_assessment"]["supported_for_this_recording"]


def test_known_r7_algorithmic_bound_cannot_be_certified_as_twelve_ms(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/acoustic_latency.py"
    recording, out = tmp_path / "run.wav", tmp_path / "run.json"
    write_pcm(recording, capture())
    p = subprocess.run([sys.executable, str(script), "analyze", str(recording), "--out", str(out),
                        "--model-label", "r7", "--transport", "wired", "--recorder-description", "recorder",
                        "--synchronized-stereo", "--warmup-s", "0", "--max-delay-ms", "80",
                        "--channel-offset-ms", "0", "--geometry-correction-ms", "0"],
                       capture_output=True, text=True)
    assert p.returncode == 2
    assert json.loads(out.read_text())["status"] == "inconclusive"


def test_report_hardlink_alias_cannot_replace_original_wav(tmp_path):
    wav, out = tmp_path / "evidence.wav", tmp_path / "alias.json"
    write_pcm(wav, capture())
    try:
        out.hardlink_to(wav)
    except OSError:
        pytest.skip("filesystem does not support hardlinks")
    original = wav.read_bytes()
    script = Path(__file__).resolve().parents[1] / "scripts/acoustic_latency.py"
    p = subprocess.run([sys.executable, str(script), "analyze", str(wav), "--out", str(out),
                        "--model-label", "synthetic", "--transport", "wired", "--recorder-description", "recorder",
                        "--synchronized-stereo", "--warmup-s", "0", "--max-delay-ms", "80"],
                       capture_output=True, text=True)
    assert p.returncode != 0
    assert wav.read_bytes() == original
