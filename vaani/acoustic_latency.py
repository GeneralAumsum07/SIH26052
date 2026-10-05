"""Acoustic arrival-time measurements on two channels of one external recorder.

These are signal measurements, not model timers. The reference is a measurement
mic beside the prototype's input mic; the response is a measurement mic beside
its output speaker. Separate recordings and the live loop's pre-playback WAVs
do not satisfy that contract. No model, SciPy or device library is needed here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import isfinite

import numpy as np


@dataclass(frozen=True)
class AnalysisConfig:
    window_s: float = 1.0
    step_s: float = 1.0
    warmup_s: float = 2.0
    max_delay_ms: float = 1000.0
    min_delay_ms: float = 2.0
    min_correlation: float = 0.35
    peak_margin: float = 0.05
    peak_exclusion_ms: float = 5.0
    min_reference_dbfs: float = -60.0
    max_clip_fraction: float = 0.001
    dropout_frame_ms: float = 20.0
    dropout_drop_db: float = 25.0
    min_accepted_windows: int = 3
    threshold_ms: float = 15.0
    calibration: bool = False

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not isinstance(value, bool) and not isfinite(value):
                raise ValueError(f"{key} must be finite")
        if min(self.window_s, self.step_s, self.max_delay_ms, self.threshold_ms) <= 0:
            raise ValueError("window, step, search range and threshold must be positive")
        if self.warmup_s < 0 or not 0 <= self.min_delay_ms < self.max_delay_ms:
            raise ValueError("invalid warmup or minimum delay")
        if not 0 < self.min_correlation <= 1 or not 0 <= self.peak_margin <= 1:
            raise ValueError("invalid correlation quality thresholds")
        if self.peak_exclusion_ms <= 0 or not 0 <= self.max_clip_fraction < 1:
            raise ValueError("invalid peak separation or clipping threshold")
        if self.min_accepted_windows < 1:
            raise ValueError("at least one accepted window is required")
        if min(self.dropout_frame_ms, self.dropout_drop_db) <= 0:
            raise ValueError("dropout frame and relative level threshold must be positive")


def normalized_correlation(reference: np.ndarray, search: np.ndarray) -> np.ndarray:
    """Centered NCC for every FULL-length candidate window in ``search``.

    Entry k is corr(reference, search[k:k+N]). This explicit convention is why
    the FFT uses a reversed reference: a delayed output has a positive lag.
    Candidate-local centering/energy prevents DC and gain differences from
    masquerading as a delay. Zero padding prevents circular wraparound.
    """
    x = np.asarray(reference, np.float64)
    y = np.asarray(search, np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) < 2 or len(y) < len(x):
        raise ValueError("correlation needs a reference and a longer 1-D search")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("correlation input contains NaN or infinity")
    n = len(x)
    x = x - x.mean()
    energy_x = float(x @ x)
    fft_n = 1 << (len(y) + n - 2).bit_length()
    dots = np.fft.irfft(np.fft.rfft(y, fft_n) * np.fft.rfft(x[::-1], fft_n), fft_n)
    dots = dots[n - 1:len(y)]
    sums = np.concatenate(([0.0], np.cumsum(y)))
    squares = np.concatenate(([0.0], np.cumsum(y * y)))
    local_sum = sums[n:] - sums[:-n]
    energy_y = np.maximum(0.0, squares[n:] - squares[:-n] - local_sum ** 2 / n)
    denom = np.sqrt(energy_x * energy_y)
    scores = np.zeros_like(dots)
    # Silence is unmeasurable; it must not generate a divide-by-zero/NaN or a
    # successful zero-delay result in the caller's JSON.
    np.divide(dots, denom, out=scores, where=denom > 1e-15)
    return np.clip(scores, -1.0, 1.0)


def _summary(values):
    if not values:
        return None
    a = np.asarray(values, np.float64)
    return {"n": len(a), "min": float(a.min()), "median": float(np.median(a)),
            "mean": float(a.mean()), "p95": float(np.percentile(a, 95)),
            "max": float(a.max())}


def analyze_capture(capture: np.ndarray, sr: int, *, config: AnalysisConfig | None = None,
                    reference_channel: int = 0, response_channel: int = 1,
                    channel_offset_ms: float | None = None,
                    geometry_correction_ms: float | None = None,
                    setup_verified: bool = False, isolation_verified: bool = False) -> dict:
    """Estimate acoustic lag in independent windows and retain EVERY failure.

    Device-boundary delay is only derived when both the recorder channel skew
    and the external acoustic path correction were supplied. Neither is silently
    assumed zero. Accepted-window statistics remain descriptive when any active
    window failed; that prevents speech deletion from improving a latency claim.
    """
    cfg = config or AnalysisConfig()
    x = np.asarray(capture, np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or not np.isfinite(x).all():
        raise ValueError("need a finite synchronized recording with at least two channels")
    if not isinstance(sr, (int, np.integer)) or sr < 8000:
        raise ValueError("sample rate must be an integer of at least 8000 Hz")
    if reference_channel == response_channel or min(reference_channel, response_channel) < 0:
        raise ValueError("reference and response must be distinct channels")
    if max(reference_channel, response_channel) >= len(x):
        raise ValueError("selected channel is absent from the recording")
    for v in (channel_offset_ms, geometry_correction_ms):
        if v is not None and not isfinite(v):
            raise ValueError("corrections must be finite measured values")
    n = int(round(cfg.window_s * sr))
    radius = int(round(cfg.max_delay_ms * sr / 1000))
    step = max(1, int(round(cfg.step_s * sr)))
    start = max(radius, int(round(cfg.warmup_s * sr)))
    stop = x.shape[1] - n - radius
    if n < 2 or radius < 1 or stop < start:
        raise ValueError("recording is too short for a full window and the complete lag search")
    ref, response = x[reference_channel], x[response_channel]
    rows, delays, inactive = [], [], 0
    exclusion = max(1, int(round(cfg.peak_exclusion_ms * sr / 1000)))
    for t in range(start, stop + 1, step):
        window = ref[t:t + n]
        rms = float(np.std(window))
        if rms < 10 ** (cfg.min_reference_dbfs / 20):
            inactive += 1
            continue
        row = {"reference_start_s": t / sr, "reference_dbfs": float(20 * np.log10(rms)),
               "accepted": False, "delay_ms": None}
        rows.append(row)
        if float(np.mean(np.abs(window) >= 0.999)) > cfg.max_clip_fraction:
            row["reason"] = "clipped_reference"
            continue
        search = response[t - radius:t + n + radius]
        if float(np.std(search)) < 1e-8:
            row["reason"] = "silent_response"
            continue
        signed = normalized_correlation(window, search)
        scores = np.abs(signed)  # Microphone polarity changes the sign, not the arrival time.
        best = int(np.argmax(scores))
        lag = best - radius
        row.update(correlation=float(signed[best]), candidate_delay_ms=lag * 1000 / sr)
        # Compare DISTINCT local peaks, not a neighboring sample of the same
        # broad speech peak. Harmonic tones and two equal echoes must fail.
        peaks = np.flatnonzero((scores[1:-1] >= scores[:-2]) & (scores[1:-1] >= scores[2:])) + 1
        # An echo just at the search limit remains a real competing arrival,
        # even though a winner at that limit is itself unmeasurable.
        if scores[0] >= scores[1]:
            peaks = np.concatenate(([0], peaks))
        if scores[-1] >= scores[-2]:
            peaks = np.concatenate((peaks, [len(scores) - 1]))
        peaks = peaks[np.abs(peaks - best) > exclusion]
        competitor = float(scores[peaks].max()) if len(peaks) else 0.0
        row["peak_margin"] = float(scores[best] - competitor)
        if scores[best] < cfg.min_correlation:
            reason = "low_correlation"
        elif best in (0, len(scores) - 1):
            reason = "search_boundary"
        elif not cfg.calibration and lag < 0:
            reason = "negative_lag_check_channels"
        elif not cfg.calibration and abs(lag * 1000 / sr) < cfg.min_delay_ms:
            reason = "near_zero_direct_leakage"
        elif row["peak_margin"] < cfg.peak_margin:
            reason = "ambiguous_peak"
        elif float(np.mean(np.abs(search[best:best + n]) >= 0.999)) > cfg.max_clip_fraction:
            reason = "clipped_response"
        else:
            reason = None
        if reason:
            row["reason"] = reason
            continue
        if not cfg.calibration:
            # A 1-second NCC near 0.7 can hide HALF the output being zero.
            # Check short aligned frames before accepting the window. Relative
            # gain (rather than an absolute output level) preserves legitimate
            # quiet recordings while detecting disappearance within a window.
            frame = max(2, int(round(cfg.dropout_frame_ms * sr / 1000)))
            count = n // frame
            if count:
                r_frames = window[:count * frame].reshape(count, frame)
                y_frames = search[best:best + count * frame].reshape(count, frame)
                r_rms, y_rms = np.std(r_frames, axis=1), np.std(y_frames, axis=1)
                active = r_rms >= max(10 ** (cfg.min_reference_dbfs / 20), rms * 0.1)
                ratios = y_rms[active] / r_rms[active]
                typical = float(np.median(ratios)) if len(ratios) else 0.0
                missing = (y_rms[active] < 1e-8) | (ratios < typical * 10 ** (-cfg.dropout_drop_db / 20))
                row.update(active_short_frames=int(active.sum()), missing_short_frames=int(missing.sum()))
                if missing.any():
                    row["reason"] = "response_dropout"
                    continue
        # Sub-sample peak interpolation reduces rounding bias; it does not
        # establish sub-sample physical accuracy of microphones or the model.
        left, middle, right = scores[best - 1:best + 2]
        curvature = left - 2 * middle + right
        delta = float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5)) if curvature else 0.0
        delay = (lag + delta) * 1000 / sr
        row.update(accepted=True, reason=None, delay_ms=delay)
        delays.append(delay)
    accepted, attempted = len(delays), len(rows)
    enough = accepted >= cfg.min_accepted_windows
    all_active_measured = accepted == attempted
    corrected = None
    corrections_known = channel_offset_ms is not None and geometry_correction_ms is not None
    if corrections_known:
        corrected = [v - channel_offset_ms - geometry_correction_ms for v in delays]
    physically_possible = corrected is None or all(v >= 0 for v in corrected) or cfg.calibration
    status = "ok" if enough and all_active_measured and physically_possible else "inconclusive"
    corrected_summary = _summary(corrected) if corrected is not None else None
    acoustic_summary = _summary(delays)
    return {
        "schema": "vaani.acoustic_latency/1",
        "quantity": "recorder_channel_offset" if cfg.calibration else "acoustic_input_to_output_arrival_delay",
        "status": status,
        "sample_rate_hz": int(sr), "sample_period_ms": 1000 / sr,
        "recording_seconds": x.shape[1] / sr,
        "reference_channel": reference_channel, "response_channel": response_channel,
        "settings": asdict(cfg), "active_windows": attempted, "inactive_windows": inactive,
        "accepted_windows": accepted, "rejected_windows": attempted - accepted,
        "accepted_fraction": accepted / attempted if attempted else 0.0,
        "acoustic_delay_ms": acoustic_summary,
        "channel_offset_ms": channel_offset_ms, "geometry_correction_ms": geometry_correction_ms,
        "corrected_system_delay_ms": corrected_summary,
        "setup_verified": bool(setup_verified),
        "isolation_verified": bool(isolation_verified),
        "threshold_assessment": {
            "limit_ms": cfg.threshold_ms,
            "quantity": "corrected_system_delay_ms",
            "all_accepted_windows_below": (corrected_summary["max"] < cfg.threshold_ms)
            if corrected_summary is not None else None,
            "supported_for_this_recording": bool(status == "ok" and corrections_known and setup_verified and isolation_verified
                and not cfg.calibration and corrected_summary["max"] < cfg.threshold_ms),
            "scope": "this recording and setup only; descriptive windows, not a worst-case guarantee",
        },
        "warnings": [message for flag, message in (
            (not corrections_known and not cfg.calibration, "Device-only delay is unknown: channel/geometry corrections missing."),
            (attempted > accepted, "Rejected active windows must not be hidden when reporting latency."),
            (not enough, "Too few accepted active windows for a successful measurement."),
            (not physically_possible, "Corrections produced a negative system delay; check calibration and geometry."),
            (not setup_verified, "External common-clock recorder setup has not been verified."),
            (not isolation_verified and not cfg.calibration, "Output-muted isolation control is missing or failed; direct sound may imitate speaker output."),
        ) if flag],
        "windows": rows,
    }


def inspect_isolation(capture: np.ndarray, sr: int, *, config: AnalysisConfig | None = None,
                      reference_channel: int = 0, response_channel: int = 1) -> dict:
    """Inspect a SOURCE-ONLY control with the prototype output physically muted.

    A direct leak can arrive at 8 ms, not just at zero. Thus any confidently
    coherent path in this control is suspect, regardless of lag/ambiguity.
    We conservatively refuse certification instead of subtracting a changing
    acoustic waveform or selecting the echo that would satisfy a target.
    """
    cfg = config or AnalysisConfig()
    report = analyze_capture(capture, sr, config=replace(cfg, calibration=True, min_delay_ms=0),
                             reference_channel=reference_channel, response_channel=response_channel)
    rows = report["windows"]
    coherent = sum(abs(r.get("correlation", 0)) >= cfg.min_correlation for r in rows)
    clipped = any(r.get("reason") == "clipped_reference" for r in rows)
    response_clipped = float(np.mean(np.abs(np.asarray(capture)[response_channel]) >= 0.999)) > cfg.max_clip_fraction
    if clipped or response_clipped or report["active_windows"] < cfg.min_accepted_windows:
        status = "inconclusive"
    else:
        status = "leak_detected" if coherent else "ok"
    return {
        "schema": "vaani.acoustic_isolation/1", "quantity": "source_only_output_muted_control",
        "status": status, "sample_rate_hz": int(sr),
        "reference_channel": reference_channel, "response_channel": response_channel,
        "settings": asdict(cfg), "active_windows": report["active_windows"],
        "coherent_leak_windows": int(coherent), "response_clipped": response_clipped,
        "windows": rows,
    }


def make_stimulus(speech: np.ndarray, sr: int, *, repeats: int = 10, gap_s: float = 2.0,
                  lead_s: float = 3.0, tail_s: float = 3.0,
                  target_dbfs: float = -25.0) -> tuple[np.ndarray, dict]:
    """Repeat real, aperiodic speech rather than clicks a denoiser may remove.

    Quiet gaps exceed the default lag-search radius so one replay cannot be
    mistaken for a neighboring replay. A peak limit avoids creating clipped
    stimulus; the manifest records any reduction from the requested RMS.
    """
    x = np.asarray(speech, np.float64)
    if x.ndim != 1 or not np.isfinite(x).all() or len(x) < sr // 2:
        raise ValueError("provide at least 0.5 seconds of finite mono speech")
    if sr < 8000 or repeats < 2 or gap_s < 2 or min(lead_s, tail_s) < 0:
        raise ValueError("need >=2 repeats, >=2 s gaps, valid padding and sample rate")
    if not all(isfinite(v) for v in (gap_s, lead_s, tail_s, target_dbfs)):
        raise ValueError("stimulus settings must be finite")
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x * x)))
    if rms < 1e-8:
        raise ValueError("silent or constant probe cannot measure an acoustic delay")
    gain = min(10 ** (target_dbfs / 20) / rms, 0.8 / np.max(np.abs(x)))
    x = (x * gain).astype(np.float32)
    lead, gap, tail = (int(round(s * sr)) for s in (lead_s, gap_s, tail_s))
    parts = [np.zeros(lead, np.float32)]
    starts = []
    pos = lead
    for i in range(repeats):
        starts.append(pos / sr)
        parts.append(x)
        pos += len(x)
        if i + 1 < repeats:
            parts.append(np.zeros(gap, np.float32)); pos += gap
    parts.append(np.zeros(tail, np.float32))
    return np.concatenate(parts), {
        "schema": "vaani.acoustic_stimulus/1", "sample_rate_hz": sr, "repeats": repeats,
        "speech_seconds": len(x) / sr, "gap_seconds": gap / sr,
        "trial_start_s": starts, "target_dbfs": target_dbfs,
        "actual_speech_dbfs": float(20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)))),
        "peak": float(np.max(np.abs(x))),
    }
