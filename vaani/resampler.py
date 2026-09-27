"""48 <-> 16 kHz resampler pairs (low-delay plan, Sections 2.4 and 4).

  r0_linphase_kaiser193_v1  the current 193-tap linear-phase Kaiser sinc (vaani.live.lowpass_fir): the control,
                            2.000 ms per conversion, 4.0 ms pair. Designed in numpy, no file.
  r1_minphase_kaiser193_v1  its minimum-phase equivalent (scipy homomorphic, half=False, n_fft 65536): identical
                            magnitude, 0.4 ms pair; phase dispersion inaudible in the speech band but not
                            waveform-transparent. The deployment default (D7).
  r2_cdelay_ls193_v1        a constrained-delay near-linear-phase LS FIR with an explicit transition-band gain
                            constraint: the transparent alternative.

Board code has no scipy, so R1 and R2 ship as versioned coefficient files (deploy/resampler/<id>.json, written by
scripts/make_resampler_fir.py) with their SHA-256 and measured delays; loading verifies the hash. A filter's delay is
taken from its file, never from its tap count (that is right only for a symmetric linear-phase FIR).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from vaani.audio_contract import RESAMPLER_R0, RESAMPLER_R1, RESAMPLER_R2

ROOT = Path(__file__).resolve().parents[1]
COEF_DIR = ROOT / "deploy" / "resampler"
FS_HI, FS_LO = 48000, 16000
IDS = (RESAMPLER_R0, RESAMPLER_R1, RESAMPLER_R2)


def coef_sha256(h: np.ndarray) -> str:
    """SHA-256 of the float64 little-endian coefficient bytes."""
    return hashlib.sha256(np.asarray(h, "<f8").tobytes()).hexdigest()


def r0_coefficients() -> np.ndarray:
    from vaani.live import lowpass_fir
    return lowpass_fir()


class ResamplerPair:
    """One 48 kHz FIR used by both conversions: decimator(ch) and interpolator(ch) build streaming Decimate3 /
    Interpolate3 with it. delay_ms: {"pair_peak_ms", "per_conversion_ms", ...} from the coefficient file (R0: exact)."""

    def __init__(self, resampler_id: str, h: np.ndarray, delays: dict, sha256: str):
        self.id, self.h, self.delays, self.sha256 = resampler_id, np.asarray(h, np.float64), dict(delays), sha256

    def decimator(self, channels: int):
        from vaani.live import Decimate3
        return Decimate3(channels, self.h)

    def interpolator(self, channels: int):
        from vaani.live import Interpolate3
        return Interpolate3(channels, self.h)

    @property
    def pair_delay_ms(self) -> float:
        return float(self.delays["pair_peak_ms"])

    def pair_delay_samples_16k(self) -> int:
        """The pair's identity-path impulse-peak delay, rounded to 16 kHz samples (for offline alignment only)."""
        return int(round(self.pair_delay_ms * FS_LO / 1000))


def load(resampler_id: str, coef_dir: Path | None = None) -> ResamplerPair:
    if resampler_id not in IDS:
        raise ValueError(f"unknown resampler {resampler_id!r}; known: {IDS}")
    if resampler_id == RESAMPLER_R0:
        h = r0_coefficients()
        return ResamplerPair(resampler_id, h, {"pair_peak_ms": 4.0, "per_conversion_ms": 2.0,
                                               "source": "linear phase: (taps-1)/2 samples per conversion"},
                             coef_sha256(h))
    path = Path(coef_dir or COEF_DIR) / f"{resampler_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path}: run scripts/make_resampler_fir.py (Task 0) to write the {resampler_id} coefficients")
    j = json.loads(path.read_text(encoding="utf-8"))
    h = np.asarray(j["coefficients"], np.float64)
    sha = coef_sha256(h)
    if sha != j["sha256"]:
        raise ValueError(f"{path}: coefficient sha256 {sha} does not match the recorded {j['sha256']}")
    return ResamplerPair(resampler_id, h, j["delays"], sha)


def apply_pair(x16: np.ndarray, pair: ResamplerPair) -> np.ndarray:
    """(ch, n) at 16 kHz -> interpolate to 48 kHz -> decimate back: the identity path through the pair."""
    x16 = np.atleast_2d(np.asarray(x16, np.float32))
    up = pair.interpolator(x16.shape[0])(x16)
    return pair.decimator(x16.shape[0])(up)
