"""Runtime guards (plan 11.3): decide, per 16 ms hop, when the reference channel must be treated as absent.

Two detectors, both causal, numpy-only and O(hop) per call, wired into `vaani.live.StreamEngine` behind the
default-off `guards=` option (the r7 default path does not run a line of this):

  RefInformativeness  the reference carries no second view of the scene: |corr(P, R)| > 0.97 and |ILD| < 1 dB on
                      the raw hop, held for 0.5 s -> validity 0 (a duplicated-mono WAV, a Y-split cable, a web
                      recording with both channels the same). Hysteresis: it recovers only after 0.5 s of hops
                      that clearly fail the test (|corr| < 0.9 or |ILD| > 2 dB). Near-silent hops hold the verdict.
  NeverVanish         the output has vanished while the input has speech: speech-band (300-3400 Hz) output energy
                      more than 25 dB below the input's for more than 0.5 s of VAD-speech hops -> crossfade to the
                      reference-absent path for `hold_s`, then hand back. Measured failure it targets: r7 on
                      low-ILD web audio attenuates active frames by a median 36-38 dB (plan 11.1, diag_webaudio).
  EnergyVAD           the cheap speech detector NeverVanish needs: speech-band log energy of the input more than
                      `snr_db` above a causal noise-floor tracker (drops to any new minimum at once, rises by
                      `rise_db_per_s`), with a `hang_s` hangover. Energy-based, not a trained VAD: it fires on any
                      non-stationary band energy (speech or not). Stationary noise is absorbed by the floor. Not
                      validated against a labelled corpus; the thresholds are the plan's, untuned.

For r7 the reference-absent path is the reference zeroed (plan 11.1: loss 0.11 vs 0.86 with the reference kept on
low-ILD input); for a model trained with a validity input it is validity 0. Every verdict change is logged as a
fallback event in the engine's telemetry.
"""
from __future__ import annotations

import numpy as np

SR, HOP, N_FFT = 16000, 256, 512
HOPS_PER_S = SR / HOP
BAND = slice(int(300 / (SR / N_FFT)), int(3400 / (SR / N_FFT)) + 1)   # rfft bins 9..108 of a 512-point frame


def _hops(seconds: float) -> int:
    return max(1, int(round(seconds * HOPS_PER_S)))


def band_db(spec: np.ndarray) -> float:
    """Speech-band energy of one complex rfft frame (257,), dB."""
    s = spec[BAND]
    return float(10 * np.log10(np.sum(s.real.astype(np.float64) ** 2 + s.imag.astype(np.float64) ** 2) + 1e-12))


class RefInformativeness:
    """Duplicated-mono detector. update(prim, ref) -> True while the reference is informative."""

    def __init__(self, corr_max: float = 0.97, ild_db: float = 1.0, hold_s: float = 0.5,
                 recover_corr: float = 0.9, recover_ild_db: float = 2.0, recover_s: float = 0.5,
                 silent_rms: float = 1e-4):
        self.corr_max, self.ild_db, self.recover_corr, self.recover_ild_db = corr_max, ild_db, recover_corr, recover_ild_db
        self.hold_hops, self.recover_hops, self.silent_rms = _hops(hold_s), _hops(recover_s), silent_rms
        self.informative, self._run = True, 0
        self.corr, self.ild = 0.0, float("inf")

    def update(self, prim: np.ndarray, ref: np.ndarray) -> bool:
        p = np.asarray(prim, np.float64); r = np.asarray(ref, np.float64)
        ep, er = float(np.mean(p * p)), float(np.mean(r * r))
        if max(ep, er) < self.silent_rms ** 2:            # silence says nothing about the channels: hold
            return self.informative
        pc, rc = p - p.mean(), r - r.mean()
        den = np.sqrt(np.sum(pc * pc) * np.sum(rc * rc))
        self.corr = float(np.sum(pc * rc) / den) if den > 0 else 0.0
        self.ild = float(10 * np.log10((ep + 1e-12) / (er + 1e-12)))
        if self.informative:
            dup = abs(self.corr) > self.corr_max and abs(self.ild) < self.ild_db
            self._run = self._run + 1 if dup else 0
            if self._run >= self.hold_hops:
                self.informative, self._run = False, 0
        else:
            clear = abs(self.corr) < self.recover_corr or abs(self.ild) > self.recover_ild_db
            self._run = self._run + 1 if clear else 0
            if self._run >= self.recover_hops:
                self.informative, self._run = True, 0
        return self.informative


class EnergyVAD:
    """Causal energy-over-floor speech detector on the input primary's speech band (module docstring)."""

    def __init__(self, snr_db: float = 6.0, rise_db_per_s: float = 3.0, hang_s: float = 0.2, min_db: float = -90.0):
        self.snr_db, self.rise = snr_db, rise_db_per_s / HOPS_PER_S
        self.hang_hops, self.min_db = _hops(hang_s), min_db
        self.floor, self._hang = None, 0

    def update(self, e_db: float) -> bool:
        if self.floor is None or e_db < self.floor:
            self.floor = e_db
        else:
            self.floor += self.rise
        if e_db > self.min_db and e_db > self.floor + self.snr_db:
            self._hang = self.hang_hops
        elif self._hang:
            self._hang -= 1
        else:
            return False
        return True


class NeverVanish:
    """update(in_db, out_db) -> True while the fallback (reference-absent path) is engaged."""

    def __init__(self, drop_db: float = 25.0, hold_s: float = 0.5, fallback_s: float = 2.0, vad: dict | None = None):
        self.drop_db, self.trigger_hops, self.fallback_hops = drop_db, _hops(hold_s), _hops(fallback_s)
        self.vad = EnergyVAD(**(vad or {}))
        self.active, self._run, self._left = False, 0, 0
        self.speech = False

    def update(self, in_db: float, out_db: float) -> bool:
        self.speech = self.vad.update(in_db)
        if self.active:
            self._left -= 1
            if self._left <= 0:
                self.active, self._run = False, 0
            return self.active
        if self.speech:                                    # non-speech hops neither count nor reset
            self._run = self._run + 1 if out_db < in_db - self.drop_db else 0
            if self._run > self.trigger_hops:              # "more than 0.5 s"
                self.active, self._left, self._run = True, self.fallback_hops, 0
        return self.active


class Guards:
    """Both guards plus the soft crossfade weight the engine applies to every reference-derived signal.

    weight(n) is a per-sample gain that moves linearly toward 0 (fallback) or 1 over `fade_hops` hops, so a verdict
    change never switches the reference in one sample. The engine reads `self.ref_ok` from the previous hop
    (informativeness is decided on the raw hop before processing; never-vanish on the previous hop's output)."""

    def __init__(self, cfg: dict | bool | None = True, telemetry=None):
        cfg = {} if cfg is True else dict(cfg or {})
        inf, nv = cfg.get("informativeness", True), cfg.get("never_vanish", True)
        self.inf = RefInformativeness(**(inf if isinstance(inf, dict) else {})) if inf else None
        self.nv = NeverVanish(**(nv if isinstance(nv, dict) else {})) if nv else None
        self.fade_hops = max(1, int(cfg.get("fade_hops", 8)))          # 128 ms crossfade
        self.telemetry = telemetry
        self.g = 1.0                                                     # weight at the end of the last hop
        self.informative, self.fallback = True, False
        self.sample = 0

    @property
    def ref_ok(self) -> bool:
        """The guards' current target: trust the reference (False = reference-absent path, validity 0)."""
        return self.informative and not self.fallback

    def _event(self, kind: str, **kw) -> None:
        if self.telemetry is not None:
            self.telemetry.event(kind, sample=self.sample, **kw)

    def pre(self, prim: np.ndarray, ref: np.ndarray):
        """Before the hop: update informativeness on the raw input; return the per-sample reference weight
        (float32, HOP) or None when the guards leave the reference untouched."""
        if self.inf is not None:
            ok = self.inf.update(prim, ref)
            if ok != self.informative:
                self.informative = ok
                self._event("ref_uninformative" if not ok else "ref_informative",
                            corr=round(self.inf.corr, 4), ild_db=round(self.inf.ild, 2))
        target = 1.0 if (self.informative and not self.fallback) else 0.0
        g0 = self.g
        if g0 == target == 1.0:
            return None
        step = 1.0 / self.fade_hops
        g1 = min(target, g0 + step) if target > g0 else max(target, g0 - step)
        self.g = g1
        return (g0 + (g1 - g0) * (np.arange(1, HOP + 1, dtype=np.float32) / HOP)).astype(np.float32)

    def post(self, P: np.ndarray, out_spec: np.ndarray) -> None:
        """After the hop: never-vanish on this frame's input spectrum P (257,) and model output (257,)."""
        self.sample += HOP
        if self.nv is None:
            return
        in_db, out_db = band_db(P), band_db(out_spec)
        was = self.fallback
        self.fallback = self.nv.update(in_db, out_db)
        if self.fallback != was:
            self._event("never_vanish" if self.fallback else "never_vanish_release",
                        in_db=round(in_db, 1), out_db=round(out_db, 1))
