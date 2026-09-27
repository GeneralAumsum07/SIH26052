"""Write the R1 and R2 resampler coefficient files and the R0/R1/R2 comparison (low-delay plan Section 2.4, Task 0).

  python scripts/make_resampler_fir.py [--out deploy/resampler] [--report results_r2/r8_ld/gate0/resampler.json]

R1  r1_minphase_kaiser193_v1: scipy.signal.minimum_phase(R0, method="homomorphic", half=False, n_fft=65536) - the
    minimum-phase equivalent of R0 (193 taps, identical magnitude).
R2  r2_cdelay_ls193_v1: a 193-tap complex least-squares FIR fitted to R0's passband magnitude with a fixed pure delay
    of 36 samples (0.75 ms per conversion) and an explicit transition-band gain constraint, enforced by Lawson-style
    reweighting until every Task 0 requirement holds on a 5 Hz grid:
      passband magnitude within 0.1 dB of R0 over 0-7 kHz; no gain above 0 dB over 7-9 kHz;
      >= 60 dB rejection over 8-9 kHz and >= 80 dB above 9 kHz;
      group delay <= 0.75 ms nominal, flat within +-0.02 ms over 0.1-6 kHz; identity SNR >= 30 dB.
Every requirement is re-measured on a 1 Hz grid and recorded; the script exits 1 if one fails (never relaxed).

Each file holds the float64 coefficients, their SHA-256 (vaani.resampler.coef_sha256) and the measured delays; a
filter's delay is read from its file, never from its tap count. The board has no scipy: these files are what ships.

Identity SNR: 4 s of LTASS-shaped noise at 16 kHz (white noise shaped flat to 500 Hz, -9 dB/octave above; this
shape reproduces the plan's R0/R1 figures within 0.4 dB), interpolated to 48 kHz and decimated back through the
streaming pair, compared with the input after the best fractional delay (0.01-sample grid) and least-squares gain.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.signal as ss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vaani import resampler as rs  # noqa: E402
from vaani.audio_contract import RESAMPLER_R0, RESAMPLER_R1, RESAMPLER_R2  # noqa: E402
from vaani.live import Decimate3, Interpolate3  # noqa: E402

FS = rs.FS_HI
R2_TAPS, R2_DELAY = 193, 36
GD_FREQS = (100, 1000, 4000, 6000, 7000)
R2_REQ = {"passband_dev_db": 0.1, "transition_max_db": 0.0, "reject_8_9k_db": -60.0, "reject_above_9k_db": -80.0,
          "gd_nominal_ms": 0.75, "gd_flat_ms": 0.02, "identity_snr_db": 30.0}


def response(h, f):
    return ss.freqz(h, worN=f, fs=FS)[1]


def db(x):
    return 20 * np.log10(np.abs(x) + 1e-20)


def group_delay_ms(h, f):
    return ss.group_delay((h, [1.0]), w=np.asarray(f, float), fs=FS)[1] / FS * 1000


def r1_design(h0):
    return ss.minimum_phase(h0, method="homomorphic", half=False, n_fft=65536)


def r2_design(h0, taps=R2_TAPS, delay=R2_DELAY, iters=200):
    fg = np.linspace(0, FS / 2, 4801)                                   # 5 Hz design grid
    E = np.exp(-2j * np.pi * np.outer(fg, np.arange(taps)) / FS)
    a0 = np.abs(response(h0, fg))
    des = a0 * np.exp(-2j * np.pi * fg * delay / FS)                   # R0's magnitude, pure delay
    w = np.where(fg <= 7000, 1.0, np.where(fg < 8000, 0.05, np.where(fg < 9000, 30.0, 100.0)))
    for it in range(iters):
        M = np.vstack([(E * w[:, None]).real, (E * w[:, None]).imag])
        b = np.concatenate([(des * w).real, (des * w).imag])
        h = np.linalg.lstsq(M, b, rcond=None)[0]
        d = db(E @ h)
        viol = ((fg >= 9000) & (d > R2_REQ["reject_above_9k_db"] - 1)) \
            | ((fg >= 8000) & (fg < 9000) & (d > R2_REQ["reject_8_9k_db"] - 1)) \
            | ((fg >= 7000) & (fg <= 9000) & (d > R2_REQ["transition_max_db"])) \
            | ((fg <= 7000) & (np.abs(d - db(a0)) > R2_REQ["passband_dev_db"] * 0.9))
        if not viol.any():
            return h, it + 1
        w = w * (1 + viol)
    raise RuntimeError("R2 design did not converge; requirements are never relaxed")


def ltass_noise(n=4 * rs.FS_LO, seed=0, slope_db_oct=-9.0, knee=500.0):
    g = np.random.default_rng(seed)
    X = np.fft.rfft(g.standard_normal(n))
    fr = np.fft.rfftfreq(n, 1 / rs.FS_LO)
    a = np.where(fr <= knee, 0.0, slope_db_oct * np.log2(np.maximum(fr, 1.0) / knee))
    x = np.fft.irfft(X * 10 ** (a / 20), n)
    return x / np.abs(x).max()


def identity_snr(h, x=None):
    """(SNR dB, best delay in 16 kHz samples) of the streaming 16 -> 48 -> 16 kHz round trip."""
    x = ltass_noise() if x is None else x
    y = Decimate3(1, h)(Interpolate3(1, h)(x[None].astype(np.float32)))[0].astype(np.float64)
    n = len(x)
    X, fr = np.fft.rfft(x), np.fft.rfftfreq(n)
    s = slice(200, n - 200)
    best, bt = -np.inf, 0.0
    for coarse in (np.arange(0, 80, 0.25), None):
        grid = coarse if coarse is not None else np.arange(bt - 0.3, bt + 0.3, 0.01)
        for tau in grid:
            r = np.fft.irfft(X * np.exp(-2j * np.pi * fr * tau), n)[s]
            g = np.dot(y[s], r) / np.dot(r, r)
            snr = 10 * np.log10(np.sum((g * r) ** 2) / np.sum((y[s] - g * r) ** 2))
            if snr > best:
                best, bt = snr, float(tau)
    return float(best), round(bt, 2)


def stream_check(h, seed=0):
    """Max abs difference between the streaming classes fed in random chunk sizes and one offline filtering."""
    g = np.random.default_rng(seed)
    x16 = g.standard_normal((2, 3000)).astype(np.float32) * 0.1
    x48 = g.standard_normal((2, 9000)).astype(np.float32) * 0.1
    off_d = np.stack([np.convolve(c, h)[:9000:3] for c in x48.astype(np.float64)])
    up = np.zeros((2, 9000)); up[:, ::3] = x16
    off_i = np.stack([np.convolve(c, 3 * h)[:9000] for c in up])
    dec, itp = Decimate3(2, h), Interpolate3(2, h)
    outs_d, outs_i, a = [], [], 0
    while a < 3000:
        k = int(g.integers(1, 200))
        k = min(k, 3000 - a)
        outs_i.append(itp(x16[:, a:a + k])); outs_d.append(dec(x48[:, 3 * a:3 * (a + k)])); a += k
    return {"decimate_max_abs": float(np.abs(np.concatenate(outs_d, 1) - off_d).max()),
            "interpolate_max_abs": float(np.abs(np.concatenate(outs_i, 1) - off_i).max())}


def measure(h, h0):
    f = np.linspace(0, FS / 2, 24001)                                  # 1 Hz measurement grid
    H, H0 = response(h, f), response(h0, f)
    d, d0 = db(H), db(H0)
    pb = f <= 7000
    gd_f = np.linspace(100, 6000, 591)
    gd = group_delay_ms(h, gd_f)
    pair = np.convolve(h, 3 * h)
    snr, lag = identity_snr(h)
    above = np.flatnonzero(d[f >= 1000] <= -3.0)
    return {"taps": len(h), "passband_dev_vs_r0_db": float(np.abs(d - d0)[pb].max()),
            "passband_ripple_db": float(d[pb].max() - d[pb].min()),
            "minus3db_hz": float(f[f >= 1000][above[0]]) if above.size else None,
            "transition_7_9k_max_db": float(d[(f >= 7000) & (f <= 9000)].max()),
            "reject_8_9k_db": float(d[(f >= 8000) & (f < 9000)].max()),
            "reject_above_9k_db": float(d[f >= 9000].max()), "stopband_above_8k_db": float(d[f >= 8000].max()),
            "group_delay_ms": {str(k): float(v) for k, v in zip(GD_FREQS, group_delay_ms(h, GD_FREQS))},
            "group_delay_0p1_6k_ms": [float(gd.min()), float(gd.max())],
            "pair_group_delay_300_4000_max_ms": float(2 * group_delay_ms(h, np.linspace(300, 4000, 371)).max()),
            "pair_peak_ms": float(np.argmax(np.abs(pair)) / FS * 1000),
            "per_conversion_peak_ms": float(np.argmax(np.abs(h)) / FS * 1000),
            "identity_snr_db": snr, "identity_best_delay_16k_samples": lag,
            "stream_vs_offline": stream_check(h)}


def r2_checks(m):
    lo, hi = m["group_delay_0p1_6k_ms"]
    return {"passband_dev_db": m["passband_dev_vs_r0_db"] <= R2_REQ["passband_dev_db"],
            "transition_max_db": m["transition_7_9k_max_db"] <= R2_REQ["transition_max_db"],
            "reject_8_9k_db": m["reject_8_9k_db"] <= R2_REQ["reject_8_9k_db"],
            "reject_above_9k_db": m["reject_above_9k_db"] <= R2_REQ["reject_above_9k_db"],
            "gd_nominal_ms": R2_DELAY / FS * 1000 <= R2_REQ["gd_nominal_ms"],
            "gd_flat_ms": max(abs(lo - R2_REQ["gd_nominal_ms"]), abs(hi - R2_REQ["gd_nominal_ms"])) <= R2_REQ["gd_flat_ms"],
            "identity_snr_db": m["identity_snr_db"] >= R2_REQ["identity_snr_db"]}


def record(rid, h, design, m):
    delays = {"pair_peak_ms": m["pair_peak_ms"], "per_conversion_peak_ms": m["per_conversion_peak_ms"],
              "group_delay_ms": m["group_delay_ms"], "group_delay_0p1_6k_ms": m["group_delay_0p1_6k_ms"],
              "pair_group_delay_300_4000_max_ms": m["pair_group_delay_300_4000_max_ms"],
              "source": "measured by scripts/make_resampler_fir.py"}
    return {"id": rid, "fs": FS, "taps": len(h), "design": design, "coefficients": [float(v) for v in h],
            "sha256": rs.coef_sha256(h), "delays": delays, "measurements": m,
            "generated_by": "python scripts/make_resampler_fir.py", "scipy": __import__("scipy").__version__,
            "numpy": np.__version__}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "deploy/resampler"))
    ap.add_argument("--report", default=str(ROOT / "results_r2/r8_ld/gate0/resampler.json"))
    a = ap.parse_args(argv)
    h0 = rs.r0_coefficients()
    h1 = r1_design(h0)
    h2, iters = r2_design(h0)
    m0, m1, m2 = measure(h0, h0), measure(h1, h0), measure(h2, h0)
    checks = r2_checks(m2)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = {}
    for rid, h, design, m in (
            # R0 is designed in numpy at load time (vaani.resampler.load); its file serves the native runtime, whose
            # control path must load the same coefficients by hash
            (RESAMPLER_R0, h0, {"method": "vaani.live.lowpass_fir", "taps": 193, "cutoff_hz": 7300.0, "beta": 8.6,
                                "note": "linear phase; the Python loader designs it in numpy and ignores this file"}, m0),
            (RESAMPLER_R1, h1, {"method": "scipy.signal.minimum_phase", "of": RESAMPLER_R0, "kwargs":
                                {"method": "homomorphic", "half": False, "n_fft": 65536}}, m1),
            (RESAMPLER_R2, h2, {"method": "complex weighted least squares, Lawson reweighting", "of": RESAMPLER_R0,
                                "taps": R2_TAPS, "pure_delay_samples": R2_DELAY, "grid_hz": 5, "iterations": iters,
                                "requirements": R2_REQ, "checks": checks}, m2)):
        p = out / f"{rid}.json"
        p.write_text(json.dumps(record(rid, h, design, m), indent=1) + "\n", encoding="utf-8")
        files[rid] = p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)
        assert rs.load(rid, out).sha256 == rs.coef_sha256(h)          # round-trips through the loader
    rep = {"generated_by": "python scripts/make_resampler_fir.py", "files": files,
           "ltass": "white noise, flat to 500 Hz, -9 dB/octave above, 4 s at 16 kHz, seed 0",
           "filters": {RESAMPLER_R0: m0, RESAMPLER_R1: m1, RESAMPLER_R2: m2}, "r2_checks": checks,
           "r2_pass": all(checks.values())}
    rp = Path(a.report); rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    for rid, m in rep["filters"].items():
        print(f"{rid}: pair peak {m['pair_peak_ms']:.3f} ms, identity SNR {m['identity_snr_db']:.1f} dB, "
              f"stopband {m['stopband_above_8k_db']:.1f} dB, passband dev {m['passband_dev_vs_r0_db']:.2g} dB")
    print("R2 checks:", checks)
    return 0 if rep["r2_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
