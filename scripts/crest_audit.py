"""Is this corpus actually impulsive? Measure before you add it.

A "gunshot" recording that has been through YouTube, a limiter, or a consumer AGC is not
impulsive any more, and training on it teaches the model nothing about transients. This
is the gate to run on any candidate impulse corpus BEFORE writing an adapter for it.

Crest factor = 20*log10(peak / RMS).

    speech                    12-18 dB   (measured here: 18.0 dB whole-file, 11.7 dB event)
    a real, uncontaminated
    gunshot / balloon burst   > 35 dB
    anything in between       has been limited somewhere in the chain

Two numbers are reported because they answer different questions:
  crest_full   over the whole file. A long silent tail inflates this -- a clip that is
               99 % silence scores well even if the transient itself is squashed.
  crest_event  over +/-100 ms around the peak. This is the honest one for a transient.

Run:
    uv run python scripts/crest_audit.py data/raw/mad/shooting
    uv run python scripts/crest_audit.py data/raw/esc50/fireworks data/raw/librispeech
    uv run python scripts/crest_audit.py --glob 'data/raw/**/*.flac' --by-parent
"""
import argparse
import glob as globmod
import os
from collections import defaultdict

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 16000
SPEECH_FULL_REF = 18.0   # whole-file crest of ordinary speech (measured)
SPEECH_EVENT_REF = 13.0  # event-window crest of ordinary speech (measured)
IMPULSE_TARGET = 35.0  # below this, the transient has been limited somewhere


def crest_db(x):
    x = np.asarray(x, np.float64)
    if x.size < 64 or not np.any(x):
        return float("nan")
    return 20 * np.log10(np.abs(x).max() / (np.sqrt((x ** 2).mean()) + 1e-12))


def crest_event_db(x, sr, half_ms=100.0):
    """Crest over the window around the loudest sample -- immune to silent padding."""
    k = int(np.argmax(np.abs(x)))
    w = int(sr * half_ms / 1000)
    return crest_db(x[max(0, k - w): k + w])


def load(path, target=None):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if target and sr != target:
        g = np.gcd(sr, target)
        x = resample_poly(x, target // g, sr // g).astype(np.float32)
        sr = target
    return x, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="directories or files to audit")
    ap.add_argument("--glob", help="glob pattern instead of paths")
    ap.add_argument("--by-parent", action="store_true", help="group by parent directory name")
    ap.add_argument("--limit", type=int, default=60, help="max files per group")
    a = ap.parse_args()

    files = []
    if a.glob:
        files = globmod.glob(a.glob, recursive=True)
    for p in a.paths:
        if os.path.isdir(p):
            for ext in ("wav", "flac", "ogg", "mp3"):
                files += globmod.glob(os.path.join(p, "**", f"*.{ext}"), recursive=True)
        else:
            files.append(p)
    if not files:
        raise SystemExit("no audio found")

    groups = defaultdict(list)
    for f in sorted(files):
        key = os.path.basename(os.path.dirname(f)) if a.by_parent else (
            a.paths[0] if len(a.paths) == 1 and os.path.isdir(a.paths[0]) else os.path.dirname(f))
        if len(groups[key]) < a.limit:
            groups[key].append(f)

    print(f"speech reference: {SPEECH_FULL_REF:.0f} dB full / {SPEECH_EVENT_REF:.0f} dB event | "
          f"impulse target: >{IMPULSE_TARGET:.0f} dB event | window +/-100 ms\n")
    print(f"{'group':<34s}{'n':>4s}{'sr':>12s}{'crest_full':>12s}{'crest_event':>13s}"
          f"{'clip%':>8s}  verdict")
    for key, fs in sorted(groups.items()):
        full, ev, clip, srs = [], [], [], set()
        for f in fs:
            try:
                x, sr = load(f)
            except Exception:
                continue
            srs.add(sr)
            clip.append(float((np.abs(x) >= 0.999).mean()) * 100)
            y, _ = load(f, SR)            # the rate the pipeline actually trains at
            full.append(crest_db(y))
            ev.append(crest_event_db(y, SR))
        if not full:
            continue
        cf, ce, cl = np.nanmean(full), np.nanmean(ev), np.mean(clip)
        if ce >= IMPULSE_TARGET:
            verdict = "IMPULSIVE - usable as a transient"
        elif ce >= 25:
            verdict = "partly limited - usable if the limitation is stated"
        elif ce >= SPEECH_EVENT_REF + 5:
            verdict = "LIMITED - only just above speech"
        else:
            verdict = "NOT IMPULSIVE - at or below speech; do not label it impulsive"
        srtxt = "/".join(str(v) for v in sorted(srs)[:2])
        print(f"{key[:33]:<34s}{len(full):>4d}{srtxt:>12s}{cf:>12.1f}{ce:>13.1f}{cl:>8.2f}  {verdict}")

    print("\nA high clip% means the recording chain already hit full scale: the original "
          "transient\nwas louder than the capture could hold, so its true crest is unknown "
          "and unrecoverable.")


if __name__ == "__main__":
    main()
