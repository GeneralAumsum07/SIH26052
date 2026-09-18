"""Corpus adapters: scan a downloaded corpus, convert to 16k mono FLAC,
return manifest rows."""
import csv
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from vaani.data.manifests import COLUMNS
from vaani.data.splits import assign

SR = 16000


def to_flac16k(src: Path, dst: Path) -> float:
    x, sr = sf.read(src, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, x, SR, subtype="PCM_16")
    return len(x) / SR


def estimate_snr_db(x: np.ndarray, sr: int, frame_ms: float = 20.0) -> float:
    """Energy-based SNR: top 30% energy frames ~ speech, bottom 10% ~ noise floor."""
    n = int(sr * frame_ms / 1000)
    frames = x[: len(x) // n * n].reshape(-1, n)
    e = (frames ** 2).mean(axis=1) + 1e-10
    e = np.sort(e)
    noise = e[: max(1, len(e) // 10)].mean()
    speech = e[-max(1, int(len(e) * 0.3)):].mean()
    return float(10 * np.log10(speech / noise))


# thresholds set on synthetic probes: white/pink/AM-hum sit <1.8 dB / <0.07, gated bursts and chirps >4x past
STATIONARY_ENERGY_STD_DB = 5.0
STATIONARY_FLUX_MAX = 0.15


def stationarity_class(x: np.ndarray, sr: int, frame_ms: float = 32.0) -> str:
    """Low frame-energy variance + low spectral flux => stationary; either high => changing."""
    n = int(sr * frame_ms / 1000)
    frames = x[: len(x) // n * n].reshape(-1, n)
    if len(frames) < 2:
        return "changing"
    e_db = 10 * np.log10((frames ** 2).mean(axis=1) + 1e-10)
    spec = np.abs(np.fft.rfft(frames * np.hanning(n), axis=1))
    spec = spec / (spec.sum(axis=1, keepdims=True) + 1e-10)  # normalise so loudness doesn't drive flux
    flux = np.sqrt(((spec[1:] - spec[:-1]) ** 2).sum(axis=1)).mean()
    stationary = e_db.std() < STATIONARY_ENERGY_STD_DB and flux < STATIONARY_FLUX_MAX
    return "stationary" if stationary else "changing"


def _row(source_id, corpus, kind, group_id, speaker_id, path, dur, licence, noise_class=""):
    sha1 = hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]
    row = dict(source_id=source_id, corpus=corpus, kind=kind, group_id=group_id,
               speaker_id=speaker_id, path=str(path), duration_s=dur, licence=licence,
               split=assign(group_id), sha1=sha1, noise_class=noise_class)
    assert set(row) == set(COLUMNS), "row shape must match manifests.COLUMNS"
    return row


def scan_librispeech(root: Path, out: Path, max_hours: float | None = None) -> list[dict]:
    """Layout: <root>/<spk>/<chapter>/<spk>-<chapter>-<utt>.flac"""
    rows, total = [], 0.0
    for f in sorted(root.rglob("*.flac")):
        spk = f.parts[-3]
        dst = out / "librispeech" / f.name
        dur = to_flac16k(f, dst) if not dst.exists() else sf.info(dst).duration
        rows.append(_row(f"ls:{f.stem}", "librispeech", "speech", f"ls-spk-{spk}", spk, dst, dur, "CC BY 4.0"))
        total += dur / 3600
        if max_hours and total >= max_hours:
            break
    return rows


def scan_commonvoice_hi(root: Path, out: Path, min_snr_db: float = 30.0) -> list[dict]:
    """Gate crowdsourced clips by estimated SNR; report retained count."""
    rows, kept, seen = [], 0, 0
    with open(root / "validated.tsv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            seen += 1
            src = root / "clips" / r["path"]
            if not src.exists():
                continue
            dst = out / "cv_hi" / (Path(r["path"]).stem + ".flac")
            dur = to_flac16k(src, dst) if not dst.exists() else sf.info(dst).duration
            x, _ = sf.read(dst, dtype="float32")
            if estimate_snr_db(x, SR) < min_snr_db:
                dst.unlink(missing_ok=True)
                continue
            kept += 1
            spk = r["client_id"][:16]
            rows.append(_row(f"cvhi:{Path(r['path']).stem}", "cv_hi", "speech", f"cv-spk-{spk}", spk, dst, dur, "CC0"))
    print(f"[cv_hi] kept {kept}/{seen} clips at SNR>={min_snr_db} dB")
    return rows


# MAD class names -> our noise taxonomy; anything unlisted is "changing".
MAD_CLASS_MAP = {
    "gunshot": "impulsive", "explosion": "impulsive", "artillery": "impulsive",
    "helicopter": "stationary", "jet": "stationary", "engine": "stationary",
    "vehicle": "stationary", "tank": "stationary",
}


def scan_mad(root: Path, out: Path) -> list[dict]:
    """Layout <root>/<class>/<clip>.wav; stem prefix before last '_' groups excerpts."""
    rows = []
    for f in sorted(root.rglob("*.wav")):
        cls = f.parent.name.lower()
        noise_class = next((v for k, v in MAD_CLASS_MAP.items() if k in cls), "changing")
        group = f"mad-{cls}-{f.stem.rsplit('_', 1)[0]}"
        dst = out / "mad" / cls / (f.stem + ".flac")
        dur = to_flac16k(f, dst) if not dst.exists() else sf.info(dst).duration
        rows.append(_row(f"mad:{cls}/{f.stem}", "mad", "noise", group, "", dst, dur, "MAD (see repo)", noise_class))
    return rows


def scan_dns_noise(root: Path, out: Path) -> list[dict]:
    """DNS clips arrive unlabelled by stationarity; classify each from its own 16k audio."""
    rows = []
    for f in sorted(root.rglob("*.wav")):
        dst = out / "dns_noise" / (f.stem + ".flac")
        dur = to_flac16k(f, dst) if not dst.exists() else sf.info(dst).duration
        x, sr = sf.read(dst, dtype="float32")
        noise_class = stationarity_class(x, sr)
        rows.append(_row(f"dnsn:{f.stem}", "dns_noise", "noise", f"dnsn-{f.stem}", "", dst, dur,
                          "DNS-4 archive noise_fullband (see DNS README per-clip licences)", noise_class))
    return rows


def scan_dns_speech(root: Path, out: Path) -> list[dict]:
    """Filenames carry a reader/book id before the first '_'."""
    rows = []
    for f in sorted(root.rglob("*.wav")):
        spk = f.stem.split("_")[0]
        dst = out / "dns_speech" / (f.stem + ".flac")
        dur = to_flac16k(f, dst) if not dst.exists() else sf.info(dst).duration
        rows.append(_row(f"dnss:{f.stem}", "dns_speech", "speech", f"dns-spk-{spk}", spk, dst, dur, "DNS-5 (per-shard)"))
    return rows
