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
                try:
                    dst.unlink(missing_ok=True)
                except PermissionError:   # Windows: another process has it open; the row is dropped either way
                    print(f"[cv_hi] could not delete {dst} (in use)")
                continue
            kept += 1
            spk = r["client_id"][:16]
            rows.append(_row(f"cvhi:{Path(r['path']).stem}", "cv_hi", "speech", f"cv-spk-{spk}", spk, dst, dur, "CC0"))
    print(f"[cv_hi] kept {kept}/{seen} clips at SNR>={min_snr_db} dB")
    return rows


# MAD label indices (cls_list in the repo's main.py). "communication" is radio speech: not noise.
MAD_CLASSES = ["communication", "shooting", "footsteps", "shelling", "vehicle", "helicopter", "fighter"]
# crest_audit 2026-09-20: shooting 16.0/12.2 dB, shelling 17.6/11.2, footsteps 21.9/17.6 (full/event) - all at or
# below speech (18/13). YouTube normalisation flattened them; they are non-stationary noise, not transients.
MAD_CLASS_MAP = {"shooting": "changing", "shelling": "changing", "footsteps": "changing",
                 "vehicle": "stationary", "helicopter": "stationary", "fighter": "stationary"}


def scan_mad(root: Path, out: Path) -> list[dict]:
    """MAD_dataset/{training,test}.csv rows: path, label idx, youtube title/url; one group per source video."""
    import csv
    rows = []
    for csv_name in ("training.csv", "test.csv"):
        with open(root / csv_name, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                cls = MAD_CLASSES[int(r["label"])]
                if cls not in MAD_CLASS_MAP:
                    continue
                f = root / r["path"]; vid = f.parent.name
                dst = out / "mad" / cls / f"{vid}_{f.stem}.flac"
                dur = to_flac16k(f, dst) if not dst.exists() else sf.info(dst).duration
                rows.append(_row(f"mad:{cls}/{vid}_{f.stem}", "mad", "noise", f"mad-{vid}", "", dst, dur,
                                 "MAD (YouTube-sourced; see repo)", MAD_CLASS_MAP[cls]))
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


# ESC-50 categories that are human vocalisations (too speech-like for a noise corpus) or impulsive enough to keep.
ESC50_EXCLUDE = {"crying_baby", "sneezing", "coughing", "laughing", "breathing", "snoring"}
# crest_audit 2026-09-20 (event crest, dB): keyboard_typing 22.1, mouse_click 21.6, can_opening 20.1, fireworks 20.0,
# footsteps 19.8, clock_tick 19.2 kept as "limited" transients (only keyboard_typing meets the >22 gate outright);
# church_bells 10.7, glass_breaking 13.0, door_wood_knock 15.6, clapping 17.4 dropped - speech-level crest.
ESC50_IMPULSIVE = {"fireworks", "mouse_click", "can_opening", "footsteps", "keyboard_typing", "clock_tick"}


def scan_esc50(root: Path, out: Path) -> list[dict]:
    """Layout <root>/meta/esc50.csv + <root>/audio/<clip>.wav; src_file groups takes of one recording."""
    import csv
    rows = []
    with open(root / "meta" / "esc50.csv", newline="") as fh:
        for m in sorted(csv.DictReader(fh), key=lambda m: m["filename"]):
            cat = m["category"]
            if cat in ESC50_EXCLUDE:
                continue
            f = root / "audio" / m["filename"]
            dst = out / "esc50" / cat / (f.stem + ".flac")
            dur = to_flac16k(f, dst) if not dst.exists() else sf.info(dst).duration
            if cat in ESC50_IMPULSIVE:
                noise_class = "impulsive"
            else:
                x, sr = sf.read(dst, dtype="float32")
                noise_class = stationarity_class(x, sr)
            rows.append(_row(f"esc50:{cat}/{f.stem}", "esc50", "noise", f"esc50-{m['src_file']}", "", dst, dur,
                              "ESC-50 (CC BY-NC per clip, see meta)", noise_class))
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


def scan_gunshots(root: Path, out: Path) -> list[dict]:
    """Zenodo 7004819 (Kabealo et al., Data in Brief 2023): <firearm>/<uuid>[_chanN]_vK.wav at 44.1k.
    Channel splits, the channel mean and every clip of one recording share the uuid, so they share a split."""
    rows = []
    for f in sorted(root.rglob("*.wav")):
        uuid = f.stem.split("_")[0]; arm = f.parent.name
        dst = out / "gunshots" / arm / (f.stem + ".flac")
        dur = to_flac16k(f, dst) if not dst.exists() else sf.info(dst).duration
        rows.append(_row(f"gun:{arm}/{f.stem}", "gunshots", "noise", f"gun-{uuid}", "", dst, dur, "CC BY 4.0", "impulsive"))
    return rows


DEMAND_PAIR = (1, 9)   # diffuse-field coherence null at ~1.44 kHz in DKITCHEN and NPARK => ~11.9 cm, our rig's 12 cm


def scan_demand(root: Path, out: Path, pair=DEMAND_PAIR) -> list[dict]:
    """DEMAND (Thiemann et al. 2013, Zenodo 1227121, CC BY-SA 4.0): <ENV>/ch01..ch16.wav, 16 kHz, 5 min, 16-mic grid.
    One stereo row per environment from the channel pair nearest our mic spacing; the mixer keeps a (n, 2) noise
    row's inter-channel relation instead of spatialising it, so this is the only measured two-mic noise we have."""
    rows = []
    for env in sorted(p for p in root.iterdir() if p.is_dir() and (p / f"ch{pair[0]:02d}.wav").exists()):
        dst = out / "demand" / (env.name + ".flac")
        if not dst.exists():
            chans = [sf.read(env / f"ch{c:02d}.wav", dtype="float32")[0] for c in pair]
            x = np.stack(chans, 1)
            dst.parent.mkdir(parents=True, exist_ok=True); sf.write(dst, x, SR, subtype="PCM_16")
        x, _ = sf.read(dst, dtype="float32")
        rows.append(_row(f"demand:{env.name}", "demand", "noise", f"demand-{env.name}", "", dst, len(x) / SR, "CC BY-SA 4.0",
                         stationarity_class(x[:, 0], SR)))
    return rows


CADRE_CREST_FAIL = {"M16_Zoom"}   # crest_audit 2026-09-20: 31.4 full / 21.8 event, under the 22 dB event line


def scan_cadre(root: Path, out: Path) -> list[dict]:
    """Cadre Gunshot Audio Forensics dataset (NIJ 2016-DN-BX-0183, 2018): <gun>/<Gun>_Zoom/ZM_<exp><A-D>_S<shot>.wav,
    96 kHz stereo from the Zoom H4N X/Y pair (phone recordings not taken: AGC). 20 firearms x 20 positions
    (0.5-150 m). One firearm folder = one recording session = one split. Terms: as-is for any researcher, cite the grant."""
    rows = []
    for f in sorted(root.rglob("ZM_*.wav")):
        arm = f.parent.name
        if arm in CADRE_CREST_FAIL:
            continue
        dst = out / "cadre" / arm / (f.stem + ".flac")
        dur = to_flac16k(f, dst) if not dst.exists() else sf.info(dst).duration
        rows.append(_row(f"cadre:{arm}/{f.stem}", "cadre", "noise", f"cadre-{arm}", "", dst, dur, "NIJ 2016-DN-BX-0183 as-is", "impulsive"))
    return rows


def scan_drone(root: Path, out: Path) -> list[dict]:
    """DroneAudioDataset (Al-Emadi et al. 2019): <set>/<class>/<clip>.wav. Only the drone folders are taken;
    the 'unknown' folders are ESC-50 and Speech Commands noise already in the pool. Recorded indoors; no licence
    file in the repo. Grouped per folder (one recording session each) so takes cannot straddle splits."""
    rows = []
    for f in sorted(root.rglob("*.wav")):
        cls = f.parent.name
        if cls == "unknown":
            continue
        dst = out / "drone" / cls / (f.stem + ".flac")
        dur = to_flac16k(f, dst) if not dst.exists() else sf.info(dst).duration
        x, sr = sf.read(dst, dtype="float32")
        rows.append(_row(f"drone:{cls}/{f.stem}", "drone", "noise", f"drone-{cls}", "", dst, dur,
                         "DroneAudioDataset (cite Al-Emadi et al., IWCMC 2019)", stationarity_class(x, sr)))
    return rows


# EARS task prefixes that are not speech: coughs/breaths/yawns, laughs/cries, singing
EARS_EXCLUDE = ("vegetative", "nonverbal", "melodic")


def scan_ears(root: Path, out: Path) -> list[dict]:
    """EARS (Richter et al., Interspeech 2024, CC BY-NC 4.0): <root>/pNNN/<task>_<...>_<style>.wav, 48 kHz anechoic.
    Read, emotional and freeform speech kept; the style suffix (whisper/loud/fast/...) rides in the source_id so the
    vocal-effort axis stays traceable. One group per speaker."""
    rows = []
    for f in sorted(root.rglob("p[0-9][0-9][0-9]/*.wav")):
        if f.stem.startswith(EARS_EXCLUDE):
            continue
        spk = f.parent.name
        dst = out / "ears" / spk / (f.stem + ".flac")
        dur = to_flac16k(f, dst) if not dst.exists() else sf.info(dst).duration
        rows.append(_row(f"ears:{spk}/{f.stem}", "ears", "speech", f"ears-spk-{spk}", spk, dst, dur, "CC BY-NC 4.0"))
    return rows


# NOISEX-92 (D7). crest_audit 2026-09-20 on the 19.98 kHz/16-bit originals: machinegun 18.7/16.9 dB full/event,
# i.e. speech-level - the 1992 DAT chain flattened the bursts, so it is "changing", never "impulsive".
# leopard/m109/machinegun in the common GitHub mirror are 8 kHz/8-bit re-encodes; the .mat files on SPIB are the originals.
NOISEX_CLASS = {"white": "stationary", "pink": "stationary", "volvo": "stationary", "leopard": "stationary",
                "m109": "stationary", "destroyerengine": "stationary", "f16": "stationary", "buccaneer1": "stationary",
                "buccaneer2": "stationary", "hfchannel": "stationary", "babble": "changing", "factory1": "changing",
                "factory2": "changing", "destroyerops": "changing", "machinegun": "changing"}
NOISEX_LICENCE = "NOISEX-92 (DRA Malvern 1992, via SPIB; redistribution terms unclear - cite Varga & Steeneken 1993)"


def scan_noisex92(root: Path, out: Path) -> list[dict]:
    """<root>/<name>.wav, 19.98 kHz 16-bit, one 235 s take per class. One row and one group per file: a take must
    not straddle splits, so each class lands whole in whichever split its hash says."""
    rows = []
    for f in sorted(root.glob("*.wav")):
        if f.stem not in NOISEX_CLASS:
            continue
        info = sf.info(f)
        assert info.samplerate >= 16000 and info.subtype == "PCM_16", f"{f.name}: {info.samplerate} Hz {info.subtype} is a lossy mirror copy"
        dst = out / "noisex92" / (f.stem + ".flac")
        dur = to_flac16k(f, dst) if not dst.exists() else sf.info(dst).duration
        rows.append(_row(f"noisex92:{f.stem}", "noisex92", "noise", f"noisex92-{f.stem}", "", dst, dur,
                         NOISEX_LICENCE, NOISEX_CLASS[f.stem]))
    return rows
