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
               speaker_id=speaker_id, path=Path(path).as_posix(), duration_s=dur, licence=licence,
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


def youtube_id(url: str) -> str:
    """The 11-character video id from a watch?v=, youtu.be/ or shorts/ URL ("" when none is found)."""
    import re
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", str(url))
    return m.group(1) if m else ""


def scan_mad(root: Path, out: Path, group_by: str = "folder") -> list[dict]:
    """MAD_dataset/{training,test}.csv rows: path, label idx, youtube title/url. group_by="folder" (default, the r1-r7
    manifests) groups per MAD folder; "video" groups per YouTube id, since 141 of 673 test rows share a video with
    training rows across folders (plan 3.6)."""
    import csv
    if group_by not in ("folder", "video"):
        raise ValueError(f"group_by must be folder or video, not {group_by!r}")
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
                grp = f"mad-{vid}" if group_by == "folder" else f"mad-yt-{youtube_id(r['youtube url']) or vid}"
                rows.append(_row(f"mad:{cls}/{vid}_{f.stem}", "mad", "noise", grp, "", dst, dur,
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
            chans, srs = zip(*(sf.read(env / f"ch{c:02d}.wav", dtype="float32") for c in pair))
            x = np.stack(chans, 1)
            if srs[0] != SR:   # SCAFE exists on Zenodo only as the 48 kHz zip
                g = np.gcd(srs[0], SR); x = resample_poly(x, SR // g, srs[0] // g, axis=0).astype(np.float32)
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


def brickwall_ratio(x: np.ndarray, sr: int, cutoff: float = 4000.0, width: float = 500.0) -> float:
    """Mean power just below `cutoff` over mean power just above it.

    Resampling from 8 kHz leaves a cliff at 4 kHz that survives upsampling into a higher-rate
    container, so a samplerate check alone cannot catch a laundered mirror copy.

    Measured 2026-09-22 on the corpus itself, all fifteen files over a 30 s probe resampled to 16 kHz:
    genuine SPIB originals span 0.14-3.27 (widest: babble), and the three real 8 kHz mirror copies
    upsampled back to 16 kHz read 9.8-11.9. The separation is about 3x, not orders of magnitude --
    resample_poly's anti-imaging filter has a finite transition band, so the cliff is steep rather than
    vertical. BRICKWALL_MAX sits between the two populations in log space.

    Treat this as a tripwire, not a proof: it catches the specific failure that already happened to this
    corpus once, and a more aggressive resampler with a wider transition band could slip under it."""
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    fr = np.fft.rfftfreq(len(x), 1 / sr)
    below = X[(fr >= cutoff - width) & (fr < cutoff)].mean()
    above = X[(fr >= cutoff) & (fr < cutoff + width)].mean()
    return float(below / max(above, 1e-30))


BRICKWALL_MAX = 6.0   # between genuine (max 3.27) and laundered (min 9.8); see brickwall_ratio for the measurement


def scan_noisex92(root: Path, out: Path) -> list[dict]:
    """<root>/<name>.wav, 19.98 kHz 16-bit, one 235 s take per class. One row and one group per file: a take must
    not straddle splits, so each class lands whole in whichever split its hash says."""
    rows = []
    for f in sorted(root.glob("*.wav")):
        if f.stem not in NOISEX_CLASS:
            continue
        info = sf.info(f)
        assert info.samplerate >= 16000 and info.subtype == "PCM_16", f"{f.name}: {info.samplerate} Hz {info.subtype} is a lossy mirror copy"
        probe, psr = sf.read(f, dtype="float32", frames=int(info.samplerate * 30), always_2d=True)
        r = brickwall_ratio(probe.mean(axis=1), psr)
        assert r < BRICKWALL_MAX, (f"{f.name}: spectral cliff at 4 kHz (ratio {r:.0f}) - this is an 8 kHz mirror "
                                   f"copy upsampled into a 16-bit container, not the SPIB original")
        dst = out / "noisex92" / (f.stem + ".flac")
        dur = to_flac16k(f, dst) if not dst.exists() else sf.info(dst).duration
        rows.append(_row(f"noisex92:{f.stem}", "noisex92", "noise", f"noisex92-{f.stem}", "", dst, dur,
                         NOISEX_LICENCE, NOISEX_CLASS[f.stem]))
    return rows


WHAM_LICENCE = "CC BY-NC 4.0"
# Binaural rig (~17 cm) puts WHAM!'s diffuse-field coherence null near 343/(2*0.17) = 1009 Hz, against
# our 12 cm rig's 1441 Hz. Deliberately not a geometry match: this corpus tests inter-channel realism,
# not our exact spacing, and the mismatch is the experiment rather than a defect. See the r6 protocol.
WHAM_SPACING_M = 0.17


def scan_wham(root: Path, out: Path, split: str = "tr") -> list[dict]:
    """WHAM! noise (Wichern et al., Interspeech 2019, CC BY-NC 4.0): <root>/{tr,cv,tt}/*.wav, 16 kHz stereo,
    ~82 h of restaurants/cafes/bars/parks from a binaural tripod rig. Kept stereo for the same reason as
    DEMAND: the mixer preserves an (n, 2) noise row's inter-channel relation instead of spatialising it.

    Only `tr` is scanned by default. `tt` is deliberately left unscanned so it stays available as an
    unseen-corpus held-out set; scanning it here would put it in a manifest and make that claim false.

    group_id is the recording session, not the clip, so assign() cannot split one session across train
    and test. WHAM! encodes the session in the filename stem (`<session>_<utt>.wav`)."""
    if split == "tt":
        raise ValueError("wham tt is reserved as a held-out generalisation set; scanning it into a manifest would spend it")
    rows = []
    for f in sorted((root / split).glob("*.wav")):
        info = sf.info(f)
        assert info.channels == 2, f"{f.name}: {info.channels} channels; the two-mic relation is the point of this corpus"
        dst = out / "wham" / split / (f.stem + ".flac")
        if not dst.exists():
            x, sr = sf.read(f, dtype="float32", always_2d=True)
            if sr != SR:   # the 48 kHz variant exists; accept it rather than failing a whole download
                g = np.gcd(sr, SR); x = resample_poly(x, SR // g, sr // g, axis=0).astype(np.float32)
            dst.parent.mkdir(parents=True, exist_ok=True); sf.write(dst, x, SR, subtype="PCM_16")
        x, _ = sf.read(dst, dtype="float32", always_2d=True)
        session = f.stem.split("_")[0]
        rows.append(_row(f"wham:{split}:{f.stem}", "wham", "noise", f"wham-{session}", "", dst, len(x) / SR,
                         WHAM_LICENCE, stationarity_class(x[:, 0], SR)))
    return rows


VEHICLE_LICENCE = "CC BY 4.0"


def scan_vehicle_interior(root: Path, out: Path) -> list[dict]:
    """Vehicle Interior Sound Dataset (Zenodo 5606504, CC BY 4.0): <root>/<class>/*.wav, 48 kHz, 5980 clips
    of 3-5 s across eight vehicle classes, no human voices.

    Clips are shorter than the evaluation crop and render_bucket_item pads speech but not noise, so a 3 s
    clip would reach mix() short. Every clip in a class is concatenated into one long file per class, which
    removes the short-clip problem and gives a group_id -- the vehicle class -- that split assignment can
    use honestly.

    Intended as held-out generalisation material, not training data: civilian road vehicles on asphalt,
    which is stationary vehicular noise of a completely different provenance from anything in the recipe."""
    rows = []
    for cls in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(cls.glob("*.wav"))
        if not files:
            continue
        dst = out / "vehicle_interior" / (cls.name + ".flac")
        if not dst.exists():
            chunks = []
            for f in files:
                x, sr = sf.read(f, dtype="float32", always_2d=True)
                x = x.mean(axis=1)
                if sr != SR:
                    g = np.gcd(sr, SR); x = resample_poly(x, SR // g, sr // g).astype(np.float32)
                chunks.append(x)
            x = np.concatenate(chunks)
            dst.parent.mkdir(parents=True, exist_ok=True); sf.write(dst, x, SR, subtype="PCM_16")
        dur = sf.info(dst).duration
        x, _ = sf.read(dst, dtype="float32")
        r = _row(f"vehicle_interior:{cls.name}", "vehicle_interior", "noise", f"vehicle-{cls.name}", "",
                 dst, dur, VEHICLE_LICENCE, stationarity_class(x, SR))
        # assign() splits 80/10/10 by group hash so a corpus that IS trained on cannot leak into eval.
        # This corpus never enters a training recipe (see results_r2/generalisation/PROTOCOL.md, enforced
        # by scripts/check_heldout.py), so there is nothing to protect against and the split only throws
        # material away: on eight groups it drew 5/1/2 and put the one stationary class into "train",
        # leaving the test split with no stationary noise at all. A held-out corpus is held out whole.
        r["split"] = "test"
        rows.append(r)
    return rows


# ---------------------------------------------------------------------------------------------------------------
# r8 / mixer v2 pools (plan 11.5 "Datasets"). Nothing here downloads: every scanner returns [] with a note when its
# root is absent, so a manifest build on a box without the corpus is a no-op rather than an error. Layouts marked
# TBD are the published ones as documented, not checked against a local copy.
# ---------------------------------------------------------------------------------------------------------------

# DroneAudioDataset stays in v1 manifests (r7 reproduction) but leaves every v2 pool: indoor, two toy drones, 0.37 h,
# no licence. scenes.V2_EXCLUDED_CORPORA enforces the same set at sampling time.
V2_DROPPED_CORPORA = {"drone"}


def _absent(root, name) -> bool:
    if root is None or not Path(root).exists():
        print(f"[{name}] {root} absent: skipped (no download is made here)")
        return True
    return False


def scan_librittsr(root: Path, out: Path, max_hours: float | None = None) -> list[dict]:
    """LibriTTS-R (Koizumi et al. 2023, CC BY 4.0): <root>/<subset>/<spk>/<chapter>/<spk>_<chapter>_*.wav, 24 kHz.
    One group per speaker; max_hours caps a curated subset (plan: 150-300 h)."""
    if _absent(root, "librittsr"):
        return []
    rows, total = [], 0.0
    for f in sorted(Path(root).rglob("*.wav")):
        spk = f.stem.split("_")[0]
        dst = out / "librittsr" / spk / (f.stem + ".flac")
        dur = to_flac16k(f, dst) if not dst.exists() else sf.info(dst).duration
        rows.append(_row(f"lttsr:{f.stem}", "librittsr", "speech", f"lttsr-spk-{spk}", spk, dst, dur, "CC BY 4.0"))
        total += dur / 3600
        if max_hours and total >= max_hours:
            break
    return rows


# FSD50K labels the v2 scenes use (scenes.FSD50K_TAGS keys). Per-clip licence: CC0 and CC BY only; NC and
# Sampling+ clips are refused so the training pool stays commercially clean.
FSD50K_LABELS = ("Gunshot_and_gunfire", "Explosion", "Siren", "Wind", "Helicopter", "Engine", "Vehicle", "Aircraft",
                 "Crowd", "Chatter")
FSD50K_IMPULSIVE = {"Gunshot_and_gunfire", "Explosion"}


def fsd50k_licence_ok(url: str) -> bool:
    u = str(url).lower()
    return ("publicdomain/zero" in u or "/licenses/by/" in u) and "-nc" not in u and "sampling" not in u


def scan_fsd50k(root: Path, out: Path, labels=FSD50K_LABELS) -> list[dict]:
    """FSD50K (Fonseca et al. 2022): <root>/FSD50K.ground_truth/{dev,eval}.csv (fname, labels, ...),
    FSD50K.metadata/{dev,eval}_clips_info_FSD50K.json (license, uploader), FSD50K.{dev,eval}_audio/<fname>.wav.
    A clip is kept under the first wanted label it carries; grouped by uploader so one recordist's takes share a split."""
    import json
    if _absent(root, "fsd50k"):
        return []
    root = Path(root); rows = []; wanted = set(labels)
    for part in ("dev", "eval"):
        gt, info = root / "FSD50K.ground_truth" / f"{part}.csv", root / "FSD50K.metadata" / f"{part}_clips_info_FSD50K.json"
        if not gt.exists() or not info.exists():
            continue
        meta = json.loads(info.read_text(encoding="utf-8"))
        with open(gt, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                labs = [lab for lab in str(r["labels"]).split(",") if lab in wanted]
                m = meta.get(str(r["fname"])) or {}
                if not labs or not fsd50k_licence_ok(m.get("license", "")):
                    continue
                lab = labs[0]
                f = root / f"FSD50K.{part}_audio" / f"{r['fname']}.wav"
                if not f.exists():
                    continue
                dst = out / "fsd50k" / lab / (f.stem + ".flac")
                dur = to_flac16k(f, dst) if not dst.exists() else sf.info(dst).duration
                x, _ = sf.read(dst, dtype="float32")
                ncls = "impulsive" if lab in FSD50K_IMPULSIVE else stationarity_class(x, SR)
                up = str(m.get("uploader") or f"clip{r['fname']}")
                rows.append(_row(f"fsd50k:{lab}/{f.stem}", "fsd50k", "noise", f"fsd50k-up-{up}", "", dst, dur,
                                 str(m.get("license")), ncls))
    return rows


def _top(f: Path, root: Path) -> str:
    rel = f.relative_to(root).parts
    return rel[0] if len(rel) > 1 else f.stem


def scan_c3gd(root: Path, out: Path) -> list[dict]:
    """C3GD (arXiv 2606.18135, CC BY 4.0; host URL TBD): layout TBD, taken as <root>/<firearm>/**/*.wav. Grouped per
    firearm until the per-session id is known (conservative: no firearm straddles splits)."""
    if _absent(root, "c3gd"):
        return []
    rows = []
    for f in sorted(Path(root).rglob("*.wav")):
        arm = _top(f, Path(root))
        dst = out / "c3gd" / arm / (f.stem + ".flac")
        dur = to_flac16k(f, dst) if not dst.exists() else sf.info(dst).duration
        rows.append(_row(f"c3gd:{arm}/{f.stem}", "c3gd", "noise", f"c3gd-{arm}", "", dst, dur, "CC BY 4.0", "impulsive"))
    return rows


def scan_avq_drone(root: Path, out: Path) -> list[dict]:
    """AVQ drone noise (CC BY 4.0): layout TBD, taken as <root>/<recording or drone>/**/*.wav, grouped by the top folder."""
    if _absent(root, "avq_drone"):
        return []
    rows = []
    for f in sorted(Path(root).rglob("*.wav")):
        top = _top(f, Path(root))
        dst = out / "avq_drone" / top / (f.stem + ".flac")
        dur = to_flac16k(f, dst) if not dst.exists() else sf.info(dst).duration
        x, _ = sf.read(dst, dtype="float32")
        rows.append(_row(f"avq_drone:{top}/{f.stem}", "avq_drone", "noise", f"avq-{top}", "", dst, dur, "CC BY 4.0",
                         stationarity_class(x, SR)))
    return rows


def scan_lombard_grid(root: Path, out: Path) -> list[dict]:
    """Lombard GRID (Alghamdi et al. 2018, CC BY 4.0): <root>/**/<spk>_<l|p>_<sentence>.wav, l = Lombard, p = plain.
    The style letter rides in the source_id (lgrid:<spk>/<l|p>/<stem>) for M11 effort classes; one group per talker."""
    if _absent(root, "lombard_grid"):
        return []
    rows = []
    for f in sorted(Path(root).rglob("*.wav")):
        parts = f.stem.split("_")
        if len(parts) < 3 or parts[1] not in ("l", "p"):
            continue
        spk, style = parts[0], parts[1]
        dst = out / "lombard_grid" / spk / (f.stem + ".flac")
        dur = to_flac16k(f, dst) if not dst.exists() else sf.info(dst).duration
        rows.append(_row(f"lgrid:{spk}/{style}/{f.stem}", "lombard_grid", "speech", f"lgrid-spk-{spk}", spk, dst, dur,
                         "CC BY 4.0"))
    return rows


def _musan_licences(d: Path) -> dict:
    """MUSAN ships a LICENSE file per source folder: '<file> <licence words...>' lines; unknown files get the folder note."""
    lic = {}
    f = d / "LICENSE"
    if f.exists():
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            p = line.split(None, 1)
            if len(p) == 2 and p[0].endswith(".wav"):
                lic[Path(p[0]).stem] = p[1].strip()
    return lic


def scan_musan_noise(root: Path, out: Path) -> list[dict]:
    """MUSAN noise (Snyder et al. 2015): <root>/noise/<source>/*.wav. One group per clip (no recordist field);
    speech and music folders are not taken."""
    if _absent(root, "musan") or not (Path(root) / "noise").exists():
        return []
    rows = []
    for d in sorted(p for p in (Path(root) / "noise").iterdir() if p.is_dir()):
        lic = _musan_licences(d)
        for f in sorted(d.glob("*.wav")):
            dst = out / "musan" / d.name / (f.stem + ".flac")
            dur = to_flac16k(f, dst) if not dst.exists() else sf.info(dst).duration
            x, _ = sf.read(dst, dtype="float32")
            rows.append(_row(f"musan:{d.name}/{f.stem}", "musan", "noise", f"musan-{f.stem}", "", dst, dur,
                             lic.get(f.stem, f"MUSAN {d.name} (see LICENSE)"), stationarity_class(x, SR)))
    return rows


def coherence_null_hz(x2: np.ndarray, sr: int = SR, nper: int = 1024) -> float:
    """First zero crossing above 200 Hz of the real coherence of a two-channel diffuse recording; spacing ~ c / (2 f)."""
    from scipy.signal import csd, welch
    f, sab = csd(x2[:, 0], x2[:, 1], fs=sr, nperseg=nper)
    _, saa = welch(x2[:, 0], fs=sr, nperseg=nper); _, sbb = welch(x2[:, 1], fs=sr, nperseg=nper)
    g = np.real(sab) / np.sqrt(saa * sbb + 1e-30)
    k = np.nonzero((g[1:] <= 0) & (f[1:] > 200))[0]
    return float(f[1 + k[0]]) if len(k) else float("nan")


def demand_pairs_by_null(env_dir: Path, spacing_m=(0.105, 0.135), probe_s: float = 30.0, n_ch: int = 16) -> list:
    """Channel pairs whose measured diffuse-field null implies a 12 cm-like spacing (null = c / 2d: 1.27-1.63 kHz).
    Measured from the audio, not a geometry table, so a wrong grid assumption cannot sneak in."""
    lo, hi = 343.0 / (2 * spacing_m[1]), 343.0 / (2 * spacing_m[0])
    chans = {}
    for c in range(1, n_ch + 1):
        f = env_dir / f"ch{c:02d}.wav"
        if f.exists():
            chans[c] = sf.read(f, dtype="float32", frames=int(sf.info(f).samplerate * probe_s))
    keep = []
    for a in sorted(chans):
        for b in sorted(chans):
            if b <= a:
                continue
            (xa, sr), (xb, _) = chans[a], chans[b]
            n = min(len(xa), len(xb))
            nu = coherence_null_hz(np.stack([xa[:n], xb[:n]], 1), sr)
            if lo <= nu <= hi:
                keep.append((a, b, nu))
    return keep


def scan_demand_pairs(root: Path, out: Path, pairs=None, spacing_m=(0.105, 0.135)) -> list[dict]:
    """DEMAND with several 12 cm-like pairs per environment (plan 11.5): one stereo row per (environment, pair).
    pairs=None measures them per environment (demand_pairs_by_null); all pairs of one environment share a group,
    since they are the same five minutes of sound. scan_demand (one pair, the r7 manifest) is unchanged."""
    if _absent(root, "demand"):
        return []
    rows = []
    for env in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        use = [(a, b) for a, b, _ in demand_pairs_by_null(env, spacing_m)] if pairs is None else list(pairs)
        for a, b in use:
            if not ((env / f"ch{a:02d}.wav").exists() and (env / f"ch{b:02d}.wav").exists()):
                continue
            dst = out / "demand_pairs" / f"{env.name}_ch{a:02d}-{b:02d}.flac"
            if not dst.exists():
                chans, srs = zip(*(sf.read(env / f"ch{c:02d}.wav", dtype="float32") for c in (a, b)))
                x = np.stack(chans, 1)
                if srs[0] != SR:
                    g = np.gcd(srs[0], SR); x = resample_poly(x, SR // g, srs[0] // g, axis=0).astype(np.float32)
                dst.parent.mkdir(parents=True, exist_ok=True); sf.write(dst, x, SR, subtype="PCM_16")
            x, _ = sf.read(dst, dtype="float32")
            rows.append(_row(f"demand:{env.name}/ch{a:02d}-{b:02d}", "demand", "noise", f"demand-{env.name}", "", dst,
                             len(x) / SR, "CC BY-SA 4.0", stationarity_class(x[:, 0], SR)))
    return rows


def scan_but_reverbdb(root: Path, out: Path) -> list[dict]:
    """BUT ReverbDB (Szoke et al. 2019, CC BY 4.0): <root>/<room>/.../RIR/*.wav measured responses. kind "rir", one
    group per room so no room straddles splits. Kept at the measured length; only resampled and made mono."""
    if _absent(root, "but_reverbdb"):
        return []
    rows = []
    for f in sorted(Path(root).rglob("*.wav")):
        rel = f.relative_to(root).parts
        if "RIR" not in rel[:-1]:
            continue
        room = rel[0]
        tag = "_".join(p for p in rel[1:-1] if p != "RIR")
        dst = out / "but_reverbdb" / room / f"{tag}_{f.stem}.flac"
        dur = to_flac16k(f, dst) if not dst.exists() else sf.info(dst).duration
        rows.append(_row(f"butrdb:{room}/{tag}/{f.stem}", "but_reverbdb", "rir", f"butrdb-{room}", "", dst, dur, "CC BY 4.0"))
    return rows


# AudioSet ontology ids for the DNS AudioSet shard filter (plan 11.5: Speech 4.9 %, Music 10.3 % of those clips)
AUDIOSET_SPEECH, AUDIOSET_MUSIC = "/m/09x0r", "/m/04rlf"


def dns_audioset_filter(rows: list[dict], label_csv: Path | None, drop=(AUDIOSET_SPEECH, AUDIOSET_MUSIC)) -> list[dict]:
    """Drop DNS AudioSet noise rows whose YouTube id carries Speech or Music in the AudioSet segments CSV
    (YTID, start_seconds, end_seconds, positive_labels). The CSV is a metadata download not on disk: without it
    the rows pass through unchanged and a note says the filter did not run (TBD). The clip-name -> YTID rule
    (first 11 characters of the stem) is the AudioSet convention, unverified on the DNS shard names (TBD)."""
    if label_csv is None or not Path(label_csv).exists():
        print(f"[dns_audioset_filter] label CSV {label_csv} absent: {len(rows)} rows unfiltered (TBD)")
        return rows
    bad = set()
    with open(label_csv, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            p = [t.strip().strip('"') for t in line.split(",", 3)]
            if len(p) == 4 and any(d in p[3] for d in drop):
                bad.add(p[0])
    return [r for r in rows if Path(str(r["path"])).stem[:11] not in bad]
