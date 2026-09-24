"""MAD speech-contamination pass (plan 11.1 / 11.5): MAD noise classes are cut from YouTube videos, many of which
also yield MAD "communication" (radio speech) clips, so a noise clip can carry speech the model is then taught to
remove. A VAD runs over every MAD noise manifest clip; the flagged list and a video-grouped, speech-free manifest are
written as NEW files (the r7 manifest data/manifests/mad.parquet is never touched).

VAD: Silero (the ONNX model bundled with faster-whisper, run by onnxruntime), the same one the web-WAV diagnosis
used, peak-normalised to 0.5 so level alone does not decide. Without faster-whisper importable it falls back to an
energy + spectral-flatness VAD, which is weaker (it flags gunfire and engines more readily); the JSON names the VAD.
Resumable: per-clip results append to a jsonl cache and are skipped on restart.

usage: python scripts/mad_speech_filter.py [--threshold 0.5] [--min-speech-s 0.5] [--min-frac 0.10]
outputs: data/manifests/mad_speech_contamination.parquet, data/manifests/mad_v2.parquet,
         results_r2/r8/mad_speech_filter.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from vaani.data.sources import MAD_CLASSES, youtube_id   # noqa: E402
from vaani.data.splits import assign   # noqa: E402

SR = 16000


def silero():
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    def run(x, thr):
        y = x / (np.abs(x).max() + 1e-9) * 0.5
        ts = get_speech_timestamps(y, VadOptions(threshold=thr, min_silence_duration_ms=200, min_speech_duration_ms=150))
        return sum(t["end"] - t["start"] for t in ts) / SR
    return "silero (faster-whisper bundle)", run


def energy_vad():
    """Weaker fallback: 20 ms frames within 15 dB of the clip's loudest, low spectral flatness, voiced band dominant."""
    def run(x, thr):
        n = 320; f = x[: len(x) // n * n].reshape(-1, n)
        if not len(f):
            return 0.0
        e = 10 * np.log10((f ** 2).mean(1) + 1e-12)
        S = np.abs(np.fft.rfft(f * np.hanning(n), axis=1)) ** 2 + 1e-12
        flat = np.exp(np.log(S).mean(1)) / S.mean(1)
        fr = np.fft.rfftfreq(n, 1 / SR); band = S[:, (fr >= 300) & (fr <= 3400)].sum(1) / S.sum(1)
        v = (e > e.max() - 15) & (flat < 0.3 * (1 + thr)) & (band > 0.6)
        return float(v.sum() * n / SR)
    return "energy+flatness (weaker fallback)", run


def mad_video_map(root: Path) -> pd.DataFrame:
    """source_id -> youtube id, and whether that video also yields MAD communication clips."""
    fr = []
    for part in ("training", "test"):
        d = pd.read_csv(root / f"{part}.csv"); d["part"] = part; fr.append(d)
    d = pd.concat(fr, ignore_index=True)
    d["vid"] = d.path.str.split("/").str[1]; d["stem"] = d.path.str.split("/").str[2].str[:-4]
    d["cls"] = [MAD_CLASSES[int(i)] for i in d.label]
    d["youtube_id"] = d["youtube url"].map(youtube_id)
    comm = set(d.loc[(d.cls == "communication") & (d.youtube_id != ""), "youtube_id"])
    d["video_has_comm"] = d.youtube_id.isin(comm) & (d.youtube_id != "")
    d["source_id"] = "mad:" + d.cls + "/" + d.vid + "_" + d.stem
    return d[d.cls != "communication"][["source_id", "vid", "youtube_id", "video_has_comm"]]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(REPO / "data/manifests/mad.parquet"))
    ap.add_argument("--mad-root", default=str(REPO / "data/download/mad/MAD_dataset"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-speech-s", type=float, default=0.5, help="flag when VAD speech >= this many seconds ...")
    ap.add_argument("--min-frac", type=float, default=0.10, help="... and >= this fraction of the clip")
    ap.add_argument("--vad", choices=["auto", "silero", "energy"], default="auto")
    ap.add_argument("--out-contam", default=str(REPO / "data/manifests/mad_speech_contamination.parquet"))
    ap.add_argument("--out-manifest", default=str(REPO / "data/manifests/mad_v2.parquet"))
    ap.add_argument("--out-json", default=str(REPO / "results_r2/r8/mad_speech_filter.json"))
    ap.add_argument("--cache", default=str(REPO / "data/manifests/.mad_vad_cache.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    src = Path(a.manifest).resolve()
    for o in (a.out_contam, a.out_manifest):
        if Path(o).resolve() == src:
            raise SystemExit("refusing to overwrite the input manifest")
    name, vad = None, None
    if a.vad in ("auto", "silero"):
        try:
            name, vad = silero()
        except Exception as e:   # noqa: BLE001 - any import failure: fall back or stop, as asked
            if a.vad == "silero":
                raise
            print(f"silero unavailable ({e}); energy fallback", flush=True)
    if vad is None:
        name, vad = energy_vad()
    man = pd.read_parquet(src)
    if a.limit:
        man = man.head(a.limit)
    cache = Path(a.cache); done = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("vad") == name and r.get("threshold") == a.threshold:
                done[r["source_id"]] = r["speech_s"]
    t0 = time.time()
    with open(cache, "a", encoding="utf-8") as fh:
        for i, r in enumerate(man.itertuples()):
            if r.source_id in done:
                continue
            x, sr = sf.read(str(REPO / str(r.path).replace("\\", "/")), dtype="float32")
            assert sr == SR
            done[r.source_id] = float(vad(x if x.ndim == 1 else x[:, 0], a.threshold))
            fh.write(json.dumps({"source_id": r.source_id, "speech_s": done[r.source_id], "vad": name,
                                 "threshold": a.threshold}) + "\n")
            if i % 500 == 0:
                fh.flush(); print(f"{i}/{len(man)} {time.time() - t0:.0f}s", flush=True)
    man = man.copy()
    man["speech_s"] = man.source_id.map(done)
    man["speech_frac"] = man.speech_s / man.duration_s.clip(lower=1e-6)
    man["flagged"] = (man.speech_s >= a.min_speech_s) & (man.speech_frac >= a.min_frac)
    vm = mad_video_map(Path(a.mad_root)) if Path(a.mad_root).exists() else None
    if vm is not None:
        man = man.merge(vm, on="source_id", how="left")
        man["youtube_id"] = man.youtube_id.fillna("")
        man["video_has_comm"] = man.video_has_comm.fillna(False).astype(bool)
    else:
        man["youtube_id"] = ""; man["video_has_comm"] = False
    contam = man[["source_id", "group_id", "youtube_id", "video_has_comm", "duration_s", "speech_s", "speech_frac", "flagged"]]
    Path(a.out_contam).parent.mkdir(parents=True, exist_ok=True)
    contam.to_parquet(a.out_contam, index=False)
    # v2 manifest: speech-flagged clips dropped, grouped (and so split) by YouTube video; folder when no id is known
    keep = man[~man.flagged].copy()
    keep["group_id"] = np.where(keep.youtube_id != "", "mad-yt-" + keep.youtube_id, keep.group_id)
    keep["split"] = keep.group_id.map(assign)
    from vaani.data.manifests import COLUMNS
    keep[COLUMNS].to_parquet(a.out_manifest, index=False)
    by_vid = man[man.youtube_id != ""].groupby("youtube_id").agg(flag=("flagged", "any"), comm=("video_has_comm", "first"))
    res = {"vad": name, "threshold": a.threshold, "flag_rule": f"speech_s >= {a.min_speech_s} and speech_frac >= {a.min_frac}",
           "clips": int(len(man)), "clips_flagged": int(man.flagged.sum()),
           "hours": float(man.duration_s.sum() / 3600), "hours_flagged": float(man.duration_s[man.flagged].sum() / 3600),
           "flagged_by_class": {k: int(v) for k, v in man[man.flagged].source_id.str.split("[:/]").str[1].value_counts().items()},
           "clips_by_class": {k: int(v) for k, v in man.source_id.str.split("[:/]").str[1].value_counts().items()},
           "videos": int(len(by_vid)), "videos_with_comm_clips": int(by_vid.comm.sum()),
           "videos_flagged": int(by_vid.flag.sum()), "videos_flagged_and_with_comm": int((by_vid.flag & by_vid.comm).sum()),
           "flag_rate_videos_with_comm": float(man[man.video_has_comm].flagged.mean()) if man.video_has_comm.any() else None,
           "flag_rate_videos_without_comm": float(man[~man.video_has_comm].flagged.mean()) if (~man.video_has_comm).any() else None,
           "speech_s_p50_p90_p99": [float(v) for v in np.percentile(man.speech_s, [50, 90, 99])],
           "v2_manifest": {"path": Path(a.out_manifest).relative_to(REPO).as_posix() if Path(a.out_manifest).is_relative_to(REPO) else a.out_manifest,
                           "rows": int(len(keep)), "groups": int(keep.group_id.nunique()),
                           "split_rows": {k: int(v) for k, v in keep.split.value_counts().items()}},
           "seconds": round(time.time() - t0, 1)}
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
