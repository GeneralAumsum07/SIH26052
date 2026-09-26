#!/usr/bin/env python3
"""Field acceptance test (gate G4): does a system keep speech on real-world audio beds and two-channel constructions?

    uv run --with numba python scripts/field_accept.py --system r7 --name r7 --workers 2
    uv run --with numba python scripts/field_accept.py --system stream:runs/r8/cascade.onnx@deploy/r8/model_config.json --name r8
    uv run --with numba python scripts/field_accept.py --system ckpt:results_r2/runs/<run>/best.pt --name <run>
    python scripts/field_accept.py --name r7 --summarise-only

`--system`: `r7` (deploy/r7/cascade.onnx through vaani.live.StreamEngine, the board path), `stream:<onnx>[@<config>]`
(any ONNX with the StreamEngine contract; config defaults to deploy/r7/model_config.json), or any spec
`vaani.eval.enhance_fn` knows (`ckpt:`, `cascade:`, `onnx:<graph>@<ckpt>`, baseline names), run offline on (2, T).

Part 1, speech injection (clean reference known). Noise beds: the web WAV (C:/Users/Rachit/Downloads/abcd.wav,
stereo, L/R = its own noise channels, up to 60 s) and MAD "communication" clips (mono; one per video, seed 0, as
scripts/score_real.py). `--n-utt` clean utterances (6 s: LibriSpeech-100h / CV-hi `--speech-split` speech, pieces of at
most 4 s with 0.3 s gaps, speech at -26 dBFS active level) are injected at each SNR in `--snrs` (re primary noise).
Constructions (p = s + g*nL on every one, so the primary input is identical across them):
  M  mono duplicated: r = p            W  web stereo: r = (s + g*nR) delayed +-0.5 ms, gain +-1 dB (speech centred)
  H4 r = s*10^(-4/20) + g*nR           H8 r = s*10^(-8/20) + g*nR
  Z  r = 0 (reference zeroed)          G  r = p*10^(-12/20)
nR is the bed's R channel; a mono bed's nR = 0.8*nL + 0.6*(the same bed at another offset), corr ~0.8 (the
diag_webaudio web-like construction). Speech loss is the diag_webaudio definition (frame_stats): per 20 ms frame
where the clean speech is within 30 dB of its max, the projection gain a = <y,c>/<c,c>; lost if 20log10(a) < -15 dB.
Criteria per bed x construction (SNRs pooled): mean loss <= 0.06; p95 per-clip loss <= 0.15; longest lost run
(max over clips) <= 0.3 s; mean STOI_out >= mean STOI_in - 0.02; mean dSNR >= +3 dB; on M, W, Z mean dSNR >=
gtcrn_pretrained's mean dSNR on the same primaries - 1 dB; on H8 at +5 dB mean SNR_out > 15, STOI > 0.85, PESQ > 2.5.

Headline rows (Rachit, 2026-09-25): single-channel audio is headlined by Z / ref_zero (reference zeroed, at
validity 0 where the model takes a validity input; r7 has none), and M / mono_dup (duplicated primary) is a labelled
stress row. The criteria are unchanged; only the report order and labels follow this.

Part 2, reference-free, on the original web WAV run four ways (as_is L/R, ref_zero, mono_dup L/L, swapped R/L); all
comparisons against ref_zero. Whisper confident-word survival (vaani.asr.WordTranscriber: faster-whisper
`--whisper-model`, default small, `--asr-device` auto; words p > 0.5 in the run's primary input, survived = same word
in the output within 0.5 s); mono survival = the same on mono_dup. Silero VAD speech seconds (vaani.asr.
vad_speech_seconds: the VAD bundled in the faster-whisper wheel, run on onnxruntime, no download; threshold 0.3,
peak-normalised, no edge padding). Both need faster-whisper (the `asr` extra); `--asr off` or not importable -> TBD.
Longest stretch attenuated > 30 dB: longest run over the active input frames (20 ms, energy within 30 dB of the p99
frame; inactive frames skipped) where output < input - 30 dB, as_is run, <= 1.0 s. Validity-flag latency: first hop at
which the engine's reference-informativeness estimate (`eng.last[<--validity-key>]`, default `ref_informative`, which
the runtime guards expose under `--guards`; falsy = uninformative) fires on as_is and mono_dup, <= 0.5 s; TBD when
the system exposes no estimate (r7; offline specs; guards off). The capture-path `validity` key is not an estimate
and is not autodetected. `proxy_survival` = 1 - atten20_frac is reported, not gated.

Reporting (Rachit, 2026-09-26): the markdown and JSON state the runtime-guard state of each part (`guards`; on for
the r8 candidates, off for the r7 baseline, whose default path runs without them), and every PESQ figure carries
`pesq_nan` (clips whose isolated PESQ child failed) and the pesq 0.0.4 garbage-read footnote. Scores are unchanged.

Per-task rows are appended to <out>/work/<name>_part1.jsonl, so a killed run resumes.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from vaani import physical  # noqa: E402

SR, HOP, F = 16000, 256, 320
NS = 6 * SR
BASE_LVL = -26.0
# env, not only a flag: spawned workers re-import this module and must see the same file
WEB_WAV = Path(os.environ.get("VAANI_WEB_WAV", "C:/Users/Rachit/Downloads/abcd.wav"))
OUT = REPO / "results_r2/field"
CONS = ["M", "W", "H4", "H8", "Z", "G"]
# report order and row labels: the single-channel headline first, the duplicated-primary stress row last
SHOW = ["Z", "W", "H4", "H8", "G", "M"]
ROW = {"Z": "single-channel headline", "M": "stress (duplicated primary)", "W": "two-channel", "H4": "two-channel",
       "H8": "two-channel", "G": "two-channel"}
SHOW2 = ["ref_zero", "as_is", "swapped", "mono_dup"]
ROW2 = {"ref_zero": "single-channel headline (comparison base)", "as_is": "as recorded", "swapped": "as recorded",
        "mono_dup": "stress (duplicated primary)"}
# eng.last keys of a reference-informativeness estimate; not ref_valid or validity (capture-path availability)
VALIDITY_KEYS = ("ref_informative", "ref_validity")
LOSS_MEAN, LOSS_P95, LOST_RUN_S, STOI_TOL, DSNR_MIN, GTCRN_TOL = 0.06, 0.15, 0.3, 0.02, 3.0, 1.0
PS_SNR, PS_STOI, PS_PESQ = 15.0, 0.85, 2.5
SURV_RATIO, ATTEN30_RUN_S, VALID_LAT_S = 0.8, 1.0, 0.5
# decision 7 (Rachit, 2026-09-26): every G4 table states its guard state; the rule is printed with it
GUARD_RULE = "rule (Rachit, 2026-09-26): guards on for the r8 candidates, off for the r7 baseline"
# decision 6 (Rachit, 2026-09-26): PESQ NaNs counted, garbage reads footnoted (results_r2/r8/native_crash/README.md)
PESQ_FOOT = ("\u2020 PESQ: pesq 0.0.4 reads out of bounds in `utterance_split` on some noise-dominated inputs. A read "
             "that faults kills the isolated PESQ child: the clip scores NaN, is counted in `pesq_nan` and is left out "
             "of the PESQ mean. A read that does not fault returns a garbage value that cannot be detected per clip: 2 "
             "of 2,280 raw noisy eval_r2_relabel/test inputs (0.09 %) under ASan "
             "(results_r2/r8/native_crash/asan/sweep_relabel_test.tsv); the rate on model outputs was not measured.")


def guard_label(on):
    return "guards on (r8-candidate setting)" if on else "guards off (r7 default path)"


def guard_state(res, requested=False):
    """Part 1 never runs the guards; Part 2 as its run recorded (a run without the key predates it: off), else --guards."""
    p2 = bool(res["part2"].get("guards", False)) if res.get("part2") else bool(requested)
    return {"part1": False, "part2": p2, "part1_label": guard_label(False), "part2_label": guard_label(p2),
            "rule": GUARD_RULE}


# ---------- signals ----------

def active_power(x, frame=F, thresh_db=-30.0):
    f = x[: len(x) // frame * frame].reshape(-1, frame).astype(np.float64)
    e = (f ** 2).mean(1) + 1e-12
    keep = e > e.max() * 10 ** (thresh_db / 10)
    return float(e[keep].mean()) if keep.any() else float(e.mean())


def fit(x, n, rng):
    """Crop (random offset) or tile to n samples, as mixer._fit."""
    if len(x) >= n:
        o = int(rng.integers(0, len(x) - n + 1)); return x[o:o + n]
    return np.tile(x, int(np.ceil(n / len(x))))[:n]


def frac_delay(x, d):
    k = np.fft.rfftfreq(len(x))
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * k * d), len(x)).astype(np.float32)


def load16(path):
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return physical.to_16k(x.T, sr)                                    # (ch, n)


def utterance(i, split, seed):
    """6 s clean speech at BASE_LVL; even items LibriSpeech-100h, odd CV-hi; deterministic in (i, split, seed)."""
    import pandas as pd
    rng = np.random.default_rng(7000 + 100003 * seed + i)
    man = pd.read_parquet(REPO / "data/manifests" / ("librispeech_100h.parquet" if i % 2 == 0 else "cv_hi.parquet"))
    man = man[man.split == split].sort_values("source_id").reset_index(drop=True)
    parts, n = [], 0
    while n < NS:
        x = load16(REPO / man.path.iloc[int(rng.integers(len(man)))].replace("\\", "/"))[0][: 4 * SR]
        parts += [x, np.zeros(int(0.3 * SR), np.float32)]; n += len(x) + int(0.3 * SR)
    s = np.concatenate(parts)[:NS]
    return (s * (10 ** (BASE_LVL / 20) / (np.sqrt(active_power(s)) + 1e-9))).astype(np.float32)


def bed_noise(bed, i, seed, mad_clips):
    """(nL, nR) NS-long noise for item i of a bed."""
    rng = np.random.default_rng(9000 + 100003 * seed + i)
    if bed == "web":
        x = load16(WEB_WAV)[:, : 60 * SR]
        o = int(rng.integers(0, max(1, x.shape[1] - NS + 1)))
        seg = x[:, o:o + NS]
        if seg.shape[1] < NS:
            seg = np.tile(seg, (1, int(np.ceil(NS / seg.shape[1]))))[:, :NS]
        return seg[0].astype(np.float32), seg[1].astype(np.float32)
    clip = load16(mad_clips[i % len(mad_clips)]["path"])[0][: 10 * SR]
    nL, other = fit(clip, NS, rng), fit(clip, NS, rng)
    other = other * np.sqrt(((nL ** 2).mean() + 1e-12) / ((other ** 2).mean() + 1e-12))
    return nL.astype(np.float32), (0.8 * nL + 0.6 * other).astype(np.float32)


def construct(s, nL, nR, cons, snr, rng):
    """(primary, reference) for one construction; the primary is s + g*nL on all of them."""
    g = np.sqrt(active_power(s) / (((nL ** 2).mean() + 1e-12) * 10 ** (snr / 10)))
    p = s + g * nL
    if cons == "M":
        r = p.copy()
    elif cons == "W":
        dg, dd = rng.uniform(-1, 1), rng.uniform(-0.5, 0.5) * SR / 1000
        r = frac_delay((s + g * nR).astype(np.float32), dd) * 10 ** (dg / 20)
    elif cons in ("H4", "H8"):
        r = s * 10 ** (-int(cons[1:]) / 20) + g * nR
    elif cons == "Z":
        r = np.zeros_like(p)
    elif cons == "G":
        r = p * 10 ** (-12 / 20)
    else:
        raise ValueError(cons)
    return np.clip(p, -1, 1).astype(np.float32), np.clip(r, -1, 1).astype(np.float32)


# ---------- metrics ----------

def frame_stats(clean, y):
    """diag_webaudio speech loss: share of speech-active frames with projected speech gain < -15 dB, and the
    longest run of lost frames in seconds."""
    k = len(clean) // F
    C, Y = (v[: k * F].reshape(k, F).astype(np.float64) for v in (clean, y))
    ec = (C ** 2).mean(1) + 1e-12
    act = 10 * np.log10(ec) > 10 * np.log10(ec.max()) - 30
    a = (Y * C).sum(1) / ((C * C).sum(1) + 1e-12)
    lost = act & (20 * np.log10(np.maximum(a, 1e-4)) < -15)
    run = best = 0
    for v in lost:
        run = run + 1 if v else 0; best = max(best, run)
    return float(lost.sum() / max(1, act.sum())), best * F / SR


def longest_atten_run(x_in, y, db=30.0):
    """Longest run, over active input frames only (inactive skipped), of frames where output < input - db; seconds."""
    n = min(len(x_in), len(y))
    ein, eout = physical._frames_db(x_in[:n]), physical._frames_db(y[:n])
    if len(ein) == 0:
        return float("nan")
    act = ein > np.percentile(ein, 99) - physical.ACTIVE_DB
    run = best = 0
    for v in (eout < ein - db)[act]:
        run = run + 1 if v else 0; best = max(best, run)
    return best * F / SR


def validity_latency(trace, key):
    """Seconds from the start until the flag first reads uninformative (falsy / < 0.5); None if never; 'absent' if
    the engine does not expose it."""
    if not trace or key is None or not any(key in d for d in trace):
        return "absent"
    for j, d in enumerate(trace):
        v = d.get(key)
        if v is not None and float(v) < 0.5:
            return j * HOP / SR
    return None


# ---------- ASR / VAD hooks (vaani.asr; faster-whisper, the `asr` extra; TBD when not importable) ----------

def have_whisper():
    from vaani import asr
    return asr.available()


def asr_hooks(mode="auto", model=None, device="auto", threads=1):
    """(transcriber, vad) for Part 2: vaani.asr.WordTranscriber and the Silero VAD bundled with faster-whisper, both
    lazy; (None, None) when mode is 'off' or faster-whisper is not importable (those criteria are then TBD)."""
    from vaani import asr
    if mode == "off" or not asr.available():
        return None, None
    return asr.WordTranscriber(model or asr.WORD_MODEL, device, threads), asr.vad_speech_seconds


def word_survival(ref_words, out_words, conf=0.5, tol=0.5):
    """Fraction of confident input words that the output transcript has (same word, start within tol s)."""
    if ref_words is None or out_words is None:
        return None
    ref = [w for w in ref_words if w[3] > conf and w[0]]
    if not ref:
        return float("nan")
    return sum(any(o[0] == w[0] and abs(o[1] - w[1]) <= tol for o in out_words) for w in ref) / len(ref)


# ---------- systems ----------

def make_system(spec, guards=False):
    """-> fn(p, r, trace=False, ref_valid=True) -> (y (T,), diag dict). Stream systems run hop by hop through
    vaani.live; guards=True turns on the plan 11.3 runtime guards (stream systems only; off = the r7 default path)."""
    if spec == "r7" or spec.startswith("stream:"):
        onnx, config = physical.ONNX, physical.CONFIG
        if spec.startswith("stream:"):
            onnx, _, cfg = spec[7:].partition("@")
            config = cfg or physical.CONFIG

        def run(p, r, trace=False, ref_valid=True):
            return physical.run_engine(np.stack([p, r]), onnx, config, 1, trace=trace, ref_valid=ref_valid,
                                       guards=True if guards else None)
        run.stream = True
        run.validity0 = lambda: physical.engine_takes_valid(onnx, config)
        return run
    from vaani import eval as veval
    f = veval.enhance_fn(spec, "cpu")

    def run(p, r, trace=False, ref_valid=True):         # offline specs take no validity: zeroed reference only
        return np.asarray(f(np.stack([p, r]).astype(np.float32)), np.float32)[: len(p)], {}
    run.stream = False
    run.validity0 = lambda: False
    return run


_SYS = {}


def system(spec, guards=False):
    if (spec, guards) not in _SYS:
        _SYS[(spec, guards)] = make_system(spec, guards)
    return _SYS[(spec, guards)]


def _init_worker():
    import torch
    torch.set_num_threads(1)


def part1_task(t):
    """All constructions (+ gtcrn_pretrained on the shared primary) for one bed x utterance x SNR."""
    from vaani import metrics
    spec, bed, i, snr, split, seed, mad_clips = t
    s = utterance(i, split, seed)
    nL, nR = bed_noise(bed, i, seed, mad_clips)
    rows, p0 = [], None
    for c in CONS:
        p, r = construct(s, nL, nR, c, snr, np.random.default_rng(11000 + 100003 * seed + 97 * i + int(snr)))
        p0 = p
        y, diag = system(spec)(p, r, ref_valid=c != "Z")         # Z = the reference-absent mode (validity 0)
        loss, run = frame_stats(s, y)
        rows.append({"bed": bed, "item": i, "snr": snr, "cons": c, "system": spec, "loss": loss, "lost_run_s": run,
                     "snr_in": metrics.snr_db(s, p), "snr_out": metrics.snr_db(s, y),
                     "stoi_in": metrics.stoi(s, p), "stoi_out": metrics.stoi(s, y), "pesq_out": metrics.pesq_wb(s, y),
                     **{k: v for k, v in diag.items() if k != "trace"}})
    y = system("gtcrn_pretrained")(p0, p0)[0]
    loss, run = frame_stats(s, y)
    rows.append({"bed": bed, "item": i, "snr": snr, "cons": "mono", "system": "gtcrn_pretrained", "loss": loss,
                 "lost_run_s": run, "snr_in": metrics.snr_db(s, p0), "snr_out": metrics.snr_db(s, y),
                 "stoi_in": metrics.stoi(s, p0), "stoi_out": metrics.stoi(s, y), "pesq_out": metrics.pesq_wb(s, y)})
    return f"{bed}|{i}|{snr}", rows


def _task_key(t):
    return f"{t[1]}|{t[2]}|{t[3]}"


def run_part1(a, work):
    cache = work / f"{a.name}_part1.jsonl"
    done = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.strip():
                k, rows = json.loads(line); done[k] = rows
    mad = []
    if "mad" in a.beds:
        import score_real
        mad = [{"clip": r["clip"], "path": str(r["path"])} for r in
               score_real.select_subset(score_real.scan_mad_communication(), a.n_utt, a.seed)]
    tasks = [(a.system, bed, i, snr, a.speech_split, a.seed, mad) for bed in a.beds for i in range(a.n_utt)
             for snr in a.snrs]
    todo = [t for t in tasks if _task_key(t) not in done]
    print(f"part 1: {len(tasks)} tasks, {len(done)} cached, {len(todo)} to run, {a.workers} worker(s)", flush=True)
    with open(cache, "a", encoding="utf-8") as fh:
        def put(res):
            k, rows = res; done[k] = rows
            fh.write(json.dumps([k, rows]) + "\n"); fh.flush()
            print(f"  {k}: " + ", ".join(f"{r['cons']} {r['loss']:.2f}" for r in rows), flush=True)
        if a.workers <= 1:
            _init_worker()
            for t in todo:
                put(part1_task(t))
        else:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(a.workers, initializer=_init_worker) as pool:
                for res in pool.imap_unordered(part1_task, todo):
                    put(res)
    return [r for t in tasks for r in done.get(_task_key(t), [])], mad


def run_part2(a, wav=None, transcriber="auto", vad="auto"):
    """The four reference-free runs of `wav`. transcriber(y) -> [(word, start, end, p)] and vad(y) -> seconds are
    injectable; "auto" = asr_hooks(--asr, --whisper-model, --asr-device), None = that criterion TBD."""
    if transcriber == "auto" or vad == "auto":
        t, v = asr_hooks(getattr(a, "asr", "auto"), getattr(a, "whisper_model", None), getattr(a, "asr_device", "auto"))
        transcriber = t if transcriber == "auto" else transcriber
        vad = v if vad == "auto" else vad
    words = (lambda y: None) if transcriber is None else transcriber
    vsec = (lambda y: None) if vad is None else vad
    wav = WEB_WAV if wav is None else wav
    x = load16(wav)
    L, R = x[0], x[1] if x.shape[0] > 1 else x[0]
    runs = {"as_is": (L, R), "ref_zero": (L, np.zeros_like(L)), "mono_dup": (L, L.copy()), "swapped": (R, L)}
    guards = bool(getattr(a, "guards", False))
    fn = system(a.system, guards)
    key = a.validity_key
    res = {"wav": str(wav), "seconds": len(L) / SR, "whisper_available": transcriber is not None,
           "vad_available": vad is not None, "asr_model": getattr(transcriber, "model_name", None),
           "asr_device": getattr(transcriber, "device", None), "guards": guards,
           "ref_zero_validity0": fn.validity0(), "runs": {}}
    in_words, in_vad = {}, {}
    for name, (p, r) in runs.items():
        y, diag = fn(p, r, trace=fn.stream, ref_valid=name != "ref_zero")
        tr = diag.get("trace") or []
        k = key or next((c for c in VALIDITY_KEYS if tr and any(c in d for d in tr)), None)
        prox = physical.attenuation_proxy(p, y)
        pid = "R" if name == "swapped" else "L"
        if pid not in in_words:                         # each primary is transcribed once
            in_words[pid], in_vad[pid] = words(p), vsec(p)
        res["runs"][name] = {"longest_atten30_s": longest_atten_run(p, y), "atten20_frac": prox["atten20_frac"],
                             "proxy_survival": 1 - prox["atten20_frac"], "vad_speech_s": vsec(y),
                             "vad_speech_in_s": in_vad[pid],
                             "word_survival": word_survival(in_words[pid], words(y)),
                             "validity_key": k, "validity_latency_s": validity_latency(tr, k) if fn.stream else "absent"}
        print(f"part 2 {name}: {res['runs'][name]}", flush=True)
    res["asr_device"] = getattr(transcriber, "device", None)   # resolved once the model has loaded
    return res


# ---------- verdicts ----------

def _v(ok):
    return "TBD" if ok is None else ("PASS" if ok else "FAIL")


def _nan_count(rows):
    """PESQ NaNs (the isolated PESQ child failed); np.nanmean leaves them out of pesq_out."""
    return int(sum(1 for r in rows if r["pesq_out"] is None or not np.isfinite(r["pesq_out"])))


def summarise_part1(rows):
    out, crit = {}, []
    beds = list(dict.fromkeys(r["bed"] for r in rows))
    for bed in beds:
        g = [r for r in rows if r["bed"] == bed and r["system"] == "gtcrn_pretrained"]
        gd = float(np.mean([r["snr_out"] - r["snr_in"] for r in g])) if g else float("nan")
        out[f"{bed}/gtcrn_pretrained"] = {"n": len(g), "dsnr_mean": gd,
                                          "loss_mean": float(np.mean([r["loss"] for r in g])) if g else float("nan")}
        for c in CONS:
            x = [r for r in rows if r["bed"] == bed and r["cons"] == c and r["system"] != "gtcrn_pretrained"]
            if not x:
                continue
            loss = np.array([r["loss"] for r in x]); dsnr = np.array([r["snr_out"] - r["snr_in"] for r in x])
            st = {"n": len(x), "loss_mean": float(loss.mean()), "loss_p95": float(np.percentile(loss, 95)),
                  "lost_run_max_s": float(max(r["lost_run_s"] for r in x)),
                  "lost_run_mean_s": float(np.mean([r["lost_run_s"] for r in x])),
                  "stoi_in": float(np.mean([r["stoi_in"] for r in x])),
                  "stoi_out": float(np.mean([r["stoi_out"] for r in x])),
                  "snr_in": float(np.mean([r["snr_in"] for r in x])), "snr_out": float(np.mean([r["snr_out"] for r in x])),
                  "dsnr_mean": float(dsnr.mean()), "pesq_out": float(np.nanmean([r["pesq_out"] for r in x])),
                  "pesq_nan": _nan_count(x)}
            cr = {"loss_mean<=0.06": st["loss_mean"] <= LOSS_MEAN, "loss_p95<=0.15": st["loss_p95"] <= LOSS_P95,
                  "lost_run<=0.3s": st["lost_run_max_s"] <= LOST_RUN_S,
                  "stoi_out>=stoi_in-0.02": st["stoi_out"] >= st["stoi_in"] - STOI_TOL,
                  "dsnr>=+3dB": st["dsnr_mean"] >= DSNR_MIN}
            if c in ("M", "W", "Z"):
                cr["dsnr>=gtcrn-1dB"] = st["dsnr_mean"] >= gd - GTCRN_TOL
            if c == "H8":
                h = [r for r in x if r["snr"] == 5.0]
                if h:
                    st["ps_snr_out"] = float(np.mean([r["snr_out"] for r in h]))
                    st["ps_stoi"] = float(np.mean([r["stoi_out"] for r in h]))
                    st["ps_pesq"] = float(np.nanmean([r["pesq_out"] for r in h])); st["ps_pesq_nan"] = _nan_count(h)
                    cr["ps_targets@+5dB"] = (st["ps_snr_out"] > PS_SNR and st["ps_stoi"] > PS_STOI
                                            and st["ps_pesq"] > PS_PESQ)
                else:
                    cr["ps_targets@+5dB"] = None
            st["criteria"] = {k: _v(v) for k, v in cr.items()}
            st["verdict"] = "PASS" if all(cr.values()) else "FAIL"
            out[f"{bed}/{c}"] = st
            crit += list(cr.values())
    return out, ("PASS" if crit and all(crit) else "FAIL")


def summarise_part2(p2):
    R = p2["runs"]; z = R["ref_zero"]
    ratio = lambda a, b: None if a is None or b is None else (a >= SURV_RATIO * b)
    lat = [R[k]["validity_latency_s"] for k in ("as_is", "mono_dup")]
    lat_ok = None if any(v == "absent" for v in lat) else all(v is not None and v <= VALID_LAT_S for v in lat)
    cr = {"word_survival>=0.8x_ref_zero": ratio(R["as_is"]["word_survival"], z["word_survival"]),
          "vad_speech_s>=0.8x_ref_zero": ratio(R["as_is"]["vad_speech_s"], z["vad_speech_s"]),
          "longest_atten30<=1.0s": R["as_is"]["longest_atten30_s"] <= ATTEN30_RUN_S,
          "mono_word_survival>=0.8x_ref_zero": ratio(R["mono_dup"]["word_survival"], z["word_survival"]),
          "validity_flag<=0.5s": lat_ok}
    verdict = "FAIL" if any(v is False for v in cr.values()) else ("TBD" if any(v is None for v in cr.values())
                                                                     else "PASS")
    return {k: _v(v) for k, v in cr.items()}, verdict


def _f(v, p=3):
    if v is None:
        return "TBD"
    if isinstance(v, str):
        return v
    return "n/a" if not np.isfinite(v) else f"{v:.{p}f}"


def _part1_rows(part1):
    """Per bed: the gtcrn reference, then the constructions in SHOW order (headline Z first, stress M last)."""
    beds = list(dict.fromkeys(k.split("/")[0] for k in part1))
    keys = [f"{b}/{c}" for b in beds for c in ["gtcrn_pretrained", *SHOW]]
    return [(k, part1[k]) for k in keys if k in part1] + [(k, v) for k, v in part1.items() if k not in keys]


def write_md(res, path):
    gs = res.get("guards") or guard_state(res)
    L = [f"# G4 field acceptance: {res['name']} (`{res['system']}`)", "",
         f"Runtime guards: Part 1 {gs['part1_label']} (Part 1 never runs them); Part 2 {gs['part2_label']}. "
         f"{gs['rule'][0].upper() + gs['rule'][1:]}.", "",
         f"Generated by `{res['command']}`"
         + (f" (tables rebuilt by `{res['summarised_by']}`)" if res.get("summarised_by") else "")
         + f". Part 1 verdict: **{res['part1_verdict']}**; Part 2 verdict: **{res['part2_verdict']}**."
         f" Definitions and criteria: scripts/field_accept.py docstring (plan 11.2 G4).", "",
         f"Part 1: {res['n_utt']} utterances ({res['speech_split']} split, seed {res['seed']}) x SNR "
         f"{res['snrs']} dB per bed; SNRs pooled per row; gtcrn_pretrained runs on the same primaries.", "",
         "Rows: `single-channel headline` = Z (reference zeroed"
         + (", validity 0" if res.get("ref_zero_validity0") else "; this system takes no validity input") +
         "); `stress` = M (reference = the primary, a duplicated mono feed), never the headline; `two-channel` = the "
         "W/H4/H8/G constructions. Every criterion applies to every row as before.", "",
         f"Part 1 table: {gs['part1_label']}.", "",
         "| row | bed/cons | n | loss mean | loss p95 | lost run max s | STOI in -> out | SNR in -> out | dSNR | "
         "PESQ out\u2020 | pesq_nan | verdict | failed criteria |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, st in _part1_rows(res["part1"]):
        if k.endswith("gtcrn_pretrained"):
            L.append(f"| baseline | {k} | {st['n']} | {_f(st['loss_mean'])} | | | | | {_f(st['dsnr_mean'], 2)} | | | ref | |")
            continue
        bad = ", ".join(c for c, v in st["criteria"].items() if v != "PASS") or "-"
        lab = ROW.get(k.split("/")[-1], "")
        L.append(f"| {'**' + lab + '**' if lab.endswith('headline') else lab} | {k} | {st['n']} | {_f(st['loss_mean'])} | {_f(st['loss_p95'])} | {_f(st['lost_run_max_s'], 2)} | "
                 f"{_f(st['stoi_in'])} -> {_f(st['stoi_out'])} | {_f(st['snr_in'], 1)} -> {_f(st['snr_out'], 1)} | "
                 f"{_f(st['dsnr_mean'], 2)} | {_f(st['pesq_out'], 2)} | {st.get('pesq_nan', 'TBD')} | {st['verdict']} | {bad} |")
    h8 = [(k, st) for k, st in res["part1"].items() if k.endswith("/H8") and "ps_snr_out" in st]
    if h8:
        L += ["", "H8 at +5 dB (PS targets SNR_out > 15, STOI > 0.85, PESQ > 2.5): " + "; ".join(
            f"{k}: {_f(st['ps_snr_out'], 1)} / {_f(st['ps_stoi'])} / {_f(st['ps_pesq'], 2)}\u2020 "
            f"(pesq_nan {st.get('ps_pesq_nan', 'TBD')})" for k, st in h8)]
    L += ["", PESQ_FOOT]
    if res.get("part2"):
        p2 = res["part2"]
        L += ["", f"## Part 2: reference-free, `{p2['wav']}` ({p2['seconds']:.1f} s)", "",
              f"Word transcriber: {p2['whisper_available']}"
              + (f" (faster-whisper `{p2.get('asr_model')}` on {p2.get('asr_device')})" if p2["whisper_available"] else "")
              + f"; Silero VAD: {p2.get('vad_available', p2['whisper_available'])}; runtime guards: "
              f"{p2.get('guards', False)}. Whisper/VAD criteria are TBD without faster-whisper (the `asr` extra).", "",
              f"Part 2 tables: {gs['part2_label']}.", "",
              f"ref_zero at validity 0: {p2.get('ref_zero_validity0', False)} (False = reference zeroed only; the "
              "system takes no validity input or the run predates the flag).", "",
              "| row | run | longest >30 dB s | atten>20dB frac | proxy survival | VAD speech s (in) | word survival | "
              "validity latency s |", "|---|---|---|---|---|---|---|---|"]
        for n in [k for k in SHOW2 if k in p2["runs"]] + [k for k in p2["runs"] if k not in SHOW2]:
            r = p2["runs"][n]; lab = ROW2.get(n, "")
            L.append(f"| {'**' + lab + '**' if 'headline' in lab else lab} | {n} | {_f(r['longest_atten30_s'], 2)} | {_f(r['atten20_frac'])} | {_f(r['proxy_survival'])} | "
                     f"{_f(r['vad_speech_s'], 2)} ({_f(r['vad_speech_in_s'], 2)}) | {_f(r['word_survival'])} | "
                     f"{_f(r['validity_latency_s'], 2) if r['validity_latency_s'] is not None else 'never'} |")
        L += ["", "| criterion | verdict |", "|---|---|"] + [f"| {k} | {v} |" for k, v in res["part2_criteria"].items()]
    L.append("")
    Path(path).write_text("\n".join(L), encoding="utf-8")


def main(argv=None):
    global WEB_WAV
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--system", default="r7")
    ap.add_argument("--name", help="output stem (default: sanitised --system)")
    ap.add_argument("--n-utt", type=int, default=20)
    ap.add_argument("--snrs", type=float, nargs="+", default=[0.0, 5.0])
    ap.add_argument("--beds", nargs="+", default=["web", "mad"], choices=["web", "mad"])
    ap.add_argument("--speech-split", default="val", help="clean speech split (val: G4 feeds selection sweeps)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--parts", nargs="+", type=int, default=[1, 2], choices=[1, 2])
    ap.add_argument("--validity-key", help="eng.last key of the reference-informativeness flag (default: autodetect)")
    ap.add_argument("--web-wav", default=str(WEB_WAV), help="the stereo web WAV (Part 1 bed and Part 2 input)")
    ap.add_argument("--guards", action="store_true",
                    help="Part 2 stream runs: plan 11.3 runtime guards on (their ref_informative flag is the validity "
                         "estimator Part 2 times); Part 1 always runs guards off, the r7 default path")
    ap.add_argument("--asr", default="auto", choices=["auto", "off"],
                    help="Part 2 Whisper/VAD hooks: auto = on when faster-whisper imports (the asr extra), off = TBD")
    ap.add_argument("--whisper-model", default=None, help="faster-whisper model for word survival (default small)")
    ap.add_argument("--asr-device", default="auto", help="auto (cuda when ctranslate2 sees one), cpu or cuda")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--allow-pure-python", action="store_true", help="run without numba (slow; smoke tests only)")
    ap.add_argument("--summarise-only", action="store_true", help="rebuild json/md from the cached part-1 rows")
    a = ap.parse_args(argv)
    WEB_WAV = Path(a.web_wav); os.environ["VAANI_WEB_WAV"] = a.web_wav
    a.name = a.name or re.sub(r"[^\w.-]+", "_", a.system)
    if a.system == "r7" or a.system.startswith("stream:"):
        from vaani.dsp import nlms
        if not nlms._HAVE_NUMBA and not a.allow_pure_python:
            sys.exit("numba not importable: run under 'uv run --with numba' (pure-Python NLMS is ~26 ms/hop)")
    out = Path(a.out); work = out / "work"; work.mkdir(parents=True, exist_ok=True)
    res = {"name": a.name, "system": a.system, "n_utt": a.n_utt, "snrs": a.snrs, "beds": a.beds,
           "speech_split": a.speech_split, "seed": a.seed,
           "command": "python scripts/field_accept.py " + " ".join(argv if argv is not None else sys.argv[1:])}
    old = out / f"{a.name}.json"
    if a.summarise_only and old.exists():               # a rebuild keeps the scoring run's provenance
        prev = json.loads(old.read_text(encoding="utf-8"))
        res["summarised_by"] = res["command"]
        res.update({k: prev[k] for k in ("command", "mad_beds", "ref_zero_validity0") if k in prev})
    if "ref_zero_validity0" not in res:
        res["ref_zero_validity0"] = system(a.system, a.guards).validity0()
    if 1 in a.parts:
        if a.summarise_only:
            a.workers = 0
        rows, mad = run_part1(a, work) if not a.summarise_only else (
            [r for line in (work / f"{a.name}_part1.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
             for r in json.loads(line)[1]], [])
        if a.summarise_only:                            # task order (the scoring run's), not cache order: same sums
            rank = {b: n for n, b in enumerate(a.beds)}
            rows.sort(key=lambda r: (rank.get(r["bed"], len(rank)), r["item"], r["snr"]))
        res["part1"], res["part1_verdict"] = summarise_part1(rows)
        if mad:
            res["mad_beds"] = [m["clip"] for m in mad]
    p2f = work / f"{a.name}_part2.json"
    if 2 in a.parts:
        if a.summarise_only and p2f.exists():
            p2 = json.loads(p2f.read_text(encoding="utf-8"))
        else:
            p2 = run_part2(a)
            p2f.write_text(json.dumps(p2, indent=1), encoding="utf-8")
        res["part2"] = p2
        res["part2_criteria"], res["part2_verdict"] = summarise_part2(p2)
    res.setdefault("part1_verdict", "not run"); res.setdefault("part2_verdict", "not run"); res.setdefault("part1", {})
    res["guards"] = guard_state(res, a.guards)
    (out / f"{a.name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    write_md(res, out / f"{a.name}.md")
    print(f"G4 {a.name}: part 1 {res['part1_verdict']}, part 2 {res['part2_verdict']} -> {out / (a.name + '.md')}")
    return res


if __name__ == "__main__":
    main()
