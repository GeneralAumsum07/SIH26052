"""Live, block-by-block runtime of the deployed system: DSP front end + exported ONNX graph + overlap-add.

This is `vaani.dsp.pipeline.run` + `vaani.export.stream_onnx` + `vaani.dsp.stft.istft` restated so that it can
run on an audio stream one 16 ms hop at a time, which the offline path cannot: it needs the whole clip up front
(the STFT's reflect padding, the full-length n_hat). The per-frame order is exactly the offline loop's and the
deployment contract's (deploy/CONTRACT.md "Per frame, in order"), and `tests/test_live.py` holds this engine to
the offline path sample for sample, so a live run is the same system the eval scores.

Numpy + onnxruntime only (no torch): it has to run on the board. numba is strongly recommended there - the pure
Python NLMS reference costs ~10 ms per hop on a desktop core, most of a 16 ms budget, before the model runs.

Framing. Frame k covers samples [kH - 256, kH + 256) (H = 256, center=True). When block k (samples [kH, kH+H))
arrives, frame k is complete, so each `process` call runs one frame and emits the overlap-add output for
samples [(k-1)H, kH): output lags input by exactly one hop, and the block buffering adds the other hop - the
contract's 32 ms algorithmic latency. The first call's output covers the left context (samples [-256, 0)).

The left context of frame 0 is zeros on a live stream. Offline, torch reflects the clip's own start into it;
`left_context` exists so the parity test can supply that and compare exactly, and has no other use.
"""
from __future__ import annotations

import json
import struct
import time
from pathlib import Path

import numpy as np

from vaani.dsp import stft
from vaani.dsp.blocking import BlockingMatrix
from vaani.dsp.controller import Controller
from vaani.dsp.features import N_FEATURES, FrameFeatures
from vaani.dsp.limiter import Limiter
from vaani.dsp.nlms import NLMS

HOP = stft.HOP
N_FFT = stft.N_FFT
SR = 16000
WINDOW = (np.hanning(stft.WIN + 1)[:-1] ** 0.5)   # periodic sqrt-Hann, as np_stft and torch.hann_window**0.5


def load_model_config(path) -> dict:
    """The DSP configuration the weights were trained behind: {"controller_on": bool, "dsp": {...}}.

    The ONNX graph does not encode it, and the board has no torch to read a checkpoint, so the export step writes
    it next to the graph (`write_model_config`). A plain JSON file keeps the board free of pickles."""
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if "controller_on" not in cfg:
        raise ValueError(f"{path}: missing 'controller_on'")
    return {"controller_on": bool(cfg["controller_on"]), "dsp": cfg.get("dsp") or {}}


def write_model_config(ckpt_path, out_path) -> dict:
    """Read the DSP configuration out of a checkpoint (needs torch; run on the dev machine, not the board)."""
    import hashlib
    import torch
    c = torch.load(ckpt_path, map_location="cpu", weights_only=True)["config"]
    sha = hashlib.sha256(Path(ckpt_path).read_bytes()).hexdigest()
    cfg = {"checkpoint": Path(ckpt_path).as_posix(), "checkpoint_sha256": sha, "model": c["model"],
           "controller_on": bool(c["controller_on"]),
           "dsp": c.get("dsp") or {}, "model_cfg": c.get("model_cfg")}
    Path(out_path).write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return cfg


class StreamEngine:
    """One stream's state: limiter, blocking matrix, NLMS, features, controller, ONNX caches, overlap-add buffer.

    Not thread-safe and not reusable across streams: build a new engine per stream (the contract zeroes every
    cache at stream start and never between frames of the same stream)."""

    def __init__(self, onnx_path, controller_on: bool = True, dsp: dict | None = None, threads: int = 1,
                 left_context: np.ndarray | None = None):
        import onnxruntime as ort
        dsp = dsp or {}
        self.controller_on = controller_on
        self.nlms, self.ff = NLMS(), FrameFeatures()
        self.ctl = Controller(**dsp.get("controller", {}))
        bk = dsp.get("blocking"); lk = dsp.get("limiter")
        self.blk = BlockingMatrix(**(bk if isinstance(bk, dict) else {})) if bk else None
        self.lim = Limiter(**(lk if isinstance(lk, dict) else {})) if lk else None
        self.gate, self.prev_hit, self.frames = 1.0, False, 0
        # the left half of the next frame: the previous hop of limited primary, limited reference and n_hat
        self.hist = (np.zeros((3, HOP), np.float32) if left_context is None
                     else np.asarray(left_context, np.float32).reshape(3, HOP).copy())
        self.ola = np.zeros(N_FFT, np.float64)

        opts = ort.SessionOptions(); opts.intra_op_num_threads = threads; opts.log_severity_level = 3
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
        ins = self.sess.get_inputs()
        if [i.name for i in ins[:2]] != ["spec6", "feats"]:
            raise ValueError(f"{onnx_path}: expected inputs spec6, feats first; got {[i.name for i in ins[:2]]}")
        self.cache_names = [i.name for i in ins[2:]]
        self.caches = [np.zeros(i.shape, np.float32) for i in ins[2:]]   # static batch-1 shapes (export contract)
        self.last = {}          # diagnostics of the most recent frame: gate, burst, reliability, limiter, ms
        self.ms = []            # per-frame processing time, ms (whole frame: DSP + model + synthesis)

    def process(self, prim: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """One 256-sample hop of both mics at 16 kHz in [-1, 1] -> 256 enhanced samples (one hop behind)."""
        t0 = time.perf_counter()
        prim = np.asarray(prim, np.float32); ref = np.asarray(ref, np.float32)
        if prim.shape != (HOP,) or ref.shape != (HOP,):
            raise ValueError(f"process() takes exactly {HOP} samples per channel, got {prim.shape}, {ref.shape}")

        hit = False
        if self.lim is not None:                    # step 0: limiter, before everything else sees the hop
            prim, ref = self.lim.process_block(prim, ref)
            hit = self.lim.engaged > 0; self.lim.engaged = 0
        r_in = ref
        if self.blk is not None:                    # step 1: blocking matrix, gated by the previous speech verdict
            r_in = self.blk.process_block(prim, ref, self.ctl.speech_adapt if self.controller_on else 0.0)
        n_hat, health = self.nlms.process_block(prim, r_in, self.gate if self.controller_on else 1.0)

        fp = np.concatenate([self.hist[0], prim])    # frame k = previous hop + this hop
        fr = np.concatenate([self.hist[1], ref])
        fn = np.concatenate([self.hist[2], n_hat])
        P = np.fft.rfft(fp * WINDOW).astype(np.complex64)
        R = np.fft.rfft(fr * WINDOW).astype(np.complex64)
        Nh = np.fft.rfft(fn * WINDOW).astype(np.complex64)

        f = self.ff.compute(fp, fr, P, R, health, self.gate)       # step 2: features (gate = previous frame's)
        burst, rel = False, 1.0
        if self.controller_on:                                     # step 3: controller -> next hop's gate
            # frame k spans hop blocks k-1 and k, so either block's limiter hit counts (as pipeline.run)
            self.gate, burst, rel = self.ctl.step(f, self.ff.diff_jump, bool(self.prev_hit or hit), self.ff.prim_margin)
            feats = f.astype(np.float32)
        else:
            feats = np.zeros(N_FEATURES, np.float32)

        spec6 = np.stack([P.real, P.imag, R.real, R.imag, Nh.real, Nh.imag], -1).astype(np.float32)[None, :, None, :]
        out = self.sess.run(None, {"spec6": spec6, "feats": feats[None, None, :],
                                   **dict(zip(self.cache_names, self.caches))})      # step 4 (+5, refiner inside)
        self.caches = out[1:]
        y = np.fft.irfft(out[0][0, :, 0, 0] + 1j * out[0][0, :, 0, 1], n=N_FFT) * WINDOW   # step 6: iSTFT + OLA
        self.ola += y
        res = self.ola[:HOP].astype(np.float32)
        self.ola[:HOP] = self.ola[HOP:]; self.ola[HOP:] = 0.0

        self.hist[0], self.hist[1], self.hist[2] = prim, ref, n_hat
        self.prev_hit = hit
        self.frames += 1
        dt = (time.perf_counter() - t0) * 1000
        self.ms.append(dt)
        self.last = {"gate": float(self.gate), "burst": bool(burst), "reliability": float(rel), "limiter": bool(hit), "ms": dt}
        return res


# ------------------------------------------------------------------------------------------------ 48 <-> 16 kHz
def lowpass_fir(taps: int = 193, cutoff_hz: float = 7300.0, fs: float = 48000.0, beta: float = 8.6) -> np.ndarray:
    """Kaiser-windowed sinc, unit DC gain. 193 taps at 48 kHz: passband to ~7 kHz, >80 dB down by 8 kHz, and a
    group delay of 96 samples (2 ms) per conversion. Designed in numpy so the board needs no scipy."""
    n = np.arange(taps) - (taps - 1) / 2
    h = np.sinc(2 * cutoff_hz / fs * n) * np.kaiser(taps, beta)
    return (h / h.sum()).astype(np.float64)


class Decimate3:
    """Streaming 48 kHz -> 16 kHz for (channels, n) blocks, n a multiple of 3. Block boundaries are invisible:
    feeding a signal in any block sizes gives the same output as one call (tests/test_live.py)."""

    def __init__(self, channels: int, h: np.ndarray | None = None):
        self.h = lowpass_fir() if h is None else h
        self.state = np.zeros((channels, len(self.h) - 1))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.shape[1] % 3:
            raise ValueError("Decimate3 needs blocks whose length is a multiple of 3")
        buf = np.concatenate([self.state, x], axis=1)
        self.state = buf[:, -(len(self.h) - 1):]
        y = np.stack([np.convolve(c, self.h, mode="valid") for c in buf])   # len(x) outputs, aligned to x
        return y[:, ::3].astype(np.float32)


class Interpolate3:
    """Streaming 16 kHz -> 48 kHz for (channels, n) blocks: zero-stuff by 3, same low-pass, gain 3."""

    def __init__(self, channels: int, h: np.ndarray | None = None):
        self.h = 3 * (lowpass_fir() if h is None else h)
        self.state = np.zeros((channels, len(self.h) - 1))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        up = np.zeros((x.shape[0], 3 * x.shape[1])); up[:, ::3] = x
        buf = np.concatenate([self.state, up], axis=1)
        self.state = buf[:, -(len(self.h) - 1):]
        return np.stack([np.convolve(c, self.h, mode="valid") for c in buf]).astype(np.float32)


# ------------------------------------------------------------------------------------------------ WAV files
def read_wav(path) -> tuple[np.ndarray, int]:
    """(channels, n) float32 in [-1, 1] and the sample rate. PCM 16/24/32-bit and IEEE float 32/64 (the eval sets
    are float32, which the stdlib `wave` module refuses)."""
    b = Path(path).read_bytes()
    if b[:4] != b"RIFF" or b[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(b):
        cid, size = b[pos:pos + 4], struct.unpack("<I", b[pos + 4:pos + 8])[0]
        body = b[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            tag, ch, sr, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            if tag == 0xFFFE:                                  # WAVE_FORMAT_EXTENSIBLE: real tag in the subformat GUID
                tag = struct.unpack("<H", body[24:26])[0]
            fmt = (tag, ch, sr, bits)
        elif cid == b"data":
            data = body
        pos += 8 + size + (size & 1)
    if fmt is None or data is None:
        raise ValueError(f"{path}: missing fmt or data chunk")
    tag, ch, sr, bits = fmt
    if tag == 3:
        x = np.frombuffer(data, {32: "<f4", 64: "<f8"}[bits]).astype(np.float32)
    elif tag == 1 and bits == 24:
        u = np.frombuffer(data, np.uint8).reshape(-1, 3).astype(np.int32)
        x = ((u[:, 0] | (u[:, 1] << 8) | (u[:, 2] << 16)) << 8 >> 8).astype(np.float32) / 2 ** 23
    elif tag == 1 and bits in (16, 32):
        x = np.frombuffer(data, {16: "<i2", 32: "<i4"}[bits]).astype(np.float32) / 2 ** (bits - 1)
    else:
        raise ValueError(f"{path}: unsupported WAV format tag {tag}, {bits} bits")
    return x.reshape(-1, ch).T.copy(), sr


def write_wav(path, x: np.ndarray, sr: int) -> None:
    """(channels, n) or (n,) float in [-1, 1] -> 16-bit PCM WAV (plays everywhere)."""
    x = np.atleast_2d(np.asarray(x, np.float32))
    pcm = (np.clip(x, -1, 1) * 32767).round().astype("<i2").T.tobytes()
    ch = x.shape[0]
    hdr = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE" + b"fmt " + struct.pack(
        "<IHHIIHH", 16, 1, ch, sr, sr * ch * 2, ch * 2, 16) + b"data" + struct.pack("<I", len(pcm))
    Path(path).write_bytes(hdr + pcm)
