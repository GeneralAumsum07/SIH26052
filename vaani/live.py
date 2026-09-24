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

Model stepping goes through `vaani.backend` (ORT CPU by default; the versioned StreamState holds the caches), and
audio transport is separate: `BoundedHopQueue` (drop-oldest overload policy), `ChannelMonitor` (reference
availability), `WavWriter` (bounded-memory recording). scripts/capture_loop.py wires them to ALSA.
"""
from __future__ import annotations

import collections
import json
import struct
import threading
import time
from pathlib import Path

import numpy as np

from vaani.dsp import stft
from vaani.dsp.blocking import BlockingMatrix
from vaani.dsp.controller import Controller
from vaani.dsp.features import FEATURE_NAMES, N_FEATURES, FrameFeatures
from vaani.dsp.limiter import Limiter
from vaani.dsp.nlms import NLMS

HOP = stft.HOP
N_FFT = stft.N_FFT
SR = 16000
WINDOW = (np.hanning(stft.WIN + 1)[:-1] ** 0.5)   # periodic sqrt-Hann, as np_stft and torch.hann_window**0.5
REF_RAMP_HOPS = 16                                   # reconnect ramp, 256 ms (dropout policy, StreamEngine docstring)
_COH = slice(FEATURE_NAMES.index("coherence_b0"), FEATURE_NAMES.index("coherence_b7") + 1)
_CLIP_REF = FEATURE_NAMES.index("clip_frac_reference")
_REF_DROPOUT = FEATURE_NAMES.index("ref_dropout")


def load_model_config(path) -> dict:
    """The DSP configuration the weights were trained behind: {"controller_on": bool, "dsp": {...}, "onnx_sha256"}.

    The ONNX graph does not encode it, and the board has no torch to read a checkpoint, so the export step writes
    it next to the graph (`write_model_config`). A plain JSON file keeps the board free of pickles."""
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if "controller_on" not in cfg:
        raise ValueError(f"{path}: missing 'controller_on'")
    return {"controller_on": bool(cfg["controller_on"]), "dsp": cfg.get("dsp") or {},
            "onnx_sha256": cfg.get("onnx_sha256")}


def write_model_config(ckpt_path, out_path, onnx_path=None) -> dict:
    """Read the DSP configuration out of a checkpoint (needs torch; run on the dev machine, not the board).
    With onnx_path, also binds the exported graph's sha256, which StreamEngine then checks on load."""
    import hashlib
    import torch
    c = torch.load(ckpt_path, map_location="cpu", weights_only=True)["config"]
    sha = hashlib.sha256(Path(ckpt_path).read_bytes()).hexdigest()
    cfg = {"checkpoint": Path(ckpt_path).as_posix(), "checkpoint_sha256": sha, "model": c["model"],
           "controller_on": bool(c["controller_on"]),
           "dsp": c.get("dsp") or {}, "model_cfg": c.get("model_cfg")}
    if onnx_path is not None:
        cfg["onnx_sha256"] = hashlib.sha256(Path(onnx_path).read_bytes()).hexdigest()
    Path(out_path).write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return cfg


class StreamEngine:
    """One stream's state: limiter, blocking matrix, NLMS, features, controller, model caches, overlap-add buffer.

    Not thread-safe. The model step goes through a `vaani.backend.Backend` (default: ORT CPU on `onnx_path`, the
    reference); the model caches live in `self.state` (a `vaani.backend.StreamState`), the DSP state on this object.
    `export_state` / `import_state` carry both, so a stream can be saved and resumed exactly.

    Reference dropout policy (`process(..., ref_valid=False)`; a runtime safety measure - the r7 weights were NOT
    trained with it, so enhancement quality during and after a dropout is unmeasured):
      invalid hop  the reference is zeroed before the limiter, so a dead or railed mic drives nothing; the blocking
                   matrix is stepped with adaptation off and its output (which would be -h*primary, i.e. speech)
                   discarded; the NLMS sees a zero reference with adaptation off, so n_hat = 0 and its weights keep
                   their pre-dropout values (plan 2.8: they blew up); feature slots coherence_b* and
                   clip_frac_reference are zeroed and ref_dropout set to 1; the reference-onset history and ratio
                   floor in FrameFeatures are held. The primary is never gated.
      reconnect    reference-derived signals (limited reference, blocking output) are scaled by a per-sample
                   linear ramp 0 -> 1 over `ref_ramp_hops` hops, the zeroed/forced feature slots blend back by the
                   same factor, and NLMS + blocking adaptation stay frozen until the ramp ends (the pre-dropout
                   weights are used meanwhile, then adaptation recalibrates them).
    With every hop valid none of this runs: the default path is the pre-contract engine, operation for operation.
    """

    def __init__(self, onnx_path, controller_on: bool = True, dsp: dict | None = None, threads: int = 1,
                 left_context: np.ndarray | None = None, *, backend=None, onnx_sha256: str | None = None,
                 allow_hash_mismatch: bool = False, ref_ramp_hops: int = REF_RAMP_HOPS, ms_window: int = 4096,
                 stage_timing: bool = False, profile_id: str | None = None):
        from vaani import backend as bk
        dsp = dsp or {}
        self.controller_on, self.dsp = controller_on, dsp
        self.ref_ramp_hops = max(1, int(ref_ramp_hops))
        self.stage_timing = stage_timing
        self.onnx_sha256 = None
        if onnx_path is not None:
            self.onnx_sha256 = verify_onnx(onnx_path, onnx_sha256, allow_hash_mismatch)
        self.backend = backend if backend is not None else bk.OrtBackend(onnx_path, threads=threads, profile_id=profile_id)
        self.config_hash = bk.config_hash(controller_on, dsp)
        self._left_context = None if left_context is None else np.asarray(left_context, np.float32).reshape(3, HOP).copy()
        self.state = self.backend.new_state(self.config_hash)
        self._init_dsp()
        self.last = {}          # diagnostics of the most recent frame: gate, burst, reliability, limiter, ms
        self.ms = collections.deque(maxlen=ms_window)   # per-frame time, ms, last `ms_window` hops (bounded: plan 2.10)
        self.telemetry = bk.Telemetry()                 # whole-hop timing over the whole run (DSP + model + synthesis)
        self.skipped = 0        # hops not processed (overload bypass or dropped input)
        self.ref_invalid_hops = 0

    # ---------------------------------------------------------------------------------------------- lifecycle
    def _init_dsp(self, keep=None):
        dsp = self.dsp
        self.nlms, self.ff = NLMS(), FrameFeatures()
        self.ctl = Controller(**dsp.get("controller", {}))
        bk = dsp.get("blocking"); lk = dsp.get("limiter")
        self.blk = BlockingMatrix(**(bk if isinstance(bk, dict) else {})) if bk else None
        self.lim = Limiter(**(lk if isinstance(lk, dict) else {})) if lk else None
        if keep is not None:                    # a soft reset keeps what the adaptive filters learned of the headset
            self.nlms.w[:] = keep[0]
            if self.blk is not None and keep[1] is not None:
                self.blk.f.w[:] = keep[1]
        self.gate, self.prev_hit, self.frames = 1.0, False, 0
        self._ramp_pos = self.ref_ramp_hops     # fully ramped: the reference is trusted
        # the left half of the next frame: the previous hop of limited primary, limited reference and n_hat
        self.hist = (np.zeros((3, HOP), np.float32) if self._left_context is None else self._left_context.copy())
        self.ola = np.zeros(N_FFT, np.float64)

    def reset(self, keep_adaptation: bool = False) -> None:
        """Start the stream over: every cache, the overlap-add tail and the frame history are zeroed, so no audio
        from before the reset can reach the output after it. keep_adaptation keeps the NLMS / blocking weights."""
        from vaani import backend as bk
        keep = (self.nlms.w.copy(), self.blk.f.w.copy() if self.blk is not None else None) if keep_adaptation else None
        self._left_context = None               # the parity-test left context belongs to the first stream start only
        self._init_dsp(keep)
        self.backend.reset(self.state)
        self.state.discontinuity_flags |= bk.DISC_RESET

    def skip(self, prim: np.ndarray | None = None, ref: np.ndarray | None = None) -> None:
        """A hop the engine does not process: overload bypass (the caller has the audio and plays it raw) or input
        that was dropped (no audio). Advances sample time, abandons the pending overlap-add half (stale by the time
        output resumes; resuming fades in over one hop instead) and keeps the adaptive and model state."""
        from vaani import backend as bk
        self.state.sample_counter += HOP
        self.state.discontinuity_flags |= bk.DISC_BYPASS if prim is not None else bk.DISC_GAP
        self.ola[:] = 0.0
        if prim is None:
            self.hist[:] = 0.0
        else:
            self.hist[0] = np.asarray(prim, np.float32)
            self.hist[1] = 0.0 if ref is None else np.asarray(ref, np.float32)
            self.hist[2] = 0.0
        self.prev_hit = False
        self.skipped += 1

    # ---------------------------------------------------------------------------------------------- state I/O
    _DSP_OBJS = ("nlms", "ff", "ctl", "lim", "blk_f")
    _ENGINE_ATTRS = ("gate", "prev_hit", "frames", "hist", "ola", "_ramp_pos")

    def _dsp_obj(self, name):
        if name == "blk_f":
            return self.blk.f if self.blk is not None else None
        return getattr(self, name)

    def export_state(self):
        """The whole stream as a host `StreamState`: model caches ("model/<name>"), DSP object state
        ("dsp/<object>/<attr>") and engine buffers ("engine/<attr>"). None-valued attributes are recorded as
        "<key>#none" (empty int8) so the header stays pure shapes and dtypes."""
        host = self.backend.to_host(self.state)
        caches = {f"model/{k}": v for k, v in host.caches.items()}
        for name in self._DSP_OBJS:
            obj = self._dsp_obj(name)
            if obj is None:
                continue
            for attr, v in vars(obj).items():
                key = f"dsp/{name}/{attr}"
                if v is None:
                    caches[key + "#none"] = np.zeros(0, np.int8)
                elif isinstance(v, (np.ndarray, np.generic)):       # first: np.float64 is also a Python float
                    caches[key] = np.array(v)
                elif isinstance(v, (bool, int, float, complex)):     # Python scalar: stays one on import (NEP 50 promotion)
                    caches[key + "#py"] = np.array(v)
        for attr in self._ENGINE_ATTRS:
            v = getattr(self, attr)
            py = isinstance(v, (bool, int, float)) and not isinstance(v, np.generic)
            caches[f"engine/{attr}" + ("#py" if py else "")] = np.array(v)
        host.caches = caches
        return host

    def import_state(self, state) -> None:
        """Resume a stream saved by `export_state` (same model profile and DSP configuration, or ValueError)."""
        from vaani import backend as bk
        if state.config_hash != self.config_hash:
            raise ValueError("stream state was built under another DSP configuration (config_hash differs)")
        model = {k[6:]: v for k, v in state.caches.items() if k.startswith("model/")}
        self.state = self.backend.from_host(bk.StreamState(state.profile_id, state.config_hash, model,
                                                           state.sample_counter, tuple(state.channel_validity),
                                                           state.discontinuity_flags))
        for key, v in state.caches.items():
            if key.startswith("dsp/"):
                _, name, attr = key.split("/", 2)
                obj = self._dsp_obj(name)
                if obj is None:
                    raise ValueError(f"state has {key} but this engine has no {name}")
                if attr.endswith("#none"):
                    setattr(obj, attr[:-5], None)
                else:
                    setattr(obj, *_restore(attr, v))
            elif key.startswith("engine/"):
                setattr(self, *_restore(key[7:], v))

    # ---------------------------------------------------------------------------------------------- one hop
    def _ref_ramp(self, ref_valid: bool):
        """Per-sample reference weight for this hop, or None when the reference is fully trusted (legacy path)."""
        if not ref_valid:
            self._ramp_pos = 0
            return np.zeros(HOP, np.float32)
        if self._ramp_pos >= self.ref_ramp_hops:
            return None
        a = ((self._ramp_pos + np.arange(HOP, dtype=np.float32) / HOP) / self.ref_ramp_hops).astype(np.float32)
        self._ramp_pos += 1
        return a

    def process(self, prim: np.ndarray, ref: np.ndarray, ref_valid: bool = True) -> np.ndarray:
        """One 256-sample hop of both mics at 16 kHz in [-1, 1] -> 256 enhanced samples (one hop behind).
        ref_valid=False applies the reference dropout policy (class docstring)."""
        t0 = time.perf_counter()
        marks = [] if self.stage_timing else None
        prim = np.asarray(prim, np.float32); ref = np.asarray(ref, np.float32)
        if prim.shape != (HOP,) or ref.shape != (HOP,):
            raise ValueError(f"process() takes exactly {HOP} samples per channel, got {prim.shape}, {ref.shape}")
        a = self._ref_ramp(bool(ref_valid))
        if not ref_valid:
            ref = np.zeros(HOP, np.float32)
            self.ref_invalid_hops += 1

        hit = False
        if self.lim is not None:                    # step 0: limiter, before everything else sees the hop
            prim, ref = self.lim.process_block(prim, ref)
            hit = self.lim.engaged > 0; self.lim.engaged = 0
        if marks is not None: marks.append(("limiter", time.perf_counter()))
        r_in = ref
        if self.blk is not None:                    # step 1: blocking matrix, gated by the previous speech verdict
            adapt = (self.ctl.speech_adapt if self.controller_on else 0.0) if a is None else 0.0
            r_in = self.blk.process_block(prim, ref, adapt)
        if a is not None:                           # dropout / reconnect: ramp every reference-derived signal
            ref = ref * a; r_in = r_in * a
        if marks is not None: marks.append(("blocking", time.perf_counter()))
        n_hat, health = self.nlms.process_block(prim, r_in, (self.gate if self.controller_on else 1.0) if a is None else 0.0)
        if marks is not None: marks.append(("nlms", time.perf_counter()))

        fp = np.concatenate([self.hist[0], prim])    # frame k = previous hop + this hop
        fr = np.concatenate([self.hist[1], ref])
        fn = np.concatenate([self.hist[2], n_hat])
        P = np.fft.rfft(fp * WINDOW).astype(np.complex64)
        R = np.fft.rfft(fr * WINDOW).astype(np.complex64)
        Nh = np.fft.rfft(fn * WINDOW).astype(np.complex64)
        if marks is not None: marks.append(("stft", time.perf_counter()))

        if a is not None and not ref_valid:         # hold the reference trackers across the dropout
            held = (None if self.ff.sub_hist_r is None else self.ff.sub_hist_r.copy(), self.ff.ratio_floor)
        f = self.ff.compute(fp, fr, P, R, health, self.gate)       # step 2: features (gate = previous frame's)
        if a is not None:
            if not ref_valid:
                self.ff.sub_hist_r, self.ff.ratio_floor = held
            w = float(a.mean())
            f[_COH] *= w; f[_CLIP_REF] *= w
            f[_REF_DROPOUT] = f[_REF_DROPOUT] * w + (1.0 - w)
        if marks is not None: marks.append(("features", time.perf_counter()))
        burst, rel = False, 1.0
        if self.controller_on:                                     # step 3: controller -> next hop's gate
            # frame k spans hop blocks k-1 and k, so either block's limiter hit counts (as pipeline.run)
            self.gate, burst, rel = self.ctl.step(f, self.ff.diff_jump, bool(self.prev_hit or hit), self.ff.prim_margin)
            feats = f.astype(np.float32)
        else:
            feats = np.zeros(N_FEATURES, np.float32)
        if marks is not None: marks.append(("controller", time.perf_counter()))

        spec6 = np.stack([P.real, P.imag, R.real, R.imag, Nh.real, Nh.imag], -1).astype(np.float32)[None, :, None, :]
        out0 = self.backend.step(spec6, feats[None, None, :], self.state)   # step 4 (+5, refiner inside)
        if marks is not None: marks.append(("model", time.perf_counter()))
        y = np.fft.irfft(out0[0, :, 0, 0] + 1j * out0[0, :, 0, 1], n=N_FFT) * WINDOW   # step 6: iSTFT + OLA
        self.ola += y
        res = self.ola[:HOP].astype(np.float32)
        self.ola[:HOP] = self.ola[HOP:]; self.ola[HOP:] = 0.0
        if marks is not None: marks.append(("istft", time.perf_counter()))

        self.hist[0], self.hist[1], self.hist[2] = prim, ref, n_hat
        self.prev_hit = hit
        self.frames += 1
        st = self.state
        st.sample_counter += HOP
        st.channel_validity = (True, bool(ref_valid))
        if not ref_valid:
            from vaani.backend import DISC_REF_DROPOUT
            st.discontinuity_flags |= DISC_REF_DROPOUT
        dt = (time.perf_counter() - t0) * 1000
        self.ms.append(dt); self.telemetry.add(dt)
        self.last = {"gate": float(self.gate), "burst": bool(burst), "reliability": float(rel), "limiter": bool(hit),
                     "ms": dt, "ref_valid": bool(ref_valid), "ref_weight": 1.0 if a is None else float(a[-1])}
        if marks is not None:
            prev, stages = t0, {}
            for name, t in marks:
                stages[name] = (t - prev) * 1000; prev = t
            self.last["stages"] = stages
        return res

    @classmethod
    def from_config(cls, onnx_path, config_path, **kw) -> "StreamEngine":
        """Engine built from a model_config.json: its controller/DSP settings and its onnx_sha256, verified."""
        cfg = load_model_config(config_path)
        kw.setdefault("onnx_sha256", cfg.get("onnx_sha256"))
        return cls(onnx_path, cfg["controller_on"], cfg["dsp"], **kw)


def _restore(attr: str, v: np.ndarray):
    """(name, value) with the exported Python/numpy scalar type or array kept, so a resumed stream is bit-exact."""
    if attr.endswith("#py"):
        return attr[:-3], v.item()
    return attr, (v.copy() if v.ndim else v[()])


def verify_onnx(onnx_path, expected: str | None = None, allow_mismatch: bool = False) -> str:
    """sha256 of the graph, checked against `expected` or else the onnx_sha256 of a model_config.json beside it.
    The DSP config is not inside the graph, so this is what binds the two (plan 2.13)."""
    from vaani.backend import file_sha256
    p = Path(onnx_path)
    source = "argument"
    if expected is None:
        side = p.with_name("model_config.json")
        if side.exists():
            expected = json.loads(side.read_text(encoding="utf-8")).get("onnx_sha256"); source = str(side)
    actual = file_sha256(p)
    if expected is not None and actual != expected:
        msg = (f"{p}: sha256 {actual} does not match onnx_sha256 {expected} from {source}. The graph and its DSP "
               "configuration are out of step; re-export and rewrite model_config.json, or pass "
               "allow_hash_mismatch=True (capture_loop: --allow-hash-mismatch) to run it anyway.")
        if not allow_mismatch:
            raise ValueError(msg)
        import warnings
        warnings.warn(msg)
    return actual


# ------------------------------------------------------------------------------------------------ transport
class HopFramer:
    """Arbitrary-size (channels, n) capture chunks -> exact 256-sample hops. Chunk boundaries are invisible to
    the engine: any chunking yields the same hops in the same order (tests/test_stream_contract.py)."""

    def __init__(self, channels: int = 2, hop: int = HOP):
        self.hop, self.buf = hop, np.zeros((channels, 0), np.float32)

    def push(self, x: np.ndarray) -> list:
        self.buf = np.concatenate([self.buf, np.asarray(x, np.float32).reshape(self.buf.shape[0], -1)], axis=1)
        k = self.buf.shape[1] // self.hop
        hops = [self.buf[:, i * self.hop:(i + 1) * self.hop] for i in range(k)]
        self.buf = self.buf[:, k * self.hop:].copy()
        return hops

    def pending(self) -> int:
        return self.buf.shape[1]


class BoundedHopQueue:
    """Capture -> engine hand-off with a fixed capacity (spec 6.4). Overload policy: when full, the OLDEST
    pending hop is dropped (never processed, never emitted), `dropped_samples` counts it, and the next hop
    carries the gap so the consumer can tell the engine (`StreamEngine.skip`). Latency stays bounded by
    capacity x 16 ms and old audio is never replayed. Thread-safe: one producer, one consumer."""

    def __init__(self, capacity: int = 8):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._q = collections.deque()
        self._cv = threading.Condition()
        self._carry = 0
        self.closed = False
        self.pushed = self.dropped_hops = self.dropped_samples = self.max_depth = 0

    def push(self, block) -> bool:
        """Enqueue one hop; returns False when it forced a drop."""
        with self._cv:
            ok = True
            if len(self._q) >= self.capacity:
                old, gap = self._q.popleft()
                self.dropped_hops += 1; self.dropped_samples += int(np.shape(old)[-1]); ok = False
                if self._q:
                    self._q[0][1] += gap + 1
                else:
                    self._carry += gap + 1
            self._q.append([block, self._carry]); self._carry = 0
            self.pushed += 1
            self.max_depth = max(self.max_depth, len(self._q))
            self._cv.notify()
            return ok

    def pop(self, timeout: float | None = None):
        """(block, hops dropped just before it), or None when closed and drained, or on timeout."""
        with self._cv:
            if not self._cv.wait_for(lambda: self._q or self.closed, timeout):
                return None
            if not self._q:
                return None
            b, gap = self._q.popleft()
            return b, gap

    def depth(self) -> int:
        with self._cv:
            return len(self._q)

    def close(self) -> None:
        with self._cv:
            self.closed = True; self._cv.notify_all()


class ChannelMonitor:
    """Capture-side availability of one channel, with hysteresis: invalid after `bad_hops` consecutive hops that
    are digitally silent (an unplugged I2S mic reads exact zeros) or railed (>= `rail_frac` of samples at full
    scale); valid again after `good_hops` consecutive good hops. A heuristic: not validated on hardware."""

    def __init__(self, bad_hops: int = 3, good_hops: int = 8, silent_peak: float = 2 ** -20, rail_frac: float = 0.25):
        self.bad_hops, self.good_hops, self.silent_peak, self.rail_frac = bad_hops, good_hops, silent_peak, rail_frac
        self.valid, self._run = True, 0

    def update(self, x: np.ndarray) -> bool:
        ax = np.abs(np.asarray(x, np.float32))
        bad = ax.max() < self.silent_peak or float((ax >= 0.999).mean()) >= self.rail_frac
        if bad == self.valid:                   # evidence against the current verdict
            self._run += 1
            if self._run >= (self.bad_hops if self.valid else self.good_hops):
                self.valid, self._run = not self.valid, 0
        else:
            self._run = 0
        return self.valid


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


class WavWriter:
    """16-bit PCM WAV streamed to disk hop by hop, so a long recording costs no memory (plan 2.10); the RIFF and
    data sizes are patched on close (a file cut off by a crash still holds every written sample)."""

    def __init__(self, path, sr: int, channels: int):
        self.path, self.sr, self.ch, self.n = Path(path), sr, channels, 0
        self.f = open(self.path, "wb")
        self.f.write(self._header(0))

    def _header(self, nbytes: int) -> bytes:
        return (b"RIFF" + struct.pack("<I", 36 + nbytes) + b"WAVE" + b"fmt " + struct.pack(
            "<IHHIIHH", 16, 1, self.ch, self.sr, self.sr * self.ch * 2, self.ch * 2, 16) + b"data" + struct.pack("<I", nbytes))

    def write(self, x: np.ndarray) -> None:
        """(channels, n) or (n,) float in [-1, 1]."""
        x = np.atleast_2d(np.asarray(x, np.float32))
        if x.shape[0] != self.ch:
            raise ValueError(f"{self.path}: {x.shape[0]} channels, writer has {self.ch}")
        self.f.write((np.clip(x, -1, 1) * 32767).round().astype("<i2").T.tobytes())
        self.n += x.shape[1]

    def close(self) -> None:
        if self.f.closed:
            return
        self.f.seek(0); self.f.write(self._header(self.n * self.ch * 2)); self.f.close()
