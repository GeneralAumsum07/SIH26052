"""Versioned streaming contract and model-stepping backends (scalable-mini spec 6.1 and 6.4).

A stream is a `StreamState`: {schema_version, profile_id, sample_counter, channel_validity, discontinuity_flags,
caches, config_hash}. A `Backend` steps one 16 ms frame of the exported streaming model and owns nothing per
stream: every cache lives in the state it is handed, so two streams on one backend cannot contaminate each other
and a stream can be saved, moved and resumed. Audio transport and DSP live in `vaani.live`, not here.

Backends:
  OrtBackend    ONNX Runtime. CPUExecutionProvider is the correctness reference (what the board runs).
                `providers=` selects CUDA/TensorRT EPs; with `io_binding=True` caches stay device-resident in
                OrtValues and are double-buffered per stream. The IO-binding code path is tested on the CPU
                device; the CUDA/TensorRT EPs themselves are UNTESTED here (no onnxruntime-gpu in .venv).
  TorchBackend  the StreamVaaniNet / StreamCascade twin on cpu or cuda, caches as persistent device tensors.
                Dev machine only (needs torch); the laptop GPU comparison in scripts/hop_benchmark.py.
  FeOrtBackend  ONNX Runtime over a VaaniFE step graph (vaani.export.export_fe: spec, [valid], state -> spec_out,
                state_out). One flat state tensor (1, S) per stream, sized by the tier; the same IO-binding path.
  FeTorchBackend  VaaniFE.step (vaani/models/vaani_fe.py) on cpu or cuda, the flat state as one device tensor.
  `open_onnx` picks OrtBackend or FeOrtBackend from the graph's input names (numpy + onnxruntime only).

Models with a validity input (VaaniFE, inputs != "p") set `takes_valid`; `step(..., valid=v)` then feeds v, the
capture-path reference validity of this frame. Validity 0 is the trained reference-absent (mono) path.

Width/profile changes happen only by building a new backend and a new state: `check_state` refuses a state whose
cache shapes differ from the backend's, so hidden state is never resized or reused across widths.

numpy only at import time (torch is imported lazily by TorchBackend), so the board path stays torch-free.
"""
from __future__ import annotations

import collections
import hashlib
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
DEADLINE_MS = 16.0                 # one 256-sample hop at 16 kHz

# discontinuity_flags bits: sticky since the stream started or the caller last cleared them
DISC_GAP = 1                       # input samples were dropped (overload queue or capture xrun)
DISC_BYPASS = 2                    # hops skipped by the overload bypass: output was the raw primary
DISC_RESET = 4                     # the state was reset mid-stream
DISC_REF_DROPOUT = 8               # the reference channel was invalid for at least one hop


def config_hash(controller_on: bool, dsp: dict | None, extra: dict | None = None) -> str:
    """sha256 of the preprocessing contract a state was built under; a state never resumes under another one."""
    doc = {"controller_on": bool(controller_on), "dsp": dsp or {}, "sr": 16000, "n_fft": 512, "hop": 256,
           "window": "periodic sqrt-hann", **(extra or {})}
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ state
@dataclass
class StreamState:
    """One stream. `caches` values are backend-native (numpy, torch tensor or OrtValue); `to_host` for numpy."""
    profile_id: str
    config_hash: str
    caches: dict = field(default_factory=dict)
    sample_counter: int = 0
    channel_validity: tuple = (True, True)          # (primary, reference) of the most recent hop
    discontinuity_flags: int = 0
    schema_version: int = SCHEMA_VERSION

    def header(self, host_caches: dict) -> dict:
        return {"schema_version": self.schema_version, "profile_id": self.profile_id,
                "config_hash": self.config_hash, "sample_counter": int(self.sample_counter),
                "channel_validity": [bool(v) for v in self.channel_validity],
                "discontinuity_flags": int(self.discontinuity_flags),
                "caches": [{"name": k, "shape": list(v.shape), "dtype": str(v.dtype)} for k, v in host_caches.items()]}

    def to_bytes(self, host_caches: dict | None = None) -> bytes:
        """.npz bytes: a JSON header (uint8) that spells out every cache's shape and dtype, then the arrays."""
        hc = {k: np.asarray(v) for k, v in (host_caches if host_caches is not None else self.caches).items()}
        buf = io.BytesIO()
        hdr = np.frombuffer(json.dumps(self.header(hc)).encode(), np.uint8)
        np.savez(buf, __header__=hdr, **{f"c{i}": v for i, v in enumerate(hc.values())})
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, b: bytes) -> "StreamState":
        """Inverse of to_bytes, with numpy caches. Refuses another schema version or a header/array mismatch."""
        z = np.load(io.BytesIO(b), allow_pickle=False)
        hdr = json.loads(z["__header__"].tobytes().decode())
        if hdr.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"stream state schema {hdr.get('schema_version')!r}; this build reads {SCHEMA_VERSION}")
        caches = {}
        for i, c in enumerate(hdr["caches"]):
            a = z[f"c{i}"]
            if list(a.shape) != c["shape"] or str(a.dtype) != c["dtype"]:
                raise ValueError(f"cache {c['name']!r}: header says {c['shape']} {c['dtype']}, array is {a.shape} {a.dtype}")
            caches[c["name"]] = a
        return cls(profile_id=hdr["profile_id"], config_hash=hdr["config_hash"], caches=caches,
                   sample_counter=int(hdr["sample_counter"]), channel_validity=tuple(hdr["channel_validity"]),
                   discontinuity_flags=int(hdr["discontinuity_flags"]), schema_version=hdr["schema_version"])

    def save(self, path, host_caches: dict | None = None) -> None:
        Path(path).write_bytes(self.to_bytes(host_caches))

    @classmethod
    def load(cls, path) -> "StreamState":
        return cls.from_bytes(Path(path).read_bytes())


# ------------------------------------------------------------------------------------------------ telemetry
class Telemetry:
    """Bounded per-step timing: exact count/mean/max/deadline misses, whole-run quantiles from a fixed 0.01 ms
    histogram (resolution-limited, never memory-limited), and exact quantiles over the last `window` steps."""

    BIN_MS, MAX_MS = 0.01, 250.0

    def __init__(self, deadline_ms: float = DEADLINE_MS, window: int = 4096):
        self.deadline_ms = deadline_ms
        self.hist = np.zeros(int(self.MAX_MS / self.BIN_MS) + 1, np.int64)   # last bin collects >= MAX_MS
        self.recent = collections.deque(maxlen=window)
        self.count, self.total_ms, self.max_ms, self.misses = 0, 0.0, 0.0, 0
        self.events = collections.deque(maxlen=window)   # fallback events, last `window` (bounded)
        self.event_counts = collections.Counter()

    def event(self, kind: str, **info) -> None:
        """A fallback event (guard verdict change, overload bypass, ...): counted over the run, kept in a ring."""
        self.event_counts[kind] += 1
        self.events.append({"kind": kind, **info})

    def add(self, ms: float) -> None:
        self.hist[min(int(ms / self.BIN_MS), len(self.hist) - 1)] += 1
        self.recent.append(ms)
        self.count += 1; self.total_ms += ms; self.max_ms = max(self.max_ms, ms)
        self.misses += ms > self.deadline_ms

    def quantile(self, q: float) -> float:
        """Whole-run quantile, upper edge of the histogram bin (conservative by at most BIN_MS)."""
        if not self.count:
            return float("nan")
        k = int(np.searchsorted(np.cumsum(self.hist), q * self.count, side="left"))
        return min((k + 1) * self.BIN_MS, self.max_ms)

    def summary(self) -> dict:
        return {"steps": self.count, "mean_ms": self.total_ms / self.count if self.count else float("nan"),
                "p50_ms": self.quantile(.5), "p95_ms": self.quantile(.95), "p99_ms": self.quantile(.99),
                "max_ms": self.max_ms, "deadline_ms": self.deadline_ms, "deadline_misses": self.misses,
                "fallback_events": dict(self.event_counts)}


# ------------------------------------------------------------------------------------------------ backends
class Backend:
    """Protocol: cache_specs(), new_state(), reset(state), step(spec6, feats, state) -> (1,257,1,2) float32,
    to_host(state) / from_host(state), and `telemetry` over step() calls. Subclasses fill in _zeros/_step/_host."""

    name = "backend"
    tested = True
    kind = "cascade"
    takes_valid = False            # True: the graph has a per-frame reference validity input

    def __init__(self, profile_id: str, cache_specs: dict, deadline_ms: float = DEADLINE_MS):
        self.profile_id = profile_id
        self._specs = {k: (tuple(s), np.dtype(d)) for k, (s, d) in cache_specs.items()}
        self.telemetry = Telemetry(deadline_ms)

    def cache_specs(self) -> dict:
        return dict(self._specs)

    def new_state(self, config_hash: str = "") -> StreamState:
        return StreamState(self.profile_id, config_hash, {k: self._zeros(s, d) for k, (s, d) in self._specs.items()})

    def reset(self, state: StreamState) -> StreamState:
        """Zero every cache in place of the state's own buffers (same shapes: a reset never changes width)."""
        self.check_state(state, host=False)
        state.caches = {k: self._zeros(s, d) for k, (s, d) in self._specs.items()}
        state.sample_counter = 0; state.discontinuity_flags |= DISC_RESET
        return state

    def check_state(self, state: StreamState, host: bool = True) -> None:
        if state.profile_id != self.profile_id:
            raise ValueError(f"state is for profile {state.profile_id!r}, backend runs {self.profile_id!r}; "
                             "change profiles by building a new state at a reset boundary")
        if set(state.caches) != set(self._specs):
            raise ValueError(f"state caches {sorted(state.caches)} != backend caches {sorted(self._specs)}")
        if host:
            for k, (s, d) in self._specs.items():
                a = state.caches[k]
                if tuple(a.shape) != s or np.dtype(a.dtype) != d:
                    raise ValueError(f"cache {k!r} is {tuple(a.shape)} {a.dtype}; backend expects {s} {d}")

    def step(self, spec6: np.ndarray, feats: np.ndarray, state: StreamState, valid: float = 1.0) -> np.ndarray:
        t0 = time.perf_counter()
        out = self._step(spec6, feats, state, valid) if self.takes_valid else self._step(spec6, feats, state)
        self.telemetry.add((time.perf_counter() - t0) * 1000)
        return out

    def to_host(self, state: StreamState) -> StreamState:
        """A copy of the state with numpy caches (what `StreamState.save` writes)."""
        return StreamState(state.profile_id, state.config_hash, {k: self._host(v) for k, v in state.caches.items()},
                           state.sample_counter, tuple(state.channel_validity), state.discontinuity_flags)

    def from_host(self, state: StreamState) -> StreamState:
        """Adopt a host (numpy) state, e.g. from `StreamState.load`, after checking shapes and dtypes."""
        self.check_state(state, host=True)
        return StreamState(state.profile_id, state.config_hash, {k: self._device(np.asarray(v)) for k, v in state.caches.items()},
                           state.sample_counter, tuple(state.channel_validity), state.discontinuity_flags)

    # subclass hooks
    def _zeros(self, shape, dtype): return np.zeros(shape, dtype)
    def _host(self, v): return np.array(v, copy=True)
    def _device(self, a): return np.array(a, copy=True)
    def _step(self, spec6, feats, state): raise NotImplementedError


class OrtBackend(Backend):
    """ONNX Runtime backend over an exported streaming graph (inputs spec6, feats, then the caches)."""

    def __init__(self, onnx_path, threads: int = 1, providers=None, io_binding: bool | None = None,
                 device: str | None = None, profile_id: str | None = None, deadline_ms: float = DEADLINE_MS):
        self.ort, self.sess, providers = _ort_session(onnx_path, threads, providers)
        ins, outs = self.sess.get_inputs(), self.sess.get_outputs()
        if [i.name for i in ins[:2]] != ["spec6", "feats"]:
            raise ValueError(f"{onnx_path}: expected inputs spec6, feats first; got {[i.name for i in ins[:2]]}")
        if len(outs) != len(ins) - 1:
            raise ValueError(f"{onnx_path}: {len(ins) - 2} caches in but {len(outs) - 1} out")
        specs = {}
        for i in ins[2:]:
            if any(not isinstance(d, int) for d in i.shape):
                raise ValueError(f"cache input {i.name!r} has a dynamic shape {i.shape}; export is static batch-1")
            specs[i.name] = (tuple(i.shape), np.float32)
        self.cache_names = [i.name for i in ins[2:]]
        self.out_names = [o.name for o in outs]
        gpu = any((p if isinstance(p, str) else p[0]) != "CPUExecutionProvider" for p in providers)
        self.io_binding = gpu if io_binding is None else bool(io_binding)
        self.device = device or ("cuda" if gpu else "cpu")
        self.providers = self.sess.get_providers()
        self.name = "ort-" + ("cpu" if not gpu else providers[0] if isinstance(providers[0], str) else providers[0][0])
        self.tested = not gpu            # GPU EPs: code path exists, never run on this machine
        self.onnx_path = str(onnx_path)
        super().__init__(profile_id or Path(onnx_path).name, specs, deadline_ms)

    def _zeros(self, shape, dtype):
        if not self.io_binding:
            return np.zeros(shape, dtype)
        return self.ort.OrtValue.ortvalue_from_numpy(np.zeros(shape, dtype), self.device, 0)

    def _host(self, v):
        return v.numpy() if isinstance(v, self.ort.OrtValue) else np.array(v, copy=True)

    def _device(self, a):
        if not self.io_binding:
            return np.array(a, copy=True)
        return self.ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(a), self.device, 0)

    def _step(self, spec6, feats, state):
        if not self.io_binding:      # the reference path: identical call to vaani.export.stream_onnx
            out = self.sess.run(None, {"spec6": spec6, "feats": feats, **{n: state.caches[n] for n in self.cache_names}})
            state.caches = dict(zip(self.cache_names, out[1:]))
            return out[0]
        # device-resident caches: bind this stream's current buffers as inputs and its spare set as outputs,
        # then swap, so nothing but spec6/feats in and the spectrum out crosses the host boundary per hop
        spare = getattr(state, "_spare", None)
        if spare is None:
            spare = {n: self._zeros(*self._specs[n]) for n in self.cache_names}
        b = self.sess.io_binding()
        b.bind_cpu_input("spec6", np.ascontiguousarray(spec6, np.float32))
        b.bind_cpu_input("feats", np.ascontiguousarray(feats, np.float32))
        for n in self.cache_names:
            b.bind_ortvalue_input(n, state.caches[n])
        b.bind_output(self.out_names[0], "cpu")
        for n, o in zip(self.cache_names, self.out_names[1:]):
            b.bind_ortvalue_output(o, spare[n])
        self.sess.run_with_iobinding(b)
        y = b.copy_outputs_to_cpu()[0]
        state._spare, state.caches = state.caches, spare
        return y


def _ort_session(onnx_path, threads, providers):
    import onnxruntime as ort
    providers = list(providers or ["CPUExecutionProvider"])
    missing = [p if isinstance(p, str) else p[0] for p in providers]
    missing = [p for p in missing if p not in ort.get_available_providers()]
    if missing:
        raise RuntimeError(f"execution provider(s) {missing} not available; this onnxruntime has "
                           f"{ort.get_available_providers()}")
    opts = ort.SessionOptions(); opts.intra_op_num_threads = threads; opts.log_severity_level = 3
    return ort, ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers), providers


class TorchBackend(Backend):
    """The PyTorch streaming twin; caches are tensors that stay on `device` between hops."""

    def __init__(self, module, cache_names, init_caches, device: str = "cpu", profile_id: str = "torch",
                 deadline_ms: float = DEADLINE_MS):
        import torch
        self.torch = torch
        self.module = module.to(device).eval()
        self.cache_names = list(cache_names)
        self.device = device
        self._init = [c.detach().cpu() for c in init_caches]
        specs = {n: (tuple(c.shape), np.float32) for n, c in zip(self.cache_names, self._init)}
        self._spec_dev = torch.zeros(1, 257, 1, 6, device=device)       # persistent input buffers: no per-hop alloc
        self._feat_dev = torch.zeros(1, 1, 18, device=device)
        self.name = f"torch-{device}"
        super().__init__(profile_id, specs, deadline_ms)

    @classmethod
    def from_checkpoint(cls, ckpt_path, device: str = "cpu", profile_id: str | None = None):
        """The same twin `vaani.export.export` traces, loaded from a trained checkpoint (read-only use of export)."""
        from vaani import export
        v, mc = export._load_batch_model(ckpt_path)
        s, caches, in_names, _ = export._stream_twin(v, mc)
        return cls(s, in_names[2:], caches, device, profile_id or Path(ckpt_path).parent.name)

    @classmethod
    def from_config(cls, model_cfg: dict, refiner_cfg: dict | None = None, device: str = "cpu", seed: int = 20260924,
                    profile_id: str | None = None):
        """An UNTRAINED cascade profile (cost measurement only, never quality)."""
        import torch
        from vaani.models.cascade import StreamCascade, init_cascade_caches
        from vaani.export import IN_NAMES
        torch.manual_seed(seed)
        rc = refiner_cfg if refiner_cfg is not None else {"hidden": 16, "past": 2}
        s = StreamCascade(model_cfg, rc).eval()
        names = list(IN_NAMES[2:]) + (["noise_cache"] if model_cfg.get("noise_floor") else []) + ["refine_cache"]
        return cls(s, names, init_cascade_caches(first_model_cfg=model_cfg, refiner_cfg=rc), device,
                   profile_id or f"C{model_cfg.get('channels', 16)}-untrained")

    def _zeros(self, shape, dtype):
        return self.torch.zeros(shape, dtype=self.torch.float32, device=self.device)

    def _host(self, v):
        return v.detach().cpu().numpy().copy()

    def _device(self, a):
        return self.torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(self.device)

    def _step(self, spec6, feats, state):
        torch = self.torch
        with torch.inference_mode():
            self._spec_dev.copy_(torch.from_numpy(np.ascontiguousarray(spec6, np.float32)))
            self._feat_dev.copy_(torch.from_numpy(np.ascontiguousarray(feats, np.float32)))
            # the twin writes some caches in place; each state owns its tensors, so streams stay independent
            out = self.module(self._spec_dev, self._feat_dev, *[state.caches[n] for n in self.cache_names])
            state.caches = dict(zip(self.cache_names, out[1:]))
            return out[0].cpu().numpy()          # .cpu() synchronises a CUDA step, so telemetry covers the kernel time


# ------------------------------------------------------------------------------------------------ VaaniFE
FE_KIND = "vaani_fe"
FE_INPUTS = (["spec", "valid", "state"], ["spec", "state"])      # vaani.models.vaani_fe.StepGraph.io_names
FE_OUTPUTS = ["spec_out", "state_out"]


def fe_profile_id(profile: str | None, state_floats: int) -> str:
    """Profile id of a VaaniFE stream: the tier name when known ("vaani_fe-mini"), else the state size."""
    return f"{FE_KIND}-{profile}" if profile else f"{FE_KIND}-s{int(state_floats)}"


def _fe_frame(spec6: np.ndarray, n_raw: int) -> np.ndarray:
    """(1,257,1,6) engine frame -> (1,n_raw,257) channels-first step input (the first n_raw raw RI channels)."""
    return np.ascontiguousarray(spec6[0, :, 0, :n_raw].T[None], np.float32)


class FeOrtBackend(Backend):
    """ONNX Runtime over a VaaniFE step graph. The whole recurrent state is one flat (1, S) float32 tensor named
    "state" (K*F*C2 GRU hidden, plus the deep-filter frame cache when df_taps > 0), so a stream's state is sized
    by its tier and a state from another tier is refused by `check_state`. Output (1,257,1,2) like OrtBackend."""

    kind = FE_KIND

    def __init__(self, onnx_path, threads: int = 1, providers=None, io_binding: bool | None = None,
                 device: str | None = None, profile_id: str | None = None, profile: str | None = None,
                 deadline_ms: float = DEADLINE_MS):
        self.ort, self.sess, providers = _ort_session(onnx_path, threads, providers)
        ins, outs = self.sess.get_inputs(), self.sess.get_outputs()
        names = [i.name for i in ins]
        if names not in FE_INPUTS or [o.name for o in outs] != FE_OUTPUTS:
            raise ValueError(f"{onnx_path}: not a VaaniFE step graph (inputs {names}, outputs {[o.name for o in outs]})")
        shp = {i.name: i.shape for i in ins}
        for n, s in shp.items():
            if any(not isinstance(d, int) for d in s):
                raise ValueError(f"input {n!r} has a dynamic shape {s}; export is static batch-1")
        if shp["spec"][0] != 1 or shp["spec"][2] != 257 or shp["state"][0] != 1:
            raise ValueError(f"{onnx_path}: spec {shp['spec']} / state {shp['state']} are not batch-1 (1,n,257) / (1,S)")
        self.takes_valid = "valid" in names
        self.n_raw = int(shp["spec"][1])
        self.state_floats = int(shp["state"][1])
        gpu = any((p if isinstance(p, str) else p[0]) != "CPUExecutionProvider" for p in providers)
        self.io_binding = gpu if io_binding is None else bool(io_binding)
        self.device = device or ("cuda" if gpu else "cpu")
        self.providers = self.sess.get_providers()
        self.name = "ort-" + ("cpu" if not gpu else providers[0] if isinstance(providers[0], str) else providers[0][0])
        self.tested = not gpu
        self.onnx_path = str(onnx_path)
        self._valid = np.ones((1, 1), np.float32)                    # reused host buffer: no per-hop alloc
        super().__init__(profile_id or fe_profile_id(profile, self.state_floats),
                         {"state": ((1, self.state_floats), np.float32)}, deadline_ms)

    _zeros = OrtBackend._zeros
    _host = OrtBackend._host
    _device = OrtBackend._device

    def _step(self, spec6, feats, state, valid=1.0):
        x = _fe_frame(spec6, self.n_raw)
        self._valid[0, 0] = valid
        if not self.io_binding:
            feeds = {"spec": x, "state": state.caches["state"]}
            if self.takes_valid:
                feeds["valid"] = self._valid
            y, s = self.sess.run(FE_OUTPUTS, feeds)
            state.caches = {"state": s}
            return y.transpose(0, 2, 1)[:, :, None]                 # (1,2,257) -> (1,257,1,2)
        # device-resident flat state, double-buffered per stream as in OrtBackend
        spare = getattr(state, "_spare", None)
        if spare is None:
            spare = {"state": self._zeros(*self._specs["state"])}
        b = self.sess.io_binding()
        b.bind_cpu_input("spec", x)
        if self.takes_valid:
            b.bind_cpu_input("valid", self._valid)
        b.bind_ortvalue_input("state", state.caches["state"])
        b.bind_output("spec_out", "cpu")
        b.bind_ortvalue_output("state_out", spare["state"])
        self.sess.run_with_iobinding(b)
        y = b.copy_outputs_to_cpu()[0]
        state._spare, state.caches = state.caches, spare
        return y.transpose(0, 2, 1)[:, :, None]


class FeTorchBackend(Backend):
    """VaaniFE.step on cpu or cuda; the flat state is one tensor that stays on `device` between hops."""

    kind = FE_KIND

    def __init__(self, model, device: str = "cpu", profile_id: str | None = None, profile: str | None = None,
                 deadline_ms: float = DEADLINE_MS):
        import torch
        self.torch = torch
        self.module = model.to(device).eval()
        self.device = device
        self.takes_valid = bool(model.uses_ref)
        self.n_raw, self.state_floats = int(model.n_raw), int(model.state_size)
        self._spec_dev = torch.zeros(1, self.n_raw, 257, device=device)
        self._valid_dev = torch.ones(1, 1, device=device)
        self.name = f"torch-{device}"
        super().__init__(profile_id or fe_profile_id(profile or _tier_of(model), self.state_floats),
                         {"state": ((1, self.state_floats), np.float32)}, deadline_ms)

    @classmethod
    def from_arch(cls, arch, seed: int = 0, device: str = "cpu", profile_id: str | None = None):
        """An UNTRAINED tier (name or arch dict), seeded exactly as vaani.export.fe_untrained: cost only."""
        from vaani import export
        return cls(export.fe_untrained(arch, seed), device, profile_id)

    @classmethod
    def from_checkpoint(cls, ckpt_path, device: str = "cpu", profile_id: str | None = None):
        from vaani import export
        return cls(export.fe_load(ckpt_path), device, profile_id)

    _zeros = TorchBackend._zeros
    _host = TorchBackend._host
    _device = TorchBackend._device

    def _step(self, spec6, feats, state, valid=1.0):
        torch = self.torch
        with torch.inference_mode():
            self._spec_dev.copy_(torch.from_numpy(_fe_frame(spec6, self.n_raw)))
            self._valid_dev.fill_(float(valid))
            y, s = self.module.step(self._spec_dev, self._valid_dev if self.takes_valid else None, state.caches["state"])
            state.caches = {"state": s}
            return y.transpose(1, 2)[:, :, None].cpu().numpy()


def _tier_of(model) -> str | None:
    """Tier name when the model's sizes are exactly a named tier's, else None."""
    from vaani.models.vaani_fe import TIERS
    c = model.cfg
    for t, d in TIERS.items():
        if all(c[k] == v for k, v in d.items()):
            return t
    return None


def graph_kind(onnx_path) -> str:
    """FE_KIND for a VaaniFE step graph, "cascade" for the r7-style spec6/feats/caches graph (input names only)."""
    import onnxruntime as ort
    so = ort.SessionOptions(); so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    names = [i.name for i in ort.InferenceSession(str(onnx_path), sess_options=so,
                                                  providers=["CPUExecutionProvider"]).get_inputs()]
    return FE_KIND if names in FE_INPUTS else "cascade"


def open_onnx(onnx_path, kind: str | None = None, profile: str | None = None, **kw) -> Backend:
    """The backend for an exported graph: FeOrtBackend for a VaaniFE step graph, else OrtBackend (unchanged r7
    path). `kind` (model_config.json) is checked against the graph when given."""
    got = graph_kind(onnx_path)
    if kind is not None and kind != got:
        raise ValueError(f"{onnx_path}: model_config kind {kind!r} but the graph is a {got!r} graph")
    if got == FE_KIND:
        return FeOrtBackend(onnx_path, profile=profile, **kw)
    return OrtBackend(onnx_path, **kw)
