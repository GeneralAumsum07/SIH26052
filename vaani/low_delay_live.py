"""Low-delay streaming reference (low-delay plan Task 6): one stream of a low-delay VaaniFE, one contract hop at a time.

LowDelayStreamEngine(contract, backend).process(primary, reference, available) takes one H-sample hop per channel
(H from the contract; 3H samples at 48 kHz when the engine owns a resampler pair) and returns one output hop. It is
the parity reference for the ORT graph and the native runtime, not the timing vehicle. Per hop (Section 3.1 order):
  0. (resampler) the 48 kHz hop is decimated to 16 kHz; availability is reduced per 16 kHz sample
  1. guards (optional) judge the raw hop; a hop they distrust is an absent hop (as the legacy trained path)
  2. the shared frontend (vaani.dsp.low_delay_frontend.LowDelayFrontend): limiter, zeroing and reconnect ramp
  3. frame validity (StreamValidity: any unavailable real sample in the 512-sample support invalidates the frame)
  4. analysis: the H new samples join the retained K - H history; the last K samples are transformed
  5. one neural step through the backend (the recurrent state, deep-filter cache included, lives in its StreamState)
  6. synthesis: the final L samples of the windowed inverse FFT, overlap-added; H samples are released
  7. (resampler) the released hop is interpolated to 48 kHz
A non-finite primary sample never reaches the recurrent model: that hop bypasses the model (its synthesis input is
the primary spectrum), the neural state is reset and the hop is flagged DISC_GAP | DISC_RESET.

Alignment: the first release precedes stream sample 0 by `contract.release_lead` = L - H samples; `run()` feeds a
whole signal (end zero-padded to whole hops, then one flush hop of known zeros) and returns the output aligned with
the offline route (vaani.enhance_low_delay.enhance_low_delay). `push()` accepts any chunk sizes (hops are aggregated
internally, so the output does not depend on the chunking); `flush()` ends a stream.

State: `export_state()` returns a vaani.backend.StreamState whose caches hold the neural state ("model/state"), the
frontend, validity, analysis and synthesis state, the guards, the pending input samples, the sample counters and, in
runners that own one, the resampler state. `import_state()` refuses a state of another profile, contract or
configuration (config_hash), even when the shapes agree.
"""
from __future__ import annotations

import numpy as np

from vaani import audio_contract as ac
from vaani import backend as bk
from vaani.dsp.low_delay_frontend import LowDelayFrontend, StreamValidity
from vaani.dsp.low_delay_stft import StreamAnalyzer, StreamSynthesizer

_SKIP = {"telemetry", "c", "dsp", "pol", "h", "a", "s_tail", "kernel", "vad_cfg"}


def _flatten(prefix: str, obj, out: dict):
    """Object/dict state -> flat {key: ndarray}. Python scalars keep their type ("#py"), None is "#none",
    nested objects recurse. Configuration (strings, contracts, coefficient arrays in _SKIP) is not state."""
    items = obj.items() if isinstance(obj, dict) else vars(obj).items()
    for k, v in items:
        if k in _SKIP or callable(v) or isinstance(v, (str, ac.AudioContract)):
            continue
        key = f"{prefix}/{k}"
        if v is None:
            out[key + "#none"] = np.zeros(0, np.int8)
        elif isinstance(v, (np.ndarray, np.generic)):
            out[key] = np.array(v)
        elif isinstance(v, (bool, int, float)):
            out[key + "#py"] = np.array(v)
        elif isinstance(v, dict) or hasattr(v, "__dict__"):
            _flatten(key, v, out)
        else:
            raise TypeError(f"cannot serialize state {key} of type {type(v).__name__}")


def _assign(root, path: list, v):
    """Set root.<path> (attributes or dict keys) to v."""
    obj = root
    for p in path[:-1]:
        obj = obj[p] if isinstance(obj, dict) else getattr(obj, p)
    if isinstance(obj, dict):
        obj[path[-1]] = v
    else:
        setattr(obj, path[-1], v)


def _value(key: str, a: np.ndarray):
    if key.endswith("#none"):
        return key[:-5], None
    if key.endswith("#py"):
        return key[:-3], a.item()
    return key, np.array(a)


class LowDelayStreamEngine:
    SUBSTATE = ("frontend", "validity", "an_p", "an_r", "synth", "guards", "dec", "interp")

    def __init__(self, contract, backend, dsp: dict | None = None, guards: dict | bool | None = None,
                 resampler=None):
        self.c = ac.get_audio_contract(contract)
        if self.c.is_legacy:
            raise ValueError("LowDelayStreamEngine runs low-delay contracts; C0 uses vaani.live.StreamEngine")
        bc = getattr(backend, "audio_contract", None)
        if bc != self.c:
            raise ValueError(f"backend runs {getattr(bc, 'audio_contract_id', None)}, engine contract is "
                             f"{self.c.audio_contract_id}")
        if getattr(backend, "kind", None) != bk.FE_KIND or getattr(backend, "n_raw", 4) != 4:
            raise ValueError("the low-delay route runs VaaniFE inputs 'pr' (4 raw channels) only")
        self.backend, self.hop = backend, self.c.hop
        self.dsp = dict(dsp or {})
        self.guards_cfg = guards or None
        self.resampler = resampler
        self.rate = 3 if resampler is not None else 1
        rid = None if resampler is None else resampler.id
        self.config_hash = bk.config_hash(False, {"dsp": self.dsp, "guards": self.guards_cfg, "resampler": rid},
                                          ac.config_extra(self.c))
        self.frontend = LowDelayFrontend(self.c, self.dsp)
        self.validity = StreamValidity(self.c)
        self.an_p, self.an_r = StreamAnalyzer(self.c), StreamAnalyzer(self.c)
        self.synth = StreamSynthesizer(self.c)
        self.state = backend.new_state(self.config_hash)
        self._spec = np.zeros((1, 257, 1, 4), np.float32)              # reused per hop: no per-hop allocation
        self.reset()

    # ---- lifecycle ---------------------------------------------------------------------------------
    def _new_guards(self):
        from vaani.guards import Guards
        return Guards(self.guards_cfg, self.backend.telemetry, hop=self.hop, window=self.c.guard_window) \
            if self.guards_cfg else None

    def reset(self) -> None:
        """A fresh stream: every piece of state back to its start (deterministic)."""
        self.frontend.reset(); self.validity.reset()
        self.an_p.reset(); self.an_r.reset(); self.synth.reset()
        self.guards = self._new_guards()
        self.dec = self.interp = None
        if self.resampler is not None:
            self.dec, self.interp = self.resampler.decimator(2), self.resampler.interpolator(1)
        self.state = self.backend.new_state(self.config_hash)
        self._pend = np.zeros((2, 0), np.float32)
        self._pend_av = np.zeros(0, bool)
        self.hops = 0
        self.last = {}

    @property
    def in_hop(self) -> int:
        return self.hop * self.rate

    # ---- one hop -------------------------------------------------------------------------------------
    def process(self, primary, reference, available=True) -> np.ndarray:
        """One hop per channel (H samples; 3H at 48 kHz with a resampler) -> one output hop."""
        p = np.asarray(primary, np.float32).reshape(-1)
        r = np.asarray(reference, np.float32).reshape(-1)
        if p.shape != (self.in_hop,) or r.shape != (self.in_hop,):
            raise ValueError(f"process() takes one {self.in_hop}-sample hop per channel, got {p.shape} / {r.shape}")
        av = np.array(np.broadcast_to(np.asarray(available, bool), (self.in_hop,)))
        x = np.stack([p, r])
        if self.dec is not None:
            av = av.reshape(-1, 3).all(1)
            x = self.dec(np.where(np.isfinite(x), x, np.float32(0.0)))  # the FIR must not smear a NaN over its span
            if not np.isfinite(p).all():
                x[0, 0] = np.nan                                         # keep the discontinuity visible to step 2
        return self._hop(x[0], x[1], av)

    def _hop(self, p, r, av) -> np.ndarray:
        if self.guards is not None:                                      # 1. distrust = an absent hop
            self.guards.pre(np.nan_to_num(p), np.nan_to_num(r))
            if not self.guards.ref_ok:
                av = np.zeros(self.hop, bool)
        fr = self.frontend.process(np.stack([p, r]), av)                 # 2.
        v = self.validity.push(fr["valid"])                              # 3.
        P = self.an_p.push(fr["mix"][0])                                 # 4.
        R = self.an_r.push(fr["mix"][1])
        if fr["discontinuity"]:                                          # never into the recurrent model
            Y = P
            self.backend.reset(self.state)
            self.state.discontinuity_flags |= bk.DISC_GAP | bk.DISC_RESET
        else:
            s = self._spec
            s[0, :, 0, 0], s[0, :, 0, 1], s[0, :, 0, 2], s[0, :, 0, 3] = P.real, P.imag, R.real, R.imag
            out = self.backend.step(s, None, self.state, v)              # 5.
            Y = out[0, :, 0, 0] + 1j * out[0, :, 0, 1]
        if self.guards is not None:
            self.guards.post(P, Y)
        y = self.synth.push(Y)                                           # 6.
        self.state.sample_counter += self.hop
        self.state.channel_validity = (not fr["discontinuity"], bool(fr["valid"].all()))
        if not fr["valid"].all():
            self.state.discontinuity_flags |= bk.DISC_REF_DROPOUT
        self.hops += 1
        self.last = {"validity": v, "limiter": fr["limiter"], "discontinuity": fr["discontinuity"]}
        if self.interp is not None:                                      # 7.
            return self.interp(y[None])[0]
        return y

    # ---- arbitrary chunks, whole signals ---------------------------------------------------------------
    def push(self, primary, reference, available=True) -> np.ndarray:
        """Any number of samples per channel -> the output of every hop completed (possibly empty)."""
        p = np.asarray(primary, np.float32).reshape(-1)
        r = np.asarray(reference, np.float32).reshape(-1)
        av = np.array(np.broadcast_to(np.asarray(available, bool), p.shape))
        self._pend = np.concatenate([self._pend, np.stack([p, r])], 1)
        self._pend_av = np.concatenate([self._pend_av, av])
        n = self.in_hop
        k = self._pend.shape[1] // n
        outs = [self.process(self._pend[0, j * n:(j + 1) * n], self._pend[1, j * n:(j + 1) * n],
                             self._pend_av[j * n:(j + 1) * n]) for j in range(k)]
        self._pend, self._pend_av = self._pend[:, k * n:].copy(), self._pend_av[k * n:].copy()
        return np.concatenate(outs) if outs else np.zeros(0, np.float32)

    def flush(self) -> np.ndarray:
        """End of stream: the pending partial hop zero-padded (known padding, available), then one hop of known
        zeros that flushes the synthesis overlap. The zero flush happens only here, never mid-stream."""
        out = []
        if self._pend.shape[1]:
            pad = self.in_hop - self._pend.shape[1]
            out.append(self.push(np.zeros(pad, np.float32), np.zeros(pad, np.float32), True))
        # the flush hop is synthetic padding: it bypasses the frontend, as the offline right padding does
        z = np.zeros(self.hop, np.float32)
        v = self.validity.push(np.ones(self.hop, bool))
        P, R = self.an_p.push(z), self.an_r.push(z)
        s = self._spec
        s[0, :, 0, 0], s[0, :, 0, 1], s[0, :, 0, 2], s[0, :, 0, 3] = P.real, P.imag, R.real, R.imag
        o = self.backend.step(s, None, self.state, v)
        y = self.synth.push(o[0, :, 0, 0] + 1j * o[0, :, 0, 1])
        out.append(self.interp(y[None])[0] if self.interp is not None else y)
        return np.concatenate(out)

    def run(self, primary, reference, available=None) -> np.ndarray:
        """A whole 16 kHz signal from a fresh stream -> output aligned with the offline route (same length)."""
        if self.resampler is not None:
            raise ValueError("run() is the 16 kHz alignment helper; with a resampler use push()/flush()")
        n = len(primary)
        self.reset()
        y = np.concatenate([self.push(primary, reference, True if available is None else available), self.flush()])
        lead = self.c.release_lead
        return y[lead:lead + n]

    # ---- state -----------------------------------------------------------------------------------------
    def export_state(self) -> bk.StreamState:
        host = self.backend.to_host(self.state)
        caches = {f"model/{k}": v for k, v in host.caches.items()}
        for name in self.SUBSTATE:
            obj = getattr(self, name)
            if obj is None:
                continue
            if name in ("dec", "interp"):
                caches[f"{name}/state"] = np.array(obj.state)
            else:
                _flatten(name, obj, caches)
        caches["engine/pend"] = self._pend.copy()
        caches["engine/pend_av"] = self._pend_av.copy()
        caches["engine/hops#py"] = np.array(self.hops)
        host.caches = caches
        return host

    def import_state(self, state: bk.StreamState) -> None:
        """Resume a stream saved by export_state; refuses another profile, contract or configuration."""
        if state.config_hash != self.config_hash:
            raise ValueError("stream state was built under another contract or configuration (config_hash differs)")
        model = {k[6:]: v for k, v in state.caches.items() if k.startswith("model/")}
        base = lambda keys: {k.split("#")[0] for k in keys}                # "#none"/"#py" mark a value's type only
        mine, theirs = base(self.export_state().caches), base(state.caches)
        if theirs != mine:
            raise ValueError(f"state caches differ from this engine's: {sorted(theirs ^ mine)[:6]}")
        self.state = self.backend.from_host(
            bk.StreamState(state.profile_id, state.config_hash, model, state.sample_counter,
                           tuple(state.channel_validity), state.discontinuity_flags), config_hash=self.config_hash)
        for key, a in state.caches.items():
            if key.startswith("model/"):
                continue
            key, v = _value(key, a)
            parts = key.split("/")
            if parts[0] == "engine":
                setattr(self, {"pend": "_pend", "pend_av": "_pend_av"}.get(parts[1], parts[1]), v)
            else:
                cur = getattr(self, parts[0])
                if isinstance(v, np.ndarray) and len(parts) == 2 and isinstance(getattr(cur, parts[1], None), np.ndarray):
                    v = v.astype(getattr(cur, parts[1]).dtype)
                _assign(self, parts, v)

    @classmethod
    def from_config(cls, onnx_path, config_path, **kw) -> "LowDelayStreamEngine":
        """Engine from a model_config.json: its contract (checked against the graph's stamp) and DSP settings."""
        from vaani import live
        cfg = live.load_model_config(config_path)
        c = ac.get_audio_contract(cfg["audio_contract"])
        if c.is_legacy:
            raise ValueError(f"{config_path}: a C0 model; use vaani.live.StreamEngine")
        live.verify_onnx(onnx_path, cfg.get("onnx_sha256"))
        b = bk.FeOrtBackend(onnx_path, profile=cfg.get("profile"), audio_contract=c.audio_contract_id,
                            threads=kw.pop("threads", 1))
        return cls(c, b, cfg["dsp"], **kw)
