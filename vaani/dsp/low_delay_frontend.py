"""One low-delay frontend (plan Section 3.3 / Task 2): the r8 front end one contract hop at a time.

LowDelayFrontend(contract, dsp_cfg).process(mix, available) takes (2, H) samples and an H-element availability mask
(H from the contract) and returns finite processed samples plus per-sample validity. Per hop, in order:
  1. non-finite reference samples count as unavailable; a non-finite primary sample is a marked discontinuity with a
     finite fallback (zeroed), and never reaches the recurrent model: the caller resets or bypasses on the flag;
  2. under ref_policy, unavailable reference samples are zeroed before the limiter (as pipeline.run);
  3. the r8 limiter, unchanged (32-sample sub-blocks, three per 96-sample hop): delay-free and chunk-invariant;
  4. under ref_policy, the reconnect ramp in samples (ramp_samples, 3072 = 192 ms), independent of the hop.
The processed samples equal C0's `front_end` output on the same input, for any hop that is a multiple of the
limiter sub-block; only the frame-validity reduction differs between contracts. Offline processing
(`process_offline`) runs the same transition logic hop by hop.

Frame validity is reduced per contract downstream (vaani.dsp.low_delay_stft.frame_validity): a frame is invalid if
any real sample of its 512-sample analysis support is unavailable. `StreamValidity` is its streaming twin.
"""
from __future__ import annotations

import numpy as np

from vaani.audio_contract import get_audio_contract
from vaani.dsp import pipeline


class LowDelayFrontend:
    def __init__(self, contract=None, dsp_cfg: dict | None = None):
        self.c = get_audio_contract(contract)
        self.dsp = dict(dsp_cfg or {})
        self.hop = self.c.hop
        self.pol = self.dsp.get("ref_policy")
        self.ramp = pipeline.ramp_samples_of(self.pol, self.c) if self.pol is not None else None
        lk = self.dsp.get("limiter")
        if lk:
            sub = lk.get("sub", 32) if isinstance(lk, dict) else 32
            if self.hop % sub:
                raise ValueError(f"hop {self.hop} is not a whole number of limiter sub-blocks ({sub})")
        self.reset()

    def reset(self):
        """Every piece of frontend state back to a fresh stream (limiter, ramp clock, counters)."""
        self.lim = pipeline.make_limiter(self.dsp)
        self.since = self.ramp if self.ramp is not None else 0   # samples since the reconnect (fully ramped)
        self.prev_avail = True
        self.sample = 0
        self.discontinuities = 0

    def process(self, mix: np.ndarray, available=True) -> dict:
        """(2, H) samples + availability (bool or (H,)) -> {"mix": (2, H) float32, "valid": (H,) bool,
        "discontinuity": bool, "limiter": bool, "sample": first sample index of the hop}."""
        mix = np.asarray(mix, np.float32)
        if mix.shape != (2, self.hop):
            raise ValueError(f"process() takes (2, {self.hop}) samples, got {mix.shape}")
        av = np.array(np.broadcast_to(np.asarray(available, bool), (self.hop,)))
        prim, ref = mix[0].copy(), mix[1].copy()
        bad_ref = ~np.isfinite(ref)
        if bad_ref.any():
            av &= ~bad_ref
            ref[bad_ref] = 0.0
        disc = not np.isfinite(prim).all()
        if disc:                                         # finite fallback; the caller must not feed it to the model
            prim = np.where(np.isfinite(prim), prim, np.float32(0.0)).astype(np.float32)
            self.discontinuities += 1
        if self.pol is not None:
            ref = np.where(av, ref, np.float32(0.0)).astype(np.float32)
        hit = False
        if self.lim is not None:
            prim, ref = self.lim.process_block(prim, ref)
            hit = self.lim.engaged > 0
            self.lim.engaged = 0
        if self.pol is not None:
            g, self.since, self.prev_avail = pipeline.ref_gain_step(av, self.since, self.prev_avail,
                                                                    ramp_samples=self.ramp)
            ref = ref * g
        out = {"mix": np.stack([prim, ref]).astype(np.float32), "valid": av, "discontinuity": disc,
               "limiter": bool(hit), "sample": self.sample}
        self.sample += self.hop
        return out

    def process_offline(self, mix: np.ndarray, available=None) -> tuple[np.ndarray, np.ndarray]:
        """(2, N) -> ((2, N) processed, (N,) availability) through the same per-hop transition logic (the last hop
        zero-padded, its padding dropped)."""
        mix = np.asarray(mix, np.float32)
        n = mix.shape[1]
        av = np.ones(n, bool) if available is None else np.asarray(available, bool)
        hops = -(-n // self.hop)
        pad = hops * self.hop - n
        mp = np.pad(mix, ((0, 0), (0, pad)))
        ap = np.pad(av, (0, pad), constant_values=True)
        outs = [self.process(mp[:, j * self.hop:(j + 1) * self.hop], ap[j * self.hop:(j + 1) * self.hop])
                for j in range(hops)]
        y = np.concatenate([o["mix"] for o in outs], 1)[:, :n]
        v = np.concatenate([o["valid"] for o in outs])[:n]
        return y, v

    # ---- state -------------------------------------------------------------------------------
    def export_state(self) -> dict:
        st = {"contract": self.c.audio_contract_id, "since": self.since, "prev_avail": self.prev_avail,
              "sample": self.sample, "discontinuities": self.discontinuities}
        if self.lim is not None:
            st["limiter"] = {k: v for k, v in vars(self.lim).items()}
        return st

    def import_state(self, st: dict):
        if st["contract"] != self.c.audio_contract_id:
            raise ValueError("frontend state belongs to a different audio contract")
        self.since, self.prev_avail = st["since"], st["prev_avail"]
        self.sample, self.discontinuities = st["sample"], st["discontinuities"]
        if self.lim is not None:
            for k, v in st["limiter"].items():
                setattr(self.lim, k, v)


class StreamValidity:
    """Streaming frame validity for a low-delay contract: frame j (computed when hop j arrives) is valid only if
    every sample of its K-sample analysis support [jH - (K - H), (j + 1)H) was available. Samples before the stream
    start are known zero padding (available). Equals low_delay_stft.frame_validity frame for frame."""

    def __init__(self, contract):
        self.c = get_audio_contract(contract)
        if self.c.is_legacy:
            raise ValueError("StreamValidity is for low-delay contracts (C0 uses pipeline.frame_avail)")
        self.reset()

    def reset(self):
        self.last_bad = -(10 ** 9)   # absolute index of the latest unavailable sample
        self.sample = 0

    def push(self, available) -> float:
        av = np.broadcast_to(np.asarray(available, bool), (self.c.hop,))
        bad = np.flatnonzero(~av)
        if bad.size:
            self.last_bad = self.sample + int(bad[-1])
        self.sample += self.c.hop
        return 1.0 if self.last_bad < self.sample - self.c.k else 0.0

    def export_state(self) -> dict:
        return {"contract": self.c.audio_contract_id, "last_bad": self.last_bad, "sample": self.sample}

    def import_state(self, st: dict):
        if st["contract"] != self.c.audio_contract_id:
            raise ValueError("validity state belongs to a different audio contract")
        self.last_bad, self.sample = st["last_bad"], st["sample"]
