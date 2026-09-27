"""Decoupled-cadence NLMS (low-delay plan Section 3.3, owner decision D5): n_hat for the low-delay contracts.

The legacy pipeline (vaani.dsp.pipeline.run, vaani.live.StreamEngine) runs NLMS, features and controller in lock step
on 256-sample blocks. A low-delay contract consumes 96- or 128-sample hops, so here the two cadences are decoupled:

  NLMS         runs sample by sample (the unchanged kernels of vaani.dsp.nlms and vaani.dsp.blocking), called on
               32-sample chunks. Every per-chunk decision (reference absent, the robust kernel's dropout sub-block)
               is taken on that 32-sample grid, which divides every contract hop (96, 128, 256), so n_hat is
               chunk-invariant: the same samples come out whatever the hop, like the limiter. The only look-ahead is
               inside the current chunk, which lies inside the current hop, so no algorithmic delay is added.
  gate         the unchanged legacy FrameFeatures and Controller on their own 256-sample cadence. When a 256-sample
               block completes, legacy frame k (samples [(k-1)256, (k+1)256), the live engine's framing: zeros before
               the stream start) is analysed, and the controller's gate and speech verdict take effect from the
               next sample on. Only completed past frames are used; the gate lags by at most one legacy frame and
               gates adaptation only. No controller retuning.

Differences from the legacy 256-sample pipeline, all forced by the finer grid: the ref_policy "absent" decision (and
the "reset" onset) is taken per 32-sample chunk instead of per 256-sample block, and the robust kernel's dropout
sub-block defaults to 32 samples (legacy 64; an explicit `sub` must divide 32). Legacy frame 0 sees zeros before the
stream start (as vaani.live.StreamEngine) instead of the offline reflect padding.

Order per chunk, as pipeline.run's per block: absent flag -> (reset onset) -> blocking matrix on the true limited
reference (its output zeroed where the reference is unavailable) -> NLMS with gate (0 while absent) -> n_hat, scaled
by the reconnect ramp under ref_policy. Features see the model-facing (limited, ramped) primary and reference.
"""
from __future__ import annotations

import numpy as np

from vaani.dsp.blocking import BlockingMatrix
from vaani.dsp.controller import Controller
from vaani.dsp.features import FrameFeatures
from vaani.dsp.nlms import NLMS

CHUNK = 32          # decision grid: divides every contract hop (96, 128, 256) and equals the limiter sub-block
BLOCK = 256         # the legacy feature / controller cadence
N_FFT = 512
WINDOW = (np.hanning(N_FFT + 1)[:-1] ** 0.5).astype(np.float64)   # periodic sqrt-Hann, as stft.np_stft


def robust_cfg(pol: dict | None):
    """The NLMS `robust` argument under a ref_policy, with the dropout sub-block on the 32-sample grid."""
    if pol is None or not pol.get("nlms"):
        return None
    rb = dict(pol["nlms"]) if isinstance(pol["nlms"], dict) else {}
    rb.setdefault("sub", CHUNK)
    if CHUNK % int(rb["sub"]):
        raise ValueError(f"the decoupled NLMS needs a robust sub-block that divides {CHUNK}, got {rb['sub']}")
    return rb


class DecoupledNLMS:
    def __init__(self, dsp_cfg: dict | None = None, controller_on: bool = True):
        dsp = dict(dsp_cfg or {})
        self.pol = dsp.get("ref_policy")
        if self.pol is not None and self.pol.get("absent", "freeze") not in ("freeze", "reset"):
            raise ValueError("ref_policy.absent must be 'freeze' or 'reset'")
        self.controller_on = bool(controller_on)
        self.robust = robust_cfg(self.pol) if self.pol is not None else None
        self.ctl_kw = dict(dsp.get("controller") or {})
        bk = dsp.get("blocking")
        self.blk_kw = (bk if isinstance(bk, dict) else {}) if bk else None
        self.reset()

    def reset(self):
        self.nlms = NLMS(robust=self.robust)
        self.ff, self.ctl = FrameFeatures(), Controller(**self.ctl_kw)
        self.blk = BlockingMatrix(**self.blk_kw) if self.blk_kw is not None else None
        self.gate = 1.0
        self.was_absent = False
        self.win = np.zeros((2, N_FFT), np.float32)      # model-facing primary / reference of the current legacy frame
        self.blk_p = np.zeros(BLOCK, np.float32)         # limited primary of the current block (NLMS health)
        self.blk_n = np.zeros(BLOCK, np.float32)         # raw n_hat of the current block
        self.pos = 0                                     # samples of the current block seen so far
        self.hit_cur, self.hit_prev = False, False       # limiter engagement in the current / previous block
        self.frames = 0                                  # legacy frames analysed
        # the most recent frame's diagnostics; fixed keys, so a stream's state layout does not change after frame 0
        self.last = {"gate": 1.0, "burst": False, "reliability": 1.0, "health": 0.0}

    def push(self, prim, ref_true, ref_model, avail=None, hits=None, gain=None) -> np.ndarray:
        """Limited primary, limited reference before the ramp (what the adaptive filters see), model-facing reference
        (after the ramp; what the features see), per-sample availability, limiter engagement per 32-sample chunk,
        and the reconnect-ramp gain (None = 1) -> n_hat for these samples (ramped under ref_policy). Any length;
        chunk boundaries fall on the absolute 32-sample grid only if every call but the last is a multiple of 32."""
        prim = np.asarray(prim, np.float32)
        n = len(prim)
        ref_true, ref_model = np.asarray(ref_true, np.float32), np.asarray(ref_model, np.float32)
        avail = np.ones(n, bool) if avail is None else np.asarray(avail, bool)
        n_ch = -(-n // CHUNK)
        hits = np.zeros(n_ch, bool) if hits is None else np.asarray(hits, bool)
        out = np.empty(n, np.float32)
        # consecutive chunks inside one 256-sample block with the same absent flag run as one kernel call: the gate
        # and the speech verdict change only at block ends, the kernels run sample by sample and the robust
        # kernel's sub-blocks stay on the 32-sample grid, so the samples are those of one call per chunk
        c, pos = 0, self.pos
        while c < n_ch:
            absent = self.pol is not None and not avail[c * CHUNK:(c + 1) * CHUNK].all()
            e, pos = c + 1, pos + CHUNK
            while e < n_ch and pos < BLOCK and (self.pol is not None and not avail[e * CHUNK:(e + 1) * CHUNK].all()) == absent:
                e, pos = e + 1, pos + CHUNK
            a, b = c * CHUNK, min(n, e * CHUNK)
            out[a:b] = self._chunk(prim[a:b], ref_true[a:b], ref_model[a:b], avail[a:b], bool(hits[c:e].any()), absent)
            pos = pos % BLOCK
            c = e
        if self.pol is not None and gain is not None:
            out = (out * np.asarray(gain, np.float32)).astype(np.float32)
        return out

    def _chunk(self, p, r, rm, av, hit, absent):
        """Whole 32-sample chunks of one block (the last may be partial) sharing the absent flag."""
        if absent and not self.was_absent and self.pol.get("absent", "freeze") == "reset":
            self.nlms.reset()
        self.was_absent = absent
        r_in = r
        if self.blk is not None:
            r_in = self.blk.process_block(p, r, 0.0 if absent else (self.ctl.speech_adapt if self.controller_on else 0.0))
            if self.pol is not None:
                r_in = r_in * av
        nh, _ = self.nlms.process_block(p, r_in, 0.0 if absent else (self.gate if self.controller_on else 1.0))
        m = len(p)
        self.blk_p[self.pos:self.pos + m], self.blk_n[self.pos:self.pos + m] = p, nh
        self.win[0, BLOCK + self.pos:BLOCK + self.pos + m], self.win[1, BLOCK + self.pos:BLOCK + self.pos + m] = p, rm
        self.hit_cur |= hit
        self.pos += m
        if self.pos == BLOCK:
            self._frame()
        return nh

    def _frame(self):
        """A 256-sample block completed: legacy frame k's features and the controller's next gate."""
        resid = self.blk_p - self.blk_n
        health = min(float((resid ** 2).mean() / ((self.blk_p ** 2).mean() + 1e-12)), 1.0)   # as NLMS.process_block
        if self.controller_on:
            fp, fr = self.win[0], self.win[1]
            P = np.fft.rfft(fp * WINDOW).astype(np.complex64)
            R = np.fft.rfft(fr * WINDOW).astype(np.complex64)
            f = self.ff.compute(fp, fr, P, R, health, self.gate)
            self.gate, burst, rel = self.ctl.step(f, self.ff.diff_jump, bool(self.hit_prev or self.hit_cur),
                                                  self.ff.prim_margin)
            self.last = {"gate": self.gate, "burst": burst, "reliability": rel, "health": health}
        self.win[:, :BLOCK] = self.win[:, BLOCK:]
        self.hit_prev, self.hit_cur = self.hit_cur, False
        self.pos = 0
        self.frames += 1
