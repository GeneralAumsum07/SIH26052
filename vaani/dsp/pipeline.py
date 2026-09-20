"""Runs NLMS + features + controller frame-synchronously over a stereo clip.
Used offline (dataset/eval) and as the reference for the embedded port.
Frame k covers samples [k*HOP - 256, k*HOP + 256) after torch-style reflect
padding, so the model's frame k and this feature vector k line up exactly.

Single causal pass: NLMS block k runs with gate[k-1] (1.0 for k=0), its
health feeds features for frame k, the controller then produces gate[k] for
block k+1. This mirrors exactly what the embedded/online port does -- there
is no non-causal "replay with final gates" step here.
"""
import numpy as np

from vaani.dsp import stft
from vaani.dsp.controller import Controller
from vaani.dsp.features import FrameFeatures, N_FEATURES
from vaani.dsp.limiter import Limiter
from vaani.dsp.nlms import NLMS


def run(mix: np.ndarray, controller_on: bool = True, dsp_cfg: dict | None = None) -> dict:
    """dsp_cfg (plan 2.7, ablatable): {"limiter": bool, "controller": {Controller kwargs}}. None = r1/r2 behaviour.
    Returns "mix" too: the limited signal when the limiter is on (what the model must see), else the input."""
    dsp_cfg = dsp_cfg or {}
    prim, ref = mix[0].astype(np.float32), mix[1].astype(np.float32)
    T = len(prim)
    # local instances only (no module-level mutable state) -> safe in DataLoader workers
    nlms, ff, ctl = NLMS(), FrameFeatures(), Controller(**dsp_cfg.get("controller", {}))
    n_blocks = (T + stft.HOP - 1) // stft.HOP  # tail: last block may be shorter than HOP
    lim_hit = np.zeros(n_blocks + 1, bool)     # per hop block: did the limiter engage (feeds the burst flag)
    if dsp_cfg.get("limiter"):
        # hop-by-hop like the port; the STFT below then runs on the limited signal
        lk = dsp_cfg["limiter"]; lim = Limiter(**(lk if isinstance(lk, dict) else {}))   # True or Limiter kwargs
        lp, lr = np.empty_like(prim), np.empty_like(ref)
        for j, i in enumerate(range(0, T, stft.HOP)):
            lp[i:i + stft.HOP], lr[i:i + stft.HOP] = lim.process_block(prim[i:i + stft.HOP], ref[i:i + stft.HOP])
            lim_hit[j] = lim.engaged > 0; lim.engaged = 0
        prim, ref = lp, lr

    P = stft.np_stft(prim); R = stft.np_stft(ref)
    n_frames = P.shape[1]
    pad = np.pad(prim, stft.N_FFT // 2, mode="reflect")
    padr = np.pad(ref, stft.N_FFT // 2, mode="reflect")

    n_hat = np.zeros(T, np.float32)
    feats = np.zeros((n_frames, N_FEATURES), np.float32)
    gates = np.ones(n_frames, np.float32)
    bursts = np.zeros(n_frames, bool)
    rel = np.ones(n_frames, np.float32)

    gate = 1.0
    health = 0.0  # only used if n_frames exceeds the sample-derived block count (tail)
    for k in range(n_frames):
        # NLMS block k, gated by the decision made for the *previous* frame
        if k < n_blocks:
            i = k * stft.HOP
            blk, health = nlms.process_block(prim[i:i + stft.HOP], ref[i:i + stft.HOP],
                                              gate if controller_on else 1.0)
            n_hat[i:i + len(blk)] = blk
        # if there are more feature frames than NLMS blocks (frame count can
        # exceed sample-derived block count near the tail), hold the last health
        a = k * stft.HOP
        f = ff.compute(pad[a:a + stft.N_FFT], padr[a:a + stft.N_FFT], P[:, k], R[:, k], health, gate)
        feats[k] = f
        if controller_on:
            # frame k spans hop blocks k-1 and k (reflect-padded framing), so either block's hit counts
            gate, b, r = ctl.step(f, ff.diff_jump, bool(lim_hit[max(k - 1, 0)] or lim_hit[min(k, n_blocks)]))
            gates[k], bursts[k], rel[k] = gate, b, r
    if not controller_on:
        feats[:] = 0.0
    return {"n_hat": n_hat, "features": feats, "gate": gates, "burst": bursts, "reliability": rel,
            "mix": np.stack([prim, ref])}
