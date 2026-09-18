"""Runs NLMS + features + controller frame-synchronously over a stereo clip.
Used offline (dataset/eval) and as the reference for the embedded port.
Frame k covers samples [k*HOP - 256, k*HOP + 256) after torch-style reflect
padding, so the model's frame k and this feature vector k line up exactly.
"""
import numpy as np

from vaani.dsp import stft
from vaani.dsp.controller import Controller
from vaani.dsp.features import FrameFeatures, N_FEATURES
from vaani.dsp.nlms import NLMS


def run(mix: np.ndarray, controller_on: bool = True) -> dict:
    prim, ref = mix[0].astype(np.float32), mix[1].astype(np.float32)
    T = len(prim)
    # local instances only (no module-level mutable state) -> safe in DataLoader workers
    nlms, ff, ctl = NLMS(), FrameFeatures(), Controller()
    n_hat = np.zeros(T, np.float32)
    gate = 1.0
    # NLMS runs in hop-sized blocks so its taps update between frames
    healths = []
    for i in range(0, T, stft.HOP):
        blk, hlt = nlms.process_block(prim[i:i + stft.HOP], ref[i:i + stft.HOP], gate if controller_on else 1.0)
        n_hat[i:i + len(blk)] = blk; healths.append(hlt)
        # controller decisions for the *next* block come from the frame ending here
    P = stft.np_stft(prim); R = stft.np_stft(ref)
    n_frames = P.shape[1]
    pad = np.pad(prim, stft.N_FFT // 2, mode="reflect"); padr = np.pad(ref, stft.N_FFT // 2, mode="reflect")
    feats = np.zeros((n_frames, N_FEATURES), np.float32)
    gates = np.ones(n_frames, np.float32); bursts = np.zeros(n_frames, bool); rel = np.ones(n_frames, np.float32)
    gate = 1.0
    for k in range(n_frames):
        a = k * stft.HOP
        f = ff.compute(pad[a:a + stft.N_FFT], padr[a:a + stft.N_FFT], P[:, k], R[:, k],
                       healths[min(k, len(healths) - 1)], gate)
        feats[k] = f
        if controller_on:
            gate, b, r = ctl.step(f)
            gates[k], bursts[k], rel[k] = gate, b, r
    if not controller_on:
        feats[:] = 0.0
    # Second pass of NLMS with the actual gate trajectory, so n_hat reflects gating.
    # (Offline this is exact; online the embedded port applies gate[k-1] to block k.)
    if controller_on:
        nlms.reset()
        for k in range(n_frames):
            i = k * stft.HOP
            if i >= T:
                break
            blk, _ = nlms.process_block(prim[i:i + stft.HOP], ref[i:i + stft.HOP], gates[k - 1] if k else 1.0)
            n_hat[i:i + len(blk)] = blk
    return {"n_hat": n_hat, "features": feats, "gate": gates, "burst": bursts, "reliability": rel}
