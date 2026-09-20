"""Tiny causal residual refiner on top of a frozen first stage (Tier 4.6 plan §6).

Sees the first-stage output Y, the primary P and reference R that stage actually consumed, and the removed residual
D = P - Y (a complex difference, not a noise label). Everything is normalised per bin by s = sqrt(|P|^2 + |Y|^2) so the
net works on shape, not level, and the correction is added back at scale: Z = Y + 0.25 * s * tanh(.). The last conv
is zero-initialised, so an untrained cascade is exactly the first stage. Tensor layout inside is (B, C, T, F) like the
rest of vaani_net; the public interfaces take the project's (B, 257, T, 2) spectra.
"""
import torch
from torch import nn
from torch.nn import functional as F

N_BINS, HIDDEN, PAST, SCALE, CLIP = 257, 16, 2, 0.25, 10.0


class ResidualRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(8, HIDDEN, 1)
        self.c1 = nn.Conv2d(HIDDEN, HIDDEN, (PAST + 1, 3))   # (time, freq): two past frames + current, three neighbouring bins
        self.c2 = nn.Conv2d(HIDDEN, 2, 1)
        nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)   # identity cascade at init
        im = torch.ones(1, 2, 1, N_BINS); im[0, 1, 0, 0] = 0; im[0, 1, 0, -1] = 0   # a real signal has no imaginary DC / Nyquist
        self.register_buffer("im_mask", im, persistent=False)

    @staticmethod
    def _features(P, R, Y):
        """(B,257,T,2) x3 -> features (B,8,T,257), scale s (B,1,T,257)."""
        P, R, Y = (x.permute(0, 3, 2, 1) for x in (P, R, Y))   # (B,2,T,F)
        s = torch.sqrt((P ** 2).sum(1, keepdim=True) + (Y ** 2).sum(1, keepdim=True) + 1e-8)
        D = P - Y
        norm = torch.clamp(torch.cat([Y, D, R], 1) / s, -CLIP, CLIP)
        mags = torch.log1p(torch.cat([(P ** 2).sum(1, keepdim=True), (Y ** 2).sum(1, keepdim=True)], 1).sqrt())
        return torch.cat([norm, mags], 1), s, Y

    def _out(self, h1, s, Y):
        delta = torch.tanh(self.c2(h1)) * self.im_mask
        return (Y + SCALE * s * delta).permute(0, 3, 2, 1)

    def forward(self, primary, reference, enhanced):
        """Whole clip; causal by construction (front-padded time axis)."""
        f, s, Y = self._features(primary, reference, enhanced)
        h0 = F.relu(self.c0(f))
        h1 = F.relu(self.c1(F.pad(h0, (1, 1, PAST, 0))))
        return self._out(h1, s, Y)

    def step(self, primary, reference, enhanced, refine_cache):
        """One frame; refine_cache (1,16,PAST,257) holds the past h0 frames, oldest first. Returns (Z, new cache)."""
        f, s, Y = self._features(primary, reference, enhanced)
        h0 = F.relu(self.c0(f))                                  # (1,16,1,257)
        x = torch.cat([refine_cache, h0], 2)                     # (1,16,PAST+1,257)
        h1 = F.relu(self.c1(F.pad(x, (1, 1, 0, 0))))             # (1,16,1,257)
        return self._out(h1, s, Y), x[:, :, 1:]


def init_refine_cache(device="cpu"):
    return torch.zeros(1, HIDDEN, PAST, N_BINS, device=device)


def count_params(m: nn.Module):
    """(total, trainable) so the report can state both, as the plan requires."""
    return sum(p.numel() for p in m.parameters()), sum(p.numel() for p in m.parameters() if p.requires_grad)
