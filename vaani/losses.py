"""Upstream GTCRN HybridLoss (kept exact so fine-tuned baselines are
apples-to-apples) plus the speech-preservation variant for `vaani_full_sp`."""
import torch, torch.nn as nn
from vaani.dsp import stft


def _compress(spec, p=0.3):
    re, im = spec[..., 0], spec[..., 1]
    mag = torch.sqrt(re ** 2 + im ** 2 + 1e-12)
    return re / mag ** (1 - p), im / mag ** (1 - p), mag ** p


class HybridLoss(nn.Module):
    """30*(re+im compressed MSE) + 70*mag^0.3 MSE + SI-SNR. Verbatim upstream."""
    def forward(self, pred, true, frame_weight=None, is_clean=None):
        pr, pi, pm = _compress(pred); tr, ti, tm = _compress(true)
        w = torch.ones_like(pm[:, 0]) if frame_weight is None else frame_weight  # (B,T)
        w = w[:, None, :]                                                        # (B,1,T)
        def wmse(a, b):
            return ((a - b) ** 2 * w).mean()
        spec_loss = 30 * (wmse(pr, tr) + wmse(pi, ti)) + 70 * wmse(pm, tm)
        y_p = stft.istft(pred); y_t = stft.istft(true)
        proj = (y_t * y_p).sum(-1, keepdim=True) * y_t / ((y_t ** 2).sum(-1, keepdim=True) + 1e-8)
        sisnr = -torch.log10(proj.norm(dim=-1) ** 2 / ((y_p - proj).norm(dim=-1) ** 2 + 1e-8) + 1e-8).mean()
        return spec_loss + sisnr


class SpeechPreservationLoss(HybridLoss):
    """HybridLoss + L1 identity penalty on clean-bucket items; burst up-weighting
    arrives via frame_weight from the caller."""
    def __init__(self, clean_l1: float = 1.0):
        super().__init__(); self.clean_l1 = clean_l1

    def forward(self, pred, true, frame_weight=None, is_clean=None):
        base = super().forward(pred, true, frame_weight, is_clean)
        if is_clean is not None and is_clean.any():
            base = base + self.clean_l1 * (pred[is_clean] - true[is_clean]).abs().mean()
        return base
