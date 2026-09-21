"""Upstream GTCRN HybridLoss (kept exact so fine-tuned baselines are
apples-to-apples) plus the speech-preservation variant for `vaani_full_sp`.
The spectral balance, compression exponent and an optional absolute-SNR term
are config fields (`loss_cfg`) for the r3 round; defaults reproduce upstream.

Against the problem statement's "SI-SNR, L1/L2 loss, and perceptual loss", the
three terms map as:

- **SI-SNR** -- the scale-invariant term in `HybridLoss.forward`, plus the optional
  absolute-SNR term `w_snr` (SI-SNR is scale-blind and the target is absolute).
- **L2** -- `wmse` on the compressed complex parts and on the compressed magnitude.
  **L1** -- `clean_l1` to the clean target in `SpeechPreservationLoss`.
- **Perceptual** -- the `mag ** p` compressed-magnitude term in `_compress`. Power-law
  magnitude compression is a perceptual weighting, not merely a numerical convenience:
  it approximates the compressive loudness response of human hearing, which is why it
  is the standard spectral loss in the DNS Challenge baselines and in GTCRN. Choosing
  `p` chooses how strongly quiet spectral detail is weighted relative to loud;
  round-3-and-later configs set `p: 0.5` against the upstream default of 0.3, which
  weights quiet detail more heavily. It is an explicitly perceptual objective and
  should be described as one, but note what it is not: it is a psychoacoustic
  magnitude weighting, not a PESQ or PMSQE surrogate, so it does not optimise a
  perceptual *metric* directly.
"""
import torch, torch.nn as nn
from vaani.dsp import stft


def absolute_snr(y_p, y_t):
    return 10 * torch.log10((y_t ** 2).sum(-1) / (((y_p - y_t) ** 2).sum(-1) + 1e-8) + 1e-8)


def snr_clamp_fraction(y_p, y_t, snr_max_db=30.):
    """Fraction of samples whose SNR term has saturated (other loss terms still act)."""
    with torch.no_grad():
        return (absolute_snr(y_p, y_t) >= snr_max_db).float().mean()


def build_loss(name="hybrid", config=None):
    if name not in {"hybrid", "speech_preservation"}:
        raise ValueError(f"unknown loss {name!r}")
    cls = SpeechPreservationLoss if name == "speech_preservation" else HybridLoss
    return cls(**(config or {}))


def _compress(spec, p=0.3):
    re, im = spec[..., 0], spec[..., 1]
    mag = torch.sqrt(re ** 2 + im ** 2 + 1e-12)
    return re / mag ** (1 - p), im / mag ** (1 - p), mag ** p


class HybridLoss(nn.Module):
    """w_complex*(re+im compressed MSE) + w_mag*mag^p MSE + SI-SNR (+ w_snr * absolute-SNR term).
    Defaults (30/70, p=0.3, w_snr=0) are the upstream loss verbatim."""
    def __init__(self, w_complex: float = 30.0, w_mag: float = 70.0, p: float = 0.3, w_snr: float = 0.0, snr_max_db: float = 30.0):
        super().__init__()
        self.w_complex, self.w_mag, self.p, self.w_snr, self.snr_max_db = w_complex, w_mag, p, w_snr, snr_max_db
        self.last_snr_clamp_fraction = torch.tensor(0.)

    def snr_term(self, y_p, y_t):
        """-mean(min(SNR_dB, snr_max)): the target is absolute SNR and SI-SNR is scale-blind. The clamp keeps
        clean-bucket items (already near-perfect) from dominating the gradient."""
        snr = absolute_snr(y_p, y_t)
        self.last_snr_clamp_fraction = (snr.detach() >= self.snr_max_db).float().mean()
        return -snr.clamp(max=self.snr_max_db).mean()

    def forward(self, pred, true, frame_weight=None, is_clean=None):
        pr, pi, pm = _compress(pred, self.p); tr, ti, tm = _compress(true, self.p)
        w = torch.ones_like(pm[:, 0]) if frame_weight is None else frame_weight  # (B,T)
        w = w[:, None, :]                                                        # (B,1,T)
        def wmse(a, b):
            return ((a - b) ** 2 * w).mean()
        spec_loss = self.w_complex * (wmse(pr, tr) + wmse(pi, ti)) + self.w_mag * wmse(pm, tm)
        y_p = stft.istft(pred); y_t = stft.istft(true)
        proj = (y_t * y_p).sum(-1, keepdim=True) * y_t / ((y_t ** 2).sum(-1, keepdim=True) + 1e-8)
        sisnr = -torch.log10(proj.norm(dim=-1) ** 2 / ((y_p - proj).norm(dim=-1) ** 2 + 1e-8) + 1e-8).mean()
        total = spec_loss + sisnr
        if self.w_snr:
            total = total + self.w_snr * self.snr_term(y_p, y_t)
        return total


class SpeechPreservationLoss(HybridLoss):
    """HybridLoss + L1 to the clean target on clean-bucket items (== input for the
    clean bucket up to room path); burst_weight is read by the trainer for frame_weight."""
    def __init__(self, burst_weight: float = 3.0, clean_l1: float = 1.0, **kw):
        super().__init__(**kw); self.burst_weight, self.clean_l1 = burst_weight, clean_l1

    def forward(self, pred, true, frame_weight=None, is_clean=None):
        base = super().forward(pred, true, frame_weight, is_clean)
        if is_clean is not None and is_clean.any():
            base = base + self.clean_l1 * (pred[is_clean] - true[is_clean]).abs().mean()
        return base
