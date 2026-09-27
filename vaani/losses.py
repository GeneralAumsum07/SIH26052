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
import numpy as np
import torch, torch.nn as nn
from vaani.dsp import stft


def absolute_snr(y_p, y_t):
    return 10 * torch.log10((y_t ** 2).sum(-1) / (((y_p - y_t) ** 2).sum(-1) + 1e-8) + 1e-8)


def snr_clamp_fraction(y_p, y_t, snr_max_db=30.):
    """Fraction of samples whose SNR term has saturated (other loss terms still act)."""
    with torch.no_grad():
        return (absolute_snr(y_p, y_t) >= snr_max_db).float().mean()


def build_loss(name="hybrid", config=None):
    cls = {"hybrid": HybridLoss, "speech_preservation": SpeechPreservationLoss, "fe": FELoss}.get(name)
    if cls is None:
        raise ValueError(f"unknown loss {name!r}")
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


class _PeakCut(nn.Module):
    """Stands in for torch_pesq's resampler, an identity at 16 kHz, which PesqLoss.raw() applies right after dividing
    both signals by their joint peak. That peak depends on the output, so the reference's path (whose |STFT| has
    NaN gradients at exact-zero bins) reached the output's gradient through it and turned all of it NaN whenever the
    output held the peak. Here the reference comes back detached and the output as `deg / peak` with the peak held
    constant: same values, and the gradient the level-invariant score should have anyway."""
    def __init__(self):
        super().__init__()
        self.deg = None

    def forward(self, x):
        if self.deg is not None and x.requires_grad:   # raw() resamples the output first, then the reference
            out, self.deg = self.deg, None
            return out
        return x.detach()


def _pesq_module():
    """A differentiable PESQ if one is importable (torch_pesq, the `train` extra), else None: the term then carries
    weight 0. fp32: its Bark/loudness tables are float64 numpy, which would promote the whole term to fp64 on the GPU."""
    try:
        from torch_pesq import PesqLoss
        m = PesqLoss(1.0, sample_rate=16000).float()
        m.resampler = _PeakCut()
        return m
    except Exception:
        return None


PESQ_MIN_RMS = 1e-4   # ~-80 dBFS: a silent target has no PESQ (torch_pesq divides by the joint peak -> NaN)
PESQ_DITHER = 1e-6    # -120 dBFS: an exactly-zero output bin has a NaN |.| gradient inside torch_pesq
PESQ_DIST_FLOOR = 1e-4   # added to the per-frame disturbance before raw()'s L6 pooling: shifts the score by <= 1e-4


def _floored_unfold(unfold):
    """raw() pools the per-frame disturbance as (unfold(d) ** 6).mean() ** (1/6); on a near-clean output d sits at
    its 1e-20 clamp, d ** 6 underflows to 0 in fp32 and pow(1/6)'s gradient at 0 is inf, and align_level's global
    power then spreads the NaN to every sample (the sanitised gradient came back all-zero). A small floor keeps
    every base normal: by the triangle inequality the pooled value moves by at most the floor."""
    def f(x, *a, **k):
        return unfold(x + PESQ_DIST_FLOOR, *a, **k)
    return f


def _sanitize_grad(g):
    return torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)


_DITHER: dict = {}
_DUMMY: dict = {}


def _dither(n, device):
    """The fixed -120 dBFS output dither, built once per (length, device) and kept there (it used to be drawn on the
    CPU from a reseeded generator and copied on every call: the same values, now without the copy)."""
    key = (n, str(device))
    d = _DITHER.get(key)
    if d is None:
        g = torch.Generator().manual_seed(0)
        d = _DITHER[key] = (PESQ_DITHER * torch.randn(n, generator=g)).to(device)
    return d


def _dummy(n, device):
    """A fixed, well-conditioned stand-in for silent-target rows (weight 0): keeps the batch shape static."""
    key = (n, str(device))
    d = _DUMMY.get(key)
    if d is None:
        g = torch.Generator().manual_seed(1)
        d = _DUMMY[key] = (0.1 * torch.randn(n, generator=g)).to(device)
    return d


# ---- torch_pesq's fixed IIR filters evaluated by FFT convolution (plan Task 4b) --------------------------------
PESQ_IR_ENERGY_TAIL = 1e-16
_IR: dict = {}


def pesq_impulse_response(b, a, tail=PESQ_IR_ENERGY_TAIL, n=1 << 15):
    """float64 impulse response of the IIR (b, a), truncated where the remaining energy falls below `tail` of the total
    (torch_pesq's float32 coefficients evaluated exactly: 544 taps for the power band-pass, 591 for the pre-emphasis)."""
    from scipy.signal import lfilter as sp_lfilter
    x = np.zeros(n); x[0] = 1.0
    h = sp_lfilter(np.asarray(b, np.float64), np.asarray(a, np.float64), x)
    e = np.cumsum((h ** 2)[::-1])[::-1]          # energy from k to the end
    keep = int(np.argmax(e < tail * e[0])) if (e < tail * e[0]).any() else n
    return h[:keep]


def _fft_lfilter(signal, a_coeffs, b_coeffs, clamp=False):
    """Drop-in for torchaudio.functional.lfilter on torch_pesq's fixed filters: a causal FFT convolution with the
    exact impulse response, parallel over samples (torchaudio's CUDA recursion runs one thread per signal)."""
    key = (a_coeffs.data_ptr(), b_coeffs.data_ptr(), str(signal.device))
    H = _IR.get(key)
    if H is None:   # one host read per filter and device, then cached
        h = pesq_impulse_response(b_coeffs.detach().cpu().double().numpy(), a_coeffs.detach().cpu().double().numpy())
        H = _IR[key] = torch.as_tensor(h, dtype=torch.float64, device=signal.device)
    H = H.to(signal.dtype)   # float32 convolution for float32 signals (the float64 IR rounded once)
    n, L = signal.shape[-1], H.shape[0]
    nfft = 1 << (n + L - 1 - 1).bit_length()
    y = torch.fft.irfft(torch.fft.rfft(signal, nfft) * torch.fft.rfft(H, nfft), nfft)[..., :n]
    return y.clamp(-1, 1) if clamp else y


def pesq_term(pesq, y_p, y_t, fft_filters=False):
    """Mean torch_pesq distortion (0 = transparent) over items whose target is audible; (value, items scored).
    Sync-free: silent-target rows are replaced by a fixed dummy and carry zero weight (torch_pesq scores row by row,
    so audible rows are unaffected, and an all-silent batch gives 0); the output gets a fixed -120 dBFS dither so a
    fully suppressed prediction still has a gradient; any non-finite gradient left inside torch_pesq is zeroed on
    this branch only: a NaN there would pass the trainer's checks and poison the weights through the optimiser.
    `items` is a 0-d device tensor. fft_filters: evaluate torch_pesq's two fixed IIR filters by FFT convolution."""
    keep = y_t.pow(2).mean(-1).sqrt() > PESQ_MIN_RMS
    pesq.to(y_p.device)   # train.py never moves the loss module; torch_pesq keeps its filters as nn.Parameters
    n = y_p.shape[-1]
    k = keep[:, None]
    dummy = _dummy(n, y_p.device)
    ref = torch.where(k, y_t, dummy).detach()
    deg = torch.where(k, y_p + _dither(n, y_p.device), dummy)
    if deg.requires_grad:
        deg.register_hook(_sanitize_grad)
        if isinstance(pesq.resampler, _PeakCut):   # raw()'s own joint-peak normalisation, with the peak constant
            peak = torch.maximum(deg.abs().amax(1, keepdim=True), ref.abs().amax(1, keepdim=True)).detach()
            pesq.resampler.deg = deg / peak
    import torch_pesq.loss as tpl
    unfold, lfilter = tpl.unfold, tpl.lfilter
    tpl.unfold = _floored_unfold(unfold)   # scoped to this call: other PesqLoss users keep the library's pooling
    if fft_filters:
        tpl.lfilter = _fft_lfilter
    try:
        v = pesq(ref, deg).float()
    finally:
        tpl.unfold, tpl.lfilter = unfold, lfilter
        if isinstance(pesq.resampler, _PeakCut):
            pesq.resampler.deg = None
    w = keep.to(v.dtype)
    items = keep.sum()
    val = torch.where(torch.isfinite(v) & keep, v, torch.zeros_like(v)).sum() / items.clamp(min=1).to(v.dtype)
    return val, items


def anti_wrap(x):
    """MP-SENet anti-wrapping: |x - 2 pi round(x / 2 pi)|, the phase distance that ignores whole turns."""
    return (x - 2 * torch.pi * torch.round(x / (2 * torch.pi))).abs()


def asym_sq(diff, kappa):
    """VoiceFilter-Lite g_asym(x)^2: x^2 where the output is louder than the target, (kappa x)^2 where it is quieter."""
    return torch.where(diff > 0, kappa * diff, diff) ** 2


def mr_stft(y_p, y_t, ffts=(256, 512, 1024)):
    """Multi-resolution STFT: spectral convergence + log-magnitude L1 per resolution, hop n/4, Hann."""
    total = y_p.new_zeros(())
    for n in ffts:
        w = torch.hann_window(n, device=y_p.device)
        P = torch.stft(y_p, n, n // 4, n, w, return_complex=True).abs() + 1e-7
        T = torch.stft(y_t, n, n // 4, n, w, return_complex=True).abs() + 1e-7
        total = total + (T - P).norm(dim=(-2, -1)).div(T.norm(dim=(-2, -1))).mean() + (T.log() - P.log()).abs().mean()
    return total / len(ffts)


class FELoss(nn.Module):
    """r8 loss (plan 11.6): the FastEnhancer mix on alpha-compressed spectra (magnitude, complex, consistency via
    iSTFT->STFT, waveform L1, PESQ when a differentiable one exists) + the repo's clamped absolute-SNR term + an
    over-suppression penalty (VoiceFilter-Lite asymmetric L2 on compressed magnitude; kappa 1 = plain magnitude MSE)
    + optional MR-STFT and anti-wrapping phase on speech-dominant bins.

    w_snr 0.002 keeps r7's SNR-to-spectral weight ratio (0.2 over w_complex + w_mag = 100) on these unit-sum weights
    (inferred starting point; ablation 6 tunes kappa). Consistency follows MP-SENet: STFT(iSTFT(pred)) against pred."""
    TERMS = ("mag", "over", "complex", "consistency", "wave", "pesq", "snr", "mrstft", "phase")

    def __init__(self, w_mag=0.3, w_complex=0.2, w_consistency=0.3, w_wave=0.2, w_pesq=0.001, w_snr=0.002,
                 snr_max_db=30.0, kappa=1.0, w_mrstft=0.0, w_phase=0.0, p=0.3, dominance_db=0.0, pesq_required=False,
                 pesq_filters="lfilter"):
        super().__init__()
        if kappa < 1:
            raise ValueError("kappa must be >= 1 (1 = symmetric magnitude loss)")
        self.w = dict(mag=w_mag, complex=w_complex, consistency=w_consistency, wave=w_wave, snr=w_snr,
                      mrstft=w_mrstft, phase=w_phase)
        self.kappa, self.p, self.snr_max_db, self.dominance_db = kappa, p, snr_max_db, dominance_db
        if pesq_filters not in ("lfilter", "fft"):
            raise ValueError("pesq_filters must be 'lfilter' or 'fft'")
        self.pesq_fft = pesq_filters == "fft"   # r8: FFT evaluation of torch_pesq's fixed filters (plan Task 4b)
        self.pesq = _pesq_module() if w_pesq else None
        if pesq_required and w_pesq and self.pesq is None:   # a box missing the `train` extra must not train without it
            raise ImportError("loss_cfg.pesq_required: torch_pesq is not importable (uv sync --extra train)")
        self.w["pesq"] = w_pesq if self.pesq is not None else 0.0   # TBD: no differentiable PESQ importable -> 0
        self.w_snr = w_snr   # the trainer reads w_snr / last_snr_clamp_fraction like HybridLoss
        self.last_snr_clamp_fraction = torch.tensor(0.)
        self.last_terms, self.last_pesq_items = {}, 0
        self.keep_live_terms, self.live_terms = False, {}   # the trainer's per-term gradient-norm logging

    def forward(self, pred, true, frame_weight=None, is_clean=None, noisy=None, y_pred=None, y_true=None,
                reanalyze=None):
        """pred/true (B,257,T,2) spectra. y_pred/y_true (optional): the waveforms those spectra are the 512/256
        analysis of; given, the waveform terms use them instead of inverting the spectra (the re-synthesis route,
        vaani.enhance_low_delay.ResynthesisFELoss), with the same terms within rounding. reanalyze (optional): the
        analysis the consistency term re-applies to y_pred (default the legacy stft.stft)."""
        pr, pi, pm = _compress(pred, self.p); tr, ti, tm = _compress(true, self.p)
        w = (torch.ones_like(pm[:, 0]) if frame_weight is None else frame_weight)[:, None, :]
        def wmean(x):
            return (x * w).mean()
        t = {}
        diff = tm - pm   # > 0: the output is quieter than the target (over-suppression)
        t["mag"] = wmean(diff ** 2)
        t["over"] = wmean(asym_sq(diff, self.kappa)) - t["mag"]   # zero at kappa 1
        t["complex"] = wmean((pr - tr) ** 2 + (pi - ti) ** 2)
        n = pred.shape[2] * stft.HOP - stft.HOP   # the frames' exact span: STFT(iSTFT(.)) keeps T frames
        # explicit overlap-add: bit-identical to stft.istft, without its host-side envelope check
        y_p = stft.istft_explicit(pred, length=n) if y_pred is None else y_pred
        y_t = stft.istft_explicit(true, length=n) if y_true is None else y_true
        cr, ci, _ = _compress((reanalyze or stft.stft)(y_p), self.p)
        t["consistency"] = wmean((cr - pr) ** 2 + (ci - pi) ** 2)
        t["wave"] = (y_p - y_t).abs().mean()
        snr = absolute_snr(y_p, y_t)
        self.last_snr_clamp_fraction = (snr.detach() >= self.snr_max_db).float().mean()
        t["snr"] = -snr.clamp(max=self.snr_max_db).mean()
        t["pesq"], self.last_pesq_items = pesq_term(self.pesq, y_p, y_t, self.pesq_fft) if self.w["pesq"] else (pred.new_zeros(()), 0)
        t["mrstft"] = mr_stft(y_p, y_t) if self.w["mrstft"] else pred.new_zeros(())
        if self.w["phase"]:
            # speech-dominant bins: clean power above the residual noise (noisy - clean) by dominance_db;
            # without the noisy spectrum, bins within 40 dB of the item's clean peak
            ec = true.pow(2).sum(-1)
            if noisy is not None:
                dom = ec > (noisy - true).pow(2).sum(-1) * 10 ** (self.dominance_db / 10)
            else:
                dom = ec > ec.amax(dim=(1, 2), keepdim=True) * 1e-4
            ok = dom & (pred.detach().pow(2).sum(-1) > 1e-12)   # atan2 has no gradient at the origin
            ph_p = torch.atan2(torch.where(ok, pred[..., 1], 1.0), torch.where(ok, pred[..., 0], 1.0))
            ph_t = torch.atan2(true[..., 1], true[..., 0])
            t["phase"] = (anti_wrap(ph_p - ph_t) * ok).sum() / ok.sum().clamp(min=1)
        else:
            t["phase"] = pred.new_zeros(())
        self.last_terms = {k: v.detach() for k, v in t.items()}   # tensors: no host sync unless logged
        self.live_terms = dict(t) if self.keep_live_terms else {}
        total = self.w["mag"] * (t["mag"] + t["over"])
        for k in ("complex", "consistency", "wave", "pesq", "snr", "mrstft", "phase"):
            if self.w[k]:
                total = total + self.w[k] * t[k]
        return total
