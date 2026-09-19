import torch
from vaani import losses
from vaani.dsp import stft


def _pair(seed=0, n=16000):
    g = torch.Generator().manual_seed(seed)
    y = torch.randn(2, n, generator=g) * 0.1
    return stft.stft(y + 0.03 * torch.randn(2, n, generator=g)), stft.stft(y)


def test_default_hybrid_is_upstream_verbatim():
    pred, true = _pair()
    a = losses.HybridLoss()(pred, true); b = losses.HybridLoss(w_complex=30, w_mag=70, p=0.3, w_snr=0.0)(pred, true)
    assert torch.allclose(a, b)


def test_snr_term_lowers_loss_as_snr_rises_and_is_clamped():
    pred, true = _pair()
    fn = losses.HybridLoss(w_snr=0.2)
    noisy = fn(pred, true).item()
    y_t = stft.istft(true)
    better = fn(stft.stft(y_t + 0.001 * torch.randn_like(y_t)), true).item()
    assert better < noisy
    # a near-perfect prediction is clamped at 30 dB: the SNR term stops rewarding it and its gradient is bounded
    p = true.clone().requires_grad_(True)
    loss = fn(p, true); loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(p.grad).all()
    assert fn.snr_term(stft.istft(p.detach()), y_t).item() == -30.0


def test_config_fields_change_the_spectral_balance():
    pred, true = _pair()
    a = losses.HybridLoss(w_complex=50, w_mag=50, p=0.5)(pred, true)
    b = losses.HybridLoss()(pred, true)
    assert not torch.allclose(a, b)
