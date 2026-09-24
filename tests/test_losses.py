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


# --- r8 FELoss (plan 11.6) ---
import math
import pytest


def test_build_loss_fe_and_unknown():
    assert isinstance(losses.build_loss("fe", {"kappa": 3.0}), losses.FELoss)
    with pytest.raises(ValueError):
        losses.build_loss("nope")
    with pytest.raises(ValueError):
        losses.FELoss(kappa=0.5)


def test_fe_perfect_prediction_known_answer():
    _, true = _pair(n=16384)   # a whole number of hops, as the 4 s training crops are
    fn = losses.FELoss(w_mrstft=0.1, w_phase=0.1)
    loss = fn(true.clone(), true)
    t = fn.last_terms
    for k in ("mag", "over", "complex", "wave", "mrstft", "phase"):
        assert abs(float(t[k])) < 1e-6, k
    assert float(t["consistency"]) < 1e-6          # a real STFT is consistent
    assert float(t["snr"]) == -30.0                # clamped at snr_max_db
    assert math.isclose(float(loss), 0.002 * -30.0, abs_tol=1e-5)


def test_asym_known_answer_and_over_term():
    d = torch.tensor([-1.0, 0.0, 2.0])
    assert torch.allclose(losses.asym_sq(d, 3.0), torch.tensor([1.0, 0.0, 36.0]))
    _, true = _pair()
    quiet, loud = true * 0.5, true * 1.5
    f1, f3 = losses.FELoss(kappa=1.0), losses.FELoss(kappa=3.0)
    f1(quiet, true); assert abs(float(f1.last_terms["over"])) < 1e-7   # kappa 1: plain magnitude MSE
    f3(quiet, true); f3_over_quiet = float(f3.last_terms["over"])
    f3(loud, true); f3_over_loud = float(f3.last_terms["over"])
    assert f3_over_quiet > 0 and abs(f3_over_loud) < 1e-7              # only over-suppression is penalised
    # (kappa^2 - 1) x the over-suppressed share of the magnitude MSE
    f3(quiet, true)
    assert math.isclose(float(f3.last_terms["over"]), 8 * float(f3.last_terms["mag"]), rel_tol=1e-4)


def test_anti_wrap_known_answer():
    x = torch.tensor([0.0, 2 * math.pi, math.pi / 2, -2 * math.pi + 0.1])
    assert torch.allclose(losses.anti_wrap(x), torch.tensor([0.0, 0.0, math.pi / 2, 0.1]), atol=1e-6)


def test_fe_gradients_finite_all_terms_and_falls_toward_target():
    pred, true = _pair()
    noisy = pred.detach()
    fn = losses.FELoss(kappa=3.0, w_mrstft=0.05, w_phase=0.05)
    p = pred.clone().requires_grad_(True)
    loss = fn(p, true, frame_weight=torch.ones(2, true.shape[2]) * 2, noisy=noisy)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    # a zero prediction (the erasure failure) must also give finite gradients
    z = torch.zeros_like(true).requires_grad_(True)
    fn(z, true, noisy=noisy).backward()
    assert torch.isfinite(z.grad).all()
    better = fn(0.5 * pred + 0.5 * true, true, noisy=noisy)
    assert float(better) < float(fn(pred, true, noisy=noisy))


def test_fe_pesq_weight_zero_without_differentiable_pesq():
    fn = losses.FELoss()
    if fn.pesq is None:
        assert fn.w["pesq"] == 0.0


def test_existing_losses_unchanged_by_registry():
    pred, true = _pair()
    assert torch.allclose(losses.build_loss("hybrid")(pred, true), losses.HybridLoss()(pred, true))
