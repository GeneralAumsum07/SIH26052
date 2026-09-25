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


def test_fe_pesq_required_fails_loudly_without_torch_pesq(monkeypatch):
    monkeypatch.setattr(losses, "_pesq_module", lambda: None)
    with pytest.raises(ImportError, match="pesq_required"):
        losses.build_loss("fe", {"pesq_required": True})
    assert losses.build_loss("fe", {"pesq_required": True, "w_pesq": 0.0}).w["pesq"] == 0.0
    assert losses.build_loss("fe").w["pesq"] == 0.0     # default: silent fallback, as before


def test_existing_losses_unchanged_by_registry():
    pred, true = _pair()
    assert torch.allclose(losses.build_loss("hybrid")(pred, true), losses.HybridLoss()(pred, true))


# --- differentiable PESQ term (torch_pesq, the `train` extra) ---
def _speechish(n=4 * 16000, sr=16000):
    """Harmonic 120 Hz voice with syllable envelope and exact-zero pauses (the case that makes torch_pesq's |.| NaN)."""
    t = torch.arange(n) / sr
    ph = 2 * math.pi * torch.cumsum(120 + 30 * torch.sin(2 * math.pi * 0.5 * t), 0) / sr
    x = sum(torch.sin(k * ph) / k for k in range(1, 20))
    env = torch.sin(2 * math.pi * 3 * t).clamp(min=0) ** 0.5 * ((t % 1.0) < 0.7)
    return (0.1 * x * env).float()


def _at_snr(c, snr_db, seed=0):
    nz = torch.randn(c.shape, generator=torch.Generator().manual_seed(seed))
    return c + nz * c.norm() / nz.norm() * 10 ** (-snr_db / 20)


def test_pesq_term_known_answer_against_reference_pesq():
    pytest.importorskip("torch_pesq")
    from pesq import pesq
    pq = losses._pesq_module()
    c = _speechish()
    prev = -1.0
    for snr in (40, 20, 10):
        d = _at_snr(c, snr)
        # torch_pesq's MOS tracks ITU P.862.2 wide-band PESQ on the same pair (measured gap <= 0.03 on these)
        assert abs(float(pq.mos(c[None], d[None])) - pesq(16000, c.numpy(), d.numpy(), "wb")) < 0.15, snr
        v, n = losses.pesq_term(pq, d[None], c[None])
        assert n == 1 and float(v) > prev   # more noise, more distortion
        prev = float(v)
    v, _ = losses.pesq_term(pq, c[None], c[None])
    assert float(v) < 0.01                  # transparent output: ~0 (the -120 dBFS dither only), vs 2.5 at 40 dB


def test_pesq_term_gradients_finite_on_silence_and_exact_zeros():
    pytest.importorskip("torch_pesq")
    pq = losses._pesq_module()
    c = _speechish()
    true = torch.stack([c, c, torch.zeros_like(c), c])
    pred = torch.stack([c, torch.zeros_like(c), _at_snr(c, 5), _at_snr(c, 10)]).requires_grad_(True)
    v, n = losses.pesq_term(pq, pred, true)
    assert n == 3                           # the silent target is left out
    v.backward()
    assert torch.isfinite(v) and torch.isfinite(pred.grad).all()
    assert float(pred.grad[2].abs().sum()) == 0.0          # left-out item: no PESQ gradient
    assert float(pred.grad[1].abs().sum()) > 0             # fully suppressed output still pulled back (dither)
    assert float(pred.grad[3].abs().sum()) > 0
    z, n0 = losses.pesq_term(pq, pred[2:3], true[2:3])
    assert n0 == 0 and float(z) == 0.0


def test_fe_pesq_term_live_when_importable():
    pytest.importorskip("torch_pesq")
    c = _speechish(n=16384 * 4)
    true = stft.stft(torch.stack([c, c]))
    pred = stft.stft(torch.stack([_at_snr(c, 5), torch.zeros_like(c)])).requires_grad_(True)
    fn = losses.FELoss()
    assert fn.w["pesq"] == 0.001
    loss = fn(pred, true)
    loss.backward()
    assert float(fn.last_terms["pesq"]) > 0 and fn.last_pesq_items == 2
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()
    no = losses.FELoss(w_pesq=0.0)
    assert no.pesq is None and float(no(pred.detach(), true)) < float(loss.detach())


def test_pesq_term_gradient_is_live_on_a_near_clean_output():
    # the near-clean case is where torch_pesq's own gradient is all-NaN (L6 pooling of an underflowed disturbance);
    # here it must be finite, non-zero and agree with central differences where the reference holds the peak
    pytest.importorskip("torch_pesq")
    from torch_pesq import PesqLoss
    g = torch.Generator().manual_seed(3)
    ref = torch.randn(2, 32000, generator=g) * 0.2
    x = (0.5 * ref + 0.01 * torch.randn(2, 32000, generator=g)).requires_grad_(True)
    pq = losses._pesq_module()
    v, _ = losses.pesq_term(pq, x, ref)
    (gx,) = torch.autograd.grad(v, x)
    assert torch.isfinite(gx).all() and float(gx.abs().sum()) > 0
    x2 = x.detach().clone().requires_grad_(True)
    dith = losses.PESQ_DITHER * torch.randn(32000, generator=torch.Generator().manual_seed(0))
    v2 = PesqLoss(1.0, sample_rate=16000).float()(ref, x2 + dith).float().mean()
    assert abs(float(v) - float(v2)) <= losses.PESQ_DIST_FLOOR   # the floor's bound on the score shift
    (gx2,) = torch.autograd.grad(v2, x2)
    assert not torch.isfinite(gx2).any()                        # the library's own: all NaN (why the floor exists)
    for d in (gx / gx.norm(), -gx / gx.norm()):
        eps = 3e-3
        with torch.no_grad():
            fd = (float(losses.pesq_term(pq, x + eps * d, ref)[0]) - float(losses.pesq_term(pq, x - eps * d, ref)[0])) / (2 * eps)
        an = float((gx * d).sum())
        assert abs(fd - an) <= 0.2 * abs(an), (fd, an)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_pesq_term_runs_on_cuda_without_moving_the_loss():
    pytest.importorskip("torch_pesq")
    pq = losses._pesq_module()
    c = _speechish().cuda()
    p = _at_snr(c.cpu(), 10).cuda().requires_grad_(True)
    v, _ = losses.pesq_term(pq, p[None], c[None])
    v.backward()
    assert v.device.type == "cuda" and torch.isfinite(p.grad).all()
