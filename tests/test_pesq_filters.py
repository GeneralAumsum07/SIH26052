"""torch_pesq's fixed filters by FFT convolution and the sync-free pesq_term (plan Task 4b)."""
import math

import numpy as np
import pytest
import torch

torch_pesq = pytest.importorskip("torch_pesq")
from scipy.signal import lfilter as sp_lfilter  # noqa: E402

from vaani import losses  # noqa: E402


def _speech(n=32000, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(n) / 16000
    x = sum(torch.sin(2 * math.pi * f * t) / k for k, f in enumerate((140, 280, 420, 1100, 2300), 1))
    return (0.1 * x * (torch.sin(2 * math.pi * 3 * t) > 0) + 0.002 * torch.randn(n, generator=g)).float()


def _legacy_pesq_term(pesq, y_p, y_t):
    """The pre-Task-4b pesq_term, verbatim in behaviour: boolean selection, CPU dither, int item count."""
    keep = y_t.pow(2).mean(-1).sqrt() > losses.PESQ_MIN_RMS
    if not bool(keep.any()):
        return y_p.new_zeros(()), 0
    ref = y_t[keep].detach()
    g = torch.Generator().manual_seed(0)
    deg = y_p[keep] + losses.PESQ_DITHER * torch.randn(y_p.shape[-1], generator=g)
    if deg.requires_grad:
        deg.register_hook(losses._sanitize_grad)
        peak = torch.maximum(deg.abs().amax(1, keepdim=True), ref.abs().amax(1, keepdim=True)).detach()
        pesq.resampler.deg = deg / peak
    import torch_pesq.loss as tpl
    unfold = tpl.unfold
    tpl.unfold = losses._floored_unfold(unfold)
    try:
        v = pesq(ref, deg).float()
    finally:
        tpl.unfold = unfold
        pesq.resampler.deg = None
    return torch.where(torch.isfinite(v), v, torch.zeros_like(v)).mean(), int(keep.sum())


def test_impulse_response_lengths():
    pq = losses._pesq_module()
    assert len(losses.pesq_impulse_response(pq.power_filter[0].double().numpy(), pq.power_filter[1].double().numpy())) == 544
    assert len(losses.pesq_impulse_response(pq.pre_filter[0].double().numpy(), pq.pre_filter[1].double().numpy())) == 591


@pytest.mark.parametrize("which", ["power_filter", "pre_filter"])
@pytest.mark.parametrize("sig", ["speech", "noise"])
def test_fft_filter_matches_float64_lfilter(which, sig):
    pq = losses._pesq_module()
    f = getattr(pq, which)
    x = _speech() if sig == "speech" else torch.randn(32000, generator=torch.Generator().manual_seed(1)) * 0.1
    x = x[None]
    ref = sp_lfilter(f[0].double().numpy(), f[1].double().numpy(), x.double().numpy()[0])
    got = losses._fft_lfilter(x, f[1], f[0]).numpy()[0]
    assert np.abs(got - ref).max() <= 1e-6 * np.abs(ref).max()
    from torchaudio.functional import lfilter
    rec = lfilter(x, f[1], f[0], clamp=False).numpy()[0]     # the float32 recursion in use, for the record
    assert np.abs(got - ref).max() <= np.abs(rec - ref).max() + 1e-12


def test_sync_free_pesq_term_equals_legacy_with_identical_skips():
    pq = losses._pesq_module()
    c = _speech()
    for true, pred in (
        (torch.stack([c, c, c]), torch.stack([c * 0.5, c + 0.01 * torch.randn(32000), torch.zeros_like(c)])),  # audible
        (torch.stack([c, torch.zeros_like(c), c]), torch.stack([c * 0.9, c, c * 0.3])),                        # mixed
        (torch.zeros(2, 32000), torch.stack([c, c])),                                                           # silent
    ):
        p1 = pred.clone().requires_grad_(True); p2 = pred.clone().requires_grad_(True)
        v1, n1 = _legacy_pesq_term(pq, p1, true)
        v2, n2 = losses.pesq_term(pq, p2, true)
        assert int(n2) == n1
        assert abs(float(v1) - float(v2)) <= 1e-6 * max(1.0, abs(float(v1)))
        if n1:
            v1.backward(); v2.backward()
            assert torch.allclose(p1.grad, p2.grad, rtol=1e-5, atol=1e-9)
            assert (p2.grad[~(true.pow(2).mean(-1).sqrt() > losses.PESQ_MIN_RMS)] == 0).all()


def test_fft_path_gradients_against_float64_reference():
    c = _speech()
    x = (c + 0.02 * torch.randn(32000, generator=torch.Generator().manual_seed(2)))[None]
    pq32 = losses._pesq_module()
    x32 = x.clone().requires_grad_(True)
    v32, _ = losses.pesq_term(pq32, x32, c[None], fft_filters=True)
    v32.backward()
    pq64 = losses._pesq_module().double()
    x64 = x.double().clone().requires_grad_(True)
    v64, _ = losses.pesq_term(pq64, x64, c[None].double(), fft_filters=True)
    v64.backward()
    assert abs(float(v32) - float(v64)) <= 1e-3 * abs(float(v64))
    rel = (x32.grad.double() - x64.grad).norm() / x64.grad.norm()
    assert float(rel) <= 1e-2
    # and the torchaudio path's value is within the float32 recursion's own error of it
    v_ta, _ = losses.pesq_term(losses._pesq_module(), x.clone(), c[None])
    assert abs(float(v_ta) - float(v64)) <= 0.05 * abs(float(v64)) + 1e-3


def test_patch_is_scoped():
    import torch_pesq.loss as tpl
    before = tpl.lfilter
    pq = losses._pesq_module()
    losses.pesq_term(pq, _speech()[None], _speech()[None], fft_filters=True)
    assert tpl.lfilter is before
    with pytest.raises(ValueError):
        losses.FELoss(pesq_filters="iir")
