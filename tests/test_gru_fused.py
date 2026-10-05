"""Fused FP32 GRU (plan Task 4b): the reference decomposition on any device, the Triton kernels on CUDA."""
import pytest
import torch

from vaani.models import gru_fused as G
from vaani.models import vaani_fe as V
from vaani import audio_contract as ac

TS = (1, 251, 501, 668, 748, 1001)


def _pair(seed=0, composed=False):
    torch.manual_seed(seed)
    rnn = torch.nn.GRU(24, 24, batch_first=True)
    if composed:   # over-parameterized weights: products of two factors
        m = V.build("mini", audio_contract=ac.ARM_A_IDS[0], overparam=True, **V.MINI_P["p18"])
        rnn = m.blocks[0].rnn
    return rnn


def _check(fn, rnn, t, b=6, device="cpu"):
    prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        rnn = rnn.to(device)
        x = torch.randn(b, t, rnn.input_size, device=device, generator=None) * 0.5
        x1 = x.clone().requires_grad_(True); x2 = x.clone().requires_grad_(True)
        ref = rnn(x1)[0]
        got = fn(rnn, x2)
        # 1e-5: triton-windows on sm_120 measured a flat ~7e-6 for T=1..1001 (rounding, not recurrence drift)
        assert ((got - ref).abs().max() / ref.abs().max()).item() <= 1e-5
        g = torch.randn_like(ref)
        params = [p for p in rnn.parameters() if p.requires_grad]
        r1 = torch.autograd.grad(ref, [x1, *params], g, allow_unused=True)
        r2 = torch.autograd.grad(got, [x2, *params], g, allow_unused=True)
        for a, bb in zip(r1, r2):
            if a is None:
                assert bb is None
                continue
            assert ((a - bb).norm() / a.norm().clamp(min=1e-30)).item() <= 1e-5
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev


@pytest.mark.parametrize("t", TS)
def test_reference_matches_nn_gru(t):
    _check(G.gru_reference, _pair(t), t, b=3 if t > 500 else 6)


def test_reference_with_composed_weights():
    _check(G.gru_reference, _pair(1, composed=True), 251, b=4)


def test_use_fused_gru_reference_keeps_state_dict_and_causality():
    torch.manual_seed(0)
    m = V.build("mini", audio_contract=ac.ARM_A_IDS[0], fp32_islands=True, **V.MINI_P["p18"]).eval()
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    x = torch.randn(2, 257, 20, 4) * 0.1
    ref = m(x, None, torch.ones(2, 20))
    G.use_fused_gru(m, "reference")
    got = m(x, None, torch.ones(2, 20))
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)
    assert m.state_dict().keys() == sd.keys()
    y = x.clone(); y[:, :, 12:] = torch.randn(2, 257, 8, 4)
    assert torch.equal(m(y, None, torch.ones(2, 20))[:, :, :12], got[:, :, :12])   # causal


def test_fused_refuses_without_cuda():
    if G.fused_available("cuda") and torch.cuda.is_available():
        pytest.skip("CUDA and triton present")
    with pytest.raises(RuntimeError, match="gru_kernel: cudnn"):
        G.gru_fused(torch.nn.GRU(24, 24, batch_first=True), torch.randn(2, 3, 24))


def test_fused_route_ema_copy_uses_its_own_weights():
    from vaani.train import EMA
    torch.manual_seed(0)
    m = V.build("mini", audio_contract=ac.ARM_A_IDS[0], **V.MINI_P["p18"])
    G.use_fused_gru(m, "reference")
    shadow = EMA(m, .999).model
    x = torch.randn(2, 5, m.blocks[0].rnn.input_size)
    before = shadow.blocks[0].rnn(x)[0].clone()
    with torch.no_grad():
        m.blocks[0].rnn.weight_ih_l0.add_(1.)
    assert torch.equal(shadow.blocks[0].rnn(x)[0], before), "EMA forward must not close over the live GRU"


@pytest.mark.skipif(not (torch.cuda.is_available() and G.triton is not None), reason="needs CUDA and triton")
@pytest.mark.parametrize("t", TS)
@pytest.mark.parametrize("composed", [False, True])
def test_triton_kernel_matches_nn_gru(t, composed):
    _check(G.gru_fused, _pair(t, composed), t, b=512, device="cuda")


@pytest.mark.skipif(not (torch.cuda.is_available() and G.triton is not None), reason="needs CUDA and triton")
@pytest.mark.parametrize("t", (251, 501))
@pytest.mark.parametrize("h", (40, 64))
def test_triton_kernel_mid_large_hidden(t, h):
    # Mid/Large GRU widths pad to HP=64, where the bwd kernel's default pipelining overflowed sm_120 shared memory
    torch.manual_seed(h)
    _check(G.gru_fused, torch.nn.GRU(h, h, batch_first=True), t, b=512, device="cuda")
