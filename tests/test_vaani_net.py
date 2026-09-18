import time
import pytest
import torch
from vaani.models import vaani_net
from vaani.models.vaani_net import VaaniNet, StreamVaaniNet, init_caches
from vaani.models.gtcrn import GTCRN
from vaani.models.modules.convert import convert_to_stream

CKPT = "vaani/models/checkpoints/model_trained_on_dns3.tar"


def _gtcrn():
    g = GTCRN().eval()
    g.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True)["model"])
    return g


def test_shapes_and_param_budget():
    m = VaaniNet()
    # Counts all parameters incl. frozen ERB banks; excludes BN running buffers (~600, foldable).
    n = sum(p.numel() for p in m.parameters())
    assert n <= vaani_net.MAX_PARAMS, n
    spec = torch.randn(2, 257, 20, 6); f = torch.randn(2, 20, 18)
    assert m(spec, f).shape == (2, 257, 20, 2)


def test_pretrained_init_matches_gtcrn_on_primary_only():
    """Step-0 equivalence: ref/n_hat channels and feats are arbitrary, yet the
    output must equal pretrained GTCRN(primary) because the new slices are zero."""
    g = _gtcrn(); v = VaaniNet.from_pretrained_gtcrn(CKPT).eval()
    torch.manual_seed(0); p = torch.randn(1, 257, 30, 2)
    spec = torch.cat([p, torch.randn(1, 257, 30, 4)], dim=-1)
    with torch.no_grad():
        assert torch.allclose(g(p), v(spec, torch.randn(1, 30, 18)), atol=1e-5)


def test_zero_init_slices_are_trainable():
    v = VaaniNet.from_pretrained_gtcrn(CKPT).train()
    w0 = v.encoder.en_convs[0].conv.weight
    assert torch.all(w0[:, 9:] == 0) and torch.all(v.encoder.film.weight == 0)
    v(torch.randn(2, 257, 12, 6), torch.randn(2, 12, 18)).abs().mean().backward()
    assert w0.grad[:, 9:].abs().sum() > 0 and v.encoder.film.weight.grad.abs().sum() > 0


def test_stream_parity():
    torch.manual_seed(0)
    v = VaaniNet.from_pretrained_gtcrn(CKPT).eval(); s = StreamVaaniNet().eval()
    # Zero-init slices would make parity vacuous; perturb them so FiLM and ref/n_hat paths are exercised.
    with torch.no_grad():
        for t in (v.encoder.en_convs[0].conv.weight[:, 9:], v.encoder.film.weight, v.encoder.film.bias):
            torch.nn.init.normal_(t, std=0.1)
    # Stream conv wrappers nest weights one level deeper; upstream's converter remaps them.
    convert_to_stream(s, v)
    spec = torch.randn(1, 257, 25, 6); f = torch.randn(1, 25, 18)
    with torch.no_grad():
        y = v(spec, f); caches = init_caches("cpu"); outs = []
        for t in range(25):
            o, *caches = s(spec[:, :, t:t + 1], f[:, t:t + 1], *caches); outs.append(o)
    assert torch.allclose(y, torch.cat(outs, 2), atol=1e-3)


def test_cuda_forward_backward_smoke():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    m = VaaniNet().cuda().train()
    spec = torch.randn(4, 257, 63, 6, device="cuda"); f = torch.randn(4, 63, 18, device="cuda")
    y = m(spec, f); y.abs().mean().backward()
    assert y.shape == (4, 257, 63, 2) and torch.isfinite(y).all()


def test_feature_scaling_keeps_film_input_bounded():
    from vaani.models.vaani_net import FEAT_SCALE
    m = VaaniNet()
    assert len(FEAT_SCALE) == 18 and m.encoder.feat_scale.shape == (18,)
    feats = torch.full((1, 4, 18), 40.0)  # dB-range extremes
    x = torch.zeros(1, 16, 4, 129)
    torch.nn.init.ones_(m.encoder.film.weight)
    out = m.encoder._cond(x, feats)
    assert out.abs().max() <= 3.0 * 18 + 1e-6 and torch.isfinite(out).all()
