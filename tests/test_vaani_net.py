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


def test_legacy_checkpoint_with_persisted_feat_scale_loads():
    from vaani.models.vaani_net import VaaniNet
    v = VaaniNet(); sd = v.state_dict(); sd["encoder.feat_scale"] = v.encoder.feat_scale.clone()
    VaaniNet().load_state_dict(sd)


# --- r3 architecture flags (plan 2.4 / 2.5): deep-filter head, coherence channel, FiLM off ---
R3 = dict(df_order=3, film=False, coh=True)


def test_r3_flags_shapes_and_budget():
    m = VaaniNet(**R3)
    assert sum(p.numel() for p in m.parameters()) <= vaani_net.MAX_PARAMS
    assert m.encoder.film is None and m.encoder.en_convs[0].conv.weight.shape[1] == 30
    spec = torch.randn(2, 257, 20, 6); f = torch.randn(2, 20, 18)
    assert m(spec, f).shape == (2, 257, 20, 2)


def test_r3_warm_start_from_r2_checkpoint_is_exact_at_step_0():
    """Zero taps + zero coherence slice + no FiLM: an r2 checkpoint loaded into the r3 shape must produce
    the r2 output (FiLM was inert in r2 by measurement; here it is exactly inert because the r2 FiLM
    weights are dropped and compared against an r2 model whose FiLM is zero)."""
    torch.manual_seed(1)
    r2 = VaaniNet.from_pretrained_gtcrn(CKPT).eval()
    with torch.no_grad():
        torch.nn.init.normal_(r2.encoder.en_convs[0].conv.weight[:, 9:], std=0.1)  # ref/n_hat slices live
    r3 = VaaniNet(**R3).warm_start(r2.state_dict()).eval()
    spec = torch.randn(1, 257, 30, 6); f = torch.randn(1, 30, 18)
    with torch.no_grad():
        assert torch.allclose(r2(spec, f), r3(spec, f), atol=1e-5)
    assert torch.all(r3.df.conv.weight == 0) and torch.all(r3.encoder.en_convs[0].conv.weight[:, 27:] == 0)


def test_r3_stream_parity_with_live_taps_and_coherence():
    torch.manual_seed(0)
    v = VaaniNet.from_pretrained_gtcrn(CKPT, **R3).eval(); s = StreamVaaniNet(**R3).eval()
    with torch.no_grad():  # zero-init slices would make parity vacuous
        for t in (v.encoder.en_convs[0].conv.weight[:, 9:], v.df.conv.weight, v.df.conv.bias):
            torch.nn.init.normal_(t, std=0.1)
    convert_to_stream(s, v)
    spec = torch.randn(1, 257, 25, 6); f = torch.randn(1, 25, 18)
    with torch.no_grad():
        y = v(spec, f); caches = init_caches("cpu"); outs = []
        for t in range(25):
            o, *caches = s(spec[:, :, t:t + 1], f[:, t:t + 1], *caches); outs.append(o)
    assert torch.allclose(y, torch.cat(outs, 2), atol=1e-3)
    # the taps did something: the output is not the plain-CRM output of the same weights
    with torch.no_grad():
        v.df.conv.weight.zero_(); v.df.conv.bias.zero_()
        assert not torch.allclose(y, v(spec, f), atol=1e-3)


def test_coherence_map_is_one_for_identical_channels_and_causal():
    from vaani.models.vaani_net import coherence_map
    x = torch.randn(1, 257, 40, 2)
    spec = torch.cat([x, x, torch.zeros(1, 257, 40, 2)], dim=-1)
    m, _ = coherence_map(spec)
    assert torch.allclose(m[:, :, 5:], torch.ones_like(m[:, :, 5:]), atol=1e-4)
    spec2 = spec.clone(); spec2[:, :, 20:] = torch.randn_like(spec2[:, :, 20:])
    m2, _ = coherence_map(spec2)
    assert torch.allclose(m[:, :, :20], m2[:, :, :20])  # frames before the change are untouched
