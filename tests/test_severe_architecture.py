import pytest
import torch

from vaani.models.vaani_net import VaaniNet, StreamVaaniNet, init_caches
from vaani.models.modules.convert import convert_to_stream


@pytest.mark.parametrize("channels", [8, 16, 32])
def test_width_floor_streaming_and_causality(channels):
    torch.manual_seed(19)
    cfg = dict(channels=channels, noise_floor=True, df_order=3, coh=True, film=False)
    m = VaaniNet(**cfg).eval()
    s = StreamVaaniNet(**cfg).eval()
    with torch.no_grad():
        m.df.conv.weight.normal_(0, 0.01)
    convert_to_stream(s, m)
    x = torch.randn(1, 257, 9, 6) * .1
    f = torch.randn(1, 9, 18)
    with torch.no_grad():
        batch = m(x, f)
        cache = init_caches(channels=channels, noise_floor=True)
        frames = []
        for t in range(9):
            y, *cache = s(x[:, :, t:t+1], f[:, t:t+1], *cache)
            frames.append(y)
        torch.testing.assert_close(batch, torch.cat(frames, 2), atol=1e-5, rtol=1e-4)
        later = x.clone(); later[:, :, 5:] += 1
        torch.testing.assert_close(batch[:, :, :5], m(later, f)[:, :, :5])
    assert cache[-1].shape == (1, 2, 257)


def test_noise_floor_warm_start_preserves_existing_checkpoint_output():
    cfg = dict(df_order=3, coh=True, film=False)
    old = VaaniNet(**cfg).eval()
    new = VaaniNet(**cfg, noise_floor=True).warm_start(old.state_dict()).eval()
    x, f = torch.randn(1, 257, 6, 6), torch.randn(1, 6, 18)
    with torch.no_grad():
        torch.testing.assert_close(new(x, f), old(x, f))
    new(x, f).square().mean().backward()
    assert new.encoder.en_convs[0].conv.weight.grad[:, 30:].abs().sum() > 0


def test_invalid_width_and_tracker_rates_rejected():
    for cfg in [dict(channels=7), dict(channels=0), dict(noise_floor_up=0), dict(noise_floor_down=1.1)]:
        with pytest.raises(ValueError):
            VaaniNet(**cfg)


def test_floor_tracker_uses_power_and_releases_faster_than_it_rises():
    from vaani.models.vaani_net import noise_floor_features
    x = torch.zeros(1, 257, 3, 6)
    x[:, :, 0, 0] = 2
    x[:, :, 1, 0] = 4
    features, state = noise_floor_features(x, up=.1, down=.5)
    assert torch.isfinite(features).all()
    assert features[0, 0, 0, 0].item() == pytest.approx(torch.log1p(torch.tensor(4.)).item())
    assert state[0, 0, 0].item() == pytest.approx(2.6)


@pytest.mark.parametrize("cascade", [False, True])
def test_nondefault_onnx_parity(tmp_path, cascade):
    from vaani import export
    from vaani.models.cascade import FrozenCascade
    mc = dict(channels=8, noise_floor=True, coh=True, df_order=3, film=False)
    rc = dict(hidden=24, past=4, scale=.4)
    m = FrozenCascade(mc, rc) if cascade else VaaniNet(**mc)
    if cascade:
        with torch.no_grad():
            m.refiner.c2.weight.normal_(0, .02)
    ck = tmp_path / "model.pt"
    torch.save(dict(model=m.state_dict(), config=dict(model="vaani_cascade" if cascade else "vaani", model_cfg=mc, refiner_cfg=rc)), ck)
    onnx = export.export(ck, tmp_path / "model.onnx")
    result = export.parity_and_timing(ck, onnx, seconds=.16)
    assert result["max_abs_err"] < 1e-4


def test_width_warm_start_rejected():
    with pytest.raises(ValueError, match="scratch"):
        VaaniNet(channels=8).warm_start(VaaniNet().state_dict())


def test_adding_coherence_keeps_existing_floor_feature_slices():
    old = VaaniNet(noise_floor=True, coh=False, film=False).eval()
    new = VaaniNet(noise_floor=True, coh=True, film=False).warm_start(old.state_dict()).eval()
    x, f = torch.randn(1, 257, 5, 6), torch.zeros(1, 5, 18)
    with torch.no_grad():
        torch.testing.assert_close(old(x, f), new(x, f), atol=1e-5, rtol=1e-4)
