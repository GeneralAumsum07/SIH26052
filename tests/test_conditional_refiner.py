import pytest
import torch
from torch import nn

from vaani.models.conditional_refiner import ConditionalRefinerRuntime
from vaani.models.residual_refiner import ResidualRefiner, init_refine_cache


class _KnownFirstStage(nn.Module):
    def forward(self, spec6, feats):
        return spec6[..., 4:6]


class _TinyCascade(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = _KnownFirstStage()
        self.refiner = ResidualRefiner(hidden=5, past=2, scale=0.4)
        generator = torch.Generator().manual_seed(18)
        with torch.no_grad():
            for parameter in self.refiner.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)


def _clip(frames=7, seed=11):
    generator = torch.Generator().manual_seed(seed)
    spec6 = torch.randn(1, 257, frames, 6, generator=generator) * 0.1
    feats = torch.randn(1, frames, 18, generator=generator)
    return spec6, feats


def test_always_mode_matches_regular_nonzero_refiner_and_resets_each_clip():
    cascade = _TinyCascade()
    runtime = ConditionalRefinerRuntime(cascade, mode="always")
    spec6, feats = _clip()
    first = cascade.first(spec6, feats)
    expected = cascade.refiner(spec6[..., 0:2], spec6[..., 2:4], first)
    torch.testing.assert_close(runtime(spec6, feats), expected, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(runtime(spec6, feats), expected, atol=1e-5, rtol=1e-4)
    assert runtime.stats["fired"] == 14 and runtime.stats["total"] == 14


def test_bypass_mode_returns_the_exact_first_stage_output():
    cascade = _TinyCascade()
    runtime = ConditionalRefinerRuntime(cascade, mode="bypass")
    spec6, feats = _clip()
    expected = cascade.first(spec6, feats)
    actual = runtime(spec6, feats)
    assert torch.equal(actual, expected)
    assert runtime.stats["fired"] == 0 and runtime.stats["total"] == spec6.shape[2]


def test_alternating_skips_keep_c0_history_identical_to_regular_steps():
    cascade = _TinyCascade()
    runtime = ConditionalRefinerRuntime(
        cascade,
        mode="conditional",
        snr_threshold_db=10.0,
        speech_threshold=0.01,
    )
    frames = 6
    enhanced = torch.ones(1, 257, frames, 2) * 0.2
    residual = torch.empty_like(enhanced)
    residual[:, :, 0::2] = 0.01  # high estimated SNR: bypass
    residual[:, :, 1::2] = 0.2   # 0 dB estimated SNR: refine
    primary = enhanced + residual
    reference = torch.linspace(-0.2, 0.2, 257)[None, :, None, None].expand_as(enhanced)
    conditional_cache = init_refine_cache(hidden=5, past=2)
    regular_cache = init_refine_cache(hidden=5, past=2)
    decisions = []
    for frame in range(frames):
        inputs = (
            primary[:, :, frame:frame + 1],
            reference[:, :, frame:frame + 1],
            enhanced[:, :, frame:frame + 1],
        )
        conditional, conditional_cache, fired = runtime.step(*inputs, conditional_cache)
        regular, regular_cache = cascade.refiner.step(*inputs, regular_cache)
        decisions.append(fired)
        torch.testing.assert_close(conditional_cache, regular_cache)
        if fired:
            torch.testing.assert_close(conditional, regular)
        else:
            assert torch.equal(conditional, inputs[2])
    assert decisions == [False, True, False, True, False, True]


def test_policy_can_use_reliability_and_reports_matrix_mac_average():
    cascade = _TinyCascade()
    runtime = ConditionalRefinerRuntime(
        cascade,
        snr_threshold_db=-20.0,
        speech_threshold=0.01,
        reliability_threshold=0.25,
    )
    enhanced = torch.ones(1, 257, 1, 2)
    primary = enhanced + 0.001
    reference = torch.zeros_like(enhanced)
    cache = init_refine_cache(hidden=5, past=2)
    _, cache, fired_high = runtime.step(primary, reference, enhanced, cache, reliability=0.9)
    _, _, fired_low = runtime.step(primary, reference, enhanced, cache, reliability=0.1)
    assert not fired_high and fired_low
    c0 = 257 * 5 * 8
    conditional = 257 * (5 * 5 * 3 * 3 + 2 * 5)
    assert runtime.stats["avg_matrix_macs_per_frame"] == pytest.approx(c0 + conditional / 2)


@pytest.mark.parametrize("kwargs", [{"mode": "sometimes"}, {"crossfade": -0.1}, {"crossfade": 1.1}])
def test_invalid_conditional_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ConditionalRefinerRuntime(_TinyCascade(), **kwargs)
