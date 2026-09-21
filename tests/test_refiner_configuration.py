import pytest
import torch

from vaani.models.cascade import FrozenCascade
from vaani.models.residual_refiner import ResidualRefiner, init_refine_cache
from vaani.models.vaani_net import VaaniNet


def _specs(frames=8, seed=9):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(1, 257, frames, 2, generator=generator) * 0.1 for _ in range(3)]


def _make_nonzero(module, seed=4):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
    return module


def test_nondefault_refiner_matches_frame_steps_and_cache_shape():
    model = _make_nonzero(ResidualRefiner(hidden=24, past=4, scale=0.5))
    primary, reference, enhanced = _specs()
    expected = model(primary, reference, enhanced)
    cache = init_refine_cache(hidden=24, past=4)
    actual = []
    for frame in range(primary.shape[2]):
        output, cache = model.step(
            primary[:, :, frame:frame + 1],
            reference[:, :, frame:frame + 1],
            enhanced[:, :, frame:frame + 1],
            cache,
        )
        actual.append(output)
    assert cache.shape == (1, 24, 4, 257)
    torch.testing.assert_close(torch.cat(actual, dim=2), expected, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden": 0},
        {"hidden": 1.5},
        {"past": 0},
        {"past": 1.5},
        {"scale": 0},
        {"scale": float("inf")},
    ],
)
def test_invalid_refiner_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ResidualRefiner(**kwargs)


def test_cascade_checkpoint_records_and_restores_refiner_configuration(tmp_path):
    first = tmp_path / "first.pt"
    torch.save(
        {"model": VaaniNet().state_dict(), "config": {"model": "vaani"}, "step": 0},
        first,
    )
    refiner_cfg = {"hidden": 24, "past": 4, "scale": 0.5}
    cascade, config = FrozenCascade.from_first_stage(first, refiner_cfg=refiner_cfg)
    assert config["refiner_cfg"] == refiner_cfg
    restored = FrozenCascade.from_config(config)
    assert restored.refiner.c0.out_channels == 24
    assert restored.refiner.c1.kernel_size == (5, 3)
    assert restored.refiner.scale == 0.5
    assert cascade.refiner.state_dict().keys() == ResidualRefiner(hidden=24, past=4, scale=0.5).state_dict().keys()


def test_legacy_cascade_configuration_uses_original_refiner_defaults():
    cascade = FrozenCascade.from_config({"model": "vaani_cascade"})
    assert cascade.refiner.c0.out_channels == 16
    assert cascade.refiner.c1.kernel_size == (3, 3)
    assert cascade.refiner.scale == 0.25
