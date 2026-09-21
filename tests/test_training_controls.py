import numpy as np
import pytest
import torch

from vaani import losses
from vaani.data.mixer import MixConfig, sample_snr
from vaani.training_controls import (
    cosine_lr_multiplier,
    make_schedule_config,
    should_stop_for_patience,
    validate_resume_schedule,
)


def test_uniform_snr_preserves_legacy_rng_draw_exactly():
    cfg = MixConfig(snr_range=(-7.0, 13.0))
    actual_rng = np.random.default_rng(8421)
    legacy_rng = np.random.default_rng(8421)

    assert sample_snr(actual_rng, cfg) == float(legacy_rng.uniform(-7.0, 13.0))
    # A later mixer branch must see the same stream, not merely the same SNR.
    assert actual_rng.random() == legacy_rng.random()


def test_low_triangular_snr_is_deterministic_and_biases_toward_low_snr():
    cfg = MixConfig(snr_range=(-10.0, 10.0), snr_sampling="triangular_low")
    a = np.random.default_rng(9)
    b = np.random.default_rng(9)
    draws_a = np.array([sample_snr(a, cfg) for _ in range(4000)])
    draws_b = np.array([sample_snr(b, cfg) for _ in range(4000)])

    assert np.array_equal(draws_a, draws_b)
    assert -10.0 <= draws_a.min() and draws_a.max() <= 10.0
    assert draws_a.mean() < -2.5  # theoretical mean is -3.33 dB


def test_stratified_snr_honours_configured_bin_weights():
    cfg = MixConfig(
        snr_range=(-10.0, 10.0),
        snr_sampling="stratified",
        snr_bins=(-10.0, -5.0, 0.0, 10.0),
        snr_weights=(0.0, 1.0, 0.0),
    )
    rng = np.random.default_rng(2)
    draws = [sample_snr(rng, cfg) for _ in range(100)]

    assert all(-5.0 <= value <= 0.0 for value in draws)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"snr_sampling": "unknown"}, "snr_sampling"),
        ({"snr_sampling": "stratified", "snr_bins": (0.0,)}, "snr_bins"),
        (
            {"snr_sampling": "stratified", "snr_bins": (-10.0, 0.0, 15.0), "snr_weights": (1.0,)},
            "snr_weights",
        ),
    ],
)
def test_snr_sampling_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        sample_snr(np.random.default_rng(0), MixConfig(**kwargs))


def test_cosine_uses_the_full_resolved_budget_and_clamps_after_it():
    assert cosine_lr_multiplier(0, warmup_steps=2, total_steps=10) == pytest.approx(0.5)
    assert cosine_lr_multiplier(1, warmup_steps=2, total_steps=10) == pytest.approx(
        0.5 * (1.0 + np.cos(np.pi / 10.0))
    )
    assert cosine_lr_multiplier(10, warmup_steps=2, total_steps=10) == pytest.approx(0.0)
    assert cosine_lr_multiplier(11, warmup_steps=2, total_steps=10) == pytest.approx(0.0)


def test_resume_rejects_a_changed_cosine_budget_or_warmup():
    saved = make_schedule_config(epochs=8, steps_per_epoch=20, warmup_steps=10)
    validate_resume_schedule(saved, make_schedule_config(epochs=8, steps_per_epoch=20, warmup_steps=10))

    with pytest.raises(RuntimeError, match="cosine schedule"):
        validate_resume_schedule(saved, make_schedule_config(epochs=16, steps_per_epoch=20, warmup_steps=10))
    with pytest.raises(RuntimeError, match="cosine schedule"):
        validate_resume_schedule(saved, make_schedule_config(epochs=8, steps_per_epoch=20, warmup_steps=11))


def test_patience_uses_resumed_history_and_minimum_improvement():
    history = [
        {"epoch": 0, "val_stoi": 0.70},
        {"epoch": 1, "val_stoi": 0.705},
        {"epoch": 2, "val_stoi": 0.704},
    ]
    assert not should_stop_for_patience(history, patience=3, min_delta=0.01)
    history.append({"epoch": 3, "val_stoi": 0.706})
    assert should_stop_for_patience(history, patience=3, min_delta=0.01)
    assert not should_stop_for_patience(history, patience=None, min_delta=0.01)


def test_snr_clamp_fraction_is_callable_and_loss_records_latest_batch():
    target = torch.ones(4, 100)
    prediction = target.clone()
    prediction[0] += 1.0
    prediction[1] += 0.1
    prediction[2] += 0.01
    prediction[3] += 0.001

    assert losses.snr_clamp_fraction(prediction, target, snr_max_db=30.0).item() == pytest.approx(0.5)
    fn = losses.HybridLoss(w_snr=0.2, snr_max_db=30.0)
    fn.snr_term(prediction, target)
    assert fn.last_snr_clamp_fraction.item() == pytest.approx(0.5)


def test_loss_factory_allows_refiner_to_use_config_instead_of_a_constant():
    fn = losses.build_loss("hybrid", {"w_complex": 12.0, "w_mag": 34.0, "w_snr": 0.8})
    assert isinstance(fn, losses.HybridLoss)
    assert (fn.w_complex, fn.w_mag, fn.w_snr) == (12.0, 34.0, 0.8)
