"""Benchmark fidelity and resource bounds, not timing thresholds on a shared host."""
import pytest
from tests.test_low_delay_training import _cfg
from tests.test_train_smoke import _tiny


def test_loader_benchmark_uses_the_training_audio_contract(tmp_path):
    from scripts.bench_loader import build_dataset
    cfg = _cfg(tmp_path, _tiny(tmp_path))
    ds = build_dataset(cfg)
    assert ds.contract.audio_contract_id == cfg["model_cfg"]["audio_contract"]


def test_failed_or_unverified_benchmark_cannot_win():
    from scripts.bench_r8_training import winners
    good = dict(config="c", batch=32, crop_s=4., status="passed", parity_passed=True, median_ms=10.)
    rows = [good, dict(good, status="failed", median_ms=1.), dict(good, parity_passed=False, median_ms=.1)]
    assert list(winners(rows).values()) == [good]


def test_graph_rejects_partial_batches_before_training():
    from vaani.train import perf_settings
    cfg = dict(loss="fe", batch_size=32, data={"epoch_len": 33}, perf={"numerics": {"cuda_graph": True}})
    with pytest.raises(ValueError, match="divisible"):
        perf_settings(cfg)


def test_graph_rejects_broadcast_target_and_input_count():
    import torch
    from vaani.train_graph import GraphedStep
    gs = object.__new__(GraphedStep)
    gs.static_in = [torch.zeros(2, 3)]
    gs.static_target, gs.static_clean = torch.zeros(2, 3), torch.zeros(2, dtype=torch.bool)
    with pytest.raises(RuntimeError, match="target"):
        gs._copy_in([torch.zeros(2, 3)], torch.zeros(1, 3), torch.zeros(2, dtype=torch.bool))
    with pytest.raises(RuntimeError, match="input count"):
        gs._copy_in([], gs.static_target, gs.static_clean)


def test_worker_budget_accounts_for_all_scorer_processes():
    from scripts.r8_resources import budget
    b = budget(cpus=256, memory_gb=500, lanes=4, reserve=8,
               train_threads=2, scorer_procs=2, scorer_threads=4, screen_workers=2)
    assert b["total_cpu_budget"] <= 256
    assert b["workers_per_lane"] == 47
    small = budget(cpus=16, memory_gb=8, lanes=4, reserve=4,
                   train_threads=2, scorer_procs=4, scorer_threads=8, screen_workers=4)
    assert small["fits"] is False
