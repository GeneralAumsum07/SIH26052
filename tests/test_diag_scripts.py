"""Smoke tests for the read-only diagnostic scripts: they must run end to end on a
tiny rendered set so a refactor of the pipeline or RenderedDataset cannot silently
break the tools the review-fix plan gates its retrain decisions on."""
import sys

import pytest
import torch

from tests.test_metrics import _mini_eval_set
from vaani.train import build_model


def _run(monkeypatch, capsys, mod, argv):
    monkeypatch.setattr(sys, "argv", [mod.__name__, *argv])
    mod.main()
    return capsys.readouterr().out


@pytest.fixture
def evalroot(tmp_path):
    return str(_mini_eval_set(tmp_path))


@pytest.fixture
def vaani_ckpt(tmp_path):
    ck = tmp_path / "best.pt"
    torch.save({"model": build_model("vaani").state_dict(), "config": {"model": "vaani", "controller_on": True}, "step": 0}, ck)
    return str(ck)


def test_ceiling_analysis_runs(monkeypatch, capsys, evalroot):
    from scripts import ceiling_analysis
    out = _run(monkeypatch, capsys, ceiling_analysis, ["--eval-root", evalroot, "--per-bucket", "1"])
    assert "architecture ceiling" in out and "stationary_0" in out


def test_diag_controller_runs(monkeypatch, capsys, evalroot):
    from scripts import diag_controller
    out = _run(monkeypatch, capsys, diag_controller, ["--eval-root", evalroot, "--per-bucket", "1"])
    assert "erle_dB" in out and "OVERALL" in out


def test_mask_phase_probe_runs(monkeypatch, capsys, evalroot, vaani_ckpt):
    from scripts import mask_phase_probe
    out = _run(monkeypatch, capsys, mask_phase_probe, ["--system", f"ckpt:{vaani_ckpt}", "--eval-root", evalroot, "--per-bucket", "1"])
    assert "stationary_0" in out


def test_diag_conditioning_runs(monkeypatch, capsys, evalroot, vaani_ckpt):
    from scripts import diag_conditioning
    out = _run(monkeypatch, capsys, diag_conditioning, ["--ckpt", vaani_ckpt, "--eval-root", evalroot, "--n", "1"])
    assert "feats zeroed" in out
