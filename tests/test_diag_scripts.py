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


def test_ceiling_analysis_runs(monkeypatch, capsys, evalroot, tmp_path):
    import json
    from scripts import ceiling_analysis
    # The fixture renders a `test` split only; ceiling_analysis defaults to `val` behind an explicit guard.
    # Output paths are overridden so a smoke test cannot overwrite the committed results/ artefacts.
    items, agg = tmp_path / "items.csv", tmp_path / "aggregate.json"
    out = _run(monkeypatch, capsys, ceiling_analysis,
               ["--eval-root", evalroot, "--split", "test", "--allow-test", "--per-bucket", "1",
                "--items-out", str(items), "--aggregate-out", str(agg)])
    assert json.loads(out)["aggregate_out"] == str(agg)
    assert any(b["bucket"] == "stationary_0" for b in json.loads(agg.read_text(encoding="utf-8"))["aggregates"])
    assert items.exists()


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


def test_diag_conditioning_r3_runs(monkeypatch, capsys, evalroot, tmp_path):
    from scripts import diag_conditioning
    cfg = {"film": False, "coh": True, "df_order": 3}
    ckpt = tmp_path / "r3.pt"
    torch.save({"model": build_model("vaani", model_cfg=cfg).state_dict(),
                "config": {"model": "vaani", "controller_on": True, "model_cfg": cfg}}, ckpt)
    out = _run(monkeypatch, capsys, diag_conditioning,
               ["--ckpt", str(ckpt), "--eval-root", evalroot, "--n", "1"])
    assert "coh zeroed" in out
    assert "FiLM disabled" in out
    assert "film_shift /" not in out


def test_coherence_ablation_preserves_spectra_and_restores_after_error():
    from scripts.diag_conditioning import zero_coherence
    model = build_model("vaani", model_cfg={"film": False, "coh": True, "df_order": 3}).eval()
    spec = torch.randn(1, 257, 4, 6)
    feats = torch.zeros(1, 4, 18)
    seen = []
    # Observe what SFE actually receives after the diagnostic's pre-hook runs.
    with torch.no_grad(), zero_coherence(model):
        handle = model.sfe.register_forward_pre_hook(lambda _m, args: seen.append(args[0].clone()))
        model(spec, feats)
        handle.remove()
    handle = model.sfe.register_forward_pre_hook(lambda _m, args: seen.append(args[0].clone()))
    with torch.no_grad():
        baseline = model(spec, feats)
    handle.remove()
    torch.testing.assert_close(seen[0][:, :9], seen[1][:, :9], rtol=0, atol=0)
    assert torch.count_nonzero(seen[0][:, 9]) == 0
    assert torch.count_nonzero(seen[1][:, 9]) > 0
    with pytest.raises(RuntimeError, match="probe"):
        with zero_coherence(model):
            raise RuntimeError("probe")
    with torch.no_grad():
        torch.testing.assert_close(model(spec, feats), baseline, rtol=0, atol=0)


def test_diag_conditioning_writes_csv_and_json(monkeypatch, capsys, evalroot, vaani_ckpt, tmp_path):
    import json
    import pandas as pd
    from scripts import diag_conditioning
    stem = tmp_path / "diag" / "cond"
    _run(monkeypatch, capsys, diag_conditioning, ["--ckpt", vaani_ckpt, "--eval-root", evalroot, "--n", "0", "--out", str(stem)])
    df = pd.read_csv(stem.with_suffix(".csv"))
    assert {"variant", "snr_out", "stoi", "pesq_wb"} <= set(df.columns) and df.variant.nunique() > 1
    assert json.loads(stem.with_suffix(".json").read_text())


def test_mask_phase_probe_writes_csv_and_json(monkeypatch, capsys, evalroot, vaani_ckpt, tmp_path):
    import json
    import pandas as pd
    from scripts import mask_phase_probe
    stem = tmp_path / "mask"
    _run(monkeypatch, capsys, mask_phase_probe, ["--system", f"ckpt:{vaani_ckpt}", "--eval-root", evalroot, "--per-bucket", "0", "--out", str(stem)])
    assert len(pd.read_csv(stem.with_suffix(".csv"))) > 0
    assert json.loads(stem.with_suffix(".json").read_text())
