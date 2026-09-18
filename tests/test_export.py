import torch
from vaani import export
from vaani.models.vaani_net import VaaniNet


def test_export_parity(tmp_path):
    ck = tmp_path / "m.pt"
    torch.save({"model": VaaniNet().state_dict(), "config": {"model": "vaani", "controller_on": True}, "step": 0}, ck)
    onnx = export.export(ck, tmp_path / "m.onnx")
    r = export.parity_and_timing(ck, onnx, seconds=1)
    assert r["max_abs_err"] < 1e-4 and r["ms_per_frame_mean"] > 0
