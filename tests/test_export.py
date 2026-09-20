import torch
from vaani import export
from vaani.models.vaani_net import VaaniNet


def test_export_parity(tmp_path):
    ck = tmp_path / "m.pt"
    torch.save({"model": VaaniNet().state_dict(), "config": {"model": "vaani", "controller_on": True}, "step": 0}, ck)
    onnx = export.export(ck, tmp_path / "m.onnx")
    r = export.parity_and_timing(ck, onnx, seconds=1)
    assert r["max_abs_err"] < 1e-4 and r["ms_per_frame_mean"] > 0


def test_export_parity_r3_architecture(tmp_path):
    # df_order 3 + coherence: the caches that only exist in the r3 signature must round-trip through ORT
    ck = tmp_path / "r3.pt"
    mc = dict(df_order=3, film=False, coh=True)
    m = VaaniNet(**mc)
    with torch.no_grad():  # zero-init taps would hide a broken df_cache path; give them weight
        m.df.conv.weight.normal_(0, 0.05); m.df.conv.bias.normal_(0, 0.05)
    torch.save({"model": m.state_dict(), "config": {"model": "vaani", "controller_on": True, "model_cfg": mc}, "step": 0}, ck)
    onnx = export.export(ck, tmp_path / "r3.onnx")
    r = export.parity_and_timing(ck, onnx, seconds=1)
    assert r["max_abs_err"] < 1e-4 and r["ms_per_frame_mean"] > 0
