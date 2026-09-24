import torch
from vaani import export
from vaani.models.vaani_net import VaaniNet


def test_refiner_cost_counts_full_frequency_convolution():
    from vaani.models.residual_refiner import ResidualRefiner
    m = ResidualRefiner().eval()
    spec = torch.zeros(1, 257, 1, 2)
    cost = export.layer_macs(m, (spec, spec, spec))
    assert sum(cost.values()) == 633248
    assert cost["c1"] == 592128


def test_gru_cost_includes_both_directions_and_recurrent_matrices():
    # Two time steps, batch three, two directions: each gate uses input and hidden matrices.
    m = torch.nn.GRU(4, 5, batch_first=True, bidirectional=True)
    assert sum(export.layer_macs(m, (torch.zeros(3, 2, 4),)).values()) == 3 * 2 * 2 * 3 * 5 * (4 + 5)


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


def test_tracked_r7_checkpoint_reexports_to_the_shipped_graph(tmp_path):
    # a clone has only results_r2/runs/, so the default must load from there with the backbone embedded
    import hashlib, onnx
    from vaani.models import cascade
    ck = export.SHIPPING_CKPT
    assert hashlib.sha256(open(ck, "rb").read()).hexdigest().startswith("121f0c3d")
    v, _ = export._load_batch_model(ck)
    assert isinstance(v, cascade.FrozenCascade)
    got = onnx.load(export.export(ck, tmp_path / "r7.onnx")); shipped = onnx.load(export.SHIPPING_ONNX)
    # bytes differ across torch versions (folded conv weights in the last ulp); topology must not
    assert [n.SerializeToString() for n in got.graph.node] == [n.SerializeToString() for n in shipped.graph.node]
    r = export.parity_and_timing(ck, export.SHIPPING_ONNX, seconds=1)
    assert r["max_abs_err"] < 1e-4
