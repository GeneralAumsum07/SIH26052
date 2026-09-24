import sys
from pathlib import Path

import onnx
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import graph_gate  # noqa: E402
from vaani import export as E  # noqa: E402

torch.set_num_threads(1)


@pytest.mark.parametrize("tier", ["mini", "mid"])
def test_export_passes_g2_with_parity(tier, tmp_path):
    m = E.fe_untrained(tier, seed=3)
    rep = E.export_fe(m, tmp_path / f"{tier}.onnx", streams=2, hops=120, seed=3)
    assert rep["parity"]["pass"] and rep["parity"]["ort_vs_torch_max_abs"] <= 1e-5
    g = graph_gate.gate(rep["onnx"], E.fe_untrained(tier, seed=3), streams=2, hops=120, seed=3)
    assert g["pass"], g["failed"]
    s = g["structure"]
    assert s["loop_nodes"] == 0 and s["scatternd"] == 0 and s["folded_nodes"] < 250
    assert g["parity"]["stream_vs_offline_max_abs"] <= 1e-5


def test_export_io_is_static_named_and_flat(tmp_path):
    m = E.fe_untrained("mini", seed=0)
    rep = E.export_fe(m, tmp_path / "mini.onnx", parity=False)
    g = onnx.load(rep["folded"]).graph
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in list(g.input) + list(g.output)}
    assert shapes == {"spec": [1, 4, 257], "valid": [1, 1], "state": [1, m.state_size],
                      "spec_out": [1, 2, 257], "state_out": [1, m.state_size]}
    assert rep["opset"] == 17 and rep["state_bytes"] == 3072


def test_ablation_arms_export_without_loops(tmp_path):
    for i, kw in enumerate([dict(inputs="p"), dict(inputs="pr_pld", mask="bounded", df_taps=3)]):
        m = E.fe_untrained({"tier": "mini", **kw}, seed=1)
        rep = E.export_fe(m, tmp_path / f"arm{i}.onnx", streams=1, hops=80, seed=1)
        assert rep["parity"]["pass"]
        s = graph_gate.structure(rep["onnx"])
        assert s["loop_nodes"] == 0 and s["scatternd"] == 0 and s["symbolic_dims"] == 0


def test_r7_graph_fails_g2_as_documented():
    p = ROOT / "deploy/r7/cascade.onnx"
    if not p.exists():
        pytest.skip("r7 graph not present")
    g = graph_gate.gate(p)
    assert not g["pass"] and {"loops", "scatternd", "nodes"} <= set(g["failed"])
    assert g["structure"]["ops"]["GRU"] == 14 and g["structure"]["scatternd"] == 18
