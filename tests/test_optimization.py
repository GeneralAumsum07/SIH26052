"""Clauses 14a/14b/14c: the export/quantize/prune path, on throwaway checkpoints.

Nothing here needs the trained Tier 4.6 checkpoint, so these run in a clean clone.
"""
import numpy as np
import pytest
import torch

from vaani import export, prune, quantize
from vaani.models.vaani_net import VaaniNet, init_caches


def _checkpoint(tmp_path, name="m.pt", **model_cfg):
    m = VaaniNet(**model_cfg)
    ck = tmp_path / name
    torch.save({"model": m.state_dict(),
                "config": {"model": "vaani", "controller_on": True, "model_cfg": model_cfg}, "step": 0}, ck)
    return ck


def test_zero_caches_reads_the_streaming_signature_off_the_graph(tmp_path):
    # The eval and quantize paths build caches from the graph alone, with no checkpoint; that must
    # agree with the module's own init_caches or every optimized run silently starts mid-stream.
    onnx = export.export(_checkpoint(tmp_path), tmp_path / "m.onnx")
    names, values = export.zero_caches(export.load_session(onnx))
    expected = init_caches()
    assert names == export.IN_NAMES[2:]
    assert [v.shape for v in values] == [tuple(c.shape) for c in expected]
    assert all(not v.any() for v in values)


def test_quantized_graph_keeps_the_streaming_signature(tmp_path):
    # An INT8 graph the embedded loop cannot feed is worthless however small it is: same input and
    # output names, same static shapes, so deploy/CONTRACT.md still describes it.
    onnx = export.export(_checkpoint(tmp_path), tmp_path / "m.onnx")
    int8 = quantize.quantize(onnx, tmp_path / "m.int8.onnx")
    a, b = export.load_session(onnx), export.load_session(int8)
    assert [(i.name, i.shape) for i in a.get_inputs()] == [(i.name, i.shape) for i in b.get_inputs()]
    assert [o.name for o in a.get_outputs()] == [o.name for o in b.get_outputs()]


def test_quantization_report_carries_the_evidence_for_a_size_claim(tmp_path):
    onnx = export.export(_checkpoint(tmp_path), tmp_path / "m.onnx")
    int8 = quantize.quantize(onnx, tmp_path / "m.int8.onnx")
    r = quantize.report(onnx, int8, seconds=1, repeats=1)
    # initializer bytes are what makes a "the file got bigger" result explicable rather than odd
    assert r["fp32"]["initializer_bytes"] > 0 and r["int8"]["initializer_bytes"] > 0
    assert r["int8"]["nodes"] > r["fp32"]["nodes"]  # dynamic quantization inserts scale/zero-point nodes
    assert np.isfinite(r["relative_err_vs_fp32_graph"])
    assert r["fp32"]["ms_per_frame_mean"] > 0 and r["int8"]["ms_per_frame_mean"] > 0


def test_pruning_never_touches_the_fixed_erb_transform(tmp_path):
    # The ERB matrices are more than half the weight elements in prunable module types. Letting the
    # budget fall on them would corrupt the band split and flatter the sparsity figure.
    m, _ = prune.load_checkpoint(_checkpoint(tmp_path))
    before = m.erb.erb_fc.weight.detach().clone(), m.erb.ierb_fc.weight.detach().clone()
    prune.global_magnitude_prune(m, 0.9)
    assert torch.equal(m.erb.erb_fc.weight, before[0]) and torch.equal(m.erb.ierb_fc.weight, before[1])
    assert not any("erb_fc" in q for _, _, q in prune.prunable_tensors(m))


def test_zero_sparsity_is_an_unmodified_control(tmp_path):
    ck = _checkpoint(tmp_path)
    m, original = prune.load_checkpoint(ck)
    stats = prune.global_magnitude_prune(m, 0.0)
    assert stats["newly_zeroed_weights"] == 0
    for k, v in m.state_dict().items():
        assert torch.equal(v, original["model"][k])


@pytest.mark.parametrize("sparsity", [0.1, 0.5])
def test_global_pruning_hits_the_requested_fraction_of_learned_weights(tmp_path, sparsity):
    m, _ = prune.load_checkpoint(_checkpoint(tmp_path))
    stats = prune.global_magnitude_prune(m, sparsity)
    assert stats["achieved_sparsity"] == pytest.approx(sparsity, abs=1e-3)
    # achieved_sparsity is the state of the weights; newly_zeroed is this call's work, and any
    # gap between them is weights that were already zero rather than sparsity pruning produced.
    assert stats["newly_zeroed_weights"] + stats["pre_existing_zeros"] == stats["zeroed_weights"]
    # The headline sparsity is over prunable weights only; over every parameter it is necessarily lower,
    # and the report must never quote the first number as if it were the second.
    assert stats["sparsity_over_all_parameters"] < stats["achieved_sparsity"]


def test_pruning_a_gru_actually_changes_its_output(tmp_path):
    """RNN modules cache `_flat_weights` at construction. Without prune._refresh_rnn the module keeps
    running on the pre-prune tensors, the state_dict looks correctly sparse, and the sweep measures
    nothing -- a silent false negative, so it gets its own test."""
    m, _ = prune.load_checkpoint(_checkpoint(tmp_path))
    spec, feats = torch.randn(1, 257, 4, 6) * 0.1, torch.randn(1, 4, 18)
    with torch.no_grad():
        before = m(spec, feats).clone()
        prune.global_magnitude_prune(m, 0.9)
        after = m(spec, feats)
    assert not torch.allclose(before, after)
    gru = next(mod for mod in m.modules() if isinstance(mod, torch.nn.GRU))
    assert gru._flat_weights[0] is gru.weight_ih_l0  # the stale-reference failure mode itself


def test_onnx_eval_backend_reproduces_the_checkpoint(tmp_path):
    # `onnx:<graph>@<ckpt>` exists so an optimized graph can be scored in SNR/STOI/PESQ. On the
    # unquantized graph it must be the same system as `ckpt:`, or every delta measured against it
    # is really an export artefact.
    from vaani.eval import enhance_fn
    ck = _checkpoint(tmp_path)
    onnx = export.export(ck, tmp_path / "m.onnx")
    rng = np.random.default_rng(0)
    mix = (rng.standard_normal((2, 16000)) * 0.05).astype(np.float32)
    a = enhance_fn(f"ckpt:{ck}", device="cpu")(mix)
    b = enhance_fn(f"onnx:{onnx}@{ck}")(mix)
    assert np.abs(a - b).max() < 1e-4


def test_onnx_eval_resets_caches_between_clips(tmp_path):
    # Carrying cache state from one clip into the next would leak an item's history into its
    # neighbour and make results depend on CSV order.
    from vaani.eval import enhance_fn
    ck = _checkpoint(tmp_path)
    onnx = export.export(ck, tmp_path / "m.onnx")
    f = enhance_fn(f"onnx:{onnx}@{ck}")
    rng = np.random.default_rng(1)
    mix = (rng.standard_normal((2, 8000)) * 0.05).astype(np.float32)
    other = (rng.standard_normal((2, 8000)) * 0.2).astype(np.float32)
    first = f(mix)
    f(other)
    assert np.array_equal(first, f(mix))
