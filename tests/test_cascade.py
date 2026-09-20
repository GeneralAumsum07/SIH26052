import copy

import torch

from vaani import export
from vaani.models.cascade import MODEL_NAME, FrozenCascade
from vaani.models.vaani_net import VaaniNet

MC = dict(df_order=3, film=False, coh=True)   # the r3/e32 architecture


def _first_stage(tmp_path, seed=0):
    torch.manual_seed(seed); m = VaaniNet(**MC)
    with torch.no_grad(): m.df.conv.weight.normal_(0, 0.05); m.df.conv.bias.normal_(0, 0.05)   # live taps, not the zero init
    p = tmp_path / "first.pt"
    torch.save({"model": m.state_dict(), "config": {"model": "vaani", "controller_on": True, "dsp": {"nlms_mu": 0.1}, "model_cfg": MC}, "step": 0}, p)
    return p


def _inputs(T=25, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 257, T, 6, generator=g) * 0.1, torch.randn(1, T, 18, generator=g)


def test_zero_init_cascade_is_the_first_stage(tmp_path):
    c, cfg = FrozenCascade.from_first_stage(_first_stage(tmp_path))
    assert cfg["model"] == MODEL_NAME and cfg["controller_on"] is True and cfg["dsp"] == {"nlms_mu": 0.1}
    assert cfg["first_stage"]["config"]["model_cfg"] == MC and len(cfg["first_stage"]["sha256"]) == 64
    spec, f = _inputs()
    with torch.no_grad(): torch.testing.assert_close(c(spec, f), c.first(spec, f))


def test_first_stage_frozen_through_training_steps(tmp_path):
    c, _ = FrozenCascade.from_first_stage(_first_stage(tmp_path)); c.train()
    assert not c.first.training and all(not p.requires_grad for p in c.first.parameters())
    before = {k: v.clone() for k, v in c.first.state_dict().items()}   # parameters and buffers (BN running stats)
    opt = torch.optim.Adam([p for p in c.parameters() if p.requires_grad], lr=1e-2)
    for seed in (1, 2):
        spec, f = _inputs(seed=seed); loss = (c(spec, f) - spec[..., 0:2]).pow(2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    after = c.first.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert c.refiner.c2.weight.abs().sum() > 0   # the refiner did move


def test_cascade_export_parity_with_nonzero_refiner(tmp_path):
    c, cfg = FrozenCascade.from_first_stage(_first_stage(tmp_path))
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():   # identity-initialised refiner would hide a broken refine_cache
        for p in c.refiner.parameters(): p.copy_(torch.randn(p.shape, generator=g) * 0.1)
    ck = tmp_path / "cascade.pt"; torch.save({"model": c.state_dict(), "config": cfg, "step": 0}, ck)
    onnx = export.export(ck, tmp_path / "tier46" / "cascade.onnx")
    import onnxruntime as ort
    s = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    assert [i.name for i in s.get_inputs()] == export.IN_NAMES + ["refine_cache"]
    assert [o.name for o in s.get_outputs()] == export.OUT_NAMES + ["refine_cache_out"]
    assert [i.shape for i in s.get_inputs()][-1] == [1, 16, 2, 257]
    r = export.parity_and_timing(ck, onnx, seconds=1)
    assert r["max_abs_err"] < 1e-4
    # the checkpoint alone rebuilds the cascade: no external first-stage file involved
    c2 = FrozenCascade.from_config(cfg); c2.load_state_dict(torch.load(ck, weights_only=True)["model"])
    spec, f = _inputs()
    with torch.no_grad(): torch.testing.assert_close(c2(spec, f), c(spec, f))
