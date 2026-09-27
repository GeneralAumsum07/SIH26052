"""Unified VaaniFE evaluation routes (plan Task 3): contract dispatch, validity, alignment, and no legacy transform or
NLMS pipeline on any low-delay route."""
import numpy as np
import pytest
import torch

from vaani import audio_contract as ac, resampler as rs
from vaani.data.dataset import front_end
from vaani.dsp import low_delay_stft as ld, pipeline, stft
from vaani.enhance_low_delay import (enhance_fe, enhance_low_delay, filter_reference, forward_fe_batch,
                                     make_fe_system)
from vaani.models import vaani_fe as V

A = ac.ARM_A_IDS[0]
DSP = {"limiter": True, "limiter_kernel": "numba", "ref_policy": {"nlms": True, "absent": "freeze", "ramp_samples": 3072}}
DSP_C0 = {"limiter": True, "ref_policy": {"nlms": True, "absent": "freeze", "ramp_frames": 12}}
LD_CFG = {"model": "vaani_fe", "controller_on": False, "dsp": DSP,
          "model_cfg": {"tier": "mini", "audio_contract": A, **{k: list(v) if isinstance(v, tuple) else v
                                                                 for k, v in V.MINI_P["p18"].items()}}}
C0_CFG = {"model": "vaani_fe", "controller_on": False, "dsp": DSP_C0, "model_cfg": {"tier": "mini"}}


class IdentityFE(torch.nn.Module):
    """Returns its primary spectrum: every route must then give back the frontend-processed primary, aligned."""
    n_raw = 4

    def forward(self, spec, feats=None, valid=None):
        self.last_valid = valid
        return spec[..., :2]


def _mix(n, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((2, n)) * 0.05).astype(np.float32)


@pytest.fixture
def no_legacy(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a low-delay route called the legacy transform or the NLMS pipeline")
    monkeypatch.setattr(stft, "istft", boom)
    monkeypatch.setattr(stft, "istft_explicit", boom)
    monkeypatch.setattr(pipeline, "run", boom)
    monkeypatch.setattr(torch, "istft", boom)


def test_low_delay_routes_never_use_legacy_transform_or_nlms(no_legacy):
    m = V.from_arch(LD_CFG["model_cfg"]).eval()
    x = _mix(8000)
    y, diag = enhance_low_delay(x, None, m, A, dsp_cfg=DSP)
    assert y.shape == (8000,) and np.isfinite(y).all()
    y2 = enhance_fe(x, None, m, LD_CFG)
    assert np.array_equal(y, y2)
    yb = forward_fe_batch(m, LD_CFG, np.stack([front_end(x, DSP, None, ac.get_audio_contract(A), per_sample=True)[0]] * 2),
                          np.ones((2, 8000), np.uint8))
    assert np.abs(yb[0] - y).max() <= 1e-6


def test_identity_alignment_all_routes():
    c = ac.get_audio_contract(A)
    x = _mix(5000, 1)
    proc, _ = front_end(x, DSP, None, c, per_sample=True)
    y, diag = enhance_low_delay(x, None, IdentityFE(), c, dsp_cfg=DSP)
    np.testing.assert_allclose(y, proc[0], atol=1e-5)
    assert diag["computable_at"][0] == c.hop - 1 and diag["release_start"][0] == c.hop - c.support
    procc, _ = front_end(x, DSP_C0, None)
    yc = enhance_fe(x, None, IdentityFE(), C0_CFG)
    np.testing.assert_allclose(yc, procc[0], atol=1e-5)
    np.testing.assert_allclose(procc, proc, atol=1e-6)          # identical preprocessing across contracts


def test_validity_is_always_passed():
    c = ac.get_audio_contract(A)
    x = _mix(4000, 2)
    av = np.ones(4000, bool); av[1000:1500] = False
    idm = IdentityFE()
    enhance_low_delay(x, av, idm, c, dsp_cfg=DSP)
    ref = ld.frame_validity(torch.from_numpy(av)[None], c)
    assert torch.equal(idm.last_valid, ref)
    enhance_fe(x, av, idm, C0_CFG)
    assert np.array_equal(idm.last_valid[0].numpy(), pipeline.frame_avail(av, 4000 // 256 + 1))


def test_batched_equals_batch_one():
    torch.manual_seed(0)
    m = V.from_arch(LD_CFG["model_cfg"]).eval()
    c = ac.get_audio_contract(A)
    xs = [front_end(_mix(6000, s), DSP, None, c, per_sample=True)[0] for s in range(4)]
    batch = forward_fe_batch(m, LD_CFG, np.stack(xs), np.ones((4, 6000), np.uint8))
    for k in range(4):
        one = forward_fe_batch(m, LD_CFG, xs[k][None], np.ones((1, 6000), np.uint8))[0]
        assert np.abs(batch[k] - one).max() <= 1e-6
    mc = V.build("mini").eval()
    xs = [front_end(_mix(6000, s), DSP_C0, None)[0] for s in range(3)]
    batch = forward_fe_batch(mc, C0_CFG, np.stack(xs), np.ones((3, 6000)))
    one = forward_fe_batch(mc, C0_CFG, xs[1][None], np.ones((1, 6000)))[0]
    assert np.abs(batch[1] - one).max() <= 1e-6


def test_checkpoint_system_and_eval_dispatch(tmp_path, no_legacy):
    torch.manual_seed(0)
    m = V.from_arch(LD_CFG["model_cfg"])
    p = tmp_path / "best.pt"
    torch.save({"model": m.state_dict(), "config": LD_CFG}, p)
    f = make_fe_system(p)
    assert f.contract == A
    x = _mix(4000, 3)
    y = f(x, None)
    assert y.shape == (4000,) and np.isfinite(y).all()


def test_eval_modules_route_fe_checkpoints(tmp_path):
    from vaani import eval as veval
    import importlib.util
    from pathlib import Path
    torch.manual_seed(0)
    p = tmp_path / "best.pt"
    torch.save({"model": V.from_arch(LD_CFG["model_cfg"]).state_dict(), "config": LD_CFG}, p)
    x = _mix(4000, 4)
    ref = make_fe_system(p)(x, None)
    assert np.array_equal(veval.enhance_fn(f"ckpt:{p}", "cpu")(x), ref)
    spec = importlib.util.spec_from_file_location("er", Path(__file__).resolve().parents[1] / "scripts/eval_refvalid.py")
    er = importlib.util.module_from_spec(spec); spec.loader.exec_module(er)
    assert np.array_equal(er.make_system(f"ckpt:{p}")(x, None), ref)
    with pytest.raises(ValueError, match="low_delay_live"):
        veval._onnx_spectrum_fn(str(p), str(p))


def test_overparam_checkpoint_is_folded_for_scoring(tmp_path):
    torch.manual_seed(0)
    cfg = dict(LD_CFG, model_cfg=dict(LD_CFG["model_cfg"], overparam=True))
    m = V.from_arch(cfg["model_cfg"])
    p = tmp_path / "op.pt"
    torch.save({"model": m.state_dict(), "config": cfg}, p)
    from vaani.enhance_low_delay import load_fe_checkpoint
    fm, _ = load_fe_checkpoint(p)
    assert not fm.overparam
    x = _mix(4000, 5)
    a = enhance_low_delay(x, None, m.eval(), A, dsp_cfg=DSP)[0]
    b = enhance_low_delay(x, None, fm, A, dsp_cfg=DSP)[0]
    assert np.abs(a - b).max() <= 1e-5


def test_resampler_in_the_loop_and_filtered_reference():
    pair = rs.load(ac.RESAMPLER_R0)
    x48 = np.repeat(_mix(3000, 6), 3, axis=1)
    y48, diag = enhance_low_delay(x48, None, IdentityFE(), A, resampler=pair, dsp_cfg={})
    assert y48.shape == (9000,) and diag["resampler"] == ac.RESAMPLER_R0
    ref = filter_reference(x48[0], pair)
    np.testing.assert_allclose(y48, ref, atol=1e-5)   # identity model: output == the reference through the same pair


def test_screen_score_items_dispatch(monkeypatch):
    """train_refiner.score_items: VaaniFE goes through the contract runner; the cache key names the contract."""
    from vaani import train_refiner as tr

    class DS:
        def __getitem__(self, i):
            rng = np.random.default_rng(i)
            c = (rng.standard_normal(8000) * 0.05).astype(np.float32)
            return {"mix": torch.from_numpy(np.stack([c + 0.01, c * 0.8])), "clean": torch.from_numpy(c)}
    monkeypatch.setattr(tr, "SCREEN_WORKERS", 0, raising=False)
    monkeypatch.setattr(tr, "_pool_map", lambda f, xs: [f(x) for x in xs])
    torch.manual_seed(0)
    m = V.from_arch(LD_CFG["model_cfg"])
    ds = DS()
    arr = tr.score_items(m, ds, [0, 1, 2], LD_CFG, torch.device("cpu"))
    assert arr.shape == (3, 3) and np.isfinite(arr[:, :2]).all()
    keys = [k for k in tr._screen_cache if k[0] is ds]
    tr.score_items(V.build("mini"), ds, [0, 1, 2], C0_CFG, torch.device("cpu"))
    assert len([k for k in tr._screen_cache if k[0] is ds]) == len(keys) + 1   # distinct cache per contract
