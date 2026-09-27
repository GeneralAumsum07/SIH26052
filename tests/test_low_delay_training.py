"""Low-delay training controls (plan Task 4): the train loop on a low-delay contract, resume keys, gradient
finiteness, exclusions, warm-up, and a synthetic optimizer step for every tier and every Stage-1/2 arm."""
import copy
import json

import numpy as np
import pytest
import torch
import yaml

from vaani import audio_contract as ac, losses, train
from vaani.enhance_low_delay import build_fe_loss, ld_model_inputs
from vaani.models import vaani_fe as V
from tests.test_train_smoke import _tiny

A, B = ac.ARM_A_IDS, ac.ARM_B_ID
DSP = {"limiter": True, "limiter_kernel": "numba", "ref_policy": {"nlms": True, "absent": "freeze", "ramp_samples": 3072}}
LOSS = dict(w_mag=0.3, w_complex=0.2, w_consistency=0.3, w_wave=0.2, w_pesq=0.0, w_snr=0.002, kappa=1.0)


def _p18(cid=A[0], **kw):
    return {"tier": "mini", "audio_contract": cid, "freq_windows": "p18", "valid_bias": True, "df_bins": 96,
            "df_lags": [0, 3, 5], "gru_init": "tc_matched", "fp32_islands": True, **kw}


def _cfg(tmp_path, m, name="ld", **over):
    exclude = tmp_path / "exclude.json"
    exclude.write_text("[]")
    cfg = dict(name=name, model="vaani_fe", controller_on=False, loss="fe", loss_cfg=dict(LOSS), dsp=DSP,
               model_cfg=_p18(),
               data=dict(manifests=[str(m)], bank=None, crop_s=1.0, epoch_len=4, mix={"p_room": 0.0},
                         exclude_groups_file=str(exclude),
                         ref_corrupt={"p": 0.5, "p_absent": 0.3}),
               val=dict(dynamic_items=2), optim=dict(lr=1e-3, warmup=1, clip=5.0), batch_size=2, epochs=1,
               max_steps=2, amp=False, device="cpu", runs_dir=str(tmp_path / "runs"), num_workers=0, seed=0,
               ema={"decay": 0.999}, perf={"numerics": {"cuda_graph": False, "gru_kernel": "cudnn", "render": "cpu",
                                                         "compile": False}})
    cfg.update(over)
    return cfg


def _run(tmp_path, cfg):
    cp = tmp_path / f"{cfg['name']}.yaml"
    yaml.safe_dump(cfg, open(cp, "w"))
    train.main(str(cp))
    return tmp_path / "runs" / cfg["name"]


def test_low_delay_train_resume_and_key_rejection(tmp_path):
    m = _tiny(tmp_path)
    cfg = _cfg(tmp_path, m)
    rd = _run(tmp_path, cfg)
    info = json.loads((rd / "run.json").read_text())
    assert info["audio_contract"] == A[0] and info["steps"] == 2 and info["perf"] == cfg["perf"]
    ck = torch.load(rd / "last.pt", weights_only=True)
    assert all(torch.isfinite(v).all() for v in ck["model"].values() if v.is_floating_point())
    cfg2 = dict(cfg, epochs=2, max_steps=4)
    changes = {
        "contract": lambda c: c["model_cfg"].update(audio_contract=A[1]),        # another support, same shapes
        "tiling": lambda c: c["model_cfg"].update(freq_windows="p32", df_bins=144, df_lags=[0, 2, 4]),
        "precision": lambda c: c["model_cfg"].update(fp32_islands=False),
        "init": lambda c: c["model_cfg"].update(gru_init="default"),
        "batch": lambda c: c.update(batch_size=4),
        "amp": lambda c: c.update(amp=True),
        "numerics": lambda c: c["perf"]["numerics"].update(cuda_graph=True),
    }
    for what, f in changes.items():
        c = copy.deepcopy(cfg2); f(c)
        cp = tmp_path / f"{cfg['name']}.yaml"
        yaml.safe_dump(c, open(cp, "w"))
        with pytest.raises(RuntimeError, match="Resume configuration changed|cosine schedule"):
            train.main(str(cp))
    # same recipe, longer run with the same max_steps schedule: resumes
    c = copy.deepcopy(cfg); c["epochs"] = 2
    yaml.safe_dump(c, open(tmp_path / f"{cfg['name']}.yaml", "w"))
    train.main(str(tmp_path / f"{cfg['name']}.yaml"))


def test_epoch_resume_agrees_with_uninterrupted_run(tmp_path):
    m = _tiny(tmp_path)
    base = _cfg(tmp_path, m, name="whole", epochs=2, max_steps=4, ema=None)
    base["data"]["epoch_len"] = 4
    whole = torch.load(_run(tmp_path, base) / "last.pt", weights_only=True)
    part = copy.deepcopy(base); part["name"] = "parts"; part["epochs"] = 1
    _run(tmp_path, part)
    part["epochs"] = 2
    resumed = torch.load(_run(tmp_path, part) / "last.pt", weights_only=True)
    assert resumed["step"] == whole["step"] == 4
    for k, v in whole["model"].items():
        if v.is_floating_point():
            torch.testing.assert_close(resumed["model"][k], v, atol=1e-6, rtol=1e-5)


def test_missing_exclusion_file_raises(tmp_path):
    m = _tiny(tmp_path)
    cfg = _cfg(tmp_path, m, name="noexcl")
    cfg["data"]["exclude_groups_file"] = str(tmp_path / "absent.json")
    with pytest.raises(FileNotFoundError, match="exclude_groups_file"):
        _run(tmp_path, cfg)


def test_nonfinite_gradients_are_rejected_and_counted(tmp_path, monkeypatch):
    from vaani import enhance_low_delay as eld
    m = _tiny(tmp_path)
    cfg = _cfg(tmp_path, m, name="nan", max_steps=3)
    cfg["data"]["epoch_len"] = 8
    orig = eld.ResynthesisFELoss.forward
    calls = {"n": 0}

    def poisoned(self, pred, *a, **k):
        loss = orig(self, pred, *a, **k)
        calls["n"] += 1
        if calls["n"] == 2:   # finite value, NaN gradient: d sqrt(x)/dx at 0 times 0
            loss = loss + torch.sqrt(pred.sum() * 0.0)
        return loss
    monkeypatch.setattr(eld.ResynthesisFELoss, "forward", poisoned)
    rd = _run(tmp_path, cfg)
    info = json.loads((rd / "run.json").read_text())
    assert info["rejected_steps"] == 1 and info["steps"] == 3
    ck = torch.load(rd / "last.pt", weights_only=True)
    assert all(torch.isfinite(v).all() for v in ck["model"].values() if v.is_floating_point())
    assert all(torch.isfinite(v).all() for v in ck["ema"].values() if v.is_floating_point())


def test_abort_after_consecutive_nonfinite_steps(tmp_path, monkeypatch):
    from vaani import enhance_low_delay as eld
    monkeypatch.setattr(train, "MAX_BAD_STEPS", 2)
    monkeypatch.setattr(eld.ResynthesisFELoss, "forward", lambda self, pred, *a, **k: pred.sum() * float("nan"))
    m = _tiny(tmp_path)
    with pytest.raises(RuntimeError, match="consecutive non-finite"):
        _run(tmp_path, _cfg(tmp_path, m, name="abort", max_steps=4))


def test_warmup_run_scores_the_declared_segment(tmp_path):
    m = _tiny(tmp_path)
    cfg = _cfg(tmp_path, m, name="warm")
    cfg["data"].update(crop_s=(7680 + 2560) / 16000, warmup_samples=7680)
    rd = _run(tmp_path, cfg)
    assert json.loads((rd / "run.json").read_text())["steps"] == 2
    bad = _cfg(tmp_path, m, name="warm_bad")
    bad["data"].update(crop_s=(7680 + 2500) / 16000, warmup_samples=7680)
    with pytest.raises(ValueError, match="whole number"):
        _run(tmp_path, bad)
    c0 = _cfg(tmp_path, m, name="warm_c0", model_cfg={"tier": "mini", "fp32_islands": True},
              dsp={"limiter": True, "ref_policy": {"nlms": True, "absent": "freeze", "ramp_frames": 12}})
    c0["data"].update(warmup_samples=7680, crop_s=1.12)
    with pytest.raises(ValueError, match="low-delay"):
        _run(tmp_path, c0)


def test_unrelated_clips_share_no_state():
    torch.manual_seed(0)
    mdl = V.from_arch(_p18()).eval()
    x = torch.randn(2, 2, 8000) * 0.05
    spec, valid = ld_model_inputs(x, None, A[0])
    y = mdl(spec, None, valid)
    x2 = x.clone(); x2[1] = torch.randn(2, 8000)
    spec2, _ = ld_model_inputs(x2, None, A[0])
    assert torch.equal(mdl(spec2, None, valid)[0], y[0])


ARMS = {
    "C0": ({"tier": "mini", "fp32_islands": True}, None),
    **{f"A_{c[-7:-3]}": (_p18(c), None) for c in A},
    "B": ({**_p18(B), "freq_windows": "p32", "df_bins": 144, "df_lags": [0, 2, 4]}, None),
    "R": ({"tier": "mini", "audio_contract": A[0], "df_bins": 96, "df_lags": [0, 3, 5], "gru_init": "tc_matched",
           "fp32_islands": True}, None),
    "A_overparam": (_p18(overparam=True), None),
    "A_gru_default": (_p18(gru_init="default"), None),
    "A_mrstft": (_p18(), {"w_mrstft": 0.05}),
    "A_native": (_p18(), {"loss_domain": "native"}),
    **{f"{t}_ld": ({"tier": t, "audio_contract": A[0], "df_bins": 96, "df_lags": [0, 3, 5], "gru_init": "tc_matched",
                    "fp32_islands": True}, None) for t in ("mid", "large", "large_plus")},
}


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_every_arm_takes_a_finite_optimizer_step(arm):
    mc, lc = ARMS[arm]
    torch.manual_seed(0)
    mdl = train.build_model("vaani_fe", model_cfg=mc)
    lc = dict(LOSS, **(lc or {}))
    dom = lc.pop("loss_domain", "resynthesis")
    loss_fn = build_fe_loss(lc, mc, dom)
    n = 16000
    clean = torch.randn(2, n) * 0.05
    mix = torch.stack([clean + torch.randn(2, n) * 0.02, clean * 0.7], 1)
    batch = {"mix": mix, "clean": clean, "meta": [{}, {}], "avail": torch.ones(2, n, dtype=torch.uint8)}
    c = ac.contract_of(mc)
    inputs, target, fw, is_clean = train.prepare_batch(batch, "vaani_fe", torch.device("cpu"), contract=c)
    opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in mdl.parameters()]
    loss = loss_fn(mdl(*inputs).float(), target, fw, is_clean)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in mdl.parameters())
    opt.step()
    assert any(not torch.equal(a, p) for a, p in zip(before, mdl.parameters()))
    assert all(torch.isfinite(p).all() for p in mdl.parameters())


def test_c0_prepare_batch_reduces_avail_to_the_legacy_labels():
    n = 16000
    av = torch.ones(2, n, dtype=torch.uint8); av[0, 3000:4000] = 0
    from vaani.dsp import pipeline
    batch = {"mix": torch.randn(2, 2, n) * 0.05, "clean": torch.randn(2, n) * 0.05, "meta": [{}, {}], "avail": av}
    inputs, target, _, _ = train.prepare_batch(batch, "vaani_fe", torch.device("cpu"))
    assert inputs[0].shape == (2, 257, 63, 4) and inputs[2].shape == (2, 63)
    for b in range(2):
        assert np.array_equal(inputs[2][b].numpy(), pipeline.frame_avail(av[b].numpy().astype(bool), 63))
