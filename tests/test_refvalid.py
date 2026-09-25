"""Reference-validity Mini (spec 6.2/6.3): model option, cascade gating, dropout-safe DSP, limiter latch fix,
reference-failure augmentation. Every option is default-off; the defaults stay the r7 system bit-exact."""
from pathlib import Path

import numpy as np
import pytest
import torch

from vaani.data import dataset, mixer
from vaani.dsp import pipeline
from vaani.dsp.limiter import Limiter
from vaani.dsp.nlms import NLMS
from vaani.models.cascade import FrozenCascade, StreamCascade
from vaani.models.modules.convert import convert_to_stream
from vaani.models.residual_refiner import init_refine_cache
from vaani.models.vaani_net import VaaniNet, StreamVaaniNet, init_caches, N_REL
from vaani.train import build_param_groups

ROOT = Path(__file__).resolve().parents[1]
R7 = Path("results_r2/runs/r7_e256_wr64/best.pt")
MC = dict(channels=16, coh=True, df_order=3, film=False, noise_floor=False)   # r7's model_cfg
SR, HOP = 16000, 256


def _inputs(T=30, B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, 257, T, 6, generator=g) * 0.1, torch.rand(B, T, 18, generator=g)


def _r7_pair():
    if not R7.exists():
        pytest.skip("r7 checkpoint not present")
    sd = torch.load(R7, map_location="cpu", weights_only=True)["model"]
    a = VaaniNet(**MC).eval(); a.load_state_dict(sd)
    return a, VaaniNet(**MC, ref_validity=True).warm_start(sd).eval()


# --- model ---

def test_default_graph_has_no_ref_conv_and_ref_validity_requires_coh():
    assert not hasattr(VaaniNet(**MC).encoder, "ref_conv") and not VaaniNet(**MC).ref_validity
    with pytest.raises(ValueError):
        VaaniNet(coh=False, ref_validity=True)


def test_warm_start_from_r7_is_r7_exactly_with_avail_one_or_none():
    a, b = _r7_pair(); spec, f = _inputs()
    with torch.no_grad():
        ya = a(spec, f)
        assert torch.equal(ya, b(spec, f)) and torch.equal(ya, b(spec, f, torch.ones(2, spec.shape[2])))
    extra = sum(p.numel() for p in b.parameters()) - sum(p.numel() for p in a.parameters())
    assert extra == N_REL * 16 * 5 and float(b.encoder.ref_conv.conv.weight.detach().abs().sum()) == 0.0


def test_absent_reference_reaches_no_input():
    _, b = _r7_pair(); spec, f = _inputs(B=1)
    with torch.no_grad():
        torch.nn.init.normal_(b.encoder.ref_conv.conv.weight, std=0.1)
        spec2 = spec.clone(); spec2[..., 2:] = torch.randn_like(spec2[..., 2:])   # different reference and n_hat
        zero = torch.zeros(1, spec.shape[2])
        assert torch.allclose(b(spec, f, zero), b(spec2, f, zero), atol=1e-6)
        assert not torch.allclose(b(spec, f), b(spec2, f), atol=1e-3)


def test_stream_matches_batch_with_an_availability_hole():
    _, b = _r7_pair(); spec, f = _inputs(T=40, B=1, seed=1)
    with torch.no_grad():
        torch.nn.init.normal_(b.encoder.ref_conv.conv.weight, std=0.1)
        av = torch.ones(1, 40); av[:, 10:25] = 0
        y = b(spec, f, av)
        s = StreamVaaniNet(**MC, ref_validity=True).eval(); convert_to_stream(s, b)
        caches = init_caches("cpu"); outs = []
        for t in range(40):
            o, *caches = s(spec[:, :, t:t + 1], f[:, t:t + 1], *caches, ref_avail=av[:, t:t + 1]); outs.append(o)
    assert (y - torch.cat(outs, 2)).abs().max() < 1e-4


def test_cascade_gates_the_refiner_reference_only_under_ref_validity():
    torch.manual_seed(0)
    c = FrozenCascade(dict(MC, ref_validity=True)).eval(); spec, f = _inputs(B=1)
    with torch.no_grad():
        for p in c.refiner.parameters():
            torch.nn.init.normal_(p, std=0.1)
        spec2 = spec.clone(); spec2[..., 2:] = torch.randn_like(spec2[..., 2:])
        zero = torch.zeros(1, spec.shape[2])
        assert torch.allclose(c(spec, f, zero), c(spec2, f, zero), atol=1e-6)
        assert torch.equal(c(spec, f), c(spec, f, torch.ones(1, spec.shape[2])))
        s = StreamCascade(dict(MC, ref_validity=True)).eval(); s.first.load_state_dict({}, strict=False)
        caches = (*init_caches("cpu"), init_refine_cache())
        z, *_ = s(spec[:, :, :1], f[:, :1], *caches, ref_avail=torch.zeros(1, 1))
        assert z.shape == (1, 257, 1, 2)


def test_ref_conv_trains_at_lr_new():
    m = VaaniNet(**MC, ref_validity=True); groups = build_param_groups(m, dict(lr=1e-4, lr_new=5e-4))
    new = {id(p) for g in groups if g["lr"] == 5e-4 for p in g["params"]}
    assert new == {id(m.encoder.ref_conv.conv.weight)}
    assert len(build_param_groups(VaaniNet(**MC), dict(lr=1e-4, lr_new=5e-4))) == 1   # r7: one group, as before


# --- DSP ---

def _dropout_scene(n=SR * 4):
    rng = np.random.default_rng(0)
    ref = (rng.standard_normal(n) * 0.05).astype(np.float32)
    prim = (np.convolve(ref, [0.0, 0.6, -0.3, 0.1])[:n] + rng.standard_normal(n) * 0.01).astype(np.float32)
    return prim, ref


def _nlms_run(f, prim, ref):
    return np.concatenate([f.process_block(prim[i:i + HOP], ref[i:i + HOP], 1.0)[0] for i in range(0, len(prim), HOP)])


@pytest.mark.parametrize("pure", [True, False])
def test_robust_nlms_survives_an_unannounced_dropout(pure):
    prim, ref = _dropout_scene(); ref_d = ref.copy(); ref_d[24000:32000] *= 1e-2   # -40 dB, capture path unaware
    rob = NLMS(robust=True, force_pure=pure); post = np.abs(_nlms_run(rob, prim, ref_d)[32000:36800]).max()
    assert post < 2 * np.abs(prim).max() and np.linalg.norm(rob.w) <= 10.0 + 1e-3
    # and it cancels a healthy reference as well as the r7 loop does (the residual is the 1e-4 uncorrelated floor)
    res = lambda f: ((prim[SR:] - _nlms_run(f, prim, ref)[SR:]) ** 2).mean()
    assert res(NLMS(robust=True, force_pure=pure)) < 1.1 * res(NLMS(force_pure=pure)) < 0.1 * (prim[SR:] ** 2).mean()


def test_robust_nlms_primary_delay_is_off_by_default_and_valid():
    assert NLMS(robust=True).robust["delay"] == 0 and NLMS().robust is None
    with pytest.raises(ValueError):
        NLMS(robust={"delay": -1})


def test_ref_policy_without_avail_is_bit_exact_and_absent_zeros_the_reference():
    prim, ref = _dropout_scene(SR * 2); x = np.stack([prim, ref])
    a = pipeline.run(x); b = pipeline.run(x, dsp_cfg={"ref_policy": {}})
    assert all(np.array_equal(a[k], b[k]) for k in a)
    av = np.ones(len(prim), bool); av[8000:16000] = False
    c = pipeline.run(x, dsp_cfg={"ref_policy": {"nlms": True, "absent": "reset"}}, ref_avail=av)
    assert np.abs(c["mix"][1, 8000:16000]).max() == 0 and np.abs(c["n_hat"][8000:16000]).max() == 0
    fa = c["ref_avail"]; assert fa.shape == (c["features"].shape[0],) and 0 < fa.sum() < len(fa)


def test_reconnect_ramps_the_reference_back():
    av = np.ones(SR, bool); av[4000:8000] = False
    g = pipeline.ref_gain(av, ramp_frames=4)
    assert g[5000] == 0 and 0 < g[8000] < 0.01 and np.all(np.diff(g[8000:8000 + 4 * HOP]) > 0) and g[9100] == 1


def test_limiter_fix_latch_releases_a_quiet_to_loud_far_field_step():
    rng = np.random.default_rng(3)
    n = SR * 2; x = rng.standard_normal(n).astype(np.float32) * 1e-3
    x[SR // 2:] *= 60   # +35.6 dB far-field step (same at both mics)
    outs = {}
    for fix in (False, True):
        lim = Limiter(fix_latch=fix); o = np.concatenate([lim.process_block(x[i:i + HOP], x[i:i + HOP])[0]
                                                           for i in range(0, n, HOP)])
        outs[fix] = o
    tail = slice(SR + SR // 2, n)   # a second after the step
    loss = lambda o: 20 * np.log10(np.sqrt((x[tail] ** 2).mean()) / np.sqrt((o[tail] ** 2).mean() + 1e-12))
    assert loss(outs[False]) > 10 and loss(outs[True]) < 1


def test_limiter_fix_latch_still_clamps_a_blast_and_default_is_unchanged():
    rng = np.random.default_rng(1); p = rng.standard_normal(SR).astype(np.float32) * 0.003; r = p.copy()
    k = SR // 2; burst = (np.exp(-np.arange(320) / 60) * 0.9).astype(np.float32); p[k:k + 320] += burst; r[k:k + 320] += burst
    run = lambda lim: np.concatenate([lim.process_block(p[i:i + HOP], r[i:i + HOP])[0] for i in range(0, SR, HOP)])
    assert np.abs(run(Limiter(fix_latch=True))[k:k + 320]).max() < 0.3 * np.abs(p[k:k + 320]).max()
    assert np.array_equal(run(Limiter()), run(Limiter(fix_latch=False)))


# --- data ---

def _mix(n=SR * 2, seed=0):
    rng = np.random.default_rng(seed)
    clean = (rng.standard_normal(n) * 0.05).astype(np.float32)
    return np.stack([clean + rng.standard_normal(n).astype(np.float32) * 0.02,
                     rng.standard_normal(n).astype(np.float32) * 0.02]).astype(np.float32), clean


@pytest.mark.parametrize("kind", dataset.REF_KINDS)
def test_every_fault_touches_only_the_reference(kind):
    m, clean = _mix(); c = dataset.ref_corrupt_config(True)
    out, avail, tr = dataset.apply_ref_fault(np.random.default_rng(0), m, clean, kind, c)
    assert np.array_equal(out[0], m[0]) and tr["kind"] == kind and out.dtype == np.float32
    assert (not avail.all()) == (kind in ("dropout", "burst")) and not np.array_equal(out[1], m[1])


def test_talker_leak_reaches_near_primary_level():
    m, clean = _mix(); c = dataset.ref_corrupt_config({"p_leak_near": 1.0})
    out, _, tr = dataset.apply_ref_fault(np.random.default_rng(5), m, clean, "leak", c)
    assert -4.0 <= tr["leak_db"] <= 1.0
    lvl = 10 * np.log10(((out[1] - m[1]) ** 2).mean() / (clean ** 2).mean())
    assert abs(lvl - tr["leak_db"]) < 0.5


def test_corruption_rates_and_determinism():
    m, clean = _mix(SR // 4); c = dataset.ref_corrupt_config({"p": 0.15, "p_absent": 0.15})
    kinds = []
    for i in range(2000):
        _, av, tr = dataset.corrupt_reference(np.random.default_rng([0, 0, i, dataset.REF_SEED]), m, clean, c)
        kinds.append(None if tr is None else tr["kind"])
    absent = np.mean([k == "dropout" for k in kinds]); hit = np.mean([k is not None for k in kinds])
    assert 0.15 < absent < 0.21 and 0.27 < hit < 0.33   # dropout also appears inside p at its weight
    a = dataset.corrupt_reference(np.random.default_rng([0, 0, 7, dataset.REF_SEED]), m, clean, c)
    b = dataset.corrupt_reference(np.random.default_rng([0, 0, 7, dataset.REF_SEED]), m, clean, c)
    assert np.array_equal(a[0], b[0]) and a[2] == b[2]


def test_config_validation():
    assert dataset.ref_corrupt_config(None) is None and dataset.ref_corrupt_config(False) is None
    for bad in ({"bogus": 1}, {"p": 0.9, "p_absent": 0.2}, {"weights": {"nope": 1}}):
        with pytest.raises(ValueError):
            dataset.ref_corrupt_config(bad)


def test_dataset_mixture_stream_is_unchanged_and_labels_ride_along(tmp_path):
    from tests.test_dataset import _tiny_manifest
    man = _tiny_manifest(tmp_path); cfg = mixer.MixConfig(p_room=0.0)
    kw = dict(crop_s=1.0, epoch_len=8, seed=0, with_dsp=True)
    plain = dataset.DynamicMixDataset([man], "train", None, cfg, **kw)
    cor = dataset.DynamicMixDataset([man], "train", None, cfg, ref_corrupt={"p": 0.0, "p_absent": 1.0}, **kw)
    a, b = plain[3], cor[3]
    assert torch.equal(a["clean"], b["clean"]) and torch.equal(a["mix"][0], b["mix"][0])
    assert "ref_avail" not in a and b["ref_avail"].shape == (b["feats"].shape[0],) and float(b["ref_avail"].sum()) == 0
    assert b["meta"]["ref_fault"]["kind"] == "dropout"
    batch = dataset.collate([cor[0], cor[1]]); assert batch["ref_avail"].shape[0] == 2


# --- scripts/eval_refvalid.py and the r8 config ---

def _eval_mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location("eval_refvalid", ROOT / "scripts" / "eval_refvalid.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_eval_conditions_touch_only_the_reference():
    ev = _eval_mod(); rng = np.random.default_rng(0); n = 8000
    clean = (0.1 * rng.standard_normal(n)).astype(np.float32)
    mix = np.stack([clean + 0.05 * rng.standard_normal(n), 0.05 * rng.standard_normal(n)]).astype(np.float32)
    for cond in ev.CONDITIONS:
        m, avail, edges = ev.apply_condition(cond, mix, clean)
        assert np.array_equal(m[0], mix[0]) and m.shape == mix.shape
        assert (avail is None) == (cond not in ("absent", "burst_dropout"))
        if cond == "burst_dropout":
            assert len(edges) == 4 and not m[1][~avail].any() and avail.sum() < n
    assert not ev.apply_condition("absent", mix, clean)[0][1].any()
    with pytest.raises(ValueError):
        ev.apply_condition("bogus", mix, clean)


def test_eval_speech_loss_bounds():
    ev = _eval_mod(); rng = np.random.default_rng(1); n = 16000
    clean = (0.1 * rng.standard_normal(n)).astype(np.float32); prim = clean + 0.01
    assert ev.frame_stats(clean, clean, prim)["speech_loss"] == 0.0
    s = ev.frame_stats(clean, np.zeros_like(clean), prim)
    assert s["speech_loss"] == 1.0 and s["longest_lost_s"] == pytest.approx(n // ev.F * ev.F / ev.SR)


def test_r8_refvalid_config_parses_and_pins_r7():
    import hashlib, yaml
    c = yaml.safe_load(open(ROOT / "configs" / "retraining" / "r8_mini_refvalid.yaml"))
    r7 = yaml.safe_load(open(ROOT / "configs" / "retraining" / "r7_e256_wr64.yaml"))
    rc = dataset.ref_corrupt_config(c["data"]["ref_corrupt"]); assert rc["p"] == 0.15 and rc["p_absent"] == 0.15
    assert c["model_cfg"] == {**r7["model_cfg"], "ref_validity": True}
    assert {k: v for k, v in c["data"].items() if k not in ("ref_corrupt", "exclude_groups_file")} == r7["data"]
    assert c["data"]["exclude_groups_file"] == "configs/data/r8_heldout_exclude.json"   # no r8 test-set leak
    assert c["loss_cfg"] == r7["loss_cfg"] and c["optim"] == r7["optim"] and c["epochs"] == r7["epochs"]
    assert c["dsp"]["ref_policy"]["absent"] in ("freeze", "reset")
    p = ROOT / c["init_from"]
    if p.exists():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == c["init_sha256"]
