"""Decoupled-cadence NLMS (low-delay plan Section 3.3, owner decision D5): n_hat for the low-delay contracts.

Chunk invariance across every contract hop, bit-exactness against the legacy pipeline where the two framings agree,
causality, the 32-sample decision grid, state continuation, and the pr_nhat route end to end: Mini-P model, offline
runner against the streaming engine (torch and ORT), the loader, prepare_batch and a training run. The spec 6.2
budget of the pr_nhat networks is recorded, not relaxed."""
import copy

import numpy as np
import pytest
import torch

from vaani import audio_contract as ac
from vaani import backend as bk
from vaani import export as E
from vaani import train
from vaani.data.dataset import front_end, nhat_front_end
from vaani.dsp import pipeline
from vaani.dsp.decoupled_nlms import CHUNK, DecoupledNLMS, robust_cfg
from vaani.dsp.low_delay_frontend import LowDelayFrontend
from vaani.enhance_low_delay import enhance_low_delay, forward_fe_batch, ld_model_inputs
from vaani.low_delay_live import LowDelayStreamEngine
from vaani.models import vaani_fe as V

torch.set_num_threads(1)
A, B, C0 = ac.ARM_A_IDS[0], ac.ARM_B_ID, ac.LEGACY_ID
CTL = {"block_margin_db": 10.0, "diff_jump_max_db": 3.0}
# the ab2_pr_nhat DSP block under a low-delay contract (ramp in samples)
DSP = {"limiter": True, "limiter_kernel": "numba", "blocking": True, "controller": CTL,
       "ref_policy": {"nlms": True, "absent": "freeze", "ramp_samples": 3072}}


def signal(n=16000 * 2, seed=0, lead_zeros=0, gap=None, blast=None):
    g = np.random.default_rng(seed)
    t = np.arange(n) / 16000
    noise = g.normal(0, 0.1, n).astype(np.float32)
    sp = (0.3 * np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 2 * t) > 0)).astype(np.float32)
    prim = sp + np.convolve(noise, [0.5, 0.3, 0.1], "same").astype(np.float32)
    ref = noise + 0.2 * np.roll(sp, 5)
    mix = np.stack([prim, ref]).astype(np.float32)
    mix[:, :lead_zeros] = 0.0
    if blast is not None:
        mix[:, blast:blast + 40] *= 40.0
    av = np.ones(n, bool)
    if gap is not None:
        av[gap[0]:gap[1]] = False
    return mix, av


def offline(cid, mix, av=None, dsp=DSP, controller_on=True):
    return LowDelayFrontend(cid, dsp, nhat=True, controller_on=controller_on).process_offline(mix, av)


# ---- the stage ---------------------------------------------------------------------------------------
def test_n_hat_does_not_depend_on_the_hop():
    mix, av = signal(gap=(9001, 12003), blast=20000)
    ys = [offline(cid, mix, av)[0] for cid in (A, ac.ARM_A_IDS[2], B, C0)]
    assert ys[0].shape == (3, mix.shape[1]) and np.abs(ys[0][2]).max() > 0
    for y in ys[1:]:
        assert np.array_equal(ys[0], y)


def test_controller_off_is_the_legacy_pipeline_bit_for_bit():
    mix, _ = signal()
    d = {"limiter": True, "blocking": True}
    y, _ = offline(A, mix, None, d, controller_on=False)
    r = pipeline.run(mix, controller_on=False, dsp_cfg=d)
    assert np.array_equal(y[2], r["n_hat"]) and np.array_equal(y[:2], r["mix"])


@pytest.mark.parametrize("cid", [A, B])
@pytest.mark.parametrize("pol", [False, True])
def test_gate_on_the_legacy_cadence_is_the_legacy_controller(cid, pol):
    """With zeros before sample 512 the offline reflect padding of legacy frame 0 is zeros too, so the two framings
    agree and every gate, speech verdict and limiter burst reaches the NLMS on the same sample as in pipeline.run."""
    mix, _ = signal(n=16000 * 3, lead_zeros=512, blast=30000)
    d = copy.deepcopy(DSP) if pol else {k: v for k, v in DSP.items() if k != "ref_policy"}
    y, _ = offline(cid, mix, None, d)
    r = pipeline.run(mix, controller_on=True, dsp_cfg=d)
    assert np.array_equal(y[2], r["n_hat"]) and np.array_equal(y[:2], r["mix"])
    assert (r["gate"] < 1).any() and (r["gate"] > 0).any()        # the controller actually gated


def test_causal_at_every_sample():
    mix, av = signal(gap=(9001, 12003))
    m2 = mix.copy()
    m2[:, 20000:] = np.random.default_rng(1).normal(0, 1, (2, mix.shape[1] - 20000))
    av2 = av.copy(); av2[25000:26000] = False
    y1, _ = offline(A, mix, av)
    y2, _ = offline(A, m2, av2)
    assert np.array_equal(y1[:, :20000], y2[:, :20000]) and not np.array_equal(y1[:, 20000:], y2[:, 20000:])


def test_gate_changes_only_on_the_256_sample_grid_and_lags_one_legacy_frame():
    """Push sample by sample: the gate is read only when a 256-sample block completes, from that block's frame."""
    mix, _ = signal(n=16000, blast=6000)
    nh = DecoupledNLMS({k: v for k, v in DSP.items() if k != "ref_policy"})
    gates = []
    for i in range(0, mix.shape[1], CHUNK):
        nh.push(mix[0, i:i + CHUNK], mix[1, i:i + CHUNK], mix[1, i:i + CHUNK])
        gates.append(nh.gate)
    g = np.asarray(gates)
    changed = np.flatnonzero(np.diff(g)) + 1                  # chunk index after which the gate changed
    assert len(changed) and all(((c + 1) * CHUNK) % 256 == 0 for c in changed)
    assert nh.frames == mix.shape[1] // 256


@pytest.mark.parametrize("absent", ["freeze", "reset"])
def test_one_call_per_chunk_equals_merged_calls(absent):
    """push merges chunks of one block that share the absent flag into one kernel call: the samples, the weights and
    the controller must be those of one call per 32-sample chunk."""
    rng = np.random.default_rng(5)
    n = 16000 * 2
    p = (rng.standard_normal(n) * 0.1).astype(np.float32)
    r = (0.6 * np.roll(p, 5) + rng.standard_normal(n) * 0.05).astype(np.float32)
    av = np.ones(n, bool)
    for a, b in ((1000, 1001), (4000, 4700), (9001, 12003), (20000, 20100)):
        av[a:b] = False
    hits = rng.random(n // CHUNK) < 0.05
    d = copy.deepcopy(DSP); d["ref_policy"]["absent"] = absent
    one, merged = DecoupledNLMS(d), DecoupledNLMS(d)
    y1 = np.concatenate([one.push(p[s:s + CHUNK], r[s:s + CHUNK], r[s:s + CHUNK], av[s:s + CHUNK],
                                  hits[s // CHUNK:s // CHUNK + 1]) for s in range(0, n, CHUNK)])
    y2 = merged.push(p, r, r, av, hits)
    assert np.array_equal(y1, y2) and np.array_equal(one.nlms.w, merged.nlms.w)
    assert (one.gate, one.last, one.frames) == (merged.gate, merged.last, merged.frames)


def test_absent_is_decided_per_32_sample_chunk():
    """One unavailable chunk freezes adaptation for that chunk only; the legacy 256-sample block would freeze 8."""
    mix, _ = signal(n=4096)
    d = copy.deepcopy(DSP); d["ref_policy"]["absent"] = "freeze"
    st = DecoupledNLMS(d, controller_on=False)
    st.push(mix[0, :1024], mix[1, :1024], mix[1, :1024])
    w0 = st.nlms.w.copy()
    av = np.zeros(CHUNK, bool)
    st.push(mix[0, 1024:1056], mix[1, 1024:1056], mix[1, 1024:1056], av)
    assert np.array_equal(st.nlms.w, w0)                      # frozen on the absent chunk
    st.push(mix[0, 1056:1088], mix[1, 1056:1088], mix[1, 1056:1088])
    assert not np.array_equal(st.nlms.w, w0)                  # adapting again on the next one


def test_robust_sub_block_is_on_the_32_sample_grid():
    assert robust_cfg({"nlms": True})["sub"] == CHUNK
    assert robust_cfg({"nlms": {"sub": 16}})["sub"] == 16
    assert robust_cfg({"absent": "freeze"}) is None
    with pytest.raises(ValueError, match="divides 32"):
        DecoupledNLMS({"ref_policy": {"nlms": {"sub": 64}}})


def test_frontend_state_continues_exactly():
    mix, av = signal(gap=(5000, 6000))
    c = ac.get_audio_contract(A)
    ref = LowDelayFrontend(c, DSP, nhat=True)
    full = [ref.process(mix[:, j * c.hop:(j + 1) * c.hop], av[j * c.hop:(j + 1) * c.hop]) for j in range(200)]
    a = LowDelayFrontend(c, DSP, nhat=True)
    for j in range(90):
        a.process(mix[:, j * c.hop:(j + 1) * c.hop], av[j * c.hop:(j + 1) * c.hop])
    b = LowDelayFrontend(c, DSP, nhat=True)
    b.import_state(a.export_state())
    for j in range(90, 200):
        o = b.process(mix[:, j * c.hop:(j + 1) * c.hop], av[j * c.hop:(j + 1) * c.hop])
        assert np.array_equal(o["n_hat"], full[j]["n_hat"]) and np.array_equal(o["mix"], full[j]["mix"])
    with pytest.raises(ValueError, match="no NLMS stage"):
        LowDelayFrontend(c, DSP, nhat=True).import_state(LowDelayFrontend(c, DSP).export_state())


def test_loader_front_end_matches_the_frontend_and_the_pr_samples():
    mix, av = signal(gap=(9001, 12003))
    c = ac.get_audio_contract(B)
    m, a, nh = nhat_front_end(mix, DSP, av, c)
    y, v = offline(B, mix, av)
    assert np.array_equal(m, y[:2]) and np.array_equal(nh, y[2]) and np.array_equal(a, v.astype(np.uint8))
    m_pr, a_pr = front_end(mix, DSP, av, c, per_sample=True)
    np.testing.assert_allclose(m, m_pr, atol=1e-6)            # the limiter within one float32 ulp, chunk-invariant
    assert np.array_equal(a, a_pr)
    with pytest.raises(ValueError):
        nhat_front_end(mix, DSP, av, ac.get_audio_contract(C0))


# ---- the pr_nhat route -------------------------------------------------------------------------------
def net(cid, seed=0):
    tiling = "p32" if cid == B else "p18"
    return E.fe_untrained({"tier": "mini", "audio_contract": cid, "inputs": "pr_nhat", **V.MINI_P[tiling]}, seed)


def test_mini_p_takes_n_hat_and_the_budget_is_recorded_not_relaxed():
    a, b = V.summary(net(A)), V.summary(net(B))
    assert net(A).n_raw == 6 and net(A).n_in == 6
    # spec 6.2: at most 60,000 training-form entries and 90.706 MMAC/s; Arm A with n_hat exceeds the entries
    assert a["params_training_form"] == 63798 and a["params_training_form"] > 60_000 and a["mmac_per_s"] <= 90.706
    assert b["params_training_form"] <= 60_000 and b["mmac_per_s"] <= 90.706


def test_model_inputs_need_n_hat_for_six_raw_planes():
    x = torch.randn(1, 2, 4000)
    with pytest.raises(ValueError, match="pr_nhat"):
        ld_model_inputs(x, None, A, 6)
    spec, _ = ld_model_inputs(torch.randn(1, 3, 4000), None, A, 6)
    assert spec.shape[-1] == 6


@pytest.mark.parametrize("cid", [A, B])
def test_offline_runner_and_streams_agree(cid, tmp_path):
    m = net(cid)
    mix, av = signal(n=9601, gap=(3000, 4200))
    off, _ = enhance_low_delay(mix, av, m, cid, dsp_cfg=DSP)
    y_t = LowDelayStreamEngine(cid, bk.FeTorchBackend(m), DSP).run(mix[0], mix[1], av)
    onnx = E.export_fe(m, tmp_path / "m.onnx", parity=False)["folded"]
    y_o = LowDelayStreamEngine(cid, bk.FeOrtBackend(onnx, audio_contract=cid), DSP).run(mix[0], mix[1], av)
    np.testing.assert_allclose(y_t, off, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(y_o, off, atol=1e-5, rtol=1e-4)
    # the n_hat channel matters: without it (zeroed) the output moves
    x, a = LowDelayFrontend(cid, DSP, nhat=True).process_offline(mix, av)
    x0 = x.copy(); x0[2] = 0
    y0 = forward_fe_batch(m, {"model_cfg": {"audio_contract": cid}}, x0[None][:, :2], a[None], x0[None][:, 2])[0]
    y1 = forward_fe_batch(m, {"model_cfg": {"audio_contract": cid}}, x[None][:, :2], a[None], x[None][:, 2])[0]
    np.testing.assert_allclose(y1, off, atol=1e-5, rtol=1e-4)
    assert np.abs(y1 - y0).max() > 1e-6


def test_stream_state_resumes_and_hash_covers_the_controller():
    m = net(A)
    mix, av = signal(n=6400, gap=(2000, 2600))
    c = ac.get_audio_contract(A)
    whole = LowDelayStreamEngine(A, bk.FeTorchBackend(m), DSP)
    y = whole.push(mix[0], mix[1], av)
    e1 = LowDelayStreamEngine(A, bk.FeTorchBackend(m), DSP)
    k = 30 * c.hop
    y1 = e1.push(mix[0, :k], mix[1, :k], av[:k])
    e2 = LowDelayStreamEngine(A, bk.FeTorchBackend(m), DSP)
    e2.import_state(e1.export_state())
    y2 = e2.push(mix[0, k:], mix[1, k:], av[k:])
    np.testing.assert_allclose(np.concatenate([y1, y2]), y, atol=1e-6)
    off = LowDelayStreamEngine(A, bk.FeTorchBackend(m), DSP, controller_on=False)
    assert off.config_hash != e1.config_hash
    with pytest.raises(ValueError, match="config_hash"):
        off.import_state(e1.export_state())
    pr = E.fe_untrained({"tier": "mini", "audio_contract": A, **V.MINI_P["p18"]}, 0)
    assert (LowDelayStreamEngine(A, bk.FeTorchBackend(pr), DSP).config_hash
            == LowDelayStreamEngine(A, bk.FeTorchBackend(pr), DSP, controller_on=False).config_hash)


def test_prepare_batch_stacks_n_hat_for_low_delay():
    n = 8000
    mix = torch.randn(2, 2, n) * 0.05
    batch = {"mix": mix, "clean": mix[:, 0], "meta": [{}, {}], "avail": torch.ones(2, n, dtype=torch.uint8),
             "n_hat": torch.randn(2, n) * 0.01}
    c = ac.get_audio_contract(A)
    (spec, _, valid), *_ = train.prepare_batch(batch, "vaani_fe", torch.device("cpu"), contract=c)
    assert spec.shape[-1] == 6 and valid.shape == spec.shape[:1] + spec.shape[2:3]
    ref, ref_valid = ld_model_inputs(torch.cat([mix, batch["n_hat"][:, None]], 1), batch["avail"], c, 6)
    assert torch.equal(spec, ref) and torch.equal(valid, ref_valid)


def test_training_run_with_n_hat(tmp_path):
    from tests.test_low_delay_training import _cfg, _p18, _run
    from tests.test_train_smoke import _tiny
    cfg = _cfg(tmp_path, _tiny(tmp_path), name="ld_nhat", controller_on=True, model_cfg=_p18(inputs="pr_nhat"),
               dsp=DSP)
    assert train.needs_dsp(cfg)
    rd = _run(tmp_path, cfg)
    ck = torch.load(rd / "last.pt", weights_only=True)
    assert ck["config"]["model_cfg"]["inputs"] == "pr_nhat"
    assert all(torch.isfinite(v).all() for v in ck["model"].values() if v.is_floating_point())
