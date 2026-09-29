"""Training-throughput implementations (plan Task 4b): exact or within rounding of the paths they replace."""
import numpy as np
import pytest
import torch

from vaani import losses, train
from vaani.data.dataset import EpochBatchSampler, EpochSampler
from vaani.dsp import stft


def test_istft_explicit_is_bit_identical():
    for dt in (torch.float32, torch.float64):
        for n in (16000, 32000, 48000, 64000, 64123):
            z = stft.stft(torch.randn(2, n, dtype=dt)) + 0.05 * torch.randn(2, 257, n // 256 + 1, 2, dtype=dt)
            assert torch.equal(stft.istft(z, length=n), stft.istft_explicit(z, length=n))
            assert torch.equal(stft.istft(z), stft.istft_explicit(z))


def test_window_is_cached_per_device():
    assert stft.window() is stft.window("cpu")


def _legacy_feloss(fe, pred, true):
    """FELoss as it was: legacy stft.istft inside, the boolean-selection pesq_term."""
    from tests.test_pesq_filters import _legacy_pesq_term
    pr, pi, pm = losses._compress(pred, fe.p); tr, ti, tm = losses._compress(true, fe.p)
    w = torch.ones_like(pm[:, 0])[:, None, :]
    wm = lambda x: (x * w).mean()
    t = {}
    diff = tm - pm
    t["mag"] = wm(diff ** 2); t["over"] = wm(losses.asym_sq(diff, fe.kappa)) - t["mag"]
    t["complex"] = wm((pr - tr) ** 2 + (pi - ti) ** 2)
    n = pred.shape[2] * stft.HOP - stft.HOP
    y_p, y_t = stft.istft(pred, length=n), stft.istft(true, length=n)
    cr, ci, _ = losses._compress(stft.stft(y_p), fe.p)
    t["consistency"] = wm((cr - pr) ** 2 + (ci - pi) ** 2)
    t["wave"] = (y_p - y_t).abs().mean()
    t["snr"] = -losses.absolute_snr(y_p, y_t).clamp(max=fe.snr_max_db).mean()
    t["pesq"] = _legacy_pesq_term(fe.pesq, y_p, y_t)[0] if fe.w["pesq"] else pred.new_zeros(())
    total = fe.w["mag"] * (t["mag"] + t["over"])
    for k in ("complex", "consistency", "wave", "pesq", "snr"):
        if fe.w[k]:
            total = total + fe.w[k] * t[k]
    return total


@pytest.mark.parametrize("case", ["audible", "mixed", "silent"])
def test_feloss_equals_legacy(case):
    pytest.importorskip("torch_pesq")
    g = torch.Generator().manual_seed(0)
    c = torch.randn(3, 32000, generator=g) * 0.1
    if case == "mixed":
        c[1] = 0
    if case == "silent":
        c[:] = 0
    y = c + torch.randn(3, 32000, generator=g) * 0.03
    pred, true = stft.stft(y), stft.stft(c)
    fe = losses.FELoss()
    a, b = fe(pred, true), _legacy_feloss(fe, pred, true)
    assert abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))


def test_prepare_batch_unchanged():
    n = 16000
    batch = {"mix": torch.randn(2, 2, n), "clean": torch.randn(2, n), "meta": [{"clean_bucket": True}, {}],
             "ref_avail": torch.ones(2, 63)}
    inputs, target, fw, ic = train.prepare_batch(batch, "vaani_fe", torch.device("cpu"))
    assert torch.equal(target, stft.stft(batch["clean"])) and ic.tolist() == [True, False]
    assert torch.equal(fw, torch.ones(2, 63)) and inputs[0].shape == (2, 257, 63, 4)


@pytest.mark.parametrize("epoch_len,bs", [(20000, 32), (10, 3), (6, 2)])
def test_epoch_batch_sampler_yields_the_epoch_sampler_batches(epoch_len, bs):
    from torch.utils.data import BatchSampler
    got = list(EpochBatchSampler(epoch_len, bs, 1, 3))
    want = []
    for e in (1, 2):
        s = EpochSampler(epoch_len); s.set_epoch(e)
        want += list(BatchSampler(s, bs, drop_last=False))
    assert got == want and len(EpochBatchSampler(epoch_len, bs, 1, 3)) == len(want)
    assert EpochBatchSampler(20000, 32, 0, 1).batches_per_epoch == 625


def test_perf_settings_validation():
    assert train.perf_settings({}) == (None, None)
    num, ops = train.perf_settings({"perf": {"numerics": {"cuda_graph": True}}})
    assert num["cuda_graph"] and num["gru_kernel"] == "cudnn" and ops["scorer"] == "inline"
    for bad in ({"perf": {"numerics": {"batch": 64}}}, {"perf": {"ops": {"scorer": "later"}}},
                {"perf": {"extra": {}}}, {"perf": {"numerics": {"render": "tpu"}}}):
        with pytest.raises(ValueError):
            train.perf_settings(bad)


def test_graphed_step_refuses_without_cuda():
    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    from vaani.train_graph import GraphedStep
    with pytest.raises(RuntimeError, match="cuda_graph: false"):
        GraphedStep(lambda *a: None, None, "cpu", False, True)


def _ld_setup(seed=0, overparam=False):
    from vaani import audio_contract as ac
    from vaani.enhance_low_delay import build_fe_loss, ld_model_inputs
    from vaani.models import vaani_fe as V
    torch.manual_seed(seed)
    mc = dict(audio_contract=ac.ARM_A_IDS[0], fp32_islands=True, gru_init="tc_matched",
              overparam=overparam, **V.MINI_P["p18"])
    m = V.build("mini", **mc).cuda()
    lf = build_fe_loss(dict(w_mag=0.3, w_complex=0.2, w_consistency=0.3, w_wave=0.2, w_pesq=0.001, w_snr=0.002,
                            pesq_filters="fft"), {"audio_contract": ac.ARM_A_IDS[0]})
    return m, lf, ld_model_inputs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_graphed_and_eager_steps_agree_over_50_steps():
    from vaani.train_graph import GraphedStep
    import copy
    m1, lf1, mk = _ld_setup()
    m2 = copy.deepcopy(m1)
    lf2 = copy.deepcopy(lf1)
    o1, o2 = torch.optim.AdamW(m1.parameters(), 1e-3, foreach=True), torch.optim.AdamW(m2.parameters(), 1e-3, foreach=True)
    gs = GraphedStep(m2, lf2, "cuda", True, True)
    g = torch.Generator(device="cuda").manual_seed(1)
    for s in range(50):
        mix = torch.randn(4, 2, 16000, device="cuda", generator=g) * 0.05
        clean = mix[:, 0] * 0.8
        inputs = mk(mix, torch.ones(4, 16000, device="cuda"), m1.contract)
        ic = torch.zeros(4, dtype=torch.bool, device="cuda")
        o1.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = m1(*inputs)
        l1 = lf1(pred.float(), clean, None, ic); l1.backward()
        o2.zero_grad(set_to_none=False)
        l2 = gs.step(m2, inputs, clean, ic)
        assert abs(float(l1) - float(l2)) <= 1e-6 * abs(float(l1))
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            if p1.grad is not None:
                assert ((p1.grad - p2.grad).norm() / p1.grad.norm().clamp(min=1e-30)) <= 1e-6
        o1.step(); o2.step()
    for (k, a), (_, b) in zip(m1.state_dict().items(), m2.state_dict().items()):
        if a.is_floating_point():
            assert ((a - b).norm() / a.norm().clamp(min=1e-30)) <= 1e-6, k


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_step_is_sync_free_under_sync_debug_mode():
    m, lf, mk = _ld_setup()
    mix = torch.randn(4, 2, 16000, device="cuda") * 0.05
    inputs = mk(mix, torch.ones(4, 16000, device="cuda"), m.contract)
    ic = torch.zeros(4, dtype=torch.bool, device="cuda")
    lf(m(*inputs).float(), mix[:, 0], None, ic).backward()   # first call builds the cached filters (one host read)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = m(*inputs)
        lf(pred.float(), mix[:, 0], None, ic).backward()
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_overparam_graph_tracks_updated_factors():
    """Products must be recomputed on every replay, including after an optimizer update."""
    from vaani.train_graph import GraphedStep
    m1, lf1, mk = _ld_setup(overparam=True)
    m2, lf2, _ = _ld_setup(overparam=True)
    gs = GraphedStep(m2, lf2, "cuda", True, True)
    opts = [torch.optim.AdamW(m.parameters(), lr=1e-3) for m in (m1, m2)]
    for step in range(50):
        mix = torch.randn(2, 2, 16000, device="cuda") * 0.05
        inputs = mk(mix, torch.ones(2, 16000, device="cuda"), m1.contract)
        target, clean = mix[:, 0] * 0.8, torch.zeros(2, dtype=torch.bool, device="cuda")
        opts[0].zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = m1(*inputs)
        eager = lf1(pred.float(), target, None, clean)
        eager.backward()
        opts[1].zero_grad(set_to_none=False)
        got = gs.step(m2, inputs, target, clean)
        torch.testing.assert_close(got, eager, rtol=1e-6, atol=1e-7, msg=f"step {step}: {float(got.detach())} vs {float(eager.detach())}")
        for (name, p), q in zip(m1.named_parameters(), m2.parameters()):
            assert (p.grad is None) == (q.grad is None), name
            if p.grad is not None:
                torch.testing.assert_close(q.grad, p.grad, rtol=1e-5, atol=1e-7, msg=name)
        for opt in opts:
            opt.step()
        for (name, p), q in zip(m1.named_parameters(), m2.parameters()):
            torch.testing.assert_close(q, p, rtol=1e-6, atol=1e-8, msg=f"weight after {step}: {name}")
    for (name, p), q in zip(m1.named_buffers(), m2.buffers()):
        torch.testing.assert_close(q, p, rtol=1e-6, atol=1e-7, msg=name)
