"""Low-delay frontend, limiter kernels, per-contract validity and seconds-based guards (plan Task 2)."""
import numpy as np
import pytest
import torch

from vaani import audio_contract as ac
from vaani.data.dataset import front_end
from vaani.dsp import low_delay_stft as ld, pipeline
from vaani.dsp.limiter import Limiter
from vaani.dsp.limiter_kernel import LimiterKernel
from vaani.dsp.low_delay_frontend import LowDelayFrontend, StreamValidity
from vaani.guards import Guards
from vaani.live import BoundedHopQueue, ChannelMonitor

A, B = ac.ARM_A_IDS[0], ac.ARM_B_ID
R8_DSP = {"limiter": True, "limiter_kernel": "numba", "ref_policy": {"nlms": True, "absent": "freeze", "ramp_samples": 3072}}
R8_DSP_LEGACY = {"limiter": True, "ref_policy": {"nlms": True, "absent": "freeze", "ramp_frames": 12}}


def _mix(n, seed=0, bursts=4):
    rng = np.random.default_rng(seed)
    p = (rng.standard_normal(n) * 0.01).astype(np.float32); r = (rng.standard_normal(n) * 0.01).astype(np.float32)
    for _ in range(bursts):
        a = int(rng.integers(0, n - 900)); amp = 10 ** rng.uniform(0.5, 2.5)
        b = (rng.standard_normal(900) * 0.01 * amp).astype(np.float32)
        p[a:a + 900] += b; r[a:a + 900] += b * np.float32(rng.uniform(0.8, 1.1))
    return np.clip(np.stack([p, r]), -1, 1).astype(np.float32)


def _avail(n, seed=0, k=3):
    rng = np.random.default_rng(seed + 100); av = np.ones(n, bool)
    for _ in range(k):
        a = int(rng.integers(0, n)); av[a:a + int(rng.integers(1, 4000))] = False
    return av


# ---- limiter kernels -------------------------------------------------------------------------
@pytest.mark.parametrize("kernel", ["numpy", "numba"])
@pytest.mark.parametrize("fix_latch", [False, True])
def test_kernels_match_limiter(kernel, fix_latch):
    for seed in range(6):
        m = _mix(32000, seed, bursts=6)
        L = Limiter(fix_latch=fix_latch)
        ref = np.concatenate([np.stack(L.process_block(m[0, i:i + 256], m[1, i:i + 256])) for i in range(0, 32000, 256)], 1)
        K = LimiterKernel(kernel=kernel, fix_latch=fix_latch)
        got = np.stack(K.process_block(m[0], m[1]))
        assert np.abs(got - ref).max() <= 1e-6
        assert (K.env, K.floor, K.run) == pytest.approx((L.env, L.floor, L.run), rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("kernel", ["loop", "numpy", "numba"])
def test_kernel_chunk_invariance_at_sub_block_multiples(kernel):
    m = _mix(16000, 3)
    outs = []
    for ch in (32, 96, 128, 256, 1600, 16000):
        K = LimiterKernel(kernel=kernel)
        outs.append(np.concatenate([np.stack(K.process_block(m[0, i:i + ch], m[1, i:i + ch])) for i in range(0, 16000, ch)], 1))
    for o in outs[1:]:
        assert np.array_equal(o, outs[0])


def test_kernel_tail_and_engaged_count():
    m = _mix(1000, 5)
    L, K = Limiter(), LimiterKernel(kernel="numba")
    a, b = np.stack(L.process_block(m[0], m[1])), np.stack(K.process_block(m[0], m[1]))
    assert np.abs(a - b).max() <= 1e-6 and L.engaged == K.engaged
    with pytest.raises(ValueError):
        LimiterKernel(kernel="fortran")


# ---- frontend ----------------------------------------------------------------------------------
@pytest.mark.parametrize("cid", [A, B, ac.LEGACY_ID])
def test_frontend_equals_c0_front_end(cid):
    n = 48000
    m, av = _mix(n, 1), _avail(n, 1)
    c0, _ = front_end(m, R8_DSP_LEGACY, av)
    fe = LowDelayFrontend(cid, R8_DSP)
    y, v = fe.process_offline(m, av)
    assert np.abs(y - c0).max() <= 1e-6
    assert np.array_equal(v, av)


@pytest.mark.parametrize("cid", [A, B])
def test_streaming_hops_equal_offline_and_every_chunking(cid):
    c = ac.get_audio_contract(cid)
    n = 40 * c.hop
    m, av = _mix(n, 2), _avail(n, 2)
    off, _ = front_end(m, R8_DSP, av, c, per_sample=True)
    fe = LowDelayFrontend(c, R8_DSP)
    st = np.concatenate([fe.process(m[:, j * c.hop:(j + 1) * c.hop], av[j * c.hop:(j + 1) * c.hop])["mix"]
                         for j in range(40)], 1)
    assert np.array_equal(st, off)
    loop, _ = front_end(m, R8_DSP_LEGACY, av)          # the C0 loop at 256-sample chunks
    assert np.abs(st - loop).max() <= 1e-6


def test_dropout_at_every_hop_offset_and_reconnect_ramp():
    c = ac.get_audio_contract(A)
    for off in range(0, c.hop, 7):
        n = 60 * c.hop
        m = _mix(n, off)
        av = np.ones(n, bool); av[5 * c.hop + off:12 * c.hop + off] = False
        fe = LowDelayFrontend(c, R8_DSP)
        y, v = fe.process_offline(m, av)
        assert np.isfinite(y).all() and np.array_equal(v, av)
        assert (y[1, ~av] == 0).all()
        r = 12 * c.hop + off   # the reconnect: the gain ramps back over exactly 3072 samples
        g = pipeline.ref_gain(av, ramp_samples=3072)
        assert g[r] == pytest.approx(1 / 3072) and g[r + 3071] == 1.0 and g[r + 3070] < 1.0


def test_whole_channel_absence_nan_reference_and_nonfinite_primary():
    c = ac.get_audio_contract(A)
    fe = LowDelayFrontend(c, R8_DSP)
    m = _mix(c.hop, 0, bursts=0)
    o = fe.process(m, False)
    assert (o["mix"][1] == 0).all() and not o["valid"].any()
    bad = m.copy(); bad[1, 10] = np.nan; bad[1, 20] = np.inf
    o = fe.process(bad, True)
    assert np.isfinite(o["mix"]).all() and not o["valid"][10] and not o["valid"][20] and o["valid"][0]
    bad = m.copy(); bad[0, 5] = np.nan
    o = fe.process(bad, True)
    assert o["discontinuity"] and np.isfinite(o["mix"]).all() and fe.discontinuities == 1
    assert np.isfinite(fe.lim.env) and np.isfinite(fe.lim.gain)


def test_repeated_resets_and_state_continuation():
    c = ac.get_audio_contract(A)
    n = 20 * c.hop
    m, av = _mix(n, 4), _avail(n, 4)
    ref = LowDelayFrontend(c, R8_DSP)
    full = [ref.process(m[:, j * c.hop:(j + 1) * c.hop], av[j * c.hop:(j + 1) * c.hop])["mix"] for j in range(20)]
    a = LowDelayFrontend(c, R8_DSP)
    part = [a.process(m[:, j * c.hop:(j + 1) * c.hop], av[j * c.hop:(j + 1) * c.hop])["mix"] for j in range(8)]
    st = a.export_state()
    b = LowDelayFrontend(c, R8_DSP); b.import_state(st)
    part += [b.process(m[:, j * c.hop:(j + 1) * c.hop], av[j * c.hop:(j + 1) * c.hop])["mix"] for j in range(8, 20)]
    assert all(np.array_equal(x, y) for x, y in zip(full, part))
    for _ in range(3):
        a.reset()
        o = [a.process(m[:, j * c.hop:(j + 1) * c.hop], av[j * c.hop:(j + 1) * c.hop])["mix"] for j in range(4)]
        assert all(np.array_equal(x, y) for x, y in zip(full[:4], o))
    with pytest.raises(ValueError):
        LowDelayFrontend(B, R8_DSP).import_state(st)


def test_ramp_frames_rejected_under_low_delay_contract():
    with pytest.raises(ValueError, match="ramp_frames"):
        LowDelayFrontend(A, R8_DSP_LEGACY)
    with pytest.raises(ValueError, match="ramp_frames"):
        front_end(_mix(1000), R8_DSP_LEGACY, None, ac.get_audio_contract(A), per_sample=True)
    assert pipeline.ramp_samples_of({"ramp_frames": 12}) == 3072
    assert pipeline.ramp_samples_of({}, ac.get_audio_contract(A)) == 3072


def test_legacy_ref_gain_unchanged_by_ramp_samples_option():
    av = _avail(20000, 7)
    assert np.array_equal(pipeline.ref_gain(av, 12), pipeline.ref_gain(av, ramp_samples=3072))
    assert np.array_equal(pipeline.ref_gain(av, 4), pipeline.ref_gain(av, ramp_samples=1024))


# ---- validity ---------------------------------------------------------------------------------
@pytest.mark.parametrize("cid", [A, B])
def test_stream_validity_equals_offline_reduction(cid):
    c = ac.get_audio_contract(cid)
    n = 50 * c.hop
    av = _avail(n, 9, k=5)
    off = ld.frame_validity(torch.from_numpy(av)[None], c)[0].numpy()
    sv = StreamValidity(c)
    got = [sv.push(av[j * c.hop:(j + 1) * c.hop]) for j in range(50)] + [sv.push(True)]   # + the flush frame
    assert np.array_equal(np.asarray(got, np.float32), off)


def test_prepare_batch_reduction_equals_legacy_labels_for_c0():
    n = 16000
    av = np.stack([_avail(n, s) for s in range(3)])
    fv = ld.frame_validity(torch.from_numpy(av), ac.LEGACY_ID).numpy()
    for b in range(3):
        assert np.array_equal(fv[b], pipeline.frame_avail(av[b], n // 256 + 1))


# ---- guards, monitors and queues in seconds --------------------------------------------------------
def _dup_then_stereo(n_s=3.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(n_s * 16000)
    p = rng.standard_normal(n).astype(np.float32) * 0.1
    r = p.copy(); r[n // 2:] = rng.standard_normal(n - n // 2).astype(np.float32) * 0.1
    return p, r


def _verdicts(hop, p, r):
    g = Guards({"never_vanish": False}, hop=hop)
    t = []
    for j in range(len(p) // hop):
        g.pre(p[j * hop:(j + 1) * hop], r[j * hop:(j + 1) * hop])
        t.append(g.informative)
        g.sample += hop
    return np.repeat(np.asarray(t), hop)


@pytest.mark.parametrize("hop", [96, 128])
def test_guard_verdict_timing_matches_legacy(hop):
    p, r = _dup_then_stereo()
    legacy, low = _verdicts(256, p, r), _verdicts(hop, p, r)
    k = min(len(legacy), len(low))
    flips_l = np.flatnonzero(np.diff(legacy[:k].astype(int)))
    flips_n = np.flatnonzero(np.diff(low[:k].astype(int)))
    assert len(flips_l) == len(flips_n) == 2   # uninformative after 0.5 s, informative again after the change
    # same verdict times up to the 256-sample window's lag and whole-hop rounding of 0.5 s (< 2 legacy hops)
    assert np.all(np.abs(flips_l - flips_n) <= 2 * 256)


def test_guard_defaults_are_the_legacy_guards():
    g = Guards(True)
    assert g.hop == 256 and g.fade_hops == 8 and g.inf.hold_hops == 31 and g.inf._buf is None
    g96 = Guards(True, hop=96)
    assert g96.fade_hops == round(0.128 * 16000 / 96) and g96.inf.hold_hops == round(0.5 * 16000 / 96)
    w = Guards({"informativeness": False, "never_vanish": False}, hop=96)
    w.fallback = True
    assert w.pre(np.zeros(96, np.float32), np.zeros(96, np.float32)).shape == (96,)


def test_channel_monitor_and_queue_in_seconds():
    m = ChannelMonitor.from_seconds(96)
    assert (m.bad_hops, m.good_hops) == (8, 22)          # >= 48 ms and >= 128 ms in 6 ms hops
    legacy = ChannelMonitor.from_seconds(256)
    assert (legacy.bad_hops, legacy.good_hops) == (3, 8)
    assert BoundedHopQueue.from_seconds(0.128, 256).capacity == 8
    assert BoundedHopQueue.from_seconds(0.024, 96).capacity == 4
