"""Runtime guards (plan 11.3): the duplicated-mono estimator and the never-vanish fallback, alone and in the engine."""
import numpy as np
import pytest

from vaani import backend as bk
from vaani import guards as gd
from vaani import live
from tests.test_live import DSP, _graph, _mix

H = live.HOP
HALF_S = int(0.5 * live.SR)


def _speechy(seconds, seed=0, noise=0.03):
    """Harmonic 'voice' gated on and off at 3 Hz (speech-band energy that comes and goes) plus white noise."""
    rng = np.random.default_rng(seed)
    n = int(seconds * live.SR); t = np.arange(n) / live.SR
    v = sum(np.sin(2 * np.pi * 220 * k * t) / k for k in range(1, 11)) * 0.1 * (np.sin(2 * np.pi * 3 * t) > 0)
    return v.astype(np.float32), (rng.standard_normal(n) * noise).astype(np.float32)


def _hops(x):
    return [x[..., j * H:(j + 1) * H] for j in range(x.shape[-1] // H)]


class _FakeBackend(bk.Backend):
    """Stateless stand-in for the model: out = gain * P. vanish=True makes the gain collapse while the reference
    carries energy (gain = 1 - 2 ||R|| / ||P||, floored at 1e-3), the failure the never-vanish guard exists for."""
    name = "fake"

    def __init__(self, vanish: bool):
        super().__init__("fake", {})
        self.vanish = vanish

    def _step(self, spec6, feats, state):
        s = spec6[0, :, 0, :]
        g = 1.0
        if self.vanish:
            ratio = np.linalg.norm(s[:, 2:4]) / (np.linalg.norm(s[:, 0:2]) + 1e-9)
            g = float(np.clip(1.0 - 2.0 * ratio, 1e-3, 1.0))
        return (s[None, :, None, 0:2] * g).astype(np.float32)


# ---------------------------------------------------------------------------------------------- estimator
def test_informativeness_fires_on_duplicated_mono_within_half_a_second():
    v, n = _speechy(2.0)
    x = v + n
    est = gd.RefInformativeness()
    fired = next(j for j, p in enumerate(_hops(x)) if not est.update(p, p.copy()))
    assert (fired + 1) * H <= HALF_S + H                     # decided by the hop that completes 0.5 s


def test_informativeness_silent_on_a_normal_two_mic_mixture():
    mix = _mix(4.0, seed=3)
    est = gd.RefInformativeness()
    assert all(est.update(p, r) for p, r in zip(_hops(mix[0]), _hops(mix[1])))


def test_informativeness_recovers_with_hysteresis():
    v, n = _speechy(3.0)
    mix = _mix(3.0, seed=4)
    est = gd.RefInformativeness()
    for p in _hops(v + n):
        est.update(p, p.copy())
    assert not est.informative
    back = [est.update(p, r) for p, r in zip(_hops(mix[0]), _hops(mix[1]))]
    k = back.index(True)
    assert k + 1 >= est.recover_hops and all(back[k:])        # needs 0.5 s of clear evidence, then stays


# ---------------------------------------------------------------------------------------------- never vanish
def test_never_vanish_fires_on_vanished_output_only():
    v, n = _speechy(4.0)
    frames = [np.fft.rfft(np.concatenate([a, b]) * live.WINDOW) for a, b in zip(_hops(v + n)[:-1], _hops(v + n)[1:])]
    nv_bad, nv_ok = gd.NeverVanish(), gd.NeverVanish()
    bad = [nv_bad.update(gd.band_db(P), gd.band_db(P * 1e-2)) for P in frames]        # 40 dB down
    ok = [nv_ok.update(gd.band_db(P), gd.band_db(P * 0.3)) for P in frames]           # 10 dB down: fine
    assert any(bad) and not any(ok)
    assert bad.index(True) > nv_bad.trigger_hops              # "more than 0.5 s" of speech hops, never sooner


def test_energy_vad_ignores_stationary_noise():
    _, n = _speechy(4.0, noise=0.05)
    vad = gd.EnergyVAD()
    frames = [np.fft.rfft(np.concatenate([a, b]) * live.WINDOW) for a, b in zip(_hops(n)[:-1], _hops(n)[1:])]
    hits = [vad.update(gd.band_db(P)) for P in frames]
    assert np.mean(hits[20:]) < 0.1


# ---------------------------------------------------------------------------------------------- in the engine
def test_guards_off_by_default_and_silent_guards_are_bit_exact():
    mix = _mix(2.0, seed=5)
    off = live.StreamEngine(None, True, DSP, backend=_FakeBackend(False))
    on = live.StreamEngine(None, True, DSP, backend=_FakeBackend(False), guards=True)
    assert off.guards is None
    y0 = np.concatenate([off.process(p, r) for p, r in zip(_hops(mix[0]), _hops(mix[1]))])
    y1 = np.concatenate([on.process(p, r) for p, r in zip(_hops(mix[0]), _hops(mix[1]))])
    assert not on.telemetry.event_counts
    assert np.array_equal(y0, y1)


@pytest.fixture(scope="module")
def graph(tmp_path_factory):
    return _graph(tmp_path_factory.mktemp("guards"))


def test_engine_zeroes_a_duplicated_reference_through_the_onnx_path(graph):
    _, onnx = graph
    v, n = _speechy(2.0)
    x = v + n
    eng = live.StreamEngine(onnx, True, DSP, guards={"never_vanish": False})
    ys, w = [], []
    for p in _hops(x):
        ys.append(eng.process(p, p.copy())); w.append(eng.last["guard_weight"])
    ev = [e for e in eng.telemetry.events if e["kind"] == "ref_uninformative"]
    assert len(ev) == 1 and ev[0]["sample"] < HALF_S
    k = int(np.argmax(np.array(w) < 1.0))
    assert np.all(np.diff(w[k:]) <= 0) and w[-1] == 0.0      # a monotone crossfade, not a switch
    assert np.min(np.diff(w)) >= -1.0 / eng.guards.fade_hops - 1e-9
    y = np.concatenate(ys)
    assert np.all(np.isfinite(y)) and np.abs(y).max() < 10 * np.abs(x).max()


def test_engine_never_vanish_crossfades_without_clicks():
    v, n = _speechy(5.0, seed=6)
    rng = np.random.default_rng(7)
    prim, ref = v + n, 0.5 * v + (rng.standard_normal(len(v)) * 0.03).astype(np.float32)
    eng = live.StreamEngine(None, True, DSP, backend=_FakeBackend(True), guards=True)
    y = np.concatenate([eng.process(p, r) for p, r in zip(_hops(prim), _hops(ref))])
    kinds = [e["kind"] for e in eng.telemetry.events]
    assert "never_vanish" in kinds and "ref_uninformative" not in kinds
    assert eng.telemetry.summary()["fallback_events"]["never_vanish"] >= 1
    first = next(e["sample"] for e in eng.telemetry.events if e["kind"] == "never_vanish")
    assert first < 3 * live.SR
    # while vanished the output is ~60 dB down; after the fallback it carries the primary again
    seg = slice(first + 16 * H, first + 40 * H)
    assert np.sqrt(np.mean(y[seg] ** 2)) > 0.3 * np.sqrt(np.mean(prim[seg.start - H:seg.stop - H] ** 2))
    # no click: the steepest output step stays within the input's own steepest step
    assert np.abs(np.diff(y)).max() < 1.5 * np.abs(np.diff(prim)).max()
    assert np.all(np.isfinite(y))


def test_reset_rebuilds_guards():
    v, n = _speechy(1.0)
    eng = live.StreamEngine(None, True, DSP, backend=_FakeBackend(False), guards=True)
    for p in _hops(v + n):
        eng.process(p, p.copy())
    assert not eng.guards.informative
    eng.reset()
    assert eng.guards.informative and eng.guards.g == 1.0
