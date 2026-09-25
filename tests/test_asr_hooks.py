"""G4 Part 2 hooks: word transcriber and VAD behind injectable interfaces; no Whisper weights are ever loaded here."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vaani import asr, live

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("field_accept", REPO / "scripts/field_accept.py")
fa = importlib.util.module_from_spec(spec); spec.loader.exec_module(fa)
SR, WIN = 16000, 8000


def fake_words(y):
    """One confident word per loud 0.5 s window, named by its window: a stand-in for Whisper word timestamps."""
    y = np.asarray(y, np.float32)
    out = []
    for k in range(len(y) // WIN):
        if np.sqrt((y[k * WIN:(k + 1) * WIN] ** 2).mean()) > 1e-3:
            out.append((f"w{k}", k * 0.5, k * 0.5 + 0.4, 0.9))
    return out


def fake_vad(y):
    return 0.5 * len(fake_words(y))


def _wav(tmp_path, seconds=4):
    rng = np.random.default_rng(0)
    x = (0.1 * rng.standard_normal((2, seconds * SR))).astype(np.float32)
    path = tmp_path / "bed.wav"
    live.write_wav(path, x, SR)
    return path


def test_word_transcriber_is_lazy_and_normalises_words():
    calls = []

    class Model:
        def transcribe(self, audio, **kw):
            calls.append(kw)
            w = [SimpleNamespace(word=" Hello,", start=0.1, end=0.4, probability=0.93),
                 SimpleNamespace(word=" don't!", start=0.5, end=0.8, probability=0.4)]
            return iter([SimpleNamespace(words=w)]), None

    loads = []
    t = asr.WordTranscriber("tiny", device="cpu", loader=lambda d, s, n: loads.append((d, s, n)) or Model())
    assert loads == []                                     # nothing loads until the first call
    assert t(np.zeros(SR)) == [("hello", 0.1, 0.4, 0.93), ("don't", 0.5, 0.8, 0.4)]
    t(np.zeros(SR))
    assert loads == [("cpu", "tiny", 1)] and calls[0]["word_timestamps"] and not calls[0]["vad_filter"]
    assert calls[0]["temperature"] == 0.0 and calls[0]["beam_size"] == 1


def test_resolve_device(monkeypatch):
    assert asr.resolve_device("cpu") == "cpu" and asr.resolve_device("cuda") == "cuda"
    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(get_cuda_device_count=lambda: 1))
    assert asr.resolve_device("auto") == "cuda"
    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(get_cuda_device_count=lambda: 0))
    assert asr.resolve_device("auto") == "cpu"


def test_vad_seconds_uses_injected_timestamps_on_normalised_audio():
    seen = {}

    def ts(y, opts):
        seen["peak"], seen["opts"] = float(np.abs(y).max()), opts
        return [{"start": 0, "end": 8000}, {"start": 16000, "end": 20000}]
    assert asr.vad_speech_seconds(np.ones(SR * 2) * 0.01, timestamps=ts) == 0.75
    assert abs(seen["peak"] - 0.5) < 1e-6 and seen["opts"]["threshold"] == 0.3 and seen["opts"]["speech_pad_ms"] == 0


def test_hooks_off_or_unavailable_are_tbd(monkeypatch):
    assert fa.asr_hooks("off") == (None, None)
    monkeypatch.setattr(asr, "available", lambda: False)
    assert fa.asr_hooks("auto") == (None, None)
    monkeypatch.setattr(asr, "available", lambda: True)
    t, v = fa.asr_hooks("auto", "base.en", "cpu")
    assert isinstance(t, asr.WordTranscriber) and t.model_name == "base.en" and t._m is None and v is asr.vad_speech_seconds


def _stream(kill_mono=True, flag_hop=None):
    """Fake stream system: passthrough, except mono_dup (r == p) is silenced; exposes ref_informative when asked."""
    def run(p, r, trace=False, ref_valid=True):
        y = np.zeros_like(p) if kill_mono and np.array_equal(p, r) else p.copy()
        n = -(-len(p) // fa.HOP)
        tr = [{"ref_informative": not (flag_hop is not None and j >= flag_hop)} for j in range(n)] if flag_hop is not None \
            else [{"gate": 1.0, "validity": 1.0}] * n
        return y, {"trace": tr} if trace else {}
    run.stream, run.validity0 = True, lambda: True
    return run


def _args(**kw):
    return SimpleNamespace(system="x", validity_key=None, guards=False, **kw)


def test_part2_criteria_become_computable_with_hooks(tmp_path, monkeypatch):
    monkeypatch.setattr(fa, "system", lambda spec, guards=False: _stream(kill_mono=True, flag_hop=10))
    p2 = fa.run_part2(_args(), _wav(tmp_path), transcriber=fake_words, vad=fake_vad)
    R = p2["runs"]
    assert R["as_is"]["word_survival"] == 1.0 and R["mono_dup"]["word_survival"] == 0.0
    assert R["as_is"]["vad_speech_s"] == R["ref_zero"]["vad_speech_s"] == 4.0 and R["mono_dup"]["vad_speech_s"] == 0.0
    assert R["as_is"]["validity_latency_s"] == 10 * fa.HOP / SR and p2["ref_zero_validity0"]
    cr, verdict = fa.summarise_part2(p2)
    assert cr["word_survival>=0.8x_ref_zero"] == "PASS" and cr["vad_speech_s>=0.8x_ref_zero"] == "PASS"
    assert cr["mono_word_survival>=0.8x_ref_zero"] == "FAIL" and cr["validity_flag<=0.5s"] == "PASS"
    assert verdict == "FAIL"
    monkeypatch.setattr(fa, "system", lambda spec, guards=False: _stream(kill_mono=False, flag_hop=10))
    cr, verdict = fa.summarise_part2(fa.run_part2(_args(), _wav(tmp_path), transcriber=fake_words, vad=fake_vad))
    assert verdict == "PASS" and set(cr.values()) == {"PASS"}


def test_capture_validity_is_not_an_estimate(tmp_path, monkeypatch):
    """Without the guards the engine only has capture-path `validity`: latency is 'absent' (TBD), never a FAIL."""
    monkeypatch.setattr(fa, "system", lambda spec, guards=False: _stream(kill_mono=False, flag_hop=None))
    p2 = fa.run_part2(_args(), _wav(tmp_path), transcriber=None, vad=None)
    assert p2["runs"]["as_is"]["validity_latency_s"] == "absent" and not p2["whisper_available"]
    cr, verdict = fa.summarise_part2(p2)
    assert cr["validity_flag<=0.5s"] == "TBD" and cr["word_survival>=0.8x_ref_zero"] == "TBD" and verdict == "TBD"


def test_ref_zero_run_is_validity_zero(tmp_path, monkeypatch):
    seen = {}

    def run(p, r, trace=False, ref_valid=True):
        seen[("zero" if not r.any() else "other", ref_valid)] = True
        return p.copy(), {}
    run.stream, run.validity0 = True, lambda: True
    monkeypatch.setattr(fa, "system", lambda spec, guards=False: run)
    fa.run_part2(_args(), _wav(tmp_path), transcriber=None, vad=None)
    assert set(seen) == {("zero", False), ("other", True)}


@pytest.mark.skipif(not asr.available(), reason="faster-whisper not importable (the asr extra)")
def test_bundled_silero_vad_runs_without_a_download():
    """Box only: the Silero model ships inside the faster-whisper wheel; silence has no speech."""
    assert asr.vad_speech_seconds(np.zeros(SR * 2, np.float32)) == 0.0


def test_guards_flag_reaches_the_stream_engine(monkeypatch):
    got = []
    monkeypatch.setattr(fa.physical, "run_engine", lambda mix, *a, **kw: got.append(kw.get("guards")) or (mix[0], {}))
    p = np.zeros(fa.HOP, np.float32)
    fa.make_system("stream:x.onnx@x.json", guards=True)(p, p)
    fa.make_system("stream:x.onnx@x.json")(p, p)
    assert got == [True, None]
