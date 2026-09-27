"""Native low-delay runtime (low-delay plan Task 7): builds native/vaani_ld with CMake against the pinned ONNX Runtime
and checks it against the Python reference.

  - golden-vector parity: every stage of the Task 6 vectors (frontend, limiter, analysis, step, state, synthesis) and
    the R0/R1/R2 polyphase resamplers, within the manifest's tolerance
  - C++ unit tests: state continuation, contract refusal, reset determinism, interleaved streams, zero host-side DSP
    allocations, the recovery bypass and the release-timeline simulation
  - WAV mode against vaani.low_delay_live.LowDelayStreamEngine (16 kHz run(), and 48 kHz with a resampler pair), exact
    lengths, and a mid-stream state restart that leaves the output bit-identical
  - simulate and live-mode refusals (a setting outside Section 4, R0 as a would-be deployment)

Skips (with the reason) when cmake, a C++ compiler or the fetched ONNX Runtime release is absent:
native/vaani_ld/deps/fetch_ort.sh fetches it.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native" / "vaani_ld"
VEC = ROOT / "deploy" / "dsp_reference" / "vectors_ld"
RJSON = ROOT / "deploy" / "resampler"
ORT = NATIVE / "_deps" / "onnxruntime-linux-{}-1.30.0"
ARM_A = "vaanife_ld_asym512_h96_s160_v1"
DSP = {"limiter": True, "limiter_kernel": "numba", "ref_policy": {"absent": "freeze", "ramp_samples": 3072}}


def _ort_dir() -> Path:
    import platform
    return Path(str(ORT).format("aarch64" if platform.machine() in ("aarch64", "arm64") else "x64"))


@pytest.fixture(scope="module")
def build() -> Path:
    if shutil.which("cmake") is None or (shutil.which("c++") is None and shutil.which("g++") is None):
        pytest.skip("cmake or a C++ compiler is not installed")
    if not (_ort_dir() / "include" / "onnxruntime_cxx_api.h").exists():
        pytest.skip(f"ONNX Runtime 1.30.0 not fetched ({_ort_dir()}); run native/vaani_ld/deps/fetch_ort.sh")
    out = NATIVE / "build-test"
    alsa = subprocess.run(["pkg-config", "--exists", "alsa"]).returncode == 0 if shutil.which("pkg-config") else False
    cfg = ["cmake", "-S", str(NATIVE), "-B", str(out), "-DCMAKE_BUILD_TYPE=Release", f"-DVLD_WITH_ALSA={'ON' if alsa else 'OFF'}"]
    if shutil.which("ninja"):
        cfg += ["-G", "Ninja"] if not (out / "CMakeCache.txt").exists() else []
    r = subprocess.run(cfg, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    r = subprocess.run(["cmake", "--build", str(out), "-j4"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    return out


def _run(exe: Path, *args, ok=(0,)) -> tuple[int, str]:
    r = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True, timeout=600)
    assert r.returncode in ok, f"{exe.name} exited {r.returncode}\n{r.stdout[-4000:]}\n{r.stderr[-2000:]}"
    return r.returncode, r.stdout


def _extract(dst: Path) -> Path:
    """The golden .npz files as <contract>/<case>/<key>.npy (the C++ tests read .npy)."""
    for f in VEC.rglob("*.npz"):
        d = dst / f.relative_to(VEC).with_suffix("")
        d.mkdir(parents=True, exist_ok=True)
        z = np.load(f)
        for k in z.files:
            np.save(d / f"{k}.npy", z[k])
    return dst


def _write_wav(path: Path, x: np.ndarray, sr: int):
    """(channels, n) float32 -> an IEEE-float WAV (the stdlib wave module writes PCM only, so write it by hand)."""
    x = np.asarray(x, "<f4")
    ch, n = x.shape
    data = x.T.tobytes()
    hdr = b"RIFF" + (36 + len(data)).to_bytes(4, "little") + b"WAVE" + b"fmt " + (16).to_bytes(4, "little")
    hdr += (3).to_bytes(2, "little") + ch.to_bytes(2, "little") + sr.to_bytes(4, "little")
    hdr += (sr * ch * 4).to_bytes(4, "little") + (ch * 4).to_bytes(2, "little") + (32).to_bytes(2, "little")
    path.write_bytes(hdr + b"data" + len(data).to_bytes(4, "little") + data)


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    b = path.read_bytes()
    assert b[:4] == b"RIFF" and b[8:12] == b"WAVE"
    ch = int.from_bytes(b[22:24], "little"); sr = int.from_bytes(b[24:28], "little")
    at = b.index(b"data")
    n = int.from_bytes(b[at + 4:at + 8], "little")
    return np.frombuffer(b[at + 8:at + 8 + n], "<f4").reshape(-1, ch).T.copy(), sr


def _signal(n: int, seed: int = 0):
    g = np.random.default_rng(seed)
    t = np.arange(n) / 16000
    p = (0.3 * np.sin(2 * np.pi * 150 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t)) + 0.05 * g.standard_normal(n))
    r = 0.04 * g.standard_normal(n)
    return p.astype(np.float32), r.astype(np.float32)


def test_golden_vector_parity(build, tmp_path):
    tol = json.loads((VEC / "manifest.json").read_text())["tolerance"]
    code, out = _run(build / "vld_golden_test", _extract(tmp_path / "npy"), VEC, RJSON, tol, ok=(0, 1))
    rows = [json.loads(l) for l in out.splitlines() if l.startswith("{")]
    bad = [r for r in rows if r.get("pass") is False or "error" in r]
    assert not bad, bad[:5]
    summary = rows[-1]
    assert code == 0 and summary["failures"] == 0
    checks = {(r["where"], r["check"]) for r in rows if "check" in r}
    # every contract, case and stage was compared, and all three resamplers
    contracts = [p.name for p in VEC.iterdir() if p.is_dir() and p.name != "resampler"]
    assert len(contracts) == 4
    for c in contracts:
        for case in ("speech_plus_noise", "ref_dropout", "limiter_burst", "nonfinite", "silence", "low_level"):
            assert (f"{c}/{case}", "synthesis") in checks and (f"{c}/{case}", "final_state") in checks
        for stage in ("frontend_out", "limiter", "analysis", "step_out", "state"):
            assert (f"{c}/speech_plus_noise", stage) in checks
    for rid in ("r0_linphase_kaiser193_v1", "r1_minphase_kaiser193_v1", "r2_cdelay_ls193_v1"):
        assert (rid, "interpolate3") in checks and (rid, "decimate3") in checks


def test_native_unit_tests(build):
    _, out = _run(build / "vld_unit_test", VEC, RJSON)
    assert "dsp allocations after warmup: 0" in out
    assert json.loads(out.strip().splitlines()[-1])["failures"] == 0


def _python_engine(contract: str, resampler=None):
    from vaani import backend as bk
    from vaani.low_delay_live import LowDelayStreamEngine
    b = bk.FeOrtBackend(VEC / contract / "model.onnx", audio_contract=contract)
    return LowDelayStreamEngine(contract, b, DSP, resampler=resampler)


@pytest.mark.parametrize("contract", [ARM_A, "vaanife_ld_asym512_h128_s160_v1"])
def test_wav_mode_matches_python_and_restarts_exactly(build, tmp_path, contract):
    n = 16000 + 37                                        # not a whole number of hops
    p, r = _signal(n, 1)
    _write_wav(tmp_path / "in.wav", np.stack([p, r]), 16000)
    exe = build / "vaani_ld_run"
    _, out = _run(exe, "wav", "--model", VEC / contract / "model.onnx", "--contract", contract,
                  "--in", tmp_path / "in.wav", "--out", tmp_path / "out.wav")
    rep = json.loads(out)
    y, sr = _read_wav(tmp_path / "out.wav")
    assert sr == 16000 and y.shape == (1, n)               # exact length, aligned like run()
    ref = _python_engine(contract).run(p, r)
    assert np.abs(y[0] - ref).max() <= 1e-5
    assert rep["hops"] == -(-n // int(contract.split("_h")[1].split("_")[0])) + 1
    # save the whole stream state mid-stream, restore it into a fresh engine: bit-identical output
    _run(exe, "wav", "--model", VEC / contract / "model.onnx", "--contract", contract,
         "--in", tmp_path / "in.wav", "--out", tmp_path / "split.wav", "--split-at", 57)
    assert np.array_equal(_read_wav(tmp_path / "split.wav")[0], y)


def test_wav_mode_48k_with_resampler_matches_python(build, tmp_path):
    from vaani import resampler as rs
    pair = rs.load("r1_minphase_kaiser193_v1")
    n48 = 3 * 9600 + 3 * 11
    g = np.random.default_rng(3)
    t = np.arange(n48) / 48000
    p = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.03 * g.standard_normal(n48)).astype(np.float32)
    r = (0.03 * g.standard_normal(n48)).astype(np.float32)
    _write_wav(tmp_path / "in48.wav", np.stack([p, r]), 48000)
    _run(build / "vaani_ld_run", "wav", "--model", VEC / ARM_A / "model.onnx", "--contract", ARM_A,
         "--in", tmp_path / "in48.wav", "--out", tmp_path / "out48.wav", "--resampler", RJSON / f"{pair.id}.json",
         "--split-at", 40)
    y, sr = _read_wav(tmp_path / "out48.wav")
    assert sr == 48000 and y.shape == (1, n48)
    eng = _python_engine(ARM_A, pair)
    ref = np.concatenate([eng.push(p, r), eng.flush()])
    lead = 3 * eng.c.release_lead
    assert np.abs(y[0] - ref[lead:lead + n48]).max() <= 1e-5


def test_wav_mode_refuses_a_graph_of_another_contract(build, tmp_path):
    p, r = _signal(3200)
    _write_wav(tmp_path / "in.wav", np.stack([p, r]), 16000)
    code, out = _run(build / "vaani_ld_run", "wav", "--model", VEC / ARM_A / "model.onnx",
                     "--contract", "vaanife_ld_asym512_h96_s144_v1", "--in", tmp_path / "in.wav",
                     "--out", tmp_path / "o.wav", ok=(1,))
    assert "stamped for" in json.loads(out)["error"]


def test_simulate_release_timeline(build):
    exe = build / "vaani_ld_run"
    for prof in ("early", "nominal", "edge", "jitter"):
        _, out = _run(exe, "simulate", "--contract", ARM_A, "--dproc", 1, "--hops", 20000, "--profile", prof,
                      "--resampler", RJSON / "r1_minphase_kaiser193_v1.json")
        rep = json.loads(out)
        res = rep["result"]
        assert res["late_hops"] == res["late_periods"] == res["stale_periods"] == res["misplaced"] == 0, prof
        assert res["played_ok"] == 20000 * 6 and res["max_ring"] <= 6 + 1
        assert res["delay_frames"] == 3 * 96 + 2 * 48          # 3H + (D_proc + one period)
        assert rep["budget"]["eligible"] and abs(rep["budget"]["total_ms"] - (10 + 1 / 3 + 2 + 0.6)) < 1e-3
    _, out = _run(exe, "simulate", "--contract", ARM_A, "--dproc", 1, "--hops", 20000, "--profile", "late")
    res = json.loads(out)["result"]
    assert res["late_hops"] == 20 and res["recoveries"] == 20 and res["late_periods"] == 20 * 6
    # settings Section 4 does not allow are refused before any run
    code, out = _run(exe, "simulate", "--contract", ARM_A, "--dproc", 7, ok=(3,))
    assert json.loads(out)["budget"]["refused"]
    code, out = _run(exe, "simulate", "--contract", ARM_A, "--period", 96, ok=(3,))
    assert any("8 ms" in w for w in json.loads(out)["budget"]["refused"])
    _run(exe, "simulate", "--contract", "vaanife_ld_asym512_h96_s128_v1", "--period", 96)


def test_live_refuses_the_control_resampler_as_a_deployment(build):
    exe = build / "vaani_ld_run"
    if "live mode unavailable" in _run(exe, "live", "--contract", ARM_A, ok=(1,))[1]:
        pytest.skip("built without ALSA")
    code, out = _run(exe, "live", "--model", VEC / ARM_A / "model.onnx", "--contract", ARM_A, "--dproc", 1,
                     "--resampler", RJSON / "r0_linphase_kaiser193_v1.json", ok=(3,))
    rep = json.loads(out)
    assert rep["refused"] and not rep["budget"]["eligible"] and rep["budget"]["total_ms"] > 13.0
