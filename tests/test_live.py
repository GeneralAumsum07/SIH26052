"""The live engine must be the system the eval scores: same DSP order, same graph, same synthesis."""
import json

import numpy as np
import torch

from vaani import export, live
from vaani.dsp import pipeline, stft
from vaani.models.vaani_net import VaaniNet

DSP = {"blocking": True, "controller": {"block_margin_db": 10.0, "diff_jump_max_db": 3.0}, "limiter": True}


def _graph(tmp_path):
    mc = dict(df_order=3, film=False, coh=True)
    m = VaaniNet(**mc)
    with torch.no_grad():   # zero-init taps would hide a broken df path
        m.df.conv.weight.normal_(0, 0.05); m.df.conv.bias.normal_(0, 0.05)
    ck = tmp_path / "m.pt"
    torch.save({"model": m.state_dict(), "step": 0,
                "config": {"model": "vaani", "controller_on": True, "dsp": DSP, "model_cfg": mc}}, ck)
    return ck, export.export(ck, tmp_path / "m.onnx")


def _mix(seconds=2.0, seed=0):
    """Speech-ish primary, correlated noise on both mics, and one loud far-field burst so the limiter,
    the blocking matrix and the controller's burst path all run."""
    rng = np.random.default_rng(seed)
    n = int(seconds * 16000); t = np.arange(n) / 16000
    noise = rng.standard_normal(n).astype(np.float32) * 0.03
    speech = (0.1 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 3 * t) > 0)).astype(np.float32)
    prim, ref = speech + noise, 0.3 * speech + np.roll(noise, 3)
    b = slice(n // 2, n // 2 + 400)
    burst = rng.standard_normal(400).astype(np.float32)
    prim[b] += burst; ref[b] += burst
    return np.stack([prim, ref]).astype(np.float32)


def _offline(onnx, mix):
    r = pipeline.run(mix, controller_on=True, dsp_cfg=DSP)
    x = torch.from_numpy(r["mix"])[None]
    spec6 = torch.cat([stft.stft(x[:, 0]), stft.stft(x[:, 1]), stft.stft(torch.from_numpy(r["n_hat"])[None])], -1).numpy()
    sess = export.load_session(onnx); names, zero = export.zero_caches(sess)
    out, _ = export.stream_onnx(sess, spec6, np.ascontiguousarray(r["features"][None], np.float32), names, zero)
    return r, stft.istft(torch.from_numpy(out), length=mix.shape[1])[0].numpy()


def test_stream_engine_matches_offline_path(tmp_path):
    _, onnx = _graph(tmp_path)
    mix = _mix()
    r, y_off = _offline(onnx, mix)
    # the offline STFT reflects the clip's own start into frame 0; hand the engine the same left context
    lc = np.stack([r["mix"][0][256:0:-1], r["mix"][1][256:0:-1], r["n_hat"][256:0:-1]])
    eng = live.StreamEngine(onnx, True, DSP, left_context=lc)
    B = mix.shape[1] // live.HOP
    y = np.concatenate([eng.process(mix[0, j * 256:(j + 1) * 256], mix[1, j * 256:(j + 1) * 256]) for j in range(B)])
    y = y[live.HOP:]                                   # one hop behind: the first hop is the left context
    assert len(y) == (B - 1) * live.HOP
    assert np.abs(y - y_off[:len(y)]).max() < 1e-5
    assert np.abs(y_off).max() > 1e-3                  # a vacuous pass on silence would prove nothing


def test_stream_engine_rejects_wrong_block_size(tmp_path):
    _, onnx = _graph(tmp_path)
    eng = live.StreamEngine(onnx, True, DSP)
    try:
        eng.process(np.zeros(255), np.zeros(255))
    except ValueError:
        return
    raise AssertionError("a short block must be refused, not silently misframed")


def test_resamplers_are_block_size_invariant_and_unity_gain():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 768 * 6)).astype(np.float32)
    one = live.Decimate3(2)(x)
    d = live.Decimate3(2)
    pieces = np.concatenate([d(x[:, i:i + 384]) for i in range(0, x.shape[1], 384)], axis=1)
    assert np.allclose(one, pieces, atol=1e-6)

    t = np.arange(16000 * 2) / 16000
    s = np.sin(2 * np.pi * 1000 * t)[None].astype(np.float32)          # 1 kHz, well inside the passband
    up = live.Interpolate3(1); down = live.Decimate3(1)
    y = np.concatenate([down(up(s[:, i:i + 256])) for i in range(0, s.shape[1], 256)], axis=1)[0]
    delay = 2 * (len(live.lowpass_fir()) - 1) // 2 // 3                # 2 x 96 samples at 48 kHz = 64 at 16 kHz
    assert np.abs(y[delay + 1000:-1000] - s[0, 1000:-1000 - delay]).max() < 0.01


def test_decimator_rejects_aliases():
    t = np.arange(48000) / 48000
    alias = np.sin(2 * np.pi * 9000 * t)[None].astype(np.float32)       # would fold to 7 kHz at 16 kHz
    y = live.Decimate3(1)(alias[:, :48000 // 768 * 768])[0][200:]
    assert 20 * np.log10(np.abs(y).max() + 1e-12) < -60


def test_wav_round_trip_and_float_wavs(tmp_path):
    x = (np.random.default_rng(2).standard_normal((2, 1000)) * 0.1).astype(np.float32)
    live.write_wav(tmp_path / "a.wav", x, 16000)
    y, sr = live.read_wav(tmp_path / "a.wav")
    assert sr == 16000 and y.shape == x.shape and np.abs(y - x).max() < 1e-4
    # the eval sets are IEEE float32 WAVs (format tag 3), which the stdlib wave module refuses
    import struct
    data = x.T.astype("<f4").tobytes()
    hdr = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 3, 2, 16000, 128000, 8, 32)
    (tmp_path / "f.wav").write_bytes(hdr + b"data" + struct.pack("<I", len(data)) + data)
    z, _ = live.read_wav(tmp_path / "f.wav")
    assert np.array_equal(z, x)


def test_capture_loop_file_mode_is_the_engine(tmp_path):
    import importlib.util, sys
    from pathlib import Path
    ck, onnx = _graph(tmp_path)
    live.write_model_config(ck, tmp_path / "cfg.json")
    assert json.loads((tmp_path / "cfg.json").read_text())["dsp"] == DSP
    mix = _mix(1.0, seed=3)
    live.write_wav(tmp_path / "in.wav", mix, 16000)
    spec = importlib.util.spec_from_file_location("capture_loop", Path(__file__).resolve().parents[1] / "scripts" / "capture_loop.py")
    cl = importlib.util.module_from_spec(spec); spec.loader.exec_module(cl)
    cl.main(["--onnx", str(onnx), "--config", str(tmp_path / "cfg.json"),
             "--in-wav", str(tmp_path / "in.wav"), "--out-wav", str(tmp_path / "out.wav")])
    out, sr = live.read_wav(tmp_path / "out.wav")
    x, _ = live.read_wav(tmp_path / "in.wav")                           # what the CLI saw: int16-quantised input
    eng = live.StreamEngine(onnx, True, DSP)
    ref = np.concatenate([eng.process(x[0, j * 256:(j + 1) * 256], x[1, j * 256:(j + 1) * 256])
                          for j in range(x.shape[1] // 256)])[256:]
    ref = np.clip(ref, -1, 1)                                            # the WAV writer clips (the burst, untrained weights)
    assert sr == 16000 and np.abs(out[0, :len(ref)] - ref).max() < 2e-4  # int16 output quantisation
