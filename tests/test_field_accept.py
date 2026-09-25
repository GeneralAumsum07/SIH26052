"""G4 field acceptance script: constructions, the speech-loss metric, verdict logic, and a tiny synthetic end-to-end."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("field_accept", REPO / "scripts/field_accept.py")
fa = importlib.util.module_from_spec(spec); spec.loader.exec_module(fa)


def _speech(n=fa.NS, seed=0):
    rng = np.random.default_rng(seed); t = np.arange(n) / fa.SR
    s = 0.05 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 1.5 * t) > 0) + 0.005 * rng.standard_normal(n)
    return s.astype(np.float32)


def test_constructions_share_the_primary_and_hit_the_snr():
    s = _speech(); rng = np.random.default_rng(1)
    nL = rng.standard_normal(fa.NS).astype(np.float32) * 0.1; nR = rng.standard_normal(fa.NS).astype(np.float32) * 0.1
    ps = {}
    for c in fa.CONS:
        p, r = fa.construct(s, nL, nR, c, 5.0, np.random.default_rng(2)); ps[c] = (p, r)
    for c in fa.CONS:
        assert np.array_equal(ps[c][0], ps["M"][0])
    p = ps["M"][0]
    assert abs(10 * np.log10(fa.active_power(s) / (((p - s) ** 2).mean())) - 5.0) < 0.2
    assert np.array_equal(ps["M"][1], p) and not ps["Z"][1].any()
    assert np.allclose(ps["G"][1], p * 10 ** (-12 / 20), atol=1e-6)
    # H8: reference speech sits 8 dB under the primary's speech
    r_sp = ps["H8"][1] - (ps["H4"][1] - s * 10 ** (-4 / 20))
    assert abs(20 * np.log10(np.std(r_sp) / np.std(s)) + 8) < 0.1


def test_speech_loss_metric():
    s = _speech()
    assert fa.frame_stats(s, s)[0] == 0.0
    loss, run = fa.frame_stats(s, s * 0.01)                      # -40 dB: every active frame lost
    assert loss == 1.0 and run > 0.3
    half = s.copy(); half[fa.NS // 2:] = 0
    assert 0.3 < fa.frame_stats(s, half)[0] < 0.7


def test_longest_atten_run_and_validity_latency():
    x = np.random.default_rng(0).standard_normal(2 * fa.SR).astype(np.float32) * 0.1
    y = x.copy(); y[fa.SR // 2: fa.SR] *= 1e-3                   # 0.5 s at -60 dB
    assert abs(fa.longest_atten_run(x, y) - 0.5) < 0.03
    assert fa.longest_atten_run(x, x) == 0.0
    tr = [{"ref_informative": True}] * 10 + [{"ref_informative": False}] * 5
    assert fa.validity_latency(tr, "ref_informative") == 10 * fa.HOP / fa.SR
    assert fa.validity_latency(tr[:10], "ref_informative") is None
    assert fa.validity_latency([{"gate": 1.0}], "ref_informative") == "absent"


def test_word_survival():
    ref = [("hello", 0.0, 0.3, 0.9), ("there", 0.4, 0.6, 0.8), ("uh", 0.7, 0.8, 0.2)]
    assert fa.word_survival(ref, [("hello", 0.1, 0.3, 0.5)]) == 0.5
    assert fa.word_survival(ref, [("hello", 2.0, 2.3, 0.5)]) == 0.0
    assert fa.word_survival(None, []) is None


def _rows(loss, dsnr, g_dsnr=5.0):
    rows = []
    for i in range(4):
        for snr in (0.0, 5.0):
            for c in fa.CONS:
                rows.append({"bed": "web", "item": i, "snr": snr, "cons": c, "system": "x", "loss": loss,
                             "lost_run_s": 0.1, "snr_in": snr, "snr_out": snr + dsnr + (20 if c == "H8" else 0),
                             "stoi_in": 0.8, "stoi_out": 0.9, "pesq_out": 3.0})
            rows.append({"bed": "web", "item": i, "snr": snr, "cons": "mono", "system": "gtcrn_pretrained",
                         "loss": 0.1, "lost_run_s": 0.1, "snr_in": snr, "snr_out": snr + g_dsnr, "stoi_in": 0.8,
                         "stoi_out": 0.85, "pesq_out": 2.0})
    return rows


def test_part1_verdicts():
    st, v = fa.summarise_part1(_rows(0.01, 6.0))
    assert v == "PASS" and st["web/M"]["verdict"] == "PASS" and "ps_targets@+5dB" in st["web/H8"]["criteria"]
    st, v = fa.summarise_part1(_rows(0.5, 6.0))
    assert v == "FAIL" and st["web/Z"]["criteria"]["loss_mean<=0.06"] == "FAIL"
    st, _ = fa.summarise_part1(_rows(0.01, 3.5, g_dsnr=6.0))     # beats +3 dB but trails gtcrn by 2.5 dB
    assert st["web/M"]["criteria"]["dsnr>=gtcrn-1dB"] == "FAIL" and st["web/H4"]["verdict"] == "PASS"


def test_part2_verdict_is_tbd_without_asr_and_fails_on_long_attenuation():
    run = lambda a30: {"longest_atten30_s": a30, "word_survival": None, "vad_speech_s": None,
                       "validity_latency_s": "absent"}
    cr, v = fa.summarise_part2({"runs": {"as_is": run(0.2), "ref_zero": run(0.1), "mono_dup": run(0.2),
                                         "swapped": run(0.2)}})
    assert v == "TBD" and cr["longest_atten30<=1.0s"] == "PASS" and cr["word_survival>=0.8x_ref_zero"] == "TBD"
    _, v = fa.summarise_part2({"runs": {"as_is": run(5.0), "ref_zero": run(0.1), "mono_dup": run(9.0),
                                        "swapped": run(1.0)}})
    assert v == "FAIL"


def test_end_to_end_synthetic_passthrough(tmp_path, monkeypatch):
    """raw passthrough on synthetic beds: loss ~0, dSNR ~0 -> part 1 FAIL on dSNR only; files written."""
    monkeypatch.setattr(fa, "utterance", lambda i, split, seed: _speech(seed=i))
    noise = lambda bed, i, seed, mad: tuple(np.random.default_rng(i + k).standard_normal(fa.NS).astype(np.float32)
                                            * 0.02 for k in (0, 1))
    monkeypatch.setattr(fa, "bed_noise", noise)
    wav = tmp_path / "bed.wav"
    from vaani import live
    live.write_wav(wav, np.stack([_speech(fa.SR), _speech(fa.SR, 1)]), fa.SR)
    monkeypatch.setattr(fa, "WEB_WAV", wav)
    real_run_part2 = fa.run_part2
    monkeypatch.setattr(fa, "run_part2", lambda a, w=wav: real_run_part2(a, w))
    res = fa.main(["--system", "raw", "--name", "smoke", "--n-utt", "1", "--snrs", "5", "--beds", "web", "--asr", "off",
                   "--workers", "1", "--out", str(tmp_path / "field")])
    assert res["part1_verdict"] == "FAIL"
    m = res["part1"]["web/M"]
    assert m["loss_mean"] < 0.05 and abs(m["dsnr_mean"]) < 0.01
    assert m["criteria"]["dsnr>=+3dB"] == "FAIL" and m["criteria"]["loss_mean<=0.06"] == "PASS"
    assert res["part2"]["runs"]["as_is"]["longest_atten30_s"] == 0.0
    assert json.loads((tmp_path / "field/smoke.json").read_text())["name"] == "smoke"
    assert (tmp_path / "field/smoke.md").exists()
    again = fa.main(["--system", "raw", "--name", "smoke", "--n-utt", "1", "--snrs", "5", "--beds", "web",
                     "--summarise-only", "--out", str(tmp_path / "field")])
    assert again["part1"]["web/M"]["loss_mean"] == m["loss_mean"]


def test_report_order_headlines_z_and_puts_m_last():
    st, _ = fa.summarise_part1(_rows(0.01, 6.0))
    keys = [k for k, _ in fa._part1_rows(st)]
    assert keys == ["web/gtcrn_pretrained", "web/Z", "web/W", "web/H4", "web/H8", "web/G", "web/M"]
    assert fa.ROW["Z"].endswith("headline") and fa.ROW["M"].startswith("stress")
    assert fa.SHOW2[0] == "ref_zero" and fa.SHOW2[-1] == "mono_dup"


def test_z_runs_at_validity_zero_and_the_rest_valid(monkeypatch):
    seen = {}

    def sysfn(p, r, trace=False, ref_valid=True):
        seen.setdefault(("Z" if not r.any() else "other"), set()).add(ref_valid)
        return p, {}
    sysfn.stream, sysfn.validity0 = True, lambda: True
    monkeypatch.setattr(fa, "system", lambda spec: sysfn)
    monkeypatch.setattr(fa, "utterance", lambda i, split, seed: _speech(seed=i))
    noise = lambda bed, i, seed, mad: tuple(np.random.default_rng(k).standard_normal(fa.NS).astype(np.float32)
                                            * 0.02 for k in (0, 1))
    monkeypatch.setattr(fa, "bed_noise", noise)
    fa.part1_task(("x", "web", 0, 5.0, "val", 0, []))
    assert seen == {"Z": {False}, "other": {True}}
