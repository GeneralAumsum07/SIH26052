"""G1 data gate script (scripts/data_gates.py) on a tiny synthetic corpus: it runs end to end on both mixer versions
and both paths, and its pieces (AUC, per-bin ILD, M2 draw, SPL round trip) give known answers."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
SR = 16000


def _gates():
    spec = importlib.util.spec_from_file_location("data_gates", REPO / "scripts" / "data_gates.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_auc_known_answers():
    g = _gates()
    assert g.auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert g.auc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 0.0
    assert g.auc(np.zeros(5), np.zeros(7)) == 0.5                      # mono items: all ties count half


def test_bins_ild_separates_a_boom_speech_from_a_diffuse_noise():
    g = _gates()
    rng = np.random.default_rng(0)
    t = np.arange(2 * SR) / SR
    clean = (np.sin(2 * np.pi * 300 * t) * (np.sin(2 * np.pi * 1.5 * t) > 0)).astype(np.float32)
    noise = rng.standard_normal((2, len(t))).astype(np.float32) * 0.1   # within the 50 dB live window
    m = np.stack([clean, clean * 10 ** (-12 / 20)]) + noise
    sp, nz = g.bins_ild(m, clean)
    assert abs(np.median(sp) - 12.0) < 1.0 and abs(np.median(nz)) < 1.5


def test_m2_draw_and_spl_round_trip():
    g = _gates()
    h = g.m2_histogram(n=5000)
    assert abs(h["share_-6_to_+3"] - 0.25) < 0.03 and h["min"] >= -20.0 and h["max"] <= 3.0
    assert g.spl_round_trip()["pass"]


def _corpus(tmp_path):
    rng = np.random.default_rng(1)
    t = np.arange(3 * SR) / SR
    d = tmp_path / "audio"; d.mkdir()
    sp_rows, nz_rows = [], []
    for i in range(3):
        f0 = 120 + 40 * i
        x = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 6)) * (np.sin(2 * np.pi * 2 * t) > -0.3)
        p = d / f"s{i}.wav"; sf.write(p, (0.1 * x).astype(np.float32), SR)
        sp_rows.append(dict(source_id=f"ls:s{i}", corpus="librispeech", kind="speech", group_id=f"g{i}", speaker_id=str(i),
                            path=str(p), duration_s=3.0, licence="x", split="train", sha1="", noise_class=""))
    for i, cls in enumerate(["helicopter", "vehicle", "shooting"]):
        p = d / f"n{i}.wav"; sf.write(p, (rng.standard_normal(4 * SR) * 0.1).astype(np.float32), SR)
        nz_rows.append(dict(source_id=f"mad:{cls}/{i}_0", corpus="mad", kind="noise", group_id=f"mad-{i}", speaker_id="",
                            path=str(p), duration_s=4.0, licence="x", split="train", sha1="", noise_class="stationary"))
    md = tmp_path / "man"; md.mkdir()
    pd.DataFrame(sp_rows).to_parquet(md / "librispeech_100h.parquet")
    pd.DataFrame(sp_rows).to_parquet(md / "cv_hi.parquet")
    pd.DataFrame(nz_rows).to_parquet(md / "mad.parquet")
    return md


def test_small_sample_run_writes_both_versions(tmp_path):
    from vaani.data import rirs
    g = _gates()
    md = _corpus(tmp_path)
    rirs.build_bank(tmp_path / "b.npz", n=2, seed=0, n_noise=3, max_len=4000, workers=1)
    out = tmp_path / "gates"
    res = g.main(["--items", "2", "--manifest-dir", str(md), "--bank", str(tmp_path / "b.npz"), "--out", str(out)])
    for v in (1, 2):
        j = json.loads((out / f"v{v}.json").read_text())
        for p in ("param", "room"):
            assert 0.0 <= j[p]["auc_ild"] <= 1.0 and j[p]["n_speech_bins"] > 0 and j[p]["n_noise_bins"] > 0
        assert j["spl_round_trip"]["pass"]
    assert "gate_pass" in res[2] and set(res[2]["m2_draw"]["modes"]) <= {"physical", "mono", "stereo", "low_ild"}
