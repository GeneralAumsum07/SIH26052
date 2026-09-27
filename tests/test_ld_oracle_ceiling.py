"""Task 1b oracle ceiling: mask definitions, transform round trips, the pairing against C0, the D2-margin flag and the
modulation index columns, on synthetic clips (the report itself runs on the frozen validation split)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import ld_oracle_ceiling as O  # noqa: E402
from vaani import audio_contract as ac  # noqa: E402

N = 8000


def _sig(seed=0):
    g = np.random.default_rng(seed)
    t = np.arange(N) / 16000
    clean = (0.3 * np.sin(2 * np.pi * 220 * t) * (1 + np.sin(2 * np.pi * 4 * t))).astype(np.float32)
    return clean, (g.normal(0, 0.1, N)).astype(np.float32)


def test_contract_set_is_c0_arm_a_and_arm_b():
    assert O.CONTRACTS == (ac.LEGACY_ID, *ac.ARM_A_IDS, ac.ARM_B_ID) and O.LD_HOPS == [96, 128]


@pytest.mark.parametrize("c", O.CONTRACTS)
@pytest.mark.parametrize("m", O.MASKS)
def test_noise_free_oracle_is_the_identity(c, m):
    clean, _ = _sig()
    y = O.oracle(clean, clean, c, m)
    assert y.shape == clean.shape and np.abs(y - clean).max() < 1e-3


@pytest.mark.parametrize("c", O.CONTRACTS)
def test_masks_are_bounded_and_help(c):
    from vaani.dsp import low_delay_stft as ld
    clean, noise = _sig()
    mix = clean + noise
    for m in O.MASKS:
        y = O.oracle(mix, clean, c, m)
        X = ld.analyze(torch.as_tensor(mix, dtype=torch.float64)[None], c)
        Y = ld.analyze(torch.as_tensor(y, dtype=torch.float64)[None], c)
        # re-analysis of an overlap-added output is not the masked spectrum, so compare total energy only
        assert float((Y ** 2).sum()) <= float((X ** 2).sum()) * 1.01
        err_in, err_out = np.mean((mix - clean) ** 2), np.mean((y - clean) ** 2)
        assert err_out < err_in


def test_score_clip_rows_and_modulation_columns():
    clean, noise = _sig()
    rows = O.score_clip({"id": "1", "bucket": "stationary_0", "noise_class": "stationary", "snr_db": 0},
                        np.stack([clean + noise, noise]), clean)
    assert len(rows) == len(O.CONTRACTS) * len(O.MASKS)
    for r in rows:
        assert all(np.isfinite(r[k]) for k in O.METRICS)
        for h in O.LD_HOPS:
            assert f"mod_idx_hop{h}_db" in r and f"mod_idx_clean_hop{h}_db" in r


def _rows(shift):
    """Synthetic scored rows: every low-delay contract equals C0 except the shifts given per (contract, metric)."""
    g = np.random.default_rng(3)
    rows = []
    for s in range(40):
        nc = ("stationary", "clean", "impulsive")[s % 3]
        base = {k: v + g.normal(0, 0.1) for k, v in (("snr_out", 18.0), ("stoi", 0.9), ("pesq_wb", 3.0))}
        for c in O.CONTRACTS:
            for m in O.MASKS:
                r = {"id": f"{s}", "bucket": f"{nc}_0", "noise_class": nc, "snr_in": 0, "fault": None,
                     "contract": c, "mask": m, **{k: v + shift.get((c, k), 0.0) for k, v in base.items()}}
                for h in O.LD_HOPS:
                    r[f"mod_idx_hop{h}_db"] = -40.0 + shift.get((c, "mod"), 0.0)
                    r[f"mod_idx_clean_hop{h}_db"] = -42.0
                rows.append(r)
    return rows


def test_flags_ceilings_below_c0_by_more_than_the_margin():
    s128 = ac.ARM_A_IDS[2]
    rep = O.summarise(_rows({(s128, "pesq_wb"): -0.2, (ac.ARM_B_ID, "stoi"): -0.002, (s128, "mod"): 5.0}), n_boot=200)
    assert rep["n_clips"] == 40
    d = rep["masks"]["irm"]["contracts"][s128]["d_vs_c0"]["pesq_wb"]
    assert d["mean"] == pytest.approx(-0.2) and d["below_c0_by_margin"] and d["margin"] == 0.05
    assert not rep["masks"]["irm"]["contracts"][ac.ARM_B_ID]["d_vs_c0"]["stoi"]["below_c0_by_margin"]
    assert sum("pesq_wb" in f and s128 in f for f in rep["flags"]) == 2 and len(rep["flags"]) == 2
    mi = rep["masks"]["ccm"]["contracts"][s128]["modulation_index"]
    assert mi["hop_hz"] == pytest.approx(166.667, abs=1e-3)
    assert mi["stationary"]["db_vs_c0"] == pytest.approx(5.0) and mi["clean"]["db_vs_clean"] == pytest.approx(7.0)
    assert O.summarise(_rows({}), n_boot=50)["flags"] == []


def test_cli_on_a_rendered_split(tmp_path):
    sf = pytest.importorskip("soundfile")
    for b, nc in (("stationary_0", "stationary"), ("clean_inf", "clean")):
        d = tmp_path / "val" / b; d.mkdir(parents=True)
        clean, noise = _sig(len(b))
        sf.write(d / "0001.mix.wav", np.stack([clean + noise, noise], 1), 16000)
        sf.write(d / "0001.clean.wav", clean, 16000)
        (d / "0001.json").write_text(json.dumps({"noise_class": nc, "snr_db": 0}))
    out = tmp_path / "oracle"
    assert O.main(["--eval-root", str(tmp_path), "--split", "val", "--n-boot", "50", "--out", str(out)]) == 0
    rep = json.loads((out / "report.json").read_text())
    assert rep["summary"]["n_clips"] == 2 and set(rep["contract_hashes"]) == set(O.CONTRACTS)
    assert "Flags for the owner" in (out / "report.md").read_text()
    assert len(pd.read_csv(out / "clips.csv")) == 2 * len(O.CONTRACTS) * len(O.MASKS)
    assert O.main(["--eval-root", str(tmp_path), "--split", "nope", "--out", str(out)]) == 2
