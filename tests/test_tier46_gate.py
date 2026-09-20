"""Paired kill gate for the Tier 4.6 second-stage candidates (plan 2026-09-20-tier46-second-stage, §2/§3)."""
import numpy as np, pandas as pd, pytest

from vaani import tier46_gate as g

COLS = "system,id,bucket,noise_class,snr_in,clipped,ref_dropout,impulse_peak_db,fault,speech_source,snr_out,si_sdr,stoi,pesq_wb,recovery_s"


def _rows(n=40, seed=0, snr=13.0, stoi=0.9, pesq=2.3, system="a", buckets=("stationary_0", "changing_5", "fault_burst_p24dB")):
    rng = np.random.default_rng(seed); out = []
    for b in buckets:
        fault = "burst_p24dB" if b.startswith("fault_") else ""
        snr_in = 0 if b.startswith("fault_") else float(b.split("_")[-1])
        for i in range(n):
            out.append(dict(system=system, id=f"{i:04d}", bucket=b, noise_class=b.split("_")[0], snr_in=snr_in, clipped=False,
                            ref_dropout=False, impulse_peak_db=24 if fault else np.nan, fault=fault or np.nan, speech_source="ls:x",
                            snr_out=snr + rng.normal(0, 0.5), si_sdr=snr, stoi=stoi + rng.normal(0, 0.005), pesq_wb=pesq + rng.normal(0, 0.05),
                            recovery_s=(0.1 if fault else np.nan)))
    return pd.DataFrame(out)


def _write(tmp_path, name, df):
    p = tmp_path / f"{name}.csv"; df.to_csv(p, index=False); return p


PROTO = {"test_keys": None}   # filled per test from the anchor frame


def _proto(df):
    return {"n_items": len(df), "keys": sorted(map(tuple, df[["bucket", "id"]].astype(str).values.tolist()))}


def test_shifted_candidate_passes_postfilter_gate(tmp_path):
    a = _rows(); c = a.copy(); c["system"] = "c"; c["snr_out"] += 0.6; c["pesq_wb"] += 0.06
    r = g.compare(_write(tmp_path, "a", a), _write(tmp_path, "c", c), _proto(a), "postfilter")
    assert r["complete"] and r["gates"]["utility"] and r["gates"]["intelligibility"] and r["pass"]
    assert abs(r["nominal"]["d_snr_out"]["mean"] - 0.6) < 1e-6


def test_snr_gain_with_stoi_loss_fails(tmp_path):
    a = _rows(); c = a.copy(); c["snr_out"] += 1.0; c["pesq_wb"] += 0.1; c["stoi"] -= 0.01
    r = g.compare(_write(tmp_path, "a", a), _write(tmp_path, "c", c), _proto(a), "postfilter")
    assert r["gates"]["utility"] and not r["gates"]["intelligibility"] and not r["pass"]


def test_thresholds_are_inclusive_at_equality(tmp_path):
    # a constant +0.25 dB / +0.02 PESQ shift with zero STOI change sits exactly on the post-filter utility line
    a = _rows(); c = a.copy(); c["snr_out"] += 0.25; c["pesq_wb"] += 0.02
    r = g.compare(_write(tmp_path, "a", a), _write(tmp_path, "c", c), _proto(a), "postfilter")
    assert r["gates"]["utility"]
    r = g.compare(_write(tmp_path, "a", a), _write(tmp_path, "c", c), _proto(a), "refiner")
    assert not r["gates"]["utility"]   # refiner needs +0.5 / +0.05


def test_pairing_ignores_row_order(tmp_path):
    a = _rows(); c = a.sample(frac=1, random_state=3).copy(); c["snr_out"] += 0.6; c["pesq_wb"] += 0.06
    r = g.compare(_write(tmp_path, "a", a), _write(tmp_path, "c", c), _proto(a), "postfilter")
    assert abs(r["nominal"]["d_snr_out"]["mean"] - 0.6) < 1e-6 and r["pass"]


def test_missing_duplicate_or_nan_rows_are_incomplete(tmp_path):
    a = _rows()
    for mutate in (lambda d: d.iloc[1:], lambda d: pd.concat([d, d.iloc[:1]]), lambda d: d.assign(stoi=d.stoi.where(d.index != 3))):
        c = mutate(a.copy())
        r = g.compare(_write(tmp_path, "a", a), _write(tmp_path, "c", c), _proto(a), "postfilter")
        assert not r["complete"] and not r["pass"]


def test_new_unrecovered_burst_fails_recovery_gate(tmp_path):
    a = _rows(); c = a.copy(); c["snr_out"] += 0.6; c["pesq_wb"] += 0.06
    c.loc[c.bucket.str.startswith("fault_burst"), "recovery_s"] = [np.inf] + [0.1] * (c.bucket.str.startswith("fault_burst").sum() - 1)
    r = g.compare(_write(tmp_path, "a", a), _write(tmp_path, "c", c), _proto(a), "postfilter")
    assert r["complete"] and not r["gates"]["recovery"] and not r["pass"]


def test_bootstrap_is_seeded_and_clustered(tmp_path):
    a = _rows(); c = a.copy(); c["snr_out"] += 0.6; c["pesq_wb"] += 0.06
    r1 = g.compare(_write(tmp_path, "a", a), _write(tmp_path, "c", c), _proto(a), "postfilter")
    r2 = g.compare(_write(tmp_path, "a", a), _write(tmp_path, "c", c), _proto(a), "postfilter")
    assert r1["nominal"]["d_snr_out"]["lo"] == r2["nominal"]["d_snr_out"]["lo"]
    assert r1["n_clusters"] == 40   # one cluster per item id, every bucket variant kept together
