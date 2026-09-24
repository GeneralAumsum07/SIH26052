"""Known answers for the clustered bootstrap and per-clip pass rate, and the r7 headline from the committed CSVs."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vaani import report

ROOT = Path(__file__).resolve().parents[1]
R7 = ROOT / "results_r2/r7/r7_e256_wr64_cascade_eval_r2.csv"
GEN = ROOT / "results_r2/r7/r7_e256_wr64_cascade_gen.csv"


def _load_breakdown():
    spec = importlib.util.spec_from_file_location("r7_breakdown", ROOT / "scripts/r7_breakdown.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_pass_all_is_strict_and_nan_fails():
    df = pd.DataFrame({"snr_out": [15.0, 15.1, 20, np.nan], "stoi": [0.9, 0.9, 0.85, 0.9], "pesq_wb": [3, 3, 3, 3]})
    assert report.pass_all(df).tolist() == [False, True, False, False]
    assert report.pass_rate(df) == 0.25
    assert np.isnan(report.pass_rate(df.iloc[:0]))


def test_cluster_ci_known_answers():
    # constant data: the interval collapses onto the mean
    assert report.cluster_ci([2.0] * 6, list("aabbcc")) == (2.0, 2.0, 2.0)
    # two clusters of equal size: every resample mean is 0, 0.5 or 1
    m, lo, hi = report.cluster_ci([0, 0, 1, 1], list("aabb"), n=2000)
    assert m == 0.5 and lo == 0.0 and hi == 1.0
    # unequal clusters: the pooled item mean is weighted by cluster size, not a mean of cluster means
    assert report.cluster_ci([0, 0, 0, 3], list("aaab"))[0] == 0.75
    # NaNs drop out before clustering
    assert report.cluster_ci([1, np.nan, 1], list("abc"))[0] == 1.0
    assert np.isnan(report.cluster_ci([np.nan], ["a"])[0])


def test_cluster_ci_is_wider_than_row_ci_for_correlated_clips():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 40)
    x = np.repeat(base, 5) + rng.normal(0, 0.01, 200)   # five near-copies of each scene
    c = np.repeat(np.arange(40), 5)
    _, rlo, rhi = report.ci(x); _, clo, chi = report.cluster_ci(x, c)
    assert (chi - clo) > 1.8 * (rhi - rlo)


def test_scene_clusters_join_snr_and_id():
    df = pd.DataFrame({"snr_in": [0.0, 0.0, 5.0, np.inf], "id": ["0001", "0001", "0001", "0001"]})
    assert report.scene_clusters(df).nunique() == 3


def test_nominal_mask_matches_main_filter():
    df = pd.DataFrame({"clipped": [False, True, False, False, False], "ref_dropout": [False] * 4 + [True],
                       "fault": [None, None, "fault_none", None, None], "snr_in": [0, 5, 5, 15, 10]})
    assert report.nominal_mask(df).tolist() == [True, False, False, False, False]


@pytest.mark.skipif(not R7.exists(), reason="committed r7 CSV absent")
def test_headline_reproduces_from_committed_csv():
    bd = _load_breakdown()
    df = bd.load(str(R7))
    nom = df[df.nominal]
    assert len(nom) == 617
    assert (round(nom.snr_out.mean(), 3), round(nom.stoi.mean(), 3), round(nom.pesq_wb.mean(), 3)) == (14.864, 0.917, 2.462)
    assert round(100 * nom.pass3.mean(), 1) == 35.5
    assert len(df) == 2280 and round(100 * df.pass3.mean(), 1) == 24.1
    burst = df[df.fault.fillna("").str.startswith("fault_burst")]
    assert len(burst) == 240 and round(100 * burst.pass3.mean(), 1) == 3.8
    rows = {r["label"]: r for r in bd.headline_rows(df, n_boot=200)}
    r = rows["Nominal (0/5/10 dB, unclipped, no fault)"]
    assert r["n"] == 617 and r["snr_out_cl"][1] < r["snr_out"][0] < r["snr_out_cl"][2]


@pytest.mark.skipif(not GEN.exists(), reason="committed eval_gen CSV absent")
def test_per_grid_reproduces_from_committed_csv():
    bd = _load_breakdown()
    gen = bd.load(str(GEN))
    nom = gen[gen.nominal]
    st, ch = nom[nom.noise_class == "stationary"], nom[nom.noise_class == "changing"]
    assert len(st) == 102 and len(ch) == 99
    assert (round(st.snr_out.mean(), 3), round(st.stoi.mean(), 3), round(st.pesq_wb.mean(), 3)) == (14.295, 0.932, 2.490)
    assert (round(ch.snr_out.mean(), 2), round(ch.stoi.mean(), 3), round(ch.pesq_wb.mean(), 3)) == (18.37, 0.970, 3.153)


def test_paired_delta_and_gap():
    bd = _load_breakdown()
    s = pd.DataFrame({"bucket": ["b", "b"], "id": ["1", "2"], "snr_out": [10.0, 12.0], "stoi": [0.9, 0.8], "pesq_wb": [2.0, 3.0]})
    raw = s.assign(snr_out=[4.0, 5.0], stoi=[0.7, 0.7], pesq_wb=[1.0, 1.0])
    m = bd.paired(s, raw)
    assert m.snr_out_d.tolist() == [6.0, 7.0] and m.pesq_wb_d.tolist() == [1.0, 2.0]
    assert bd.gap_ci([3, 3, 3], [1, 1])[:3] == (2.0, 2.0, 2.0)
    with pytest.raises(pd.errors.MergeError):
        bd.paired(pd.concat([s, s]), raw)   # duplicated keys must not silently double-count


def test_regenerates_both_reports(tmp_path):
    bd = _load_breakdown()
    if not (R7.exists() and GEN.exists()): pytest.skip("committed r7 CSVs absent")
    import sys
    argv = sys.argv
    sys.argv = ["r7_breakdown", "--r7", str(R7), "--gen", str(GEN), "--raw", str(ROOT / "results_r2/raw.csv"),
                "--raw-gen", str(tmp_path / "missing.csv"), "--out", str(tmp_path / "bd.md"),
                "--gen-out", str(tmp_path / "pg.md"), "--n-boot", "100"]
    try: bd.main()
    finally: sys.argv = argv
    bdt, pgt = (tmp_path / "bd.md").read_text(encoding="utf-8"), (tmp_path / "pg.md").read_text(encoding="utf-8")
    assert "| Nominal (0/5/10 dB, unclipped, no fault) | 617 | 14.864" in bdt and "35.5%" in bdt
    assert "| stationary (registered) | 102 | 14.295" in pgt and "TBD:" in pgt
