import numpy as np

from scripts.ceiling_analysis import aggregate_rows, erb_project_mask, oracle_spectra, safe_complex_ratio


def test_clamp_audit_uses_loss_epsilon_and_serializes_missing_metrics(tmp_path, monkeypatch):
    import json
    from scripts import ceiling_analysis as ca
    monkeypatch.setattr(ca.metrics, "stoi", lambda *args: float("nan"))
    monkeypatch.setattr(ca.metrics, "pesq_wb", lambda *args: float("nan"))
    clean = np.ones(1600, dtype=np.float32) * .1
    values = ca.finite_metrics(clean, clean)
    assert values["clamp_at_20"] == values["clamp_at_30"] == 1
    rows = [dict(bucket="clean", variant="system", **values)]
    payload = dict(aggregates=ca.aggregate_rows(rows))
    path = tmp_path / "audit.json"
    ca.write_aggregate(path, payload)
    saved = json.loads(path.read_text())
    assert saved["aggregates"][0]["mean_stoi"] is None
    assert saved["aggregates"][0]["mean_clamp_at_30"] == 1


def test_safe_complex_ratio_and_oracles_stay_finite_at_silent_bins():
    mix = np.array([[0j, 2 + 0j], [0j, 1 + 1j]], dtype=np.complex64)
    clean = np.array([[0j, 1 + 0j], [0j, 1j]], dtype=np.complex64)
    system = np.array([[0j, 0.5 + 0j], [0j, 0.25 + 0.25j]], dtype=np.complex64)
    variants = oracle_spectra(mix, clean, system, projector=np.eye(2, dtype=np.float32))
    assert set(variants) == {"raw", "oracle_irm", "oracle_iam", "oracle_complex_unrestricted",
                             "erb_projected_complex_mask_diagnostic", "system",
                             "oracle_phase_with_system_magnitude"}
    assert all(np.isfinite(value).all() for value in variants.values())
    np.testing.assert_allclose(safe_complex_ratio(clean, mix)[0, 0], 0j)


def test_unrestricted_complex_mask_reconstructs_non_silent_reference():
    mix = np.array([[2 + 1j, -1 + 3j]], dtype=np.complex64)
    clean = np.array([[1 - 2j, 4 + 0.5j]], dtype=np.complex64)
    got = oracle_spectra(mix, clean)["oracle_complex_unrestricted"]
    np.testing.assert_allclose(got, clean, rtol=1e-6, atol=1e-6)


def test_erb_projection_preserves_low_bins_and_projects_high_bins():
    mask = np.array([[1 + 2j], [3 + 4j], [5 + 6j]], dtype=np.complex64)
    projector = np.eye(3, dtype=np.float32)
    extended = np.vstack([np.zeros((65, 1), np.complex64), mask])
    got = erb_project_mask(extended, projector)
    np.testing.assert_array_equal(got[:65], extended[:65])
    np.testing.assert_array_equal(got[65:], mask)


def test_aggregate_counts_each_metric_validity_independently():
    rows = [
        {"bucket": "b", "variant": "raw", "snr_db": 1.0, "stoi": np.nan, "pesq_wb": 2.0},
        {"bucket": "b", "variant": "raw", "snr_db": np.nan, "stoi": 0.5, "pesq_wb": np.nan},
    ]
    got = aggregate_rows(rows)[0]
    assert got["n_items"] == 2
    assert (got["n_valid_snr_db"], got["n_valid_stoi"], got["n_valid_pesq_wb"]) == (1, 1, 1)
    assert (got["mean_snr_db"], got["mean_stoi"], got["mean_pesq_wb"]) == (1.0, 0.5, 2.0)
