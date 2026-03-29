"""Robustness and edge-case tests for DHFF v2 framework.

These tests validate the new capabilities introduced in the v2 gap-closure work:
reproducibility, CSV I/O, JSON export, continuous confidence, 3D scaffolding,
coverage calibration, and SCExtractorConfig behaviour.
"""
from __future__ import annotations

import json
import math
import os
import tempfile

import numpy as np
import pytest

from dhff.core.types import ObservationPoint, ScatteringCenter
from dhff.scattering_center import SCExtractorConfig
from dhff.io import RCSMeasurementLoader
from dhff.io.csv_loader import SimulationCSVLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SMALL_ENGINE_KWARGS = dict(
    scenario_name="simple_missing_feature",
    total_measurement_budget=30,
    candidate_grid_density=10,
    n_freq_candidates=10,
)


def _make_engine(**kwargs):
    from dhff.pipeline import DHFFEngine
    kw = {**_SMALL_ENGINE_KWARGS, **kwargs}
    return DHFFEngine(**kw)


def _write_synthetic_csv(path, n=20):
    """Write a tiny synthetic RCS CSV for loader tests."""
    pts = [
        ObservationPoint(
            theta=0.3 + 0.1 * i,
            phi=0.0,
            freq_hz=8e9 + i * 0.2e9,
        )
        for i in range(n)
    ]
    vals = np.array([complex(0.01 * math.cos(i), 0.01 * math.sin(i)) for i in range(n)])
    RCSMeasurementLoader.write_csv(path, pts, vals, snr_db=25.0)
    return pts, vals


# ---------------------------------------------------------------------------
# Phase 1 / Reproducibility
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_reproducibility():
    """Two runs with the same random_seed produce identical complex_nmse."""
    e1 = _make_engine(random_seed=42)
    e2 = _make_engine(random_seed=42)
    r1 = e1.run()
    r2 = e2.run()
    assert r1["error_metrics"]["complex_nmse"] == pytest.approx(
        r2["error_metrics"]["complex_nmse"], rel=1e-6
    ), "Results differ between two seeded runs"


# ---------------------------------------------------------------------------
# Phase 2 / SCExtractorConfig
# ---------------------------------------------------------------------------

def test_sc_config_relaxes_for_low_snr():
    """SCExtractorConfig(snr_db=15) produces a lower amplitude threshold."""
    default = SCExtractorConfig()
    relaxed = SCExtractorConfig(snr_db=15).effective()
    assert relaxed.amplitude_threshold_db < default.amplitude_threshold_db
    assert relaxed.merge_distance_m > default.merge_distance_m
    assert relaxed.min_peak_ratio < default.min_peak_ratio


def test_sc_config_no_change_above_25db():
    """SNR >= 25 dB should not alter thresholds."""
    cfg = SCExtractorConfig(snr_db=30)
    eff = cfg.effective()
    assert eff.amplitude_threshold_db == cfg.amplitude_threshold_db
    assert eff.merge_distance_m == cfg.merge_distance_m


@pytest.mark.slow
def test_sc_config_changes_pipeline_behavior():
    """Engine with aggressive merge distance should still complete."""
    from dhff.scattering_center import SCExtractorConfig
    engine = _make_engine(sc_config=SCExtractorConfig(merge_distance_m=0.30))
    results = engine.run()
    assert "fused_model" in results


# ---------------------------------------------------------------------------
# Phase 3 / CSV I/O
# ---------------------------------------------------------------------------

def test_csv_loader_round_trip():
    """Write synthetic RCS to CSV, reload, check values match to rtol=1e-5."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
        path = tf.name
    try:
        pts, vals = _write_synthetic_csv(path)
        loader = RCSMeasurementLoader(path)
        loaded_pts, loaded_vals = loader.load()
        assert len(loaded_pts) == len(pts)
        np.testing.assert_allclose(loaded_vals, vals, rtol=1e-5)
    finally:
        os.unlink(path)


def test_csv_loader_rejects_bad_header():
    """CSV missing required columns raises ValueError."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as tf:
        tf.write("theta_rad,phi_rad\n0.5,0.0\n")
        path = tf.name
    try:
        with pytest.raises(ValueError, match="missing required columns"):
            RCSMeasurementLoader(path).load()
    finally:
        os.unlink(path)


def test_csv_loader_rejects_theta_out_of_range():
    """theta_rad outside (0, π) raises ValueError."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as tf:
        tf.write("theta_rad,phi_rad,freq_hz,rcs_real,rcs_imag\n")
        tf.write("0.0,0.0,9e9,0.01,0.0\n")  # theta=0 is invalid
        path = tf.name
    try:
        with pytest.raises(ValueError, match="theta_rad"):
            RCSMeasurementLoader(path).load()
    finally:
        os.unlink(path)


def test_csv_loader_freq_filter():
    """freq_range_hz filter silently drops out-of-band rows."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
        path = tf.name
    try:
        pts, vals = _write_synthetic_csv(path, n=20)
        # Only keep rows in [9 GHz, 10 GHz]
        loader = RCSMeasurementLoader(path, freq_range_hz=(9e9, 10e9))
        loaded_pts, loaded_vals = loader.load()
        for p in loaded_pts:
            assert 9e9 <= p.freq_hz <= 10e9
    finally:
        os.unlink(path)


def test_csv_loader_snr_median():
    """median_snr_db returns correct value from CSV."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
        path = tf.name
    try:
        pts, vals = _write_synthetic_csv(path)
        loader = RCSMeasurementLoader(path)
        loader.load()
        assert loader.median_snr_db == pytest.approx(25.0, abs=0.1)
    finally:
        os.unlink(path)


def test_simulation_csv_loader_is_subclass():
    """SimulationCSVLoader is a subclass of RCSMeasurementLoader."""
    assert issubclass(SimulationCSVLoader, RCSMeasurementLoader)


def test_csv_loader_missing_file():
    """FileNotFoundError raised for nonexistent path."""
    with pytest.raises(FileNotFoundError):
        RCSMeasurementLoader("/nonexistent/path/rcs.csv").load()


# ---------------------------------------------------------------------------
# Phase 4 / Continuous confidence
# ---------------------------------------------------------------------------

def test_confidence_increases_with_samples():
    """More frequency samples → higher confidence (same violation score)."""
    from dhff.acquisition.classifier import _compute_confidence
    kk_result = {"kk_violation_score": 0.6}  # clear violation
    c_few = _compute_confidence(kk_result, n_freq_samples=4)
    c_many = _compute_confidence(kk_result, n_freq_samples=32)
    assert c_many > c_few


def test_confidence_near_boundary_is_low():
    """KK violation ≈ threshold → confidence close to 0.5."""
    from dhff.acquisition.classifier import _compute_confidence
    # violation=0.3 is exactly on the threshold (margin=0)
    kk_result = {"kk_violation_score": 0.3}
    c = _compute_confidence(kk_result, n_freq_samples=32)
    assert c < 0.65, f"Confidence {c} should be low near decision boundary"


def test_confidence_none_kk_returns_half():
    """No KK result → confidence 0.5."""
    from dhff.acquisition.classifier import _compute_confidence
    assert _compute_confidence(None, n_freq_samples=10) == 0.5


# ---------------------------------------------------------------------------
# Phase 5 / JSON and CSV export
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_json_export_contains_anomalies():
    """export_results_json produces valid JSON with required top-level keys."""
    engine = _make_engine(random_seed=7)
    results = engine.run()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "report.json")
        engine.export_results_json(results, path)
        with open(path) as fh:
            data = json.load(fh)
    for key in ("scenario", "freq_range_hz", "total_measurements",
                "error_metrics", "improvement_factor", "anomalies", "timestamp"):
        assert key in data, f"Key '{key}' missing from JSON export"
    assert isinstance(data["anomalies"], list)


@pytest.mark.slow
def test_csv_export_row_count():
    """CSV export row count equals number of detected anomalies."""
    import csv as csv_mod
    engine = _make_engine(random_seed=7)
    results = engine.run()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "anomalies.csv")
        engine.export_results_csv(results, path)
        with open(path) as fh:
            rows = list(csv_mod.DictReader(fh))
    assert len(rows) == len(results["anomalies_detected"])


# ---------------------------------------------------------------------------
# Phase 6 / Coverage calibration
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_coverage_68_is_measured():
    """coverage_68 is a finite float in (0, 1) after a real pipeline run."""
    engine = _make_engine(random_seed=13)
    results = engine.run()
    cov = results["error_metrics"].get("coverage_68")
    assert cov is not None, "coverage_68 not in error_metrics"
    assert math.isfinite(cov), f"coverage_68={cov} is not finite"
    assert 0.0 <= cov <= 1.0, f"coverage_68={cov} outside [0, 1]"


@pytest.mark.slow
def test_coverage_calibration_flag_present():
    """coverage_calibration_flag is one of the three valid values."""
    engine = _make_engine(random_seed=13)
    results = engine.run()
    flag = results["error_metrics"].get("coverage_calibration_flag")
    assert flag in ("well_calibrated", "over_confident", "under_confident", "unknown")


# ---------------------------------------------------------------------------
# Phase 7 / 3D scaffolding
# ---------------------------------------------------------------------------

def test_scattering_center_z_defaults_to_zero():
    """ScatteringCenter z defaults to 0.0 and can be set explicitly."""
    sc_default = ScatteringCenter(x=0.1, y=0.2, amplitude=1 + 0j)
    assert sc_default.z == 0.0

    sc_3d = ScatteringCenter(x=0.1, y=0.2, z=0.05, amplitude=1 + 0j)
    assert sc_3d.z == pytest.approx(0.05)


def test_aspect_angle_roll_defaults_to_zero():
    """AspectAngle roll_rad defaults to 0.0."""
    from dhff.core.types import AspectAngle
    aa = AspectAngle(theta=0.5, phi=0.0)
    assert aa.roll_rad == 0.0


def test_scattering_feature_z_defaults_to_zero():
    """ScatteringFeature z defaults to 0.0."""
    from dhff.synthetic.scatterer import ScatteringFeature
    feat = ScatteringFeature(
        x=0.1, y=0.2, base_amplitude=0.5 + 0j,
        freq_dependence="specular",
        angular_pattern="isotropic",
    )
    assert feat.z == 0.0


@pytest.mark.slow
def test_3d_scaffold_no_regression():
    """Pipeline still works correctly with the new z/roll_rad scaffold fields."""
    engine = _make_engine(random_seed=99)
    results = engine.run()
    # ScatteringCenters from get_scattering_centers should include z attribute
    centers = results.get("fused_model").discrepancy_model.get_parametric_centers() \
        if hasattr(results.get("fused_model", None), "discrepancy_model") \
        else []
    for c in centers:
        assert hasattr(c, "z"), "ScatteringCenter missing z attribute"
    assert "error_metrics" in results
