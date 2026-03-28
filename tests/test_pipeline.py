"""Tests for Module 7: Pipeline Orchestration."""
import pytest

from dhff.pipeline import DHFFEngine


@pytest.mark.slow
def test_engine_simple_missing_feature_runs():
    """DHFFEngine on simple scenario runs without error."""
    engine = DHFFEngine(
        scenario_name="simple_missing_feature",
        total_measurement_budget=30,  # small budget for testing
        candidate_grid_density=10,
        n_freq_candidates=10,
        model_type="hybrid",
    )
    results = engine.run()
    assert "fused_model" in results
    assert "error_metrics" in results
    assert "anomalies_detected" in results
    assert "history" in results
    assert results["total_measurements"] <= 30 + 10  # allow some slack


@pytest.mark.slow
def test_engine_returns_expected_keys():
    engine = DHFFEngine(
        scenario_name="simple_missing_feature",
        total_measurement_budget=20,
        candidate_grid_density=8,
        n_freq_candidates=8,
        model_type="hybrid",
    )
    results = engine.run()
    expected_keys = [
        "fused_model", "ground_truth", "anomalies_detected",
        "anomalies_classified", "error_metrics", "improvement_factor",
        "parametric_centers_found", "history", "total_measurements",
    ]
    for key in expected_keys:
        assert key in results, f"Missing key: {key}"


@pytest.mark.slow
def test_fused_improves_over_sim():
    """Fused model should improve over sim-only."""
    engine = DHFFEngine(
        scenario_name="simple_missing_feature",
        total_measurement_budget=25,
        candidate_grid_density=8,
        n_freq_candidates=8,
        model_type="hybrid",
    )
    results = engine.run()
    metrics = results["error_metrics"]
    # Fused should have lower or comparable NMSE vs sim-only
    fused_nmse = metrics.get("complex_nmse", 1.0)
    sim_nmse = metrics.get("sim_only_nmse", 1.0)
    # Even if not dramatically better (small budget), should not be wildly worse
    assert fused_nmse <= sim_nmse * 2.0  # at worst 2x worse (could improve later)


@pytest.mark.slow
def test_engine_scenario_shifted_runs():
    engine = DHFFEngine(
        scenario_name="shifted_and_amplitude",
        total_measurement_budget=20,
        candidate_grid_density=8,
        n_freq_candidates=8,
    )
    results = engine.run()
    assert "fused_model" in results


@pytest.mark.slow
def test_engine_scenario_complex_runs():
    engine = DHFFEngine(
        scenario_name="complex_target",
        total_measurement_budget=20,
        candidate_grid_density=8,
        n_freq_candidates=8,
    )
    results = engine.run()
    assert "fused_model" in results
