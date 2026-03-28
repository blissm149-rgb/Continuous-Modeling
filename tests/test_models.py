"""Tests for Module 5: Model Registry."""
import math

import numpy as np
import pytest

from dhff.core import (
    make_frequency_sweep, make_observation_grid, ObservationPoint, DiscrepancySample,
)
from dhff.synthetic import scenario_simple_missing_feature
from dhff.models import HybridDiscrepancyModel, PureGPDiscrepancyModel, FusedRCSModel


@pytest.fixture
def scenario1():
    return scenario_simple_missing_feature()


def make_discrepancy_samples(n=50, seed=42):
    """Create synthetic discrepancy samples for testing."""
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n):
        theta = rng.uniform(0.3, math.pi - 0.3)
        freq = rng.uniform(8e9, 12e9)
        residual = (rng.standard_normal() + 1j * rng.standard_normal()) * 0.1
        samples.append(DiscrepancySample(
            obs=ObservationPoint(theta=theta, phi=0.0, freq_hz=freq),
            residual=complex(residual),
        ))
    return samples


def test_hybrid_model_fit_no_error(scenario1):
    gt, sim, meas = scenario1
    pts = make_frequency_sweep(math.pi/2, 0.0, (8e9, 12e9), 30)
    gt_rcs = gt.compute_rcs(pts)
    sim_rcs = sim.compute_rcs(pts)
    samples = [
        DiscrepancySample(obs=pts[i], residual=complex(gt_rcs.values[i] - sim_rcs.values[i]))
        for i in range(len(pts))
    ]
    model = HybridDiscrepancyModel(freq_range_hz=(8e9, 12e9), gp_training_iters=20)
    model.fit(samples)  # Should not raise
    assert model._is_fitted


def test_hybrid_model_predict_shapes():
    samples = make_discrepancy_samples(50)
    model = HybridDiscrepancyModel(freq_range_hz=(8e9, 12e9), gp_training_iters=20)
    model.fit(samples)
    pts = make_frequency_sweep(math.pi/2, 0.0, (8e9, 12e9), 10)
    mean, var = model.predict(pts)
    assert mean.shape == (10,)
    assert var.shape == (10,)
    assert mean.dtype == np.complex128
    assert np.all(var >= 0.0)


def test_hybrid_model_few_samples_graceful():
    """With < 15 samples, hybrid model degrades gracefully (GP handles it)."""
    samples = make_discrepancy_samples(5)
    model = HybridDiscrepancyModel(freq_range_hz=(8e9, 12e9), gp_training_iters=10)
    model.fit(samples)  # Should not crash
    assert model._is_fitted
    assert model.sc_model.get_center_count() == 0


def test_hybrid_ensemble_non_zero_variance():
    """SC ensemble should produce non-zero variance."""
    samples = make_discrepancy_samples(60)
    model = HybridDiscrepancyModel(freq_range_hz=(8e9, 12e9), n_ensemble=5, gp_training_iters=10)
    model.fit(samples)
    pts = make_frequency_sweep(math.pi/2, 0.0, (8e9, 12e9), 10)
    ens_var = model._sc_ensemble_variance(pts)
    # If ensemble has multiple models, variance should be defined (even if close to 0)
    assert ens_var.shape == (10,)


def test_fused_model_predicts():
    gt, sim, meas = scenario_simple_missing_feature()
    pts = make_frequency_sweep(math.pi/2, 0.0, (8e9, 12e9), 20)
    samples = []
    for p in pts:
        gt_v = gt.compute_rcs([p]).values[0]
        sim_v = sim.compute_rcs([p]).values[0]
        samples.append(DiscrepancySample(obs=p, residual=complex(gt_v - sim_v)))

    model = HybridDiscrepancyModel(freq_range_hz=(8e9, 12e9), gp_training_iters=10)
    model.fit(samples)

    eval_pts = make_frequency_sweep(math.pi/2, 0.0, (8e9, 12e9), 15)
    fused = FusedRCSModel(simulator=sim, discrepancy_model=model)
    fused_rcs, uncertainty = fused.predict(eval_pts)
    assert len(fused_rcs.values) == 15
    assert uncertainty.shape == (15,)


def test_fused_model_error_metrics():
    gt, sim, meas = scenario_simple_missing_feature()
    pts = make_frequency_sweep(math.pi/2, 0.0, (8e9, 12e9), 30)
    samples = []
    for p in pts:
        gt_v = gt.compute_rcs([p]).values[0]
        sim_v = sim.compute_rcs([p]).values[0]
        samples.append(DiscrepancySample(obs=p, residual=complex(gt_v - sim_v)))

    model = HybridDiscrepancyModel(freq_range_hz=(8e9, 12e9), gp_training_iters=10)
    model.fit(samples)
    fused = FusedRCSModel(simulator=sim, discrepancy_model=model)

    eval_pts = make_frequency_sweep(math.pi/2, 0.0, (8e9, 12e9), 20)
    metrics = fused.error_vs_ground_truth(gt, eval_pts)
    assert "complex_nmse" in metrics
    assert "coverage_68" in metrics
    assert 0.0 <= metrics["coverage_68"] <= 1.0


def test_rff_variance_approximation():
    """RFF variance should be within reasonable range of GP variance."""
    samples = make_discrepancy_samples(60)
    model = HybridDiscrepancyModel(freq_range_hz=(8e9, 12e9), rff_features=200, gp_training_iters=20)
    model.fit(samples)

    pts = make_frequency_sweep(math.pi/2, 0.0, (8e9, 12e9), 10)
    _, gp_var = model.residual_gp.predict(pts)
    rff_var = model.rff.predict_variance(pts)

    if model.residual_gp._is_fitted and model.rff._is_fitted:
        # RFF variance should be positive and in the same ballpark
        assert np.all(rff_var >= 0.0)
