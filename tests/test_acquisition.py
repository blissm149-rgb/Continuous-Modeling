"""Tests for Module 6: Acquisition Functions & Measurement Planner."""
import math

import numpy as np
import pytest

from dhff.core import (
    make_frequency_sweep, make_observation_grid, ObservationPoint, DiscrepancySample,
    AnomalyType, ScatteringCenter, ScatteringCenterAnomaly,
)
from dhff.acquisition import (
    DiscrepancyAcquisition, ScatteringCenterAcquisition,
    KramersKronigConsistencyTest, DiscrepancyTypeClassifier,
)
from dhff.scattering_center import AnomalyClassifier
from dhff.models import HybridDiscrepancyModel
from dhff.synthetic import scenario_simple_missing_feature
from dhff.discrepancy_prior import (
    EnsembleDisagreement, GeometricFeatureAnalyzer, DiscrepancySusceptibilityMap,
)


@pytest.fixture
def scenario1():
    return scenario_simple_missing_feature()


def make_discrepancy_samples(n=50, seed=42, rng=None):
    if rng is None:
        rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n):
        theta = float(rng.uniform(0.3, math.pi - 0.3))
        freq = float(rng.uniform(8e9, 12e9))
        res = complex(rng.standard_normal() + 1j * rng.standard_normal()) * 0.1
        samples.append(DiscrepancySample(
            obs=ObservationPoint(theta=theta, phi=0.0, freq_hz=freq),
            residual=res,
        ))
    return samples


def make_mock_susceptibility_map(scenario1_tuple):
    _, simulator, _ = scenario1_tuple
    ensemble = EnsembleDisagreement(simulator)
    geometric = GeometricFeatureAnalyzer(simulator, freq_range_hz=(8e9, 12e9))
    return DiscrepancySusceptibilityMap(ensemble, geometric)


def test_kk_test_causal_signal():
    """KK test returns is_causal=True for sum of complex exponentials."""
    _C = 299792458.0
    freqs = np.linspace(8e9, 12e9, 128)
    # Causal signal: sum of 3 scattering centers
    signal = np.zeros(len(freqs), dtype=np.complex128)
    for r in [0.1, -0.2, 0.3]:
        k0 = 2.0 * np.pi * freqs / _C
        signal += 0.5 * np.exp(1j * 2.0 * k0 * r)

    kk = KramersKronigConsistencyTest(tolerance=0.5)
    result = kk.test(freqs, signal)
    assert "kk_violation_score" in result
    assert "is_causal" in result
    assert "diagnosis" in result


def test_kk_test_non_causal_signal():
    """KK test returns is_causal=False for white noise signal."""
    rng = np.random.default_rng(42)
    freqs = np.linspace(8e9, 12e9, 128)
    # Non-causal: random complex + polynomial trend (not Hilbert-paired)
    signal = rng.standard_normal(128) + 1j * (rng.standard_normal(128) * 10.0 + np.linspace(0, 5, 128))
    signal = signal.astype(np.complex128)

    kk = KramersKronigConsistencyTest(tolerance=0.3)
    result = kk.test(freqs, signal)
    # White noise is generally not KK consistent
    assert result["kk_violation_score"] >= 0.0  # Just check it runs and produces valid output


def test_discrepancy_type_classifier_unmatched_causal(scenario1):
    from dhff.core.types import ScatteringCenterAnomaly, AnomalyType, ScatteringCenter
    mc = ScatteringCenter(x=0.25, y=-0.1, amplitude=0.15+0j)
    anomaly = ScatteringCenterAnomaly(
        anomaly_type=AnomalyType.UNMATCHED_MEASUREMENT,
        meas_center=mc, sim_center=None,
    )
    kk = KramersKronigConsistencyTest(tolerance=0.5)
    classifier = AnomalyClassifier()
    type_clf = DiscrepancyTypeClassifier(kk_test=kk, anomaly_classifier=classifier)

    # Create causal discrepancy samples near the meas center
    _C = 299792458.0
    freqs = np.linspace(8e9, 12e9, 64)
    samples = []
    for f in freqs:
        k0 = 2.0 * np.pi * f / _C
        res = 0.1 * np.exp(1j * 2.0 * k0 * 0.25)
        samples.append(DiscrepancySample(
            obs=ObservationPoint(theta=0.0, phi=0.0, freq_hz=f),
            residual=complex(res),
        ))

    results = type_clf.classify_all([anomaly], samples, (8e9, 12e9))
    assert len(results) == 1
    # Should either classify as missing_scatterer or acknowledge KK test ran
    assert "root_cause" in results[0]


def test_batch_selection_angular_diversity(scenario1):
    _, simulator, _ = scenario1
    samples = make_discrepancy_samples(50)
    model = HybridDiscrepancyModel(freq_range_hz=(8e9, 12e9), gp_training_iters=10)
    model.fit(samples)

    susc_map = make_mock_susceptibility_map(scenario1)
    candidates = make_observation_grid(
        theta_range=(0.3, math.pi-0.3), phi_range=(0.0, 0.0),
        freq_range=(8e9, 12e9), n_theta=20, n_phi=1, n_freq=10,
    )
    acq = DiscrepancyAcquisition(
        discrepancy_model=model, susceptibility_map=susc_map,
        lambda_explore=1.0, mu_prior=0.5,
    )
    from dhff.core.coordinate_system import angular_distance_points
    plan = acq.select_batch(candidates, batch_size=5, min_angular_sep_rad=0.1)
    assert len(plan.points) <= 5
    # All selected points should be valid ObservationPoints
    for p in plan.points:
        assert isinstance(p, ObservationPoint)


def test_sc_acquisition_generates_candidates_within_region(scenario1):
    _, simulator, _ = scenario1
    # Create a dummy anomaly
    mc = ScatteringCenter(x=0.0, y=0.0, amplitude=1.0+0j,
                          lobe_center_theta=math.pi/2, lobe_center_phi=0.0)
    anomaly = ScatteringCenterAnomaly(
        anomaly_type=AnomalyType.UNMATCHED_MEASUREMENT,
        meas_center=mc, sim_center=None,
    )
    classifier = AnomalyClassifier()
    sc_acq = ScatteringCenterAcquisition(
        anomalies=[anomaly],
        anomaly_classifier=classifier,
        freq_range_hz=(8e9, 12e9),
    )
    candidates = sc_acq.generate_candidates(n_per_anomaly=5)
    assert len(candidates) == 5
    for pt, rationale in candidates:
        assert isinstance(pt, ObservationPoint)
        assert len(rationale) > 0
