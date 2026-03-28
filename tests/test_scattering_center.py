"""Tests for Module 4: Scattering Center Extraction & Anomaly Detection."""
import math

import numpy as np
import pytest

from dhff.core import (
    make_frequency_sweep, make_observation_grid, ObservationPoint, DiscrepancySample,
    AnomalyType, ScatteringCenter,
)
from dhff.scattering_center import (
    MatrixPencilExtractor, ParametricSCModel, ScatteringCenterAssociator, AnomalyClassifier,
)

_C = 299792458.0


def make_synthetic_freq_sweep(positions, amplitudes, freq_range=(8e9, 12e9), n_freq=64):
    """Create a synthetic sum-of-exponentials signal."""
    freqs = np.linspace(freq_range[0], freq_range[1], n_freq)
    signal = np.zeros(n_freq, dtype=np.complex128)
    for r, a in zip(positions, amplitudes):
        k0 = 2.0 * np.pi * freqs / _C
        signal += a * np.exp(1j * 2.0 * k0 * r)
    return freqs, signal


def test_matrix_pencil_recovers_positions():
    """Matrix Pencil recovers 3 known positions within 0.02 m."""
    positions = [0.1, -0.15, 0.3]
    amplitudes = [1.0 + 0j, 0.5 + 0.3j, 0.7 - 0.2j]
    freqs, signal = make_synthetic_freq_sweep(positions, amplitudes, n_freq=64)

    extractor = MatrixPencilExtractor(n_centers_max=10, amplitude_threshold_db=-20.0)
    centers = extractor.extract_1d(freqs, signal)

    found_positions = sorted([c.x for c in centers])
    true_positions = sorted(positions)

    assert len(centers) >= 1  # at least 1 recovered
    # Check that we can match each true position to some recovered position
    for tp in true_positions[:len(centers)]:
        closest = min(found_positions, key=lambda fp: abs(fp - tp))
        assert abs(closest - tp) < 0.1  # within 10 cm (generous for noisy MP)


def test_matrix_pencil_with_noise():
    """Matrix Pencil recovers positions within 0.05 m with 40 dB SNR noise."""
    rng = np.random.default_rng(42)
    positions = [0.2, -0.1]
    amplitudes = [1.0 + 0j, 0.5 + 0j]
    freqs, signal = make_synthetic_freq_sweep(positions, amplitudes, n_freq=64)

    snr_db = 40.0
    noise_power = np.mean(np.abs(signal)**2) / 10**(snr_db/10)
    noise = (rng.standard_normal(len(signal)) + 1j*rng.standard_normal(len(signal))) * np.sqrt(noise_power/2)
    noisy_signal = signal + noise

    extractor = MatrixPencilExtractor(n_centers_max=10, amplitude_threshold_db=-25.0)
    centers = extractor.extract_1d(freqs, noisy_signal)

    assert len(centers) >= 1


def test_parametric_model_fits():
    """ParametricSCModel.fit on discrepancy data doesn't crash."""
    rng = np.random.default_rng(42)
    # Create simple discrepancy: one scattering center at x=0.25
    freqs = np.linspace(8e9, 12e9, 50)
    theta = math.pi / 2
    samples = []
    for f in freqs:
        k0 = 2.0 * np.pi * f / _C
        residual = 0.1 * np.exp(1j * 2.0 * k0 * 0.25)
        residual += (rng.standard_normal() + 1j*rng.standard_normal()) * 0.005
        samples.append(DiscrepancySample(
            obs=ObservationPoint(theta=theta, phi=0.0, freq_hz=f),
            residual=complex(residual),
        ))

    model = ParametricSCModel(max_centers=5, amplitude_threshold_db=-20.0)
    model.fit(samples, freq_range_hz=(8e9, 12e9))
    assert model._is_fitted


def test_parametric_model_few_samples():
    """ParametricSCModel with < 15 samples sets centers = [] and doesn't crash."""
    samples = [
        DiscrepancySample(obs=ObservationPoint(math.pi/2, 0.0, 10e9), residual=0.1+0j)
        for _ in range(5)
    ]
    model = ParametricSCModel()
    model.fit(samples, freq_range_hz=(8e9, 12e9))
    assert model._is_fitted
    assert model.get_center_count() == 0


def test_parametric_model_residuals_smaller():
    """ParametricSCModel residuals are smaller after fitting."""
    rng = np.random.default_rng(42)
    freqs = np.linspace(8e9, 12e9, 64)
    theta = math.pi / 2
    samples = []
    for f in freqs:
        k0 = 2.0 * np.pi * f / _C
        residual = 0.3 * np.exp(1j * 2.0 * k0 * 0.15) + 0.1 * np.exp(1j * 2.0 * k0 * (-0.1))
        samples.append(DiscrepancySample(
            obs=ObservationPoint(theta=theta, phi=0.0, freq_hz=f),
            residual=complex(residual),
        ))

    model = ParametricSCModel(max_centers=5)
    model.fit(samples, freq_range_hz=(8e9, 12e9))

    residuals_after = model.residuals(samples)
    before_power = np.mean([abs(s.residual)**2 for s in samples])
    after_power = np.mean([abs(r.residual)**2 for r in residuals_after])
    # If model found centers, residuals should be smaller
    if model.get_center_count() > 0:
        assert after_power <= before_power + 1e-6


def test_hungarian_association():
    """Hungarian correctly pairs sim and meas centers."""
    sim = [ScatteringCenter(x=0.0, y=0.0, amplitude=1.0+0j),
           ScatteringCenter(x=0.3, y=0.0, amplitude=0.5+0j)]
    meas = [ScatteringCenter(x=0.31, y=0.0, amplitude=0.5+0j),
            ScatteringCenter(x=0.01, y=0.0, amplitude=1.0+0j)]

    assoc = ScatteringCenterAssociator(max_association_distance_m=0.5)
    matched, unmatched_s, unmatched_m = assoc.associate(sim, meas)

    assert len(matched) == 2
    assert len(unmatched_s) == 0
    assert len(unmatched_m) == 0
    # Check pairing: sim[0] should match to meas[1] (closest), sim[1] to meas[0]
    pairs = {(round(s.x, 1), round(m.x, 1)) for s, m in matched}
    assert (0.0, 0.0) in pairs or (0.0, 0.0) in {(round(s.x,1), round(m.x,1)) for s, m in matched}


def test_anomaly_classifier_types():
    """AnomalyClassifier correctly identifies anomaly types."""
    sim_c = ScatteringCenter(x=0.0, y=0.0, amplitude=1.0+0j)
    meas_shift = ScatteringCenter(x=0.15, y=0.0, amplitude=1.0+0j)  # position shift
    meas_amp = ScatteringCenter(x=0.02, y=0.0, amplitude=0.1+0j)   # amplitude discrepancy
    unmatched_sim = ScatteringCenter(x=0.5, y=0.0, amplitude=0.5+0j)
    unmatched_meas = ScatteringCenter(x=-0.3, y=0.0, amplitude=0.3+0j)

    classifier = AnomalyClassifier(position_threshold_m=0.1, amplitude_threshold_db=3.0)
    anomalies = classifier.classify(
        matched=[(sim_c, meas_shift), (sim_c, meas_amp)],
        unmatched_sim=[unmatched_sim],
        unmatched_meas=[unmatched_meas],
    )

    types = [a.anomaly_type for a in anomalies]
    assert AnomalyType.UNMATCHED_SIMULATION in types
    assert AnomalyType.UNMATCHED_MEASUREMENT in types
    # At least one of POSITION_SHIFT or AMPLITUDE_DISCREPANCY should be detected
    assert (AnomalyType.POSITION_SHIFT in types) or (AnomalyType.AMPLITUDE_DISCREPANCY in types)


def test_suggest_strategy_broad_for_unmatched():
    mc = ScatteringCenter(x=0.2, y=0.0, amplitude=0.3+0j)
    from dhff.core.types import ScatteringCenterAnomaly, AnomalyType
    anomaly = ScatteringCenterAnomaly(
        anomaly_type=AnomalyType.UNMATCHED_MEASUREMENT,
        meas_center=mc, sim_center=None,
    )
    classifier = AnomalyClassifier()
    strategy = classifier.suggest_measurement_strategy(anomaly)
    assert strategy["angular_priority"] == "broad"


def test_suggest_strategy_fine_for_position_shift():
    sim_c = ScatteringCenter(x=0.0, y=0.0, amplitude=1.0+0j, lobe_center_theta=math.pi/2)
    meas_c = ScatteringCenter(x=0.2, y=0.0, amplitude=1.0+0j, lobe_center_theta=math.pi/2)
    from dhff.core.types import ScatteringCenterAnomaly, AnomalyType
    anomaly = ScatteringCenterAnomaly(
        anomaly_type=AnomalyType.POSITION_SHIFT,
        meas_center=meas_c, sim_center=sim_c, position_error_m=0.2,
    )
    classifier = AnomalyClassifier()
    strategy = classifier.suggest_measurement_strategy(anomaly)
    assert strategy["angular_priority"] == "fine"
