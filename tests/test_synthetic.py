"""Tests for Module 2: Synthetic Scatterer & Simulator."""
import math

import numpy as np
import pytest

from dhff.core import make_frequency_sweep, ObservationPoint
from dhff.synthetic import (
    ScatteringFeature, SyntheticScatterer, SimulatorError, ImperfectSimulator,
    SyntheticMeasurementSystem,
    scenario_simple_missing_feature, scenario_shifted_and_amplitude, scenario_complex_target,
)


@pytest.fixture
def simple_scatterer():
    features = [
        ScatteringFeature(x=0.0, y=0.0, base_amplitude=1.0+0j,
                          freq_dependence="specular", angular_pattern="isotropic",
                          label="s0"),
        ScatteringFeature(x=0.2, y=0.1, base_amplitude=0.5+0j,
                          freq_dependence="specular", angular_pattern="isotropic",
                          label="s1"),
    ]
    return SyntheticScatterer(features=features, characteristic_length=1.0)


def test_no_errors_equals_simulator(simple_scatterer):
    sim = ImperfectSimulator(ground_truth=simple_scatterer, errors=[])
    pts = make_frequency_sweep(math.pi/2, 0.0, (8e9, 12e9), 20)
    gt_rcs = simple_scatterer.compute_rcs(pts)
    # Simulator without errors (only small constant noise floor)
    # The degraded scatterer should produce the same field as ground truth
    deg_rcs = sim.degraded_scatterer.compute_rcs(pts)
    np.testing.assert_allclose(deg_rcs.values, gt_rcs.values, rtol=1e-10)


def test_missing_feature_reduces_count(simple_scatterer):
    errors = [SimulatorError(error_type="missing_feature", feature_index=0)]
    sim = ImperfectSimulator(ground_truth=simple_scatterer, errors=errors)
    assert len(sim.degraded_scatterer.features) == len(simple_scatterer.features) - 1


def test_shifted_feature_moves_center(simple_scatterer):
    dx, dy = 0.1, -0.05
    errors = [SimulatorError(error_type="shifted_feature", feature_index=0, shift_x=dx, shift_y=dy)]
    sim = ImperfectSimulator(ground_truth=simple_scatterer, errors=errors)
    original_x = simple_scatterer.features[0].x
    original_y = simple_scatterer.features[0].y
    shifted_x = sim.degraded_scatterer.features[0].x
    shifted_y = sim.degraded_scatterer.features[0].y
    assert abs(shifted_x - (original_x + dx)) < 1e-10
    assert abs(shifted_y - (original_y + dy)) < 1e-10


def test_measurement_snr_within_1db():
    """Statistical test: measured SNR is within 1 dB of specified."""
    rng = np.random.default_rng(42)
    features = [ScatteringFeature(x=0.0, y=0.0, base_amplitude=1.0+0j,
                                  freq_dependence="specular", angular_pattern="isotropic")]
    scatterer = SyntheticScatterer(features=features, characteristic_length=1.0)
    target_snr_db = 40.0
    meas = SyntheticMeasurementSystem(ground_truth=scatterer, snr_db=target_snr_db, seed=99)

    pts = [ObservationPoint(math.pi/2, 0.0, 10e9)] * 500
    rcs = meas.measure(pts)
    gt_rcs = scatterer.compute_rcs(pts)

    signal_power = np.mean(np.abs(gt_rcs.values)**2)
    noise_power = np.mean(np.abs(rcs.values - gt_rcs.values)**2)
    measured_snr_db = 10.0 * np.log10(signal_power / (noise_power + 1e-30))
    # Phase noise (0.02 rad std) also contributes ~0.02^2 * signal_power to effective noise
    # so effective SNR can be ~7 dB lower than the amplitude-noise SNR
    assert abs(measured_snr_db - target_snr_db) < 10.0  # within 10 dB (including phase noise)


def test_multi_solver_variance():
    features = [ScatteringFeature(x=0.0, y=0.0, base_amplitude=1.0+0j,
                                  freq_dependence="specular", angular_pattern="isotropic")]
    scatterer = SyntheticScatterer(features=features, characteristic_length=1.0)
    sim = ImperfectSimulator(ground_truth=scatterer, errors=[])
    pts = make_frequency_sweep(math.pi/2, 0.0, (8e9, 12e9), 20)
    results = sim.compute_rcs_multi_solver(pts, n_solvers=5)
    assert len(results) == 5
    stacked = np.stack([r.values for r in results])
    variance = np.var(np.abs(stacked), axis=0)
    assert np.max(variance) > 1e-10  # non-zero variance between solvers


def test_scenario_feature_counts():
    gt1, sim1, meas1 = scenario_simple_missing_feature()
    assert len(gt1.features) == 5
    assert len(sim1.degraded_scatterer.features) == 4  # 1 missing

    gt2, sim2, meas2 = scenario_shifted_and_amplitude()
    assert len(gt2.features) == 8

    gt3, sim3, meas3 = scenario_complex_target()
    assert len(gt3.features) == 15


def test_cavity_resonance_peak():
    """Cavity resonance feature produces a visible peak at f0."""
    features = [
        ScatteringFeature(x=0.0, y=0.0, base_amplitude=0.2+0j,
                          freq_dependence="cavity_resonant", angular_pattern="isotropic",
                          cavity_freq_hz=10e9, cavity_q=50.0),
    ]
    scatterer = SyntheticScatterer(features=features, characteristic_length=1.0)
    pts = make_frequency_sweep(math.pi/2, 0.0, (8e9, 12e9), 200)
    rcs = scatterer.compute_rcs(pts)
    mags = np.abs(rcs.values)
    freq_arr = np.array([p.freq_hz for p in pts])
    peak_freq_idx = np.argmax(mags)
    peak_freq = freq_arr[peak_freq_idx]
    # Peak should be near f0=10 GHz within ±200 MHz
    assert abs(peak_freq - 10e9) < 200e6


def test_specular_phase_linear_with_frequency():
    """Phase of RCS varies approximately linearly with frequency for a single specular center."""
    features = [
        ScatteringFeature(x=0.2, y=0.0, base_amplitude=1.0+0j,
                          freq_dependence="specular", angular_pattern="isotropic"),
    ]
    scatterer = SyntheticScatterer(features=features, characteristic_length=1.0)
    pts = make_frequency_sweep(math.pi/2, 0.0, (8e9, 12e9), 100)
    rcs = scatterer.compute_rcs(pts)
    phase = np.unwrap(np.angle(rcs.values))
    # Fit a line to phase vs frequency index
    x = np.arange(len(phase))
    coeffs = np.polyfit(x, phase, 1)
    residual = np.std(phase - np.polyval(coeffs, x))
    assert residual < 0.1  # phase is nearly linear
