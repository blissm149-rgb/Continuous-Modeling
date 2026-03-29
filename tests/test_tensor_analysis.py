"""Tests for dhff/tensor_analysis/ — all fast (no slow marker)."""
from __future__ import annotations

import cmath
import math

import numpy as np
import pytest

from dhff.core.types import ObservationPoint
from dhff.tensor_analysis import (
    TensorSensitivityMap,
    GradientAnalyzer,
    ISARAnalyzer,
    SpectralAnalyzer,
    CancellationDetector,
)


# ---------------------------------------------------------------------------
# Toy tensor builders
# ---------------------------------------------------------------------------

def _az():   return np.linspace(0.2, math.pi - 0.2, 11)
def _el():   return np.linspace(-0.3, 0.3, 5)
def _freq(): return np.linspace(8e9, 12e9, 10)


def _single_scatterer_tensor(x=0.25, y=-0.10, amp=1.0, az=None, el=None, freq=None):
    """Construct an analytic (az × el × freq) tensor from a single point scatterer."""
    az   = az   if az   is not None else _az()
    el   = el   if el   is not None else _el()
    freq = freq if freq is not None else _freq()
    c = 3e8
    T = np.zeros((len(az), len(el), len(freq)), dtype=complex)
    for i, a in enumerate(az):
        for j, e in enumerate(el):
            for k, f in enumerate(freq):
                k0 = 2 * math.pi * f / c
                phase = 2.0 * k0 * (x * math.cos(a) + y * math.sin(a))
                T[i, j, k] = amp * math.cos(e) * cmath.exp(1j * phase)
    return T


def _two_scatterer_tensor(x1=0.25, y1=-0.1, x2=0.40, y2=0.05, az=None, el=None, freq=None):
    az   = az   if az   is not None else _az()
    el   = el   if el   is not None else _el()
    freq = freq if freq is not None else _freq()
    return (
        _single_scatterer_tensor(x1, y1, amp=1.0, az=az, el=el, freq=freq)
        + _single_scatterer_tensor(x2, y2, amp=0.8, az=az, el=el, freq=freq)
    )


def _lorentzian_tensor(f0=10e9, Q=15, amp=1.0, az=None, el=None, freq=None):
    """Tensor with a cavity Lorentzian across frequency at all angles."""
    az   = az   if az   is not None else _az()
    el   = el   if el   is not None else _el()
    freq = freq if freq is not None else _freq()
    T = np.zeros((len(az), len(el), len(freq)), dtype=complex)
    for k, f in enumerate(freq):
        lor = 1.0 / (1.0 + 1j * Q * (f / f0 - f0 / f))
        T[:, :, k] = amp * lor
    return T


def _constant_tensor(az=None, el=None, freq=None):
    az   = az   if az   is not None else _az()
    el   = el   if el   is not None else _el()
    freq = freq if freq is not None else _freq()
    return np.ones((len(az), len(el), len(freq)), dtype=complex)


def _make_tsm(tensor, az=None, el=None, freq=None):
    az   = az   if az   is not None else _az()
    el   = el   if el   is not None else _el()
    freq = freq if freq is not None else _freq()
    return TensorSensitivityMap(tensor, az, el, freq)


# ---------------------------------------------------------------------------
# GradientAnalyzer
# ---------------------------------------------------------------------------

def test_gradient_constant_tensor_zero_score():
    """All-constant tensor → amplitude gradient = 0 everywhere."""
    T = _constant_tensor()
    out = GradientAnalyzer().compute(T, _az(), _el(), _freq())
    np.testing.assert_allclose(out["amplitude_gradient"], 0.0, atol=1e-10)


def test_gradient_single_scatterer_nonzero():
    """Single-scatterer tensor has non-zero gradient (lobe edge regions)."""
    T = _single_scatterer_tensor()
    out = GradientAnalyzer().compute(T, _az(), _el(), _freq())
    assert np.max(out["amplitude_gradient"]) > 0.0


def test_phase_curvature_at_resonance_higher_than_flat():
    """Lorentzian frequency profile → higher mean phase curvature than flat."""
    T_lor   = _lorentzian_tensor()
    T_flat  = _constant_tensor()
    az, el, freq = _az(), _el(), _freq()
    curv_lor  = GradientAnalyzer().compute(T_lor,  az, el, freq)["phase_curvature"]
    curv_flat = GradientAnalyzer().compute(T_flat, az, el, freq)["phase_curvature"]
    assert np.mean(curv_lor) > np.mean(curv_flat), (
        "Lorentzian should produce higher phase curvature than flat"
    )


# ---------------------------------------------------------------------------
# ISARAnalyzer
# ---------------------------------------------------------------------------

def test_isar_single_scatterer_sidelobe_in_range():
    """Single-point scatterer → ISAR sidelobe ratio is a float in [0, 1]."""
    T = _single_scatterer_tensor()
    ratio, isar = ISARAnalyzer().compute_slice(T[:, 0, :])
    assert 0.0 <= ratio <= 1.0, f"Sidelobe ratio out of range: {ratio:.4f}"
    assert isar.ndim == 2


def test_isar_two_scatterers_higher_sidelobe_than_one():
    """Two scatterers → higher sidelobe floor than single scatterer.

    Use a larger grid for reliable ISAR resolution.
    """
    az   = np.linspace(0.2, math.pi - 0.2, 31)
    freq = np.linspace(8e9, 12e9, 30)
    el   = _el()
    T1 = _single_scatterer_tensor(az=az, el=el, freq=freq)
    T2 = _two_scatterer_tensor(az=az, el=el, freq=freq)
    r1, _ = ISARAnalyzer().compute_slice(T1[:, 0, :])
    r2, _ = ISARAnalyzer().compute_slice(T2[:, 0, :])
    assert r2 >= r1, (
        f"Two-scatterer sidelobe ratio {r2:.4f} should be ≥ single {r1:.4f}"
    )


def test_isar_compute_shape():
    """ISARAnalyzer.compute returns shape (N_az, N_el, N_freq)."""
    T  = _single_scatterer_tensor()
    az, el, freq = _az(), _el(), _freq()
    out = ISARAnalyzer().compute(T, az, el, freq)
    assert out.shape == T.shape


def test_isar_compute_range():
    """ISARAnalyzer.compute returns values in [0, 1]."""
    T = _two_scatterer_tensor()
    out = ISARAnalyzer().compute(T, _az(), _el(), _freq())
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


# ---------------------------------------------------------------------------
# SpectralAnalyzer
# ---------------------------------------------------------------------------

def test_spectral_variance_resonant_higher_than_flat():
    """Lorentzian peak → higher spectral variance than constant amplitude."""
    T_lor  = _lorentzian_tensor()
    T_flat = _constant_tensor()
    sa = SpectralAnalyzer()
    var_lor  = sa.compute(T_lor,  _freq())["spectral_variance"].mean()
    var_flat = sa.compute(T_flat, _freq())["spectral_variance"].mean()
    assert var_lor > var_flat, (
        f"Resonant variance {var_lor:.6f} should exceed flat {var_flat:.6f}"
    )


def test_resonance_count_detects_single_peak():
    """Single Lorentzian in-band → n_peaks ≥ 1 at some (az, el) pixel."""
    T = _lorentzian_tensor(f0=10e9, Q=15)
    out = SpectralAnalyzer().compute(T, _freq())
    assert np.max(out["resonance_count"]) >= 1


def test_spectral_output_shape():
    """SpectralAnalyzer returns correct shape."""
    T = _single_scatterer_tensor()
    out = SpectralAnalyzer().compute(T, _freq())
    assert out["spectral_variance"].shape == T.shape
    assert out["resonance_count"].shape   == T.shape


# ---------------------------------------------------------------------------
# CancellationDetector
# ---------------------------------------------------------------------------

def test_cancellation_at_null():
    """Manually-inserted amplitude null → high cancellation score at that voxel."""
    # Construct a tensor that is uniformly ~1.0 except at az_idx=2, where
    # amplitude drops to 0.01 (a clear spatial null surrounded by high-amplitude regions).
    T = np.ones((7, 3, 8), dtype=complex)
    null_az_idx = 3
    T[null_az_idx, :, :] = 0.01 + 0.001j

    scores = CancellationDetector(window=(3, 1, 3)).compute(T)
    # Score at the null should be higher than average
    null_score   = float(scores[null_az_idx].mean())
    nonnull_mean = float(scores[np.arange(len(T)) != null_az_idx].mean())
    assert null_score > nonnull_mean, (
        f"Null score {null_score:.3f} should exceed non-null mean {nonnull_mean:.3f}"
    )


def test_cancellation_constant_is_zero():
    """Constant tensor has no local minima → cancellation score ≈ 0."""
    T = _constant_tensor()
    scores = CancellationDetector().compute(T)
    # Uniform tensor: local mean == value → score = 1 - 1 = 0
    assert np.allclose(scores, 0.0, atol=1e-6)


def test_cancellation_output_shape():
    T = _single_scatterer_tensor()
    assert CancellationDetector().compute(T).shape == T.shape


# ---------------------------------------------------------------------------
# TensorSensitivityMap
# ---------------------------------------------------------------------------

def test_sensitivity_map_output_range():
    """compute() returns values in [0, 1]."""
    T   = _single_scatterer_tensor()
    tsm = _make_tsm(T)
    az, el, freq = _az(), _el(), _freq()
    pts = [ObservationPoint(theta=e, phi=a, freq_hz=f)
           for a in az[::3] for e in el[::2] for f in freq[::3]]
    scores = tsm.compute(pts)
    assert np.all(scores >= 0.0), "Scores below 0"
    assert np.all(scores <= 1.0), "Scores above 1"


def test_sensitivity_map_correct_length():
    """compute() returns array of correct length."""
    T   = _single_scatterer_tensor()
    tsm = _make_tsm(T)
    pts = [ObservationPoint(theta=0.5, phi=1.0, freq_hz=9e9)] * 7
    assert len(tsm.compute(pts)) == 7


def test_sensitivity_map_empty_points():
    """compute([]) returns empty array."""
    T   = _single_scatterer_tensor()
    tsm = _make_tsm(T)
    out = tsm.compute([])
    assert len(out) == 0


def test_sensitivity_map_combined_score_shape():
    """Combined score grid has correct shape."""
    T   = _single_scatterer_tensor()
    az, el, freq = _az(), _el(), _freq()
    tsm = TensorSensitivityMap(T, az, el, freq)
    grid = tsm.get_combined_score_grid()
    assert grid.shape == T.shape


def test_sensitivity_map_per_method_keys():
    """get_per_method_scores() returns all four method keys."""
    T   = _single_scatterer_tensor()
    tsm = _make_tsm(T)
    keys = set(tsm.get_per_method_scores().keys())
    assert keys == {"gradient", "isar", "spectral", "cancellation"}


def test_select_candidates_returns_n():
    """select_initial_measurements returns exactly n points."""
    from dhff.core import make_observation_grid
    T   = _single_scatterer_tensor()
    tsm = _make_tsm(T)
    grid = make_observation_grid(
        theta_range=(0.3, 2.8), phi_range=(0.3, 2.8),
        freq_range=(8e9, 12e9), n_theta=5, n_phi=5, n_freq=4,
    )
    plan = tsm.select_initial_measurements(grid, n_measurements=5)
    assert len(plan.points) == 5


def test_select_candidates_angular_diversity():
    """Selected candidates respect minimum angular separation."""
    from dhff.core import make_observation_grid
    from dhff.core.coordinate_system import angular_distance_points
    T   = _single_scatterer_tensor()
    tsm = _make_tsm(T)
    grid = make_observation_grid(
        theta_range=(0.3, 2.8), phi_range=(0.3, 2.8),
        freq_range=(8e9, 12e9), n_theta=8, n_phi=8, n_freq=4,
    )
    n = 6
    plan = tsm.select_initial_measurements(grid, n_measurements=n)
    min_sep = math.pi / (2.0 * n)
    # Relax threshold slightly (map uses /2 fallback)
    for i, p1 in enumerate(plan.points):
        for j, p2 in enumerate(plan.points):
            if i >= j:
                continue
            d = angular_distance_points(p1, p2)
            assert d >= min_sep / 2.0, (
                f"Points {i} and {j} too close: {math.degrees(d):.1f}° < "
                f"{math.degrees(min_sep/2):.1f}°"
            )


def test_get_top_points_length():
    """get_top_points(n) returns n tuples."""
    T   = _single_scatterer_tensor()
    tsm = _make_tsm(T)
    assert len(tsm.get_top_points(n=5)) == 5


def test_engine_accepts_rcs_tensor_input():
    """DHFFEngine runs to completion when rcs_tensor_input is provided."""
    from dhff.pipeline import DHFFEngine
    T  = _single_scatterer_tensor()
    az, el, freq = _az(), _el(), _freq()
    engine = DHFFEngine(
        scenario_name="simple_missing_feature",
        total_measurement_budget=20,
        candidate_grid_density=8,
        n_freq_candidates=8,
        rcs_tensor_input=dict(tensor=T, az_rad=az, el_rad=el, freq_hz=freq),
        random_seed=42,
    )
    results = engine.run()
    assert "fused_model" in results
    assert "error_metrics" in results
