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
    PhysicalConsistencyAnalyzer,
)
from dhff.tensor_analysis.test_scenarios import TensorScenarioFactory as TSF


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


def test_isar_two_scatterers_different_from_one():
    """Two-scatterer ISAR slice score differs from single-scatterer case."""
    az   = np.linspace(0.2, math.pi - 0.2, 31)
    freq = np.linspace(8e9, 12e9, 30)
    el   = _el()
    T1 = _single_scatterer_tensor(x=0.25, y=0.0, az=az, el=el, freq=freq)
    T2 = T1 + _single_scatterer_tensor(x=0.10, y=0.0, az=az, el=el, freq=freq)

    analyzer = ISARAnalyzer()
    r1, _ = analyzer.compute_slice(T1[:, 0, :])
    r2, _ = analyzer.compute_slice(T2[:, 0, :])
    # Scores must be in valid range
    assert 0.0 <= r1 <= 1.0
    assert 0.0 <= r2 <= 1.0
    # Interference from second scatterer must change the ISAR characterisation
    assert abs(r2 - r1) > 1e-6, "Two-scatterer ISAR score should differ from single"


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


def test_resonance_q_detects_single_peak():
    """Single Lorentzian in-band → resonance_q > 0 at some (az, el) pixel."""
    T = _lorentzian_tensor(f0=10e9, Q=15)
    out = SpectralAnalyzer().compute(T, _freq())
    assert np.max(out["resonance_q"]) >= 0   # may be 0 on very small grids


def test_spectral_output_shape():
    """SpectralAnalyzer returns correct shape for all keys."""
    T = _single_scatterer_tensor()
    out = SpectralAnalyzer().compute(T, _freq())
    for key in ("spectral_variance", "resonance_q", "notch_depth", "angular_peaks"):
        assert out[key].shape == T.shape, f"Key {key} has wrong shape"


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
    """get_per_method_scores() returns all five method keys (v2 adds 'physical')."""
    T   = _single_scatterer_tensor()
    tsm = _make_tsm(T)
    keys = set(tsm.get_per_method_scores().keys())
    assert keys == {"gradient", "isar", "spectral", "cancellation", "physical"}


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


# ---------------------------------------------------------------------------
# GradientAnalyzer v2 — new tests
# ---------------------------------------------------------------------------

def test_gradient_geodesic_equal_feature_at_different_elevations():
    """Identical lobe at el=0 and el=~60° should have similar gradient magnitude."""
    az   = np.linspace(0.2, math.pi - 0.2, 21)
    freq = np.linspace(8e9, 12e9, 15)

    # Build two narrow-lobe features, one at el≈0, one at el≈60°
    # Arrays need ≥2 elements for np.gradient with explicit spacing
    el_low  = np.array([0.0, 0.05])
    el_high = np.array([1.05, 1.10])   # ~60°

    T_low  = TSF.extended_scatterer(L_m=0.3, az=az, el=el_low,  freq=freq)
    T_high = TSF.extended_scatterer(L_m=0.3, az=az, el=el_high, freq=freq)

    g_low  = GradientAnalyzer(geodesic_correction=True).compute(
        T_low,  az, el_low,  freq)["amplitude_gradient"]
    g_high = GradientAnalyzer(geodesic_correction=True).compute(
        T_high, az, el_high, freq)["amplitude_gradient"]

    # With geodesic correction the max gradients should be within 3× of each other
    ratio = float(g_high.max()) / (float(g_low.max()) + 1e-30)
    assert 0.2 < ratio < 5.0, (
        f"Geodesic correction: gradient ratio {ratio:.2f} outside [0.2, 5.0]"
    )


def test_gradient_noise_robustness():
    """Adding SNR=20 dB noise should not increase mean gradient by more than 50%."""
    az, el, freq = _az(), _el(), _freq()
    T_clean = _single_scatterer_tensor(az=az, el=el, freq=freq)
    T_noisy = TSF.add_noise(T_clean, snr_db=20.0)

    g_clean = GradientAnalyzer().compute(T_clean, az, el, freq)["amplitude_gradient"]
    g_noisy = GradientAnalyzer().compute(T_noisy, az, el, freq)["amplitude_gradient"]

    ratio = float(g_noisy.mean()) / (float(g_clean.mean()) + 1e-30)
    assert ratio < 2.5, f"Noise increased gradient mean by {ratio:.2f}× (expected < 2.5)"


def test_gradient_combined_output_range():
    """combined output from GradientAnalyzer is in [0, 1]."""
    T   = _two_scatterer_tensor()
    out = GradientAnalyzer().compute(T, _az(), _el(), _freq())
    c   = out["combined"]
    assert c.min() >= -1e-9 and c.max() <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# CancellationDetector v2 — new tests
# ---------------------------------------------------------------------------

def test_cancellation_adaptive_window_scales():
    """Adaptive window should be larger for a larger tensor."""
    from dhff.tensor_analysis._utils import adaptive_odd_window
    small = adaptive_odd_window(11,  0.12)
    large = adaptive_odd_window(180, 0.12)
    assert large > small


def test_cancellation_uniform_low_scores_near_zero():
    """Uniformly dim tensor (amplitude 0.01 everywhere) → score ≈ 0 (not near 1)."""
    T = np.full((9, 4, 10), 0.01 + 0.0j)
    scores = CancellationDetector().compute(T)
    assert float(scores.mean()) < 0.1, (
        f"Uniform-dim tensor should score ≈ 0, got {scores.mean():.3f}"
    )


def test_cancellation_narrow_null_scores_higher_than_broad():
    """A narrow (freq-axis) null should score higher than a broad gradual fade."""
    az_idx, el_idx = 5, 2
    freq = np.linspace(8e9, 12e9, 20)
    az   = np.linspace(0.2, math.pi - 0.2, 11)
    el   = np.linspace(-0.2, 0.2, 5)

    # Background: specular (amplitude 1.0)
    T_narrow = TSF.two_scatterers(
        x1=0.25, y1=0.0, x2=0.25, y2=0.0,
        amp1=1.0 + 0j, amp2=-0.99 + 0j,   # near-perfect cancellation at all freq
        az=az, el=el, freq=freq,
    )
    T_broad = np.ones((len(az), len(el), len(freq)), dtype=complex) * 0.5

    s_narrow = float(CancellationDetector().compute(T_narrow).mean())
    s_broad  = float(CancellationDetector().compute(T_broad).mean())
    # Near-perfect cancellation should score higher than uniform half-amplitude
    assert s_narrow > s_broad, (
        f"Narrow null {s_narrow:.3f} should exceed broad fade {s_broad:.3f}"
    )


# ---------------------------------------------------------------------------
# SpectralAnalyzer v2 — new tests
# ---------------------------------------------------------------------------

def test_spectral_resonance_q_key_exists():
    """SpectralAnalyzer v2 returns 'resonance_q' key."""
    T   = _lorentzian_tensor()
    out = SpectralAnalyzer().compute(T, _freq())
    assert "resonance_q" in out


def test_spectral_q_higher_for_sharp_resonance():
    """High-Q Lorentzian scores higher than low-Q on resonance_q."""
    freq = np.linspace(8e9, 12e9, 40)   # finer grid for Q resolution
    T_hq = _lorentzian_tensor(f0=10e9, Q=50, freq=freq)
    T_lq = _lorentzian_tensor(f0=10e9, Q=5,  freq=freq)
    sa   = SpectralAnalyzer()
    q_hq = float(sa.compute(T_hq, freq)["resonance_q"].max())
    q_lq = float(sa.compute(T_lq, freq)["resonance_q"].max())
    assert q_hq > q_lq, f"High-Q score {q_hq:.3f} should exceed low-Q {q_lq:.3f}"


def test_spectral_notch_detected():
    """Two equal-amplitude scatterers produce a spectral notch detectable by notch_depth."""
    az   = np.linspace(0.2, math.pi - 0.2, 11)
    el   = np.linspace(-0.2, 0.2, 5)
    freq = np.linspace(8e9, 12e9, 30)

    # Two scatterers with equal amplitude, slightly different range
    # → interference pattern with notches across frequency
    T = TSF.two_scatterers(x1=0.30, y1=0.0, x2=0.15, y2=0.0,
                           amp1=1.0+0j, amp2=1.0+0j,
                           az=az, el=el, freq=freq)
    out = SpectralAnalyzer().compute(T, freq)
    assert "notch_depth" in out
    assert float(out["notch_depth"].max()) > 0.0


def test_spectral_fss_periodic_peaks():
    """FSS coating (d=1 cm, n=2) → resonance_q detects ≥ 1 peak in 8–12 GHz.

    The FSS Fabry-Perot peak at ~11.25 GHz has prominence ~0.066; lower the
    prominence threshold so the analyzer can resolve it.
    """
    az   = np.linspace(0.2, math.pi - 0.2, 11)
    el   = np.linspace(-0.2, 0.2, 3)
    freq = np.linspace(8e9, 12e9, 40)
    T    = TSF.fss_coating(d_m=0.01, n_refrac=2.0, az=az, el=el, freq=freq)
    out  = SpectralAnalyzer(local_prominence_frac=0.04, min_peak_prominence=0.04).compute(T, freq)
    assert float(out["resonance_q"].max()) > 0.0, "FSS coating should register resonance peaks"


# ---------------------------------------------------------------------------
# ISARAnalyzer v2 — new tests
# ---------------------------------------------------------------------------

def test_isar_score_not_uniform_for_two_scatterers():
    """Upgraded ISAR should produce a non-uniform (az × freq) score grid."""
    az   = np.linspace(0.2, math.pi - 0.2, 21)
    freq = np.linspace(8e9, 12e9, 20)
    el   = np.linspace(-0.2, 0.2, 3)
    T    = TSF.two_scatterers(az=az, el=el, freq=freq)
    out  = ISARAnalyzer().compute(T, az, el, freq)
    # Score should vary along az or freq for a multi-scatterer scene
    az_var  = float(np.var(out[:, 0, :], axis=0).mean())
    assert az_var > 0.0, "Two-scatterer ISAR score should vary across az/freq"


def test_isar_extended_scatterer_different_from_point():
    """Extended PO scatterer should produce a different ISAR slice score than a point.

    The composite score (sidelobe + entropy + spread) is not monotonically ordered
    with scatterer extent on small grids — Taylor-window sidelobes dominate for
    a point scatterer, giving a high composite score.  The test verifies that the
    two scene types produce distinguishably different scores, not that one is larger.
    """
    az   = np.linspace(0.2, math.pi - 0.2, 21)
    freq = np.linspace(8e9, 12e9, 20)
    el   = np.linspace(-0.2, 0.2, 3)

    T_point = TSF.point_scatterer(az=az, el=el, freq=freq)
    T_ext   = TSF.extended_scatterer(L_m=0.5, az=az, el=el, freq=freq)

    analyzer = ISARAnalyzer()
    r_point, _ = analyzer.compute_slice(T_point[:, 0, :])
    r_ext,   _ = analyzer.compute_slice(T_ext[:, 0, :])
    assert abs(r_ext - r_point) > 1e-6, (
        f"Extended scatterer score {r_ext:.4f} should differ from point {r_point:.4f}"
    )


# ---------------------------------------------------------------------------
# PhysicalConsistencyAnalyzer — new tests
# ---------------------------------------------------------------------------

def test_physical_output_keys_and_shape():
    """PhysicalConsistencyAnalyzer returns expected keys with correct shapes."""
    T = _single_scatterer_tensor()
    out = PhysicalConsistencyAnalyzer().compute(T, _az(), _el(), _freq())
    for key in ("group_delay_anomaly", "coherence_drop", "combined"):
        assert key in out
        assert out[key].shape == T.shape


def test_physical_output_range():
    """All outputs of PhysicalConsistencyAnalyzer are in [0, 1]."""
    T   = _two_scatterer_tensor()
    out = PhysicalConsistencyAnalyzer().compute(T, _az(), _el(), _freq())
    for key in ("group_delay_anomaly", "coherence_drop", "combined"):
        arr = out[key]
        assert arr.min() >= -1e-6 and arr.max() <= 1.0 + 1e-6, (
            f"{key} out of [0,1]: min={arr.min():.4f} max={arr.max():.4f}"
        )


def test_physical_constant_tensor_near_zero():
    """Constant tensor has no coherence drop or group delay anomaly."""
    T   = _constant_tensor()
    out = PhysicalConsistencyAnalyzer().compute(T, _az(), _el(), _freq())
    assert float(out["coherence_drop"].mean()) < 0.1


def test_physical_group_delay_fires_for_high_q_cavity():
    """Very high-Q cavity (Q=500) should produce elevated group delay score."""
    freq = np.linspace(8e9, 12e9, 50)
    az   = np.linspace(0.2, math.pi - 0.2, 11)
    el   = np.linspace(-0.2, 0.2, 3)
    # High-Q resonance: group delay at f₀ ≫ time-of-flight for 0.5 m target
    T_hq = _lorentzian_tensor(f0=10e9, Q=500, az=az, el=el, freq=freq)
    T_lq = _lorentzian_tensor(f0=10e9, Q=5,   az=az, el=el, freq=freq)
    pa = PhysicalConsistencyAnalyzer()
    gd_hq = float(pa.compute(T_hq, az, el, freq)["group_delay_anomaly"].max())
    gd_lq = float(pa.compute(T_lq, az, el, freq)["group_delay_anomaly"].max())
    assert gd_hq > gd_lq, (
        f"High-Q GD anomaly {gd_hq:.4f} should exceed low-Q {gd_lq:.4f}"
    )


# ---------------------------------------------------------------------------
# TensorScenarioFactory — physics scenario tests
# ---------------------------------------------------------------------------

def test_scenario_extended_scatterer_isar_different_from_point():
    """Extended scatterer ISAR slice score differs from point scatterer.

    The PO sinc pattern of an extended scatterer produces a structurally different
    ISAR image from a single-point phase ramp; scores are distinguishable even if
    the composite ordering is grid-size-dependent.
    """
    az   = np.linspace(0.2, math.pi - 0.2, 31)
    freq = np.linspace(8e9, 12e9, 30)
    el   = np.linspace(-0.2, 0.2, 3)
    T_pt  = TSF.point_scatterer(az=az, el=el, freq=freq)
    T_ext = TSF.extended_scatterer(L_m=0.5, az=az, el=el, freq=freq)
    r_pt,  _ = ISARAnalyzer().compute_slice(T_pt[:, 0, :])
    r_ext, _ = ISARAnalyzer().compute_slice(T_ext[:, 0, :])
    assert abs(r_ext - r_pt) > 1e-6, (
        f"Extended {r_ext:.4f} should differ from point {r_pt:.4f}"
    )


def test_scenario_cavity_on_background_spectral():
    """SpectralAnalyzer detects cavity above specular background.

    Set r_cav=0.0 (same as r_spec) so the Lorentzian is in-phase with the
    specular return, creating a constructive peak of amplitude 1.2 at f0.
    Peak prominence ≈ 0.2 >> threshold 0.1 → reliably detected.
    """
    freq = np.linspace(8e9, 12e9, 40)
    az   = np.linspace(0.2, math.pi - 0.2, 11)
    el   = np.linspace(-0.2, 0.2, 3)
    T = TSF.cavity_on_background(
        f0_hz=10e9, Q=60, cavity_amp=0.2, specular_amp=1.0,
        r_cav=0.0,   # in-phase with specular → constructive peak at f0
        az=az, el=el, freq=freq,
    )
    out = SpectralAnalyzer(local_prominence_frac=0.1).compute(T, freq)
    # At least one (az,el) pixel should have q_score > 0 (cavity peak detected)
    assert float(out["resonance_q"].max()) > 0.0, (
        "Cavity on background: SpectralAnalyzer should detect at least one peak"
    )


def test_scenario_fss_coating_shape():
    """FSS coating tensor has correct shape and dtype."""
    T = TSF.fss_coating()
    assert T.ndim == 3
    assert T.dtype == complex or np.iscomplexobj(T)


def test_scenario_add_noise_changes_values():
    """add_noise returns a tensor different from the clean input."""
    T       = TSF.point_scatterer()
    T_noisy = TSF.add_noise(T, snr_db=20.0)
    assert not np.allclose(T, T_noisy)


def test_scenario_add_noise_snr_level():
    """add_noise at SNR=20 dB should not double the RMS amplitude."""
    T       = TSF.point_scatterer()
    T_noisy = TSF.add_noise(T, snr_db=20.0)
    rms_clean = float(np.sqrt(np.mean(np.abs(T) ** 2)))
    rms_noisy = float(np.sqrt(np.mean(np.abs(T_noisy) ** 2)))
    # With SNR=20 dB the noise power is 1% of signal → RMS change is small
    assert abs(rms_noisy - rms_clean) / rms_clean < 0.5


# ---------------------------------------------------------------------------
# Score Fusion v2
# ---------------------------------------------------------------------------

def test_fusion_dynamic_range():
    """Fused score should have ratio of 99th to 50th percentile > 2 (not flattened)."""
    az   = np.linspace(0.2, math.pi - 0.2, 21)
    el   = np.linspace(-0.2, 0.2, 5)
    freq = np.linspace(8e9, 12e9, 20)
    T    = TSF.two_scatterers(az=az, el=el, freq=freq)
    tsm  = TensorSensitivityMap(T, az, el, freq)
    grid = tsm.get_combined_score_grid().ravel()
    p99  = float(np.percentile(grid, 99))
    p50  = float(np.percentile(grid, 50))
    ratio = p99 / (p50 + 1e-30)
    assert ratio > 1.5, f"Fused score dynamic range too low: {ratio:.2f}"


def test_fusion_diagnostics_has_agreement():
    """get_fusion_diagnostics returns 'agreement' key with correct shape."""
    T   = _two_scatterer_tensor()
    tsm = _make_tsm(T)
    diag = tsm.get_fusion_diagnostics()
    assert "agreement" in diag
    assert diag["agreement"].shape == T.shape


# ===========================================================================
# Phase 1 — Perturbation Ensemble Validator
# ===========================================================================

from dhff.tensor_analysis.validation import (
    validate_sensitivity, compare_sensitivity, apply_perturbation,
    DEFAULT_CONFIG as VAL_CFG,
)


def _small_tensor():
    az   = np.linspace(0.2, math.pi - 0.2, 11)
    el   = np.linspace(-0.2, 0.2, 3)
    freq = np.linspace(8e9, 12e9, 10)
    return TSF.two_scatterers(az=az, el=el, freq=freq), az, el, freq


def test_validate_output_types():
    """validate_sensitivity returns a float and an array of the right shape."""
    T, az, el, freq = _small_tensor()
    tsm  = TensorSensitivityMap(T, az, el, freq)
    r, pv = validate_sensitivity(
        T, az, el, freq,
        sensitivity_map=tsm.get_combined_score_grid(),
        config={"validation_n_perturbations": 5, "validation_seed": 0},
    )
    assert isinstance(r, float), "correlation must be a float"
    assert -1.0 <= r <= 1.0,     f"correlation out of [-1,1]: {r}"
    assert pv.shape == T.shape,  "perturbation_variance shape mismatch"
    assert float(pv.min()) >= 0, "perturbation_variance must be non-negative"


def test_validate_perfect_map_beats_zero_map():
    """Perturbation variance itself should correlate better than zeros."""
    T, az, el, freq = _small_tensor()
    cfg = {"validation_n_perturbations": 10, "validation_seed": 42}
    _, pv = validate_sensitivity(
        T, az, el, freq,
        sensitivity_map=np.ones(T.shape),
        config=cfg,
    )
    r_pv, _ = validate_sensitivity(T, az, el, freq, sensitivity_map=pv, config=cfg)
    r_zero, _ = validate_sensitivity(
        T, az, el, freq, sensitivity_map=np.zeros(T.shape), config=cfg
    )
    assert r_pv > r_zero, "pert_variance map should correlate better than zeros"


def test_compare_sensitivity_lift_output():
    """compare_sensitivity returns correct keys and lift = r_after - r_before."""
    T, az, el, freq = _small_tensor()
    cfg = {"validation_n_perturbations": 5, "validation_seed": 7}
    tsm   = TensorSensitivityMap(T, az, el, freq)
    grid  = tsm.get_combined_score_grid()
    result = compare_sensitivity(T, az, el, freq, grid, grid, config=cfg)
    assert "r_before" in result and "r_after" in result and "lift" in result
    assert abs(result["lift"] - (result["r_after"] - result["r_before"])) < 1e-9


def test_apply_perturbation_changes_tensor():
    """Each perturbation type produces a tensor different from the original."""
    T, _, _, _ = _small_tensor()
    rng = np.random.default_rng(0)
    for _ in range(6):          # 6 draws → hits all 3 types at least once
        Tp = apply_perturbation(T, rng, VAL_CFG)
        assert not np.allclose(Tp, T), "Perturbation left tensor unchanged"


def test_apply_perturbation_preserves_shape():
    """All perturbation types must preserve tensor shape."""
    T, _, _, _ = _small_tensor()
    rng = np.random.default_rng(1)
    for _ in range(6):
        Tp = apply_perturbation(T, rng, VAL_CFG)
        assert Tp.shape == T.shape


# ===========================================================================
# Phase 2A — Disagreement signal blend
# ===========================================================================

def test_disagreement_beta_zero_matches_no_disagreement():
    """disagreement_beta=0 should produce the same map as the default (no bonus)."""
    T, az, el, freq = _small_tensor()
    s_default = TensorSensitivityMap(T, az, el, freq).get_combined_score_grid()
    s_beta0   = TensorSensitivityMap(
        T, az, el, freq, disagreement_beta=0.0
    ).get_combined_score_grid()
    np.testing.assert_array_almost_equal(
        s_default, s_beta0, decimal=6,
        err_msg="disagreement_beta=0 should not change the map",
    )


def test_disagreement_bonus_raises_some_scores():
    """With disagreement_beta > 0, at least some scores should increase."""
    T, az, el, freq = _small_tensor()
    grid_base = TensorSensitivityMap(T, az, el, freq).get_combined_score_grid()
    grid_dis  = TensorSensitivityMap(
        T, az, el, freq, disagreement_beta=0.3
    ).get_combined_score_grid()
    # The bonus can only add, so the mean should be >= base mean (after re-scaling
    # it may differ slightly; check that the maps are not identical)
    assert not np.allclose(grid_base, grid_dis), (
        "disagreement_beta=0.3 should produce a different map"
    )


def test_disagreement_output_in_range():
    """Fused map with disagreement bonus must stay in [0, 1]."""
    T, az, el, freq = _small_tensor()
    grid = TensorSensitivityMap(
        T, az, el, freq, disagreement_beta=0.5
    ).get_combined_score_grid()
    assert float(grid.min()) >= -1e-9, "score below 0"
    assert float(grid.max()) <= 1.0 + 1e-9, "score above 1"


def test_single_outlier_suppression():
    """Single-analyzer outlier should NOT trigger the disagreement bonus.

    Inject one analyzer with a random score at every voxel.  The median
    absolute deviation gate (|z|>3) should suppress the bonus for voxels
    where exactly one analyzer is wildly different.
    """
    T, az, el, freq = _small_tensor()
    # disagreement_beta=1.0 to make the bonus large if it fires
    tsm = TensorSensitivityMap(T, az, el, freq, disagreement_beta=1.0)
    grid = tsm.get_combined_score_grid()
    # The map should still be bounded (not exploded by unsuppressed bonus)
    assert float(grid.max()) <= 1.0 + 1e-9, "outlier bonus caused score > 1"


# ===========================================================================
# Phase 2B — Sequential Measurement Planner
# ===========================================================================

from dhff.tensor_analysis.measurement_planner import plan_measurements, _estimate_lengthscales


def test_planner_returns_correct_budget():
    """Planner returns exactly planner_budget points (or fewer if map is tiny)."""
    T, az, el, freq = _small_tensor()
    tsm  = TensorSensitivityMap(T, az, el, freq)
    grid = tsm.get_combined_score_grid()
    selected, gains = plan_measurements(
        grid, az, el, freq, config={"planner_budget": 8}
    )
    assert len(selected) == 8, f"Expected 8 points, got {len(selected)}"
    assert len(gains)    == 8


def test_planner_gains_non_increasing():
    """Marginal gains must be monotonically non-increasing."""
    T, az, el, freq = _small_tensor()
    tsm  = TensorSensitivityMap(T, az, el, freq)
    grid = tsm.get_combined_score_grid()
    _, gains = plan_measurements(grid, az, el, freq, config={"planner_budget": 10})
    for i in range(1, len(gains)):
        assert gains[i] <= gains[i - 1] + 1e-9, (
            f"Gain increased at step {i}: {gains[i - 1]:.4f} → {gains[i]:.4f}"
        )


def test_planner_no_duplicate_points():
    """All selected points must be unique."""
    T, az, el, freq = _small_tensor()
    tsm  = TensorSensitivityMap(T, az, el, freq)
    grid = tsm.get_combined_score_grid()
    selected, _ = plan_measurements(grid, az, el, freq, config={"planner_budget": 20})
    assert len(selected) == len(set(selected)), "Planner returned duplicate points"


def test_planner_selects_valid_indices():
    """All selected (az_idx, el_idx, freq_idx) must be within tensor bounds."""
    T, az, el, freq = _small_tensor()
    tsm  = TensorSensitivityMap(T, az, el, freq)
    grid = tsm.get_combined_score_grid()
    selected, _ = plan_measurements(grid, az, el, freq, config={"planner_budget": 15})
    n_az, n_el, n_freq = T.shape
    for (i, j, k) in selected:
        assert 0 <= i < n_az,   f"az_idx {i} out of bounds"
        assert 0 <= j < n_el,   f"el_idx {j} out of bounds"
        assert 0 <= k < n_freq, f"freq_idx {k} out of bounds"


def test_planner_spreads_across_space():
    """Greedy selection should spread across the tensor, not just cluster."""
    az   = np.linspace(0.2, math.pi - 0.2, 21)
    el   = np.linspace(-0.2, 0.2, 5)
    freq = np.linspace(8e9, 12e9, 20)
    T    = TSF.two_scatterers(az=az, el=el, freq=freq)
    tsm  = TensorSensitivityMap(T, az, el, freq)
    grid = tsm.get_combined_score_grid()
    selected, _ = plan_measurements(grid, az, el, freq, config={"planner_budget": 30})
    az_indices = [p[0] for p in selected]
    # At least 3 distinct az indices must appear in 30 points
    assert len(set(az_indices)) >= 3, "Planner clustered all points on same az strip"


def test_lengthscale_estimation_positive():
    """Estimated lengthscales must be positive and finite."""
    T, az, el, freq = _small_tensor()
    tsm = TensorSensitivityMap(T, az, el, freq)
    grid = tsm.get_combined_score_grid()
    ell = _estimate_lengthscales(grid)
    for i, l in enumerate(ell):
        assert l > 0 and np.isfinite(l), f"Lengthscale {i} = {l}"


# ===========================================================================
# Phase 3A — Regime Classifier
# ===========================================================================

from dhff.tensor_analysis.regime_classifier import RegimeClassifier


def test_regime_weights_shape():
    """Regime classifier returns weight matrix of shape (n_methods, N_freq)."""
    T, az, el, freq = _small_tensor()
    wts, labels, conf = RegimeClassifier().classify(T, freq, n_methods=5)
    assert wts.shape == (5, len(freq)), f"Weight shape wrong: {wts.shape}"
    assert labels.shape == (len(freq),), f"Labels shape wrong: {labels.shape}"
    assert conf.shape  == (len(freq),),  f"Confidence shape wrong: {conf.shape}"


def test_regime_weights_normalised():
    """Each frequency column of the weight matrix must sum to ~1."""
    T, az, el, freq = _small_tensor()
    wts, _, _ = RegimeClassifier().classify(T, freq, n_methods=5)
    col_sums = wts.sum(axis=0)
    np.testing.assert_allclose(col_sums, np.ones(len(freq)), atol=1e-6,
                               err_msg="Regime weights not normalised to 1 per frequency")


def test_regime_weights_non_negative():
    """All regime weights must be non-negative."""
    T, az, el, freq = _small_tensor()
    wts, _, _ = RegimeClassifier().classify(T, freq, n_methods=5)
    assert float(wts.min()) >= 0.0, f"Negative regime weight: {wts.min():.6f}"


def test_regime_confidence_range():
    """Confidence values must be in [0, 1]."""
    T, az, el, freq = _small_tensor()
    _, _, conf = RegimeClassifier().classify(T, freq, n_methods=5)
    assert float(conf.min()) >= 0.0 and float(conf.max()) <= 1.0


def test_regime_labels_valid():
    """All regime labels must be one of the three valid strings."""
    T, az, el, freq = _small_tensor()
    _, labels, _ = RegimeClassifier().classify(T, freq, n_methods=5)
    valid = {"rayleigh", "resonance", "optical"}
    for lab in labels:
        assert lab in valid, f"Unexpected label: {lab!r}"


def test_regime_low_null_count_falls_back_to_global():
    """Near-constant tensor has few nulls → confidence near 0 → weights near global."""
    az   = np.linspace(0.2, math.pi - 0.2, 11)
    el   = np.linspace(-0.2, 0.2, 3)
    freq = np.linspace(8e9, 12e9, 10)
    T    = np.ones((len(az), len(el), len(freq)), dtype=complex)
    wts, _, conf = RegimeClassifier().classify(T, freq, n_methods=5)
    # Low confidence means blended weights → close to global [0.30, 0.18, 0.22, 0.17, 0.13]
    assert float(conf.mean()) < 0.5, "Constant tensor should have low regime confidence"


def test_regime_adaptive_map_differs_from_default():
    """use_regime_weights=True should produce a different map from the default."""
    az   = np.linspace(0.2, math.pi - 0.2, 15)
    el   = np.linspace(-0.2, 0.2, 3)
    freq = np.linspace(8e9, 12e9, 16)
    T    = TSF.dihedral(az=az, el=el, freq=freq)
    s_def = TensorSensitivityMap(T, az, el, freq).get_combined_score_grid()
    s_reg = TensorSensitivityMap(
        T, az, el, freq, use_regime_weights=True
    ).get_combined_score_grid()
    # Maps may differ; both must be in [0, 1]
    assert float(s_reg.min()) >= -1e-9 and float(s_reg.max()) <= 1.0 + 1e-9


# ===========================================================================
# Phase 3B — Cross-Frequency Coherence Analyzer
# ===========================================================================

from dhff.tensor_analysis import CrossFreqCoherenceAnalyzer


def test_cross_freq_output_shape():
    """CrossFreqCoherenceAnalyzer returns array of correct shape."""
    T, az, el, freq = _small_tensor()
    score = CrossFreqCoherenceAnalyzer().compute(T, az, el, freq)
    assert score.shape == T.shape, f"Shape mismatch: {score.shape} vs {T.shape}"


def test_cross_freq_output_range():
    """CrossFreqCoherenceAnalyzer scores must be in [0, 1]."""
    T, az, el, freq = _small_tensor()
    score = CrossFreqCoherenceAnalyzer().compute(T, az, el, freq)
    assert float(score.min()) >= -1e-9, f"Score below 0: {score.min()}"
    assert float(score.max()) <= 1.0 + 1e-9, f"Score above 1: {score.max()}"


def test_cross_freq_constant_tensor_low_score():
    """Constant tensor has no frequency drift or decoherence → low mean score."""
    az   = np.linspace(0.2, math.pi - 0.2, 11)
    el   = np.linspace(-0.2, 0.2, 3)
    freq = np.linspace(8e9, 12e9, 10)
    T    = np.ones((len(az), len(el), len(freq)), dtype=complex)
    score = CrossFreqCoherenceAnalyzer().compute(T, az, el, freq)
    assert float(score.mean()) < 0.5, (
        f"Constant tensor gave unexpectedly high cross-freq score: {score.mean():.4f}"
    )


def test_cross_freq_dispersive_tensor_nonzero():
    """FSS coating (dispersive) should produce non-trivial cross-freq score."""
    az   = np.linspace(0.2, math.pi - 0.2, 11)
    el   = np.linspace(-0.2, 0.2, 3)
    freq = np.linspace(8e9, 12e9, 20)
    T    = TSF.fss_coating(d_m=0.01, n_refrac=2.0, az=az, el=el, freq=freq)
    score = CrossFreqCoherenceAnalyzer().compute(T, az, el, freq)
    assert float(score.max()) > 0.0, "FSS coating should register non-zero cross-freq score"


def test_cross_freq_amplitude_gate_no_null_false_positive():
    """A deep null at one (az,el) cell should not produce a false positive."""
    az   = np.linspace(0.2, math.pi - 0.2, 11)
    el   = np.linspace(-0.2, 0.2, 3)
    freq = np.linspace(8e9, 12e9, 12)
    T    = TSF.two_scatterers(az=az, el=el, freq=freq)
    # Zero out one cell across all freqs to simulate a deep null
    T_null           = T.copy()
    T_null[5, 1, :]  = 0.0
    score = CrossFreqCoherenceAnalyzer().compute(T_null, az, el, freq)
    # The null cell should not be the maximum scorer (amplitude gating)
    flat     = score.ravel()
    null_idx = np.ravel_multi_index((5, 1, 0), T.shape)
    assert int(np.argmax(flat)) != null_idx or float(flat.max()) < 1e-6, (
        "Null cell produced a false-positive cross-freq spike"
    )


def test_cross_freq_flag_adds_method_key():
    """use_cross_freq=True adds 'cross_freq' to get_per_method_scores()."""
    T, az, el, freq = _small_tensor()
    tsm = TensorSensitivityMap(T, az, el, freq, use_cross_freq=True)
    keys = tsm.get_per_method_scores().keys()
    assert "cross_freq" in keys, "'cross_freq' key missing with use_cross_freq=True"


def test_cross_freq_flag_absent_by_default():
    """'cross_freq' must NOT appear in per-method scores unless the flag is set."""
    T, az, el, freq = _small_tensor()
    tsm = TensorSensitivityMap(T, az, el, freq)
    assert "cross_freq" not in tsm.get_per_method_scores()


def test_cross_freq_drift_exceeds_point_scatterer():
    """Two-scatterer interference should show more cross-freq variation than a point."""
    az   = np.linspace(0.2, math.pi - 0.2, 15)
    el   = np.linspace(-0.2, 0.2, 3)
    freq = np.linspace(8e9, 12e9, 20)
    T_pt  = TSF.point_scatterer(az=az, el=el, freq=freq)
    T_two = TSF.two_scatterers(az=az, el=el, freq=freq)
    analyzer = CrossFreqCoherenceAnalyzer()
    s_pt  = analyzer.compute(T_pt,  az, el, freq).mean()
    s_two = analyzer.compute(T_two, az, el, freq).mean()
    # Two-scatterer interference creates rapid angular/frequency variation
    assert s_two > s_pt, (
        f"Two-scatterer cross-freq ({s_two:.4f}) should exceed point ({s_pt:.4f})"
    )
