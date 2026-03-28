"""Tests for dhff/cad module: primitives and feature extractor."""
from __future__ import annotations

import math

import pytest

from dhff.cad import (
    CadFeatureExtractor,
    FlatPanel,
    EdgeSegment,
    CavityVolume,
    ConvexSurface,
)
from dhff.synthetic import SyntheticScatterer

_C = 299_792_458.0
_F_CENTER = 10e9
_LAMBDA = _C / _F_CENTER


# ---------------------------------------------------------------------------
# FlatPanel tests
# ---------------------------------------------------------------------------

def test_flat_panel_lobe_center_matches_normal():
    p = FlatPanel(x=0.1, y=0.0, width_m=0.10, height_m=0.08, normal_theta_rad=1.2)
    feat = CadFeatureExtractor().extract([p])[0]
    assert feat.lobe_center_theta == pytest.approx(1.2)


def test_flat_panel_lobe_width_scales_with_size():
    small = FlatPanel(x=0.0, y=0.0, width_m=0.05, height_m=0.04, normal_theta_rad=math.pi / 2)
    large = FlatPanel(x=0.0, y=0.0, width_m=0.20, height_m=0.15, normal_theta_rad=math.pi / 2)
    ext = CadFeatureExtractor()
    feat_small = ext.extract([small])[0]
    feat_large = ext.extract([large])[0]
    assert feat_small.lobe_width_rad > feat_large.lobe_width_rad


def test_flat_panel_narrow_lobe_for_large_panel():
    # 10-lambda panel → lobe_width < 0.1 → "narrow_lobe"
    large = FlatPanel(x=0.0, y=0.0, width_m=10 * _LAMBDA, height_m=8 * _LAMBDA,
                      normal_theta_rad=math.pi / 2)
    feat = CadFeatureExtractor().extract([large])[0]
    assert feat.angular_pattern == "narrow_lobe"


# ---------------------------------------------------------------------------
# EdgeSegment tests
# ---------------------------------------------------------------------------

def test_edge_freq_dependence_is_edge():
    e = EdgeSegment(x=0.0, y=0.0, length_m=0.10, edge_theta_rad=math.pi / 2)
    feat = CadFeatureExtractor().extract([e])[0]
    assert feat.freq_dependence == "edge"


def test_edge_broad_lobe():
    e = EdgeSegment(x=0.0, y=0.0, length_m=0.10, edge_theta_rad=math.pi / 2)
    feat = CadFeatureExtractor().extract([e])[0]
    assert feat.angular_pattern == "broad_lobe"


# ---------------------------------------------------------------------------
# CavityVolume tests
# ---------------------------------------------------------------------------

def test_cavity_f0_rectangular_te101():
    a = 0.015
    d = 0.020
    expected_f0 = _C / (2.0 * math.sqrt(a ** 2 + d ** 2))
    c = CavityVolume(x=0.25, y=-0.1, interior_dim_a_m=a, interior_dim_b_m=0.010,
                     depth_m=d, aperture_area_m2=0.0002)
    feat = CadFeatureExtractor().extract([c])[0]
    assert feat.cavity_freq_hz == pytest.approx(expected_f0, rel=0.01)


def test_cavity_q_decreases_with_aperture():
    # Use aperture areas small enough that both Q values are above the clamp minimum of 2.
    # f0 ≈ 6 GHz, V = 3e-6 m³ → Q = f0*V/(C*A); for A=1e-6: Q≈60, for A=1e-5: Q≈6
    base = dict(x=0.0, y=0.0, interior_dim_a_m=0.015, interior_dim_b_m=0.010, depth_m=0.020)
    small_apt = CavityVolume(**base, aperture_area_m2=1e-6)
    large_apt = CavityVolume(**base, aperture_area_m2=1e-5)
    ext = CadFeatureExtractor()
    feat_small = ext.extract([small_apt])[0]
    feat_large = ext.extract([large_apt])[0]
    assert feat_small.cavity_q > feat_large.cavity_q


def test_cavity_uncertainty_is_high():
    c = CavityVolume(x=0.0, y=0.0, interior_dim_a_m=0.015, interior_dim_b_m=0.010,
                     depth_m=0.020, aperture_area_m2=0.0002)
    feat = CadFeatureExtractor().extract([c])[0]
    assert feat.amplitude_uncertainty_db >= 8.0
    assert feat.freq_param_uncertainty >= 0.1


# ---------------------------------------------------------------------------
# ConvexSurface tests
# ---------------------------------------------------------------------------

def test_creeping_wave_broad_lobe():
    s = ConvexSurface(x=0.1, y=-0.2, radius_m=0.08, arc_length_m=0.12,
                      surface_theta_rad=math.pi / 4)
    feat = CadFeatureExtractor().extract([s])[0]
    assert feat.lobe_width_rad >= math.pi / 2


# ---------------------------------------------------------------------------
# Metadata propagation tests
# ---------------------------------------------------------------------------

def test_geometry_source_label_set():
    primitives = [
        FlatPanel(x=0.0, y=0.0, width_m=0.10, height_m=0.08, normal_theta_rad=1.0),
        EdgeSegment(x=0.1, y=0.0, length_m=0.10, edge_theta_rad=1.0),
        CavityVolume(x=0.2, y=0.0, interior_dim_a_m=0.015, interior_dim_b_m=0.01,
                     depth_m=0.02, aperture_area_m2=0.0002),
        ConvexSurface(x=0.3, y=0.0, radius_m=0.05, arc_length_m=0.10, surface_theta_rad=1.0),
    ]
    features = CadFeatureExtractor().extract(primitives)
    sources = [f.geometry_source for f in features]
    assert sources == ["FlatPanel", "EdgeSegment", "CavityVolume", "ConvexSurface"]


def test_uncertainty_fields_survive_get_scattering_centers():
    p = FlatPanel(x=0.0, y=0.0, width_m=0.10, height_m=0.08, normal_theta_rad=1.0,
                  manufacturing_tolerance_m=0.003)
    feat = CadFeatureExtractor().extract([p])[0]
    scatterer = SyntheticScatterer(features=[feat], characteristic_length=0.5)
    centers = scatterer.get_scattering_centers()
    assert centers[0].geometry_source == "FlatPanel"
    assert centers[0].position_uncertainty_m == pytest.approx(0.003)


@pytest.mark.slow
def test_cad_derived_scenario_runs_in_engine():
    from dhff.pipeline.engine import DHFFEngine
    engine = DHFFEngine(scenario_name="cad_derived")
    engine.setup()
    assert len(engine.ground_truth.features) > 0
