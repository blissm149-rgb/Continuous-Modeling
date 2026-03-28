"""Module 2: Predefined test scenarios."""
from __future__ import annotations

import math

from .scatterer import ScatteringFeature, SyntheticScatterer
from .simulator import ImperfectSimulator, SimulatorError
from .measurement import SyntheticMeasurementSystem


def scenario_simple_missing_feature() -> tuple[SyntheticScatterer, ImperfectSimulator, SyntheticMeasurementSystem]:
    """Scenario 1: 5 scattering centers, simulator is missing 1 of them."""
    features = [
        ScatteringFeature(
            x=0.0, y=0.0, base_amplitude=0.5 + 0j,
            freq_dependence="specular", angular_pattern="specular_lobe",
            lobe_center_theta=math.pi / 2, lobe_center_phi=0.0, lobe_width_rad=0.4,
            label="spec0",
        ),
        ScatteringFeature(
            x=0.3, y=0.1, base_amplitude=0.2 + 0j,
            freq_dependence="specular", angular_pattern="specular_lobe",
            lobe_center_theta=math.pi / 3, lobe_center_phi=0.0, lobe_width_rad=0.4,
            label="spec1",
        ),
        ScatteringFeature(
            x=-0.2, y=0.15, base_amplitude=0.1 + 0j,
            freq_dependence="edge", angular_pattern="broad_lobe",
            lobe_center_theta=2 * math.pi / 3, lobe_center_phi=0.0, lobe_width_rad=0.8,
            label="edge2",
        ),
        ScatteringFeature(
            x=0.1, y=-0.2, base_amplitude=0.08 + 0j,
            freq_dependence="edge", angular_pattern="broad_lobe",
            lobe_center_theta=math.pi / 4, lobe_center_phi=0.0, lobe_width_rad=0.8,
            label="edge3",
        ),
        ScatteringFeature(
            x=0.25, y=-0.1, base_amplitude=0.45 + 0j,
            freq_dependence="cavity_resonant", angular_pattern="isotropic",
            lobe_center_theta=math.pi / 2, lobe_center_phi=0.0, lobe_width_rad=0.5,
            cavity_freq_hz=10e9, cavity_q=15.0,
            label="cavity4",
        ),
    ]

    ground_truth = SyntheticScatterer(features=features, characteristic_length=1.0)
    errors = [SimulatorError(error_type="missing_feature", feature_index=4)]
    simulator = ImperfectSimulator(ground_truth=ground_truth, errors=errors)
    meas_system = SyntheticMeasurementSystem(
        ground_truth=ground_truth, snr_db=40.0, phase_noise_std_rad=0.02
    )
    return ground_truth, simulator, meas_system


def scenario_shifted_and_amplitude() -> tuple[SyntheticScatterer, ImperfectSimulator, SyntheticMeasurementSystem]:
    """Scenario 2: 8 scattering centers. Simulator has 2 shifted, 1 with wrong amplitude."""
    import math
    features = [
        ScatteringFeature(x=0.0, y=0.0, base_amplitude=0.5+0j,
                          freq_dependence="specular", angular_pattern="specular_lobe",
                          lobe_center_theta=math.pi/2, lobe_center_phi=0.0, lobe_width_rad=0.3, label="s0"),
        ScatteringFeature(x=0.3, y=0.1, base_amplitude=0.3+0j,
                          freq_dependence="specular", angular_pattern="specular_lobe",
                          lobe_center_theta=math.pi/3, lobe_center_phi=0.0, lobe_width_rad=0.3, label="s1"),
        ScatteringFeature(x=-0.2, y=0.2, base_amplitude=0.15+0j,
                          freq_dependence="specular", angular_pattern="specular_lobe",
                          lobe_center_theta=2*math.pi/3, lobe_center_phi=0.0, lobe_width_rad=0.3, label="s2"),
        ScatteringFeature(x=0.1, y=-0.25, base_amplitude=0.2+0j,
                          freq_dependence="specular", angular_pattern="specular_lobe",
                          lobe_center_theta=math.pi/4, lobe_center_phi=0.0, lobe_width_rad=0.3, label="s3"),
        ScatteringFeature(x=-0.35, y=0.05, base_amplitude=0.1+0j,
                          freq_dependence="edge", angular_pattern="broad_lobe",
                          lobe_center_theta=math.pi/2, lobe_center_phi=0.0, lobe_width_rad=0.8, label="e4"),
        ScatteringFeature(x=0.4, y=-0.2, base_amplitude=0.08+0j,
                          freq_dependence="edge", angular_pattern="broad_lobe",
                          lobe_center_theta=math.pi/2, lobe_center_phi=0.0, lobe_width_rad=0.8, label="e5"),
        ScatteringFeature(x=-0.1, y=0.3, base_amplitude=0.12+0j,
                          freq_dependence="cavity_resonant", angular_pattern="isotropic",
                          cavity_freq_hz=9e9, cavity_q=40.0, label="cav6"),
        ScatteringFeature(x=0.2, y=0.15, base_amplitude=0.07+0j,
                          freq_dependence="creeping", angular_pattern="broad_lobe",
                          lobe_center_theta=math.pi/2, lobe_center_phi=0.0, lobe_width_rad=1.0,
                          label="cr7"),
    ]
    ground_truth = SyntheticScatterer(features=features, characteristic_length=1.0)
    errors = [
        SimulatorError(error_type="shifted_feature", feature_index=1, shift_x=0.08, shift_y=0.03),
        SimulatorError(error_type="shifted_feature", feature_index=5, shift_x=-0.05, shift_y=0.02),
        SimulatorError(error_type="amplitude_error", feature_index=3, amplitude_scale=0.3),
    ]
    simulator = ImperfectSimulator(ground_truth=ground_truth, errors=errors)
    meas_system = SyntheticMeasurementSystem(
        ground_truth=ground_truth, snr_db=35.0, phase_noise_std_rad=0.03
    )
    return ground_truth, simulator, meas_system


def scenario_cad_derived() -> tuple[SyntheticScatterer, ImperfectSimulator, SyntheticMeasurementSystem]:
    """Scenario 4: features derived from CAD geometry primitives.

    Geometry: two flat panels, one edge, one small cavity, one convex surface.
    Simulator error: cavity is missing (same challenge as simple_missing_feature
    but position/parameters come from physics formulas, not hand-coding).
    """
    # Lazy import avoids circular dependency (cad → synthetic.scatterer → synthetic → scenarios → cad)
    from dhff.cad import CadFeatureExtractor, FlatPanel, EdgeSegment, CavityVolume, ConvexSurface

    primitives = [
        FlatPanel(x=0.0, y=0.0, width_m=0.12, height_m=0.08,
                  normal_theta_rad=math.pi / 2, label="panel_main"),
        FlatPanel(x=0.3, y=0.1, width_m=0.06, height_m=0.05,
                  normal_theta_rad=math.pi / 3, label="panel_side"),
        EdgeSegment(x=-0.2, y=0.15, length_m=0.15,
                    edge_theta_rad=2 * math.pi / 3, label="leading_edge"),
        CavityVolume(x=0.25, y=-0.1,
                     interior_dim_a_m=0.015, interior_dim_b_m=0.010, depth_m=0.020,
                     aperture_area_m2=0.0002, label="inlet_cavity"),
        ConvexSurface(x=0.1, y=-0.2, radius_m=0.08, arc_length_m=0.12,
                      surface_theta_rad=math.pi / 4, label="nose_surface"),
    ]
    extractor = CadFeatureExtractor(freq_range_hz=(8e9, 12e9), f_center=10e9)
    features = extractor.extract(primitives)

    ground_truth = SyntheticScatterer(features=features, characteristic_length=0.5)
    # Simulator is missing the cavity feature
    cavity_index = next(i for i, f in enumerate(features) if "cavity" in f.label)
    errors = [SimulatorError(error_type="missing_feature", feature_index=cavity_index)]
    simulator = ImperfectSimulator(ground_truth=ground_truth, errors=errors)
    meas_system = SyntheticMeasurementSystem(
        ground_truth=ground_truth, snr_db=38.0, phase_noise_std_rad=0.025
    )
    return ground_truth, simulator, meas_system


def scenario_complex_target() -> tuple[SyntheticScatterer, ImperfectSimulator, SyntheticMeasurementSystem]:
    """Scenario 3: 15 features, complex errors."""
    import math
    import numpy as np

    # Build 15 features systematically
    features = []
    rng = np.random.default_rng(123)

    # 6 specular features
    for i in range(6):
        x = rng.uniform(-0.4, 0.4)
        y = rng.uniform(-0.3, 0.3)
        amp = rng.uniform(0.05, 0.5)
        theta_c = rng.uniform(0.3, math.pi - 0.3)
        features.append(ScatteringFeature(
            x=x, y=y, base_amplitude=amp+0j,
            freq_dependence="specular", angular_pattern="specular_lobe",
            lobe_center_theta=theta_c, lobe_center_phi=0.0, lobe_width_rad=0.3, label=f"s{i}",
        ))

    # 4 edge features
    for i in range(4):
        x = rng.uniform(-0.4, 0.4)
        y = rng.uniform(-0.3, 0.3)
        amp = rng.uniform(0.03, 0.15)
        theta_c = rng.uniform(0.3, math.pi - 0.3)
        features.append(ScatteringFeature(
            x=x, y=y, base_amplitude=amp+0j,
            freq_dependence="edge", angular_pattern="broad_lobe",
            lobe_center_theta=theta_c, lobe_center_phi=0.0, lobe_width_rad=0.8, label=f"e{i}",
        ))

    # 3 cavity features
    for i, f0 in enumerate([8.5e9, 10e9, 11.5e9]):
        x = rng.uniform(-0.3, 0.3)
        y = rng.uniform(-0.2, 0.2)
        amp = rng.uniform(0.05, 0.2)
        features.append(ScatteringFeature(
            x=x, y=y, base_amplitude=amp+0j,
            freq_dependence="cavity_resonant", angular_pattern="isotropic",
            cavity_freq_hz=f0, cavity_q=40.0, label=f"cav{i}",
        ))

    # 2 creeping wave features
    for i in range(2):
        x = rng.uniform(-0.4, 0.4)
        y = rng.uniform(-0.3, 0.3)
        amp = rng.uniform(0.02, 0.1)
        features.append(ScatteringFeature(
            x=x, y=y, base_amplitude=amp+0j,
            freq_dependence="creeping", angular_pattern="broad_lobe",
            lobe_center_theta=math.pi/2, lobe_center_phi=0.0, lobe_width_rad=1.0, label=f"cr{i}",
        ))

    ground_truth = SyntheticScatterer(features=features, characteristic_length=1.0)
    errors = [
        SimulatorError(error_type="missing_feature", feature_index=2),
        SimulatorError(error_type="missing_feature", feature_index=9),
        SimulatorError(error_type="shifted_feature", feature_index=0, shift_x=0.06, shift_y=0.02),
        SimulatorError(error_type="shifted_feature", feature_index=4, shift_x=-0.04, shift_y=0.03),
        SimulatorError(error_type="shifted_feature", feature_index=7, shift_x=0.05, shift_y=-0.02),
        SimulatorError(error_type="amplitude_error", feature_index=1, amplitude_scale=0.4),
        SimulatorError(error_type="amplitude_error", feature_index=5, amplitude_scale=2.5),
        SimulatorError(error_type="solver_noise", noise_floor_dbsm=-50.0),
    ]
    simulator = ImperfectSimulator(ground_truth=ground_truth, errors=errors)
    meas_system = SyntheticMeasurementSystem(
        ground_truth=ground_truth, snr_db=30.0, phase_noise_std_rad=0.05
    )
    return ground_truth, simulator, meas_system
