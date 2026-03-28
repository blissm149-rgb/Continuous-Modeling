"""Module 2: Imperfect CAD simulator."""
from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass

import numpy as np

from dhff.core.types import ComplexRCS, ObservationPoint
from .scatterer import ScatteringFeature, SyntheticScatterer


@dataclass
class SimulatorError:
    """A specific error injected into the simulator."""
    error_type: str          # "missing_feature", "shifted_feature", "amplitude_error",
                             # "frequency_bias", "angular_bias", "solver_noise"
    feature_index: int = -1  # which feature is affected (-1 for global errors)
    shift_x: float = 0.0
    shift_y: float = 0.0
    amplitude_scale: float = 1.0
    phase_bias_rad: float = 0.0
    noise_floor_dbsm: float = -60.0


class ImperfectSimulator:
    """Simulates what a CAD-based EM solver would produce."""

    def __init__(self, ground_truth: SyntheticScatterer, errors: list[SimulatorError]):
        self.ground_truth = ground_truth
        self.errors = errors
        self._degraded_scatterer: SyntheticScatterer | None = None
        self._noise_floor_dbsm: float = -80.0  # default
        self._build_degraded_scatterer()

    def _build_degraded_scatterer(self):
        features = [copy.deepcopy(f) for f in self.ground_truth.features]

        # Collect indices to remove and modifications to apply
        missing_indices = set()
        for err in self.errors:
            if err.error_type == "solver_noise":
                self._noise_floor_dbsm = err.noise_floor_dbsm
            elif err.error_type == "missing_feature":
                if 0 <= err.feature_index < len(features):
                    missing_indices.add(err.feature_index)

        # Apply non-removal errors to a temporary list
        modified_features = []
        for i, feat in enumerate(features):
            if i in missing_indices:
                continue
            f = copy.deepcopy(feat)
            for err in self.errors:
                if err.feature_index != i:
                    continue
                if err.error_type == "shifted_feature":
                    f = ScatteringFeature(
                        x=f.x + err.shift_x,
                        y=f.y + err.shift_y,
                        base_amplitude=f.base_amplitude,
                        freq_dependence=f.freq_dependence,
                        angular_pattern=f.angular_pattern,
                        lobe_center_theta=f.lobe_center_theta,
                        lobe_center_phi=f.lobe_center_phi,
                        lobe_width_rad=f.lobe_width_rad,
                        cavity_freq_hz=f.cavity_freq_hz,
                        cavity_q=f.cavity_q,
                        label=f.label,
                        geometry_source=f.geometry_source,
                        position_uncertainty_m=f.position_uncertainty_m,
                        amplitude_uncertainty_db=f.amplitude_uncertainty_db,
                        freq_param_uncertainty=f.freq_param_uncertainty,
                    )
                elif err.error_type == "amplitude_error":
                    new_amp = f.base_amplitude * err.amplitude_scale * np.exp(1j * err.phase_bias_rad)
                    f = ScatteringFeature(
                        x=f.x, y=f.y,
                        base_amplitude=new_amp,
                        freq_dependence=f.freq_dependence,
                        angular_pattern=f.angular_pattern,
                        lobe_center_theta=f.lobe_center_theta,
                        lobe_center_phi=f.lobe_center_phi,
                        lobe_width_rad=f.lobe_width_rad,
                        cavity_freq_hz=f.cavity_freq_hz,
                        cavity_q=f.cavity_q,
                        label=f.label,
                        geometry_source=f.geometry_source,
                        position_uncertainty_m=f.position_uncertainty_m,
                        amplitude_uncertainty_db=f.amplitude_uncertainty_db,
                        freq_param_uncertainty=f.freq_param_uncertainty,
                    )
                elif err.error_type == "angular_bias":
                    f = ScatteringFeature(
                        x=f.x, y=f.y,
                        base_amplitude=f.base_amplitude,
                        freq_dependence=f.freq_dependence,
                        angular_pattern=f.angular_pattern,
                        lobe_center_theta=f.lobe_center_theta + err.shift_x,  # reuse shift_x for theta bias
                        lobe_center_phi=f.lobe_center_phi + err.shift_y,
                        lobe_width_rad=f.lobe_width_rad,
                        cavity_freq_hz=f.cavity_freq_hz,
                        cavity_q=f.cavity_q,
                        label=f.label,
                        geometry_source=f.geometry_source,
                        position_uncertainty_m=f.position_uncertainty_m,
                        amplitude_uncertainty_db=f.amplitude_uncertainty_db,
                        freq_param_uncertainty=f.freq_param_uncertainty,
                    )
            modified_features.append(f)

        self._degraded_scatterer = SyntheticScatterer(
            features=modified_features,
            characteristic_length=self.ground_truth.characteristic_length,
        )

    @property
    def degraded_scatterer(self) -> SyntheticScatterer:
        return self._degraded_scatterer

    def compute_rcs(self, points: list[ObservationPoint]) -> ComplexRCS:
        """Compute the simulated (imperfect) RCS."""
        rcs = self._degraded_scatterer.compute_rcs(points)

        # Add solver noise
        noise_power_linear = 10.0 ** (self._noise_floor_dbsm / 10.0)
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(len(points)) + 1j * rng.standard_normal(len(points))
        noise *= np.sqrt(noise_power_linear / 2.0)

        return ComplexRCS(
            observation_points=list(points),
            values=(rcs.values + noise).astype(np.complex128),
        )

    def compute_rcs_multi_solver(
        self, points: list[ObservationPoint], n_solvers: int = 3
    ) -> list[ComplexRCS]:
        """Simulate multiple 'solvers' with different random perturbations."""
        results = []
        base_rcs = self._degraded_scatterer.compute_rcs(points)

        for solver_idx in range(n_solvers):
            rng = np.random.default_rng(solver_idx + 1000)
            perturbed_values = base_rcs.values.copy()

            # Per-feature amplitude and phase jitter
            for feat_idx, feat in enumerate(self._degraded_scatterer.features):
                amp_jitter_db = rng.uniform(-0.5, 0.5)
                phase_jitter_rad = rng.uniform(-5 * np.pi / 180, 5 * np.pi / 180)
                amp_scale = 10 ** (amp_jitter_db / 20.0) * np.exp(1j * phase_jitter_rad)

                # Recompute this feature's contribution and apply jitter
                theta_arr = np.array([p.theta for p in points])
                phi_arr = np.array([p.phi for p in points])
                freq_arr = np.array([p.freq_hz for p in points])
                k0 = 2.0 * np.pi * freq_arr / 299792458.0

                from .scatterer import _ANG_PATTERN_FUNCS, _FREQ_DEP_FUNCS, angular_gain_isotropic
                ang_func = _ANG_PATTERN_FUNCS.get(feat.angular_pattern, angular_gain_isotropic)
                G = ang_func(theta_arr, phi_arr, feat.lobe_center_theta, feat.lobe_center_phi, feat.lobe_width_rad)
                freq_func = _FREQ_DEP_FUNCS.get(feat.freq_dependence, lambda f, ft: np.ones_like(f, dtype=np.complex128))
                F = freq_func(freq_arr, feat)
                phase_term = np.exp(1j * 2.0 * k0 * (feat.x * np.cos(theta_arr) + feat.y * np.sin(theta_arr)))

                original_contrib = feat.base_amplitude * G * F * phase_term
                perturbed_values += original_contrib * (amp_scale - 1.0)  # add the delta

            # Independent solver noise
            noise_power = 10.0 ** (self._noise_floor_dbsm / 10.0)
            noise = rng.standard_normal(len(points)) + 1j * rng.standard_normal(len(points))
            noise *= np.sqrt(noise_power / 2.0)

            results.append(ComplexRCS(
                observation_points=list(points),
                values=(perturbed_values + noise).astype(np.complex128),
            ))

        return results
