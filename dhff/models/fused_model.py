"""Module 5: Fused RCS model (sim + discrepancy)."""
from __future__ import annotations

import numpy as np

from dhff.core.types import ComplexRCS, ObservationPoint
from .base import DiscrepancyModel


class FusedRCSModel:
    """The final fused model: sigma_fused = sigma_sim + delta_model."""

    def __init__(
        self,
        simulator,
        discrepancy_model: DiscrepancyModel,
    ):
        self.simulator = simulator
        self.discrepancy_model = discrepancy_model

    def predict(
        self, points: list[ObservationPoint]
    ) -> tuple[ComplexRCS, np.ndarray]:
        """Returns (fused_rcs, uncertainty) at the given points."""
        sim_rcs = self.simulator.compute_rcs(points)
        disc_mean, disc_var = self.discrepancy_model.predict(points)

        fused_values = (sim_rcs.values + disc_mean).astype(np.complex128)
        fused_rcs = ComplexRCS(observation_points=list(points), values=fused_values)
        return fused_rcs, disc_var

    def error_vs_ground_truth(
        self, ground_truth, points: list[ObservationPoint]
    ) -> dict:
        """Compute error metrics against known ground truth."""
        gt_rcs = ground_truth.compute_rcs(points)
        sim_rcs = self.simulator.compute_rcs(points)
        fused_rcs, uncertainty = self.predict(points)

        truth = gt_rcs.values
        sim = sim_rcs.values
        fused = fused_rcs.values

        # Complex NMSE
        truth_power = np.mean(np.abs(truth)**2)
        complex_nmse = np.mean(np.abs(fused - truth)**2) / (truth_power + 1e-30)
        sim_only_nmse = np.mean(np.abs(sim - truth)**2) / (truth_power + 1e-30)

        # Magnitude RMSE in dB
        mag_fused_db = 10.0 * np.log10(np.abs(fused) + 1e-30)
        mag_truth_db = 10.0 * np.log10(np.abs(truth) + 1e-30)
        mag_rmse_db = float(np.sqrt(np.mean((mag_fused_db - mag_truth_db)**2)))

        # Phase RMSE
        phase_diff = np.angle(fused * np.conj(truth))
        phase_rmse_rad = float(np.sqrt(np.mean(phase_diff**2)))

        # Coverage metrics
        std_est = np.sqrt(np.maximum(uncertainty, 1e-30))
        abs_error = np.abs(fused - truth)
        coverage_68 = float(np.mean(abs_error < std_est))
        coverage_95 = float(np.mean(abs_error < 2.0 * std_est))

        return {
            "complex_nmse": float(complex_nmse),
            "mag_rmse_db": mag_rmse_db,
            "phase_rmse_rad": phase_rmse_rad,
            "coverage_68": coverage_68,
            "coverage_95": coverage_95,
            "sim_only_nmse": float(sim_only_nmse),
        }
