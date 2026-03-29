"""Module 6: Acquisition functions for discrepancy-hunting measurement selection."""
from __future__ import annotations

import numpy as np

from dhff.core.types import MeasurementPlan, ObservationPoint
from dhff.core.coordinate_system import angular_distance_points


class DiscrepancyAcquisition:
    """Composite acquisition function:
    alpha(x) = E[|delta(x)|^2] + lambda * Uncertainty(x) + mu * D_prior(x)
    """

    def __init__(
        self,
        discrepancy_model,
        susceptibility_map,
        lambda_explore: float = 1.0,
        mu_prior: float = 0.5,
    ):
        self.discrepancy_model = discrepancy_model
        self.susceptibility_map = susceptibility_map
        self.lambda_explore = lambda_explore
        self.mu_prior = mu_prior

    def evaluate(self, candidates: list[ObservationPoint]) -> np.ndarray:
        """Compute acquisition function at all candidate points."""
        power = self.discrepancy_model.predicted_discrepancy_power(candidates)
        uncertainty = self.discrepancy_model.acquisition_uncertainty(candidates)
        prior = self.susceptibility_map.compute(candidates)

        def normalize(arr):
            m = np.max(arr) + 1e-30
            return arr / m

        alpha = (
            normalize(power)
            + self.lambda_explore * normalize(uncertainty)
            + self.mu_prior * normalize(prior)
        )
        return alpha

    def select_batch(
        self,
        candidates: list[ObservationPoint],
        batch_size: int,
        min_angular_sep_rad: float = 0.05,
    ) -> MeasurementPlan:
        """Select a batch of points with angular diversity."""
        scores = self.evaluate(candidates)
        sorted_idx = np.argsort(scores)[::-1]

        selected_indices = []
        selected_points = []
        selected_scores = []

        # Try with initial min_sep
        current_sep = min_angular_sep_rad
        remaining = list(sorted_idx)

        while len(selected_points) < batch_size and remaining:
            for idx in list(remaining):
                if len(selected_points) >= batch_size:
                    break
                pt = candidates[idx]
                too_close = any(
                    angular_distance_points(pt, sp) < current_sep
                    for sp in selected_points
                )
                if not too_close:
                    selected_indices.append(idx)
                    selected_points.append(pt)
                    selected_scores.append(float(scores[idx]))
                    remaining.remove(idx)

            if len(selected_points) < batch_size and remaining:
                current_sep /= 2.0
                if current_sep < 1e-4:
                    # Just take remaining top candidates
                    for idx in remaining:
                        if len(selected_points) >= batch_size:
                            break
                        selected_indices.append(idx)
                        selected_points.append(candidates[idx])
                        selected_scores.append(float(scores[idx]))
                    break

        # Compute per-point components for rationale
        power_n = scores / (np.max(scores) + 1e-30)
        rationale = []
        for k, idx in enumerate(selected_indices):
            p = candidates[idx]
            pwr = float(power_n[idx]) if idx < len(power_n) else 0.0
            sc = selected_scores[k]
            rationale.append(
                f"Score={sc:.3f} at theta={p.theta:.2f}, freq={p.freq_hz/1e9:.1f}GHz"
            )

        return MeasurementPlan(
            points=selected_points,
            scores=selected_scores,
            rationale=rationale,
        )


class ScatteringCenterAcquisition:
    """Acquisition based on scattering center anomaly analysis."""

    def __init__(
        self,
        anomalies,
        anomaly_classifier,
        freq_range_hz: tuple[float, float],
        theta_range: tuple[float, float] = (0.1, 3.04),  # ~pi-0.1
        seed: int | None = None,
    ):
        self.anomalies = anomalies
        self.anomaly_classifier = anomaly_classifier
        self.freq_range_hz = freq_range_hz
        self.theta_range = theta_range
        self._seed = seed

    def generate_candidates(
        self, n_per_anomaly: int = 5
    ) -> list[tuple[ObservationPoint, str]]:
        """Generate measurement candidates tailored to each anomaly."""
        import math
        rng = np.random.default_rng(self._seed if self._seed is not None else 0)
        candidates = []

        for anomaly in self.anomalies:
            strategy = self.anomaly_classifier.suggest_measurement_strategy(anomaly)
            theta_range = strategy.get("suggested_theta_range") or self.theta_range
            freq_range = strategy.get("suggested_freq_range") or self.freq_range_hz
            if freq_range is None:
                freq_range = self.freq_range_hz

            for _ in range(n_per_anomaly):
                theta = float(rng.uniform(theta_range[0], theta_range[1]))
                freq = float(rng.uniform(freq_range[0], freq_range[1]))
                pt = ObservationPoint(theta=theta, phi=0.0, freq_hz=freq)
                rationale = (
                    f"Anomaly {anomaly.anomaly_type.name}: {strategy['strategy']}"
                )
                candidates.append((pt, rationale))

        return candidates
