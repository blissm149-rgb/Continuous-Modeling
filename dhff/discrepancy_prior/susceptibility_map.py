"""Module 3: Combined discrepancy susceptibility map."""
from __future__ import annotations

import math

import numpy as np

from dhff.core.types import MeasurementPlan, ObservationPoint
from dhff.core.coordinate_system import angular_distance_points


class DiscrepancySusceptibilityMap:
    """Combines ensemble disagreement and geometric feature analysis into D_prior."""

    def __init__(
        self,
        ensemble,
        geometric,
        w_ensemble: float = 0.5,
        w_geometric: float = 0.5,
    ):
        self.ensemble = ensemble
        self.geometric = geometric
        self.w_ensemble = w_ensemble
        self.w_geometric = w_geometric

    def compute(self, points: list[ObservationPoint]) -> np.ndarray:
        """Compute the combined discrepancy prior at each point."""
        ensemble_scores = self.ensemble.compute(points)  # already [0,1]
        geometric_raw = self.geometric.predict_susceptibility(points)
        geo_max = np.max(geometric_raw) + 1e-30
        geometric_scores = geometric_raw / geo_max

        d_prior = self.w_ensemble * ensemble_scores + self.w_geometric * geometric_scores
        d_max = np.max(d_prior) + 1e-30
        return d_prior / d_max

    def select_initial_measurements(
        self, candidate_points: list[ObservationPoint], n_measurements: int
    ) -> MeasurementPlan:
        """Select top-n measurements by D_prior with angular diversity enforcement."""
        scores = self.compute(candidate_points)
        sorted_idx = np.argsort(scores)[::-1]  # descending

        min_sep = math.pi / (2.0 * max(n_measurements, 1))
        selected_indices = []
        selected_points = []

        for idx in sorted_idx:
            if len(selected_points) >= n_measurements:
                break
            candidate = candidate_points[idx]
            too_close = False
            for sp in selected_points:
                if angular_distance_points(candidate, sp) < min_sep:
                    too_close = True
                    break
            if not too_close:
                selected_indices.append(idx)
                selected_points.append(candidate)

        # If we didn't get enough, relax separation
        if len(selected_points) < n_measurements:
            min_sep /= 2.0
            for idx in sorted_idx:
                if len(selected_points) >= n_measurements:
                    break
                if idx in selected_indices:
                    continue
                candidate = candidate_points[idx]
                too_close = False
                for sp in selected_points:
                    if angular_distance_points(candidate, sp) < min_sep:
                        too_close = True
                        break
                if not too_close:
                    selected_indices.append(idx)
                    selected_points.append(candidate)

        # Final fallback: take top-n without diversity
        if len(selected_points) < n_measurements:
            for idx in sorted_idx:
                if len(selected_points) >= n_measurements:
                    break
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    selected_points.append(candidate_points[idx])

        plan_scores = [float(scores[i]) for i in selected_indices[:n_measurements]]
        plan_points = selected_points[:n_measurements]
        rationale = [
            f"High D_prior={plan_scores[k]:.3f} at theta={plan_points[k].theta:.2f}, phi={plan_points[k].phi:.2f}, freq={plan_points[k].freq_hz/1e9:.1f}GHz"
            for k in range(len(plan_points))
        ]
        return MeasurementPlan(points=plan_points, scores=plan_scores, rationale=rationale)
