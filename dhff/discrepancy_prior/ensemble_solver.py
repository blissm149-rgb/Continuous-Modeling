"""Module 3: Multi-solver ensemble disagreement."""
from __future__ import annotations

import numpy as np

from dhff.core.types import ObservationPoint


class EnsembleDisagreement:
    """Compute inter-solver disagreement as a proxy for simulation unreliability."""

    def __init__(self, simulator):
        self.simulator = simulator

    def compute(
        self, points: list[ObservationPoint], n_solvers: int = 5
    ) -> np.ndarray:
        """Returns normalized disagreement score at each observation point (values in [0,1])."""
        results = self.simulator.compute_rcs_multi_solver(points, n_solvers=n_solvers)
        stacked = np.stack([r.values for r in results])  # (n_solvers, N)

        mu = np.mean(stacked, axis=0)  # (N,)
        diff_sq = np.mean(np.abs(stacked - mu[np.newaxis, :]) ** 2, axis=0)  # (N,)
        denom = np.abs(mu) ** 2 + 1e-30
        disagreement = diff_sq / denom  # normalized variance

        # Clip at 99th percentile then normalize to [0, 1]
        clip_val = np.percentile(disagreement, 99)
        disagreement = np.clip(disagreement, 0.0, clip_val)
        max_val = np.max(disagreement) + 1e-30
        return disagreement / max_val
