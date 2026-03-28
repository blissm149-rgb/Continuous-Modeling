"""Module 5: Abstract base class for discrepancy models."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from dhff.core.types import DiscrepancySample, ObservationPoint


class DiscrepancyModel(ABC):
    """Abstract base class for all discrepancy models."""

    @abstractmethod
    def fit(self, samples: list[DiscrepancySample]) -> None:
        """Fit the model to observed discrepancy samples."""
        ...

    @abstractmethod
    def predict(self, points: list[ObservationPoint]) -> tuple[np.ndarray, np.ndarray]:
        """Predict discrepancy at new points.

        Returns:
            mean: np.ndarray complex128 (N,)
            variance: np.ndarray float64 (N,)
        """
        ...

    @abstractmethod
    def acquisition_uncertainty(self, points: list[ObservationPoint]) -> np.ndarray:
        """Fast uncertainty estimate for acquisition function evaluation."""
        ...

    @abstractmethod
    def predicted_discrepancy_power(self, points: list[ObservationPoint]) -> np.ndarray:
        """E[|delta|^2] = |E[delta]|^2 + Var[delta]."""
        ...
