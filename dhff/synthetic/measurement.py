"""Module 2: Synthetic measurement system."""
from __future__ import annotations

import numpy as np

from dhff.core.types import ComplexRCS, ObservationPoint
from .scatterer import SyntheticScatterer


class SyntheticMeasurementSystem:
    """Simulates taking measurements of the ground-truth scatterer with realistic noise."""

    def __init__(
        self,
        ground_truth: SyntheticScatterer,
        snr_db: float = 40.0,
        phase_noise_std_rad: float = 0.02,
        calibration_error: complex = 1.0 + 0j,
        seed: int = 7,
    ):
        self.ground_truth = ground_truth
        self.snr_db = snr_db
        self.phase_noise_std_rad = phase_noise_std_rad
        self.calibration_error = calibration_error
        self._rng = np.random.default_rng(seed)

    def measure(self, points: list[ObservationPoint]) -> ComplexRCS:
        """Take a (noisy) measurement at the specified observation points."""
        rcs = self.ground_truth.compute_rcs(points)
        values = rcs.values.copy()

        for i, v in enumerate(values):
            signal_power = np.abs(v) ** 2
            noise_power = signal_power / (10.0 ** (self.snr_db / 10.0)) + 1e-30
            # Complex Gaussian noise CN(0, noise_power)
            noise = (
                self._rng.standard_normal() + 1j * self._rng.standard_normal()
            ) * np.sqrt(noise_power / 2.0)
            # Phase noise
            phase_noise = self._rng.normal(0.0, self.phase_noise_std_rad)
            values[i] = (v + noise) * np.exp(1j * phase_noise) * self.calibration_error

        return ComplexRCS(observation_points=list(points), values=values.astype(np.complex128))

    def measure_single(self, point: ObservationPoint) -> complex:
        """Take a single measurement."""
        rcs = self.measure([point])
        return complex(rcs.values[0])
