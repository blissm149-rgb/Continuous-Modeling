"""Configuration dataclass for ParametricSCModel extraction thresholds.

All hard-coded numeric constants previously scattered across
parametric_model.py are gathered here so callers can tune them in one place
without editing library code.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SCExtractorConfig:
    """Thresholds and limits for the parametric scattering-center extractor.

    The defaults reproduce the original hard-coded behaviour.  Set ``snr_db``
    to enable automatic relaxation of several thresholds when data are noisier
    than the 40 dB synthetic benchmark.

    Adaptive relaxation (applied when ``snr_db`` is set and < 25 dB):
      - ``amplitude_threshold_db``  lowered by ``(25 − snr_db) × 0.8`` dB
      - ``min_peak_ratio``          reduced toward 1.1 (floor)
      - ``merge_distance_m``        widened slightly
    """
    # ── Feature detection thresholds ──────────────────────────────────────
    amplitude_threshold_db: float = -25.0
    """Ignore scattering centers weaker than this (dB relative to strongest)."""

    min_samples_to_fit: int = 15
    """Minimum number of discrepancy samples required before fitting."""

    merge_distance_m: float = 0.12
    """Merge extracted centers that are closer than this distance (metres)."""

    # ── Grid-search / spectral-peak extraction ─────────────────────────────
    grid_step_m: float = 0.05
    """Step size for the coarse (x, y) grid search (metres)."""

    grid_half_extent_m: float = 0.6
    """Half-extent of the (x, y) search grid (metres from origin)."""

    spectral_bandwidth_hz: float = 1.5e9
    """Only use samples within ±this bandwidth of the spectral peak."""

    min_peak_ratio: float = 1.5
    """Spectral peak must exceed the median spectrum by at least this factor."""

    max_range_m: float = 2.0
    """Reject extracted / triangulated positions beyond this range (metres)."""

    # ── Optimiser limits ──────────────────────────────────────────────────
    lm_max_nfev: int = 200
    """Maximum function evaluations for Levenberg-Marquardt refinement."""

    # ── SNR-adaptive relaxation ───────────────────────────────────────────
    snr_db: float | None = None
    """When set, relax several thresholds proportionally for noisy data."""

    def effective(self) -> "SCExtractorConfig":
        """Return a copy with SNR-adaptive adjustments applied."""
        if self.snr_db is None or self.snr_db >= 25.0:
            return self
        delta = 25.0 - self.snr_db
        return SCExtractorConfig(
            amplitude_threshold_db=self.amplitude_threshold_db - delta * 0.8,
            min_samples_to_fit=self.min_samples_to_fit,
            merge_distance_m=self.merge_distance_m + 0.02 * (delta / 5.0),
            grid_step_m=self.grid_step_m,
            grid_half_extent_m=self.grid_half_extent_m,
            spectral_bandwidth_hz=self.spectral_bandwidth_hz,
            min_peak_ratio=max(1.1, self.min_peak_ratio - 0.3 * (delta / 5.0)),
            max_range_m=self.max_range_m,
            lm_max_nfev=self.lm_max_nfev,
            snr_db=self.snr_db,
        )
