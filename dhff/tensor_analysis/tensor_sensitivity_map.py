"""TensorSensitivityMap: simulation-only sensitivity analysis.

Derives a ranked list of informative measurement points from a raw
(azimuth × elevation × frequency) complex RCS tensor.  No CAD geometry
labels or running simulator required.

Interface-compatible with DiscrepancySusceptibilityMap:
    .compute(points)                   → float[0,1] per point
    .select_initial_measurements(...)  → MeasurementPlan

Coordinate convention
---------------------
tensor axis 0  → azimuth   (az_rad  → ObservationPoint.phi)
tensor axis 1  → elevation (el_rad  → ObservationPoint.theta)
tensor axis 2  → frequency (freq_hz → ObservationPoint.freq_hz)

Four sensitivity signals
------------------------
gradient     (weight 0.35): amplitude / phase-curvature gradients
             → lobe edges, resonance flanks
isar         (weight 0.20): ISAR sidelobe floor
             → multi-scatterer interference density
spectral     (weight 0.25): spectral entropy + resonance peak count
             → frequency-selective features (cavities, coatings)
cancellation (weight 0.20): near-null amplitude nodes
             → maximally sensitive to geometry errors
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from dhff.core.types import MeasurementPlan, ObservationPoint
from dhff.core.coordinate_system import angular_distance_points

from .gradient_analyzer    import GradientAnalyzer
from .isar_analyzer        import ISARAnalyzer
from .spectral_analyzer    import SpectralAnalyzer
from .cancellation_detector import CancellationDetector


def _normalise(arr: np.ndarray) -> np.ndarray:
    """Normalise a float array to [0, 1]; returns zeros if all-zero."""
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-30:
        return np.zeros_like(arr, dtype=float)
    return ((arr - mn) / (mx - mn)).astype(float)


class TensorSensitivityMap:
    """
    Sensitivity map derived purely from an (az × el × freq) complex RCS tensor.

    Parameters
    ----------
    rcs_tensor : np.ndarray, shape (N_az, N_el, N_freq), complex128
        Complex scattering coefficients from the EM simulation.
    az_rad : np.ndarray, shape (N_az,)
        Azimuth angles in radians (monotonically increasing).
    el_rad : np.ndarray, shape (N_el,)
        Elevation angles in radians (monotonically increasing).
    freq_hz : np.ndarray, shape (N_freq,)
        Frequencies in Hz (monotonically increasing).
    weights : dict[str, float] | None
        Override any subset of the default per-method weights:
        {"gradient": 0.35, "isar": 0.20, "spectral": 0.25, "cancellation": 0.20}
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "gradient":     0.35,
        "isar":         0.20,
        "spectral":     0.25,
        "cancellation": 0.20,
    }

    def __init__(
        self,
        rcs_tensor: np.ndarray,
        az_rad:     np.ndarray,
        el_rad:     np.ndarray,
        freq_hz:    np.ndarray,
        weights:    dict[str, float] | None = None,
    ) -> None:
        rcs_tensor = np.asarray(rcs_tensor, dtype=complex)
        if rcs_tensor.ndim != 3:
            raise ValueError(
                f"rcs_tensor must be 3-D (N_az, N_el, N_freq), got shape {rcs_tensor.shape}"
            )
        self._az    = np.asarray(az_rad,  dtype=float)
        self._el    = np.asarray(el_rad,  dtype=float)
        self._freq  = np.asarray(freq_hz, dtype=float)
        self._w     = {**self.DEFAULT_WEIGHTS, **(weights or {})}
        self._tensor = rcs_tensor

        # Compute the dense sensitivity grid once at init time
        self._per_method: dict[str, np.ndarray] = {}
        self._score_grid = self._build_score_grid(rcs_tensor)  # (N_az, N_el, N_freq)

        # RegularGridInterpolator: axes order (el, az, freq)
        # so that ObservationPoint.theta=el and .phi=az map naturally.
        self._interp = RegularGridInterpolator(
            (self._el, self._az, self._freq),
            self._score_grid.transpose(1, 0, 2),  # (el, az, freq)
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_score_grid(self, tensor: np.ndarray) -> np.ndarray:
        """Run all 4 analyzers and return the weighted combined score grid."""
        w = self._w

        # 1. Gradient
        grad_out = GradientAnalyzer().compute(tensor, self._az, self._el, self._freq)
        grad_score = (
            0.5 * _normalise(grad_out["amplitude_gradient"])
            + 0.5 * _normalise(grad_out["phase_curvature"])
        )
        self._per_method["gradient"] = grad_score

        # 2. ISAR
        isar_score = _normalise(
            ISARAnalyzer().compute(tensor, self._az, self._el, self._freq)
        )
        self._per_method["isar"] = isar_score

        # 3. Spectral
        spec_out = SpectralAnalyzer().compute(tensor, self._freq)
        spec_score = (
            0.6 * _normalise(spec_out["spectral_variance"])
            + 0.4 * _normalise(spec_out["resonance_count"])
        )
        self._per_method["spectral"] = spec_score

        # 4. Cancellation
        canc_score = _normalise(CancellationDetector().compute(tensor))
        self._per_method["cancellation"] = canc_score

        combined = (
            w["gradient"]     * grad_score
            + w["isar"]         * isar_score
            + w["spectral"]     * spec_score
            + w["cancellation"] * canc_score
        )
        return _normalise(combined)   # (N_az, N_el, N_freq)

    # ------------------------------------------------------------------
    # Public interface (DiscrepancySusceptibilityMap-compatible)
    # ------------------------------------------------------------------

    def compute(self, points: list[ObservationPoint]) -> np.ndarray:
        """Score each ObservationPoint.  Returns float array in [0, 1].

        Mapping convention: ObservationPoint.theta → elevation axis,
                            ObservationPoint.phi   → azimuth axis.
        """
        if not points:
            return np.empty(0, dtype=float)
        coords = np.array([[p.theta, p.phi, p.freq_hz] for p in points])
        return np.clip(self._interp(coords).astype(float), 0.0, 1.0)

    def select_initial_measurements(
        self,
        candidate_points: list[ObservationPoint],
        n_measurements:   int,
    ) -> MeasurementPlan:
        """Select top-n measurement points by sensitivity with angular diversity.

        Mirrors DiscrepancySusceptibilityMap.select_initial_measurements.
        """
        scores      = self.compute(candidate_points)
        sorted_idx  = np.argsort(scores)[::-1]
        min_sep     = math.pi / (2.0 * max(n_measurements, 1))

        selected_indices: list[int] = []
        selected_points:  list[ObservationPoint] = []

        def _can_add(candidate: ObservationPoint, sep: float) -> bool:
            return all(
                angular_distance_points(candidate, sp) >= sep
                for sp in selected_points
            )

        for idx in sorted_idx:
            if len(selected_points) >= n_measurements:
                break
            if _can_add(candidate_points[idx], min_sep):
                selected_indices.append(int(idx))
                selected_points.append(candidate_points[idx])

        # Relax separation once if short
        if len(selected_points) < n_measurements:
            min_sep /= 2.0
            for idx in sorted_idx:
                if len(selected_points) >= n_measurements:
                    break
                if idx in selected_indices:
                    continue
                if _can_add(candidate_points[idx], min_sep):
                    selected_indices.append(int(idx))
                    selected_points.append(candidate_points[idx])

        # Final fallback
        if len(selected_points) < n_measurements:
            for idx in sorted_idx:
                if len(selected_points) >= n_measurements:
                    break
                if idx not in selected_indices:
                    selected_indices.append(int(idx))
                    selected_points.append(candidate_points[idx])

        plan_scores = [float(scores[i]) for i in selected_indices[:n_measurements]]
        plan_points = selected_points[:n_measurements]
        rationale = [
            f"TensorSensitivity={plan_scores[k]:.3f} "
            f"at theta={plan_points[k].theta:.2f} phi={plan_points[k].phi:.2f} "
            f"freq={plan_points[k].freq_hz/1e9:.1f}GHz"
            for k in range(len(plan_points))
        ]
        return MeasurementPlan(
            points=plan_points, scores=plan_scores, rationale=rationale
        )

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def get_per_method_scores(self) -> dict[str, np.ndarray]:
        """Return per-method normalised score grids (N_az, N_el, N_freq)."""
        return dict(self._per_method)

    def get_combined_score_grid(self) -> np.ndarray:
        """Return the combined score grid (N_az, N_el, N_freq)."""
        return self._score_grid.copy()

    def get_isar_image(self, el_idx: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (isar_power, az_axis_m, range_axis_m) for elevation slice el_idx.

        The cross-range and range axes are in metres assuming a circular aperture.
        """
        import warnings
        analyzer = ISARAnalyzer()
        _, isar = analyzer.compute_slice(self._tensor[:, el_idx, :])

        c = 3e8
        N_az, N_freq = self._tensor.shape[0], self._tensor.shape[2]
        df       = (self._freq[-1] - self._freq[0]) / max(N_freq - 1, 1)
        daz      = (self._az[-1]  - self._az[0])   / max(N_az  - 1, 1)
        f_center = float(np.mean(self._freq))

        range_res = c / (2.0 * df * N_freq) if df > 0 else 1.0
        cr_res    = c / (2.0 * f_center * daz * N_az) if (f_center * daz) > 0 else 1.0

        rows, cols = isar.shape
        range_axis = (np.arange(rows) - rows // 2) * range_res
        cr_axis    = (np.arange(cols) - cols // 2) * cr_res
        return isar, cr_axis, range_axis

    def get_top_points(
        self,
        n: int = 10,
    ) -> list[tuple[float, float, float, float]]:
        """Return list of (az_rad, el_rad, freq_hz, score) for top-n sensitive points."""
        flat = self._score_grid.ravel()
        top_idx = np.argsort(flat)[::-1][:n]
        results = []
        N_az, N_el, N_freq = self._score_grid.shape
        for fi in top_idx:
            i  = fi // (N_el * N_freq)
            j  = (fi // N_freq) % N_el
            k  = fi % N_freq
            results.append((
                float(self._az[i]),
                float(self._el[j]),
                float(self._freq[k]),
                float(self._score_grid[i, j, k]),
            ))
        return results
