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

Sensitivity signals
-------------------
gradient     (weight 0.30): Gaussian-smoothed amplitude gradient + full 3D
                            phase curvature with geodesic elevation correction.
isar         (weight 0.18): ISAR complexity — sidelobe ratio + spatial entropy
                            + centroid spread; structured 2D broadcast.
spectral     (weight 0.22): Q-weighted resonance count + spectral variance +
                            anti-resonance (notch) detection.
cancellation (weight 0.17): Adaptive-window null depth + bandwidth sharpness.
physical     (weight 0.13): Group-delay anomaly + angular coherence drop.
cross_freq   (weight 0.13): Feature drift + inter-frequency angular decoherence
                            (enabled when use_cross_freq=True).

Score fusion
------------
1. Robust scaling of each method score to its 98th percentile.
2. Agreement-amplifying blend: (1-λ)·linear + λ·geometric_mean.
3. Optional disagreement bonus: CV × mean × β rewards regime-transition zones
   where analyzers disagree (single-outlier suppressed via MAD z-score gate).
4. Optional regime-adaptive weights: per-frequency weight vector from
   RegimeClassifier replaces global weights when use_regime_weights=True.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from dhff.core.types import MeasurementPlan, ObservationPoint
from dhff.core.coordinate_system import angular_distance_points

from .gradient_analyzer              import GradientAnalyzer
from .isar_analyzer                  import ISARAnalyzer
from .spectral_analyzer              import SpectralAnalyzer
from .cancellation_detector          import CancellationDetector
from .physical_consistency_analyzer  import PhysicalConsistencyAnalyzer
from ._utils                         import robust_scale

# Module-level score cache (keyed on tensor bytes); holds at most 4 entries.
# Each entry stores (score_grid, per_method_dict) so the per-method scores are
# available even when _compute_score_grid is skipped on a cache hit.
_SCORE_CACHE: OrderedDict[bytes, tuple[np.ndarray, dict]] = OrderedDict()
_CACHE_MAX = 4


def _cache_key(
    tensor: np.ndarray,
    az: np.ndarray,
    el: np.ndarray,
    freq: np.ndarray,
    flags: bytes = b"",
) -> bytes:
    return tensor.tobytes() + az.tobytes() + el.tobytes() + freq.tobytes() + flags


class TensorSensitivityMap:
    """
    Sensitivity map derived purely from an (az × el × freq) complex RCS tensor.

    Parameters
    ----------
    rcs_tensor : np.ndarray, shape (N_az, N_el, N_freq), complex128
    az_rad     : np.ndarray, shape (N_az,), monotonically increasing
    el_rad     : np.ndarray, shape (N_el,), monotonically increasing
    freq_hz    : np.ndarray, shape (N_freq,), monotonically increasing
    weights    : dict[str, float] | None — override any default method weight
    fusion_lambda             : Blend between linear (0) and geometric mean (1).
    robust_scale_percentile   : Winsorising percentile for per-method scaling.
    sharpen_temperature       : Score sharpening temperature (< 1 = sharper top).
    disagreement_beta         : Weight of the disagreement bonus term (0 = off).
    use_cross_freq            : Enable the CrossFreqCoherenceAnalyzer (6th signal).
    use_regime_weights        : Enable per-frequency regime-adaptive weights.
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "gradient":     0.30,
        "isar":         0.18,
        "spectral":     0.22,
        "cancellation": 0.17,
        "physical":     0.13,
    }

    def __init__(
        self,
        rcs_tensor: np.ndarray,
        az_rad:     np.ndarray,
        el_rad:     np.ndarray,
        freq_hz:    np.ndarray,
        weights:    dict[str, float] | None = None,
        fusion_lambda: float = 0.4,
        robust_scale_percentile: float = 98.0,
        sharpen_temperature: float = 1.0,
        disagreement_beta: float = 0.0,
        use_cross_freq: bool = False,
        use_regime_weights: bool = False,
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
        self._lambda = fusion_lambda
        self._rs_pct = robust_scale_percentile
        self._temp   = sharpen_temperature
        self._dis_beta        = float(disagreement_beta)
        self._use_cross_freq  = bool(use_cross_freq)
        self._use_regime_wts  = bool(use_regime_weights)

        # Compute dense score grid (cached)
        self._per_method: dict[str, np.ndarray] = {}
        self._score_grid = self._build_score_grid(rcs_tensor)

        # RegularGridInterpolator: axes (el, az, freq) to match ObservationPoint
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
        flags = (
            f"xf={self._use_cross_freq}"
            f":rw={self._use_regime_wts}"
            f":db={self._dis_beta:.4f}"
        ).encode()
        key = _cache_key(tensor, self._az, self._el, self._freq, flags)
        if key in _SCORE_CACHE:
            cached_grid, cached_per_method = _SCORE_CACHE[key]
            self._per_method = {k: v.copy() for k, v in cached_per_method.items()}
            return cached_grid.copy()
        result = self._compute_score_grid(tensor)
        _SCORE_CACHE[key] = (result.copy(), {k: v.copy() for k, v in self._per_method.items()})
        if len(_SCORE_CACHE) > _CACHE_MAX:
            _SCORE_CACHE.popitem(last=False)
        return result

    def _compute_score_grid(self, tensor: np.ndarray) -> np.ndarray:
        """Run analyzers and return weighted, fused score grid in [0,1]."""
        w = self._w

        # 1. Gradient
        grad_out = GradientAnalyzer().compute(tensor, self._az, self._el, self._freq)
        self._per_method["gradient"] = robust_scale(grad_out["combined"], self._rs_pct)

        # 2. ISAR
        isar_raw = ISARAnalyzer().compute(tensor, self._az, self._el, self._freq)
        self._per_method["isar"] = robust_scale(isar_raw, self._rs_pct)

        # 3. Spectral
        spec_out = SpectralAnalyzer().compute(tensor, self._freq)
        self._per_method["spectral"] = robust_scale(
            0.30 * robust_scale(spec_out["spectral_variance"], self._rs_pct)
            + 0.35 * robust_scale(spec_out["resonance_q"],     self._rs_pct)
            + 0.20 * robust_scale(spec_out["angular_peaks"],   self._rs_pct)
            + 0.15 * robust_scale(spec_out["notch_depth"],     self._rs_pct),
            self._rs_pct,
        )

        # 4. Cancellation
        canc_raw = CancellationDetector().compute(tensor)
        self._per_method["cancellation"] = robust_scale(canc_raw, self._rs_pct)

        # 5. Physical consistency
        phys_out = PhysicalConsistencyAnalyzer().compute(
            tensor, self._az, self._el, self._freq
        )
        self._per_method["physical"] = robust_scale(phys_out["combined"], self._rs_pct)

        # 6. Cross-frequency coherence (optional — Phase 3B feature flag)
        methods = ["gradient", "isar", "spectral", "cancellation", "physical"]
        if self._use_cross_freq:
            from .cross_freq_coherence import CrossFreqCoherenceAnalyzer
            xf_raw = CrossFreqCoherenceAnalyzer().compute(
                tensor, self._az, self._el, self._freq
            )
            self._per_method["cross_freq"] = robust_scale(xf_raw, self._rs_pct)
            methods.append("cross_freq")
            if "cross_freq" not in self._w:
                # Default: take 0.13 from gradient proportionally and renorm later
                self._w = {**self._w, "cross_freq": 0.13}

        # ── Build weight vector ───────────────────────────────────────────────
        scores_arr = np.stack(
            [self._per_method[m] for m in methods], axis=0
        )  # (N_methods, N_az, N_el, N_freq)

        if self._use_regime_wts:
            from .regime_classifier import RegimeClassifier
            # regime_weights: (N_methods, N_freq)
            regime_wts, _, _ = RegimeClassifier().classify(
                tensor, self._freq, n_methods=len(methods)
            )
            # Broadcast over (az, el): (N_methods, 1, 1, N_freq)
            wts_bcast = regime_wts[:, None, None, :]
            linear_part = (scores_arr * wts_bcast).sum(axis=0)
            log_sum     = (np.log(scores_arr + 1e-30) * wts_bcast).sum(axis=0)
        else:
            wts = np.array([self._w.get(m, 0.0) for m in methods], dtype=float)
            wts /= wts.sum()
            # Reshape for broadcasting: (N_methods, 1, 1, 1)
            wts_bcast   = wts[:, None, None, None]
            linear_part = (scores_arr * wts_bcast).sum(axis=0)
            log_sum     = (np.log(scores_arr + 1e-30) * wts_bcast).sum(axis=0)

        geom_part = np.exp(log_sum)
        fused = (1.0 - self._lambda) * linear_part + self._lambda * geom_part

        # ── Phase 2A: Disagreement bonus ─────────────────────────────────────
        if self._dis_beta > 0.0:
            mean_s   = scores_arr.mean(axis=0)
            std_s    = scores_arr.std(axis=0)
            median_s = np.median(scores_arr, axis=0)
            mad_s    = np.median(np.abs(scores_arr - median_s[None]), axis=0) + 1e-12
            z_scores = np.abs(scores_arr - median_s[None]) / mad_s[None]
            n_outliers = (z_scores > 3.0).sum(axis=0)

            cv = std_s / (mean_s + 1e-12)
            disagreement_bonus = np.where(
                n_outliers == 1,
                0.0,
                cv * mean_s * self._dis_beta,
            )
            fused = fused + disagreement_bonus

        # ── Optional sharpening ───────────────────────────────────────────────
        if 0.0 < self._temp < 1.0:
            fused = fused ** (1.0 / self._temp)

        return np.clip(robust_scale(fused, self._rs_pct), 0.0, 1.0)

    # ------------------------------------------------------------------
    # Public interface (DiscrepancySusceptibilityMap-compatible)
    # ------------------------------------------------------------------

    def compute(self, points: list[ObservationPoint]) -> np.ndarray:
        """Score each ObservationPoint.  Returns float array in [0, 1].

        Mapping: ObservationPoint.theta → elevation, .phi → azimuth.
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
        """Select top-n measurement points by sensitivity with angular diversity."""
        scores     = self.compute(candidate_points)
        sorted_idx = np.argsort(scores)[::-1]
        min_sep    = math.pi / (2.0 * max(n_measurements, 1))

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

        if len(selected_points) < n_measurements:
            for idx in sorted_idx:
                if len(selected_points) >= n_measurements:
                    break
                if idx not in selected_indices:
                    selected_indices.append(int(idx))
                    selected_points.append(candidate_points[idx])

        plan_scores = [float(scores[i]) for i in selected_indices[:n_measurements]]
        plan_points = selected_points[:n_measurements]
        rationale   = [
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
        """Return per-method robust-scaled score grids (N_az, N_el, N_freq)."""
        return dict(self._per_method)

    def get_combined_score_grid(self) -> np.ndarray:
        """Return the fused score grid (N_az, N_el, N_freq)."""
        return self._score_grid.copy()

    def get_fusion_diagnostics(self) -> dict:
        """Return per-method scores plus pointwise agreement metric.

        Useful for understanding which analyzer drove each top-scored point.
        """
        w = self._w
        methods = list(self._per_method.keys())
        wts = np.array([w.get(m, 0.0) for m in methods], dtype=float)
        wts /= wts.sum() + 1e-30

        log_sum = sum(
            wt * np.log(self._per_method[m] + 1e-30)
            for wt, m in zip(wts, methods)
        )
        agreement = np.exp(log_sum)   # geometric mean = agreement metric

        return {
            "per_method": dict(self._per_method),
            "agreement":  agreement,
        }

    def get_isar_image(self, el_idx: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (isar_power, az_axis_m, range_axis_m) for elevation slice el_idx."""
        analyzer = ISARAnalyzer()
        _, isar  = analyzer.compute_slice(self._tensor[:, el_idx, :])

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
        """Return list of (az_rad, el_rad, freq_hz, score) for top-n points."""
        flat    = self._score_grid.ravel()
        top_idx = np.argsort(flat)[::-1][:n]
        N_az, N_el, N_freq = self._score_grid.shape
        results = []
        for fi in top_idx:
            i = fi // (N_el * N_freq)
            j = (fi // N_freq) % N_el
            k = fi % N_freq
            results.append((
                float(self._az[i]),
                float(self._el[j]),
                float(self._freq[k]),
                float(self._score_grid[i, j, k]),
            ))
        return results
