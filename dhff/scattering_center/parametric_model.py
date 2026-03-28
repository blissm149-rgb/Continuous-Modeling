"""Module 4: Parametric scattering center discrepancy model."""
from __future__ import annotations

import warnings

import numpy as np
import scipy.optimize

from dhff.core.types import DiscrepancySample, ObservationPoint, ScatteringCenter
from .extractor import MatrixPencilExtractor

_C = 299792458.0
_F_REF = 10e9


def _eval_sc_model(centers: list[ScatteringCenter], points: list[ObservationPoint]) -> np.ndarray:
    """Evaluate a list of ScatteringCenter at given observation points."""
    from dhff.synthetic.scatterer import (
        _ANG_PATTERN_FUNCS, _FREQ_DEP_FUNCS, angular_gain_isotropic,
        ScatteringFeature,
    )
    N = len(points)
    values = np.zeros(N, dtype=np.complex128)
    theta_arr = np.array([p.theta for p in points])
    phi_arr = np.array([p.phi for p in points])
    freq_arr = np.array([p.freq_hz for p in points])
    k0 = 2.0 * np.pi * freq_arr / _C

    for sc in centers:
        ang_func = _ANG_PATTERN_FUNCS.get(sc.angular_pattern, angular_gain_isotropic)
        G = ang_func(theta_arr, phi_arr, sc.lobe_center_theta, sc.lobe_center_phi, sc.lobe_width_rad)

        # Fake ScatteringFeature for freq dep funcs
        feat = ScatteringFeature(
            x=sc.x, y=sc.y, base_amplitude=sc.amplitude,
            freq_dependence=sc.freq_dependence, angular_pattern=sc.angular_pattern,
            cavity_freq_hz=sc.cavity_freq_hz, cavity_q=sc.cavity_q,
        )
        freq_func = _FREQ_DEP_FUNCS.get(sc.freq_dependence, lambda f, ft: np.ones_like(f, dtype=np.complex128))
        F = freq_func(freq_arr, feat)

        phase_term = np.exp(1j * 2.0 * k0 * (sc.x * np.cos(theta_arr) + sc.y * np.sin(theta_arr)))
        values += sc.amplitude * G * F * phase_term

    return values


class ParametricSCModel:
    """Represent the discrepancy field as a sum of scattering centers."""

    def __init__(
        self,
        max_centers: int = 15,
        amplitude_threshold_db: float = -25.0,
    ):
        self.max_centers = max_centers
        self.amplitude_threshold_db = amplitude_threshold_db
        self.centers: list[ScatteringCenter] = []
        self._is_fitted = False

    def fit(
        self,
        samples: list[DiscrepancySample],
        freq_range_hz: tuple[float, float],
    ) -> None:
        """Fit the parametric model to observed discrepancy data."""
        if len(samples) < 15:
            self.centers = []
            self._is_fitted = True
            return

        extractor = MatrixPencilExtractor(
            n_centers_max=self.max_centers,
            amplitude_threshold_db=self.amplitude_threshold_db,
        )

        # Group samples by angle (cluster nearby angles within 0.02 rad)
        angle_clusters: dict[int, list[DiscrepancySample]] = {}
        angles_list = []  # (theta, phi) for each cluster

        for s in samples:
            theta, phi = s.obs.theta, s.obs.phi
            found_cluster = -1
            for ci, (ct, cp) in enumerate(angles_list):
                if abs(theta - ct) < 0.02 and abs(phi - cp) < 0.02:
                    found_cluster = ci
                    break
            if found_cluster == -1:
                found_cluster = len(angles_list)
                angles_list.append((theta, phi))
                angle_clusters[found_cluster] = []
            angle_clusters[found_cluster].append(s)

        # Extract from frequency sweeps
        all_extracted = []
        for ci, cluster_samples in angle_clusters.items():
            if len(cluster_samples) < 10:
                continue
            cluster_sorted = sorted(cluster_samples, key=lambda s: s.obs.freq_hz)
            freq_arr = np.array([s.obs.freq_hz for s in cluster_sorted])
            res_arr = np.array([s.residual for s in cluster_sorted])
            extracted = extractor.extract_1d(freq_arr, res_arr)
            all_extracted.extend(extracted)

        if not all_extracted:
            self.centers = []
            self._is_fitted = True
            return

        # Merge centers at similar positions
        merged_centers = _merge_centers(all_extracted, distance_threshold=0.05)

        # Keep only up to max_centers
        merged_centers = merged_centers[:self.max_centers]

        # Nonlinear refinement with LM
        if len(merged_centers) > 0 and len(samples) >= 15:
            merged_centers = self._refine_with_lm(merged_centers, samples)

        self.centers = merged_centers
        self._is_fitted = True

    def _refine_with_lm(
        self, initial_centers: list[ScatteringCenter], samples: list[DiscrepancySample]
    ) -> list[ScatteringCenter]:
        """Nonlinear LM refinement of center positions and amplitudes."""
        points = [s.obs for s in samples]
        target = np.array([s.residual for s in samples])

        K = len(initial_centers)
        if K == 0:
            return initial_centers

        # Parameter vector: [x_k, y_k, amp_real_k, amp_imag_k] for each center
        x0 = []
        for sc in initial_centers:
            x0.extend([sc.x, sc.y, sc.amplitude.real, sc.amplitude.imag])
        x0 = np.array(x0)

        theta_arr = np.array([p.theta for p in points])
        phi_arr = np.array([p.phi for p in points])
        freq_arr = np.array([p.freq_hz for p in points])
        k0 = 2.0 * np.pi * freq_arr / _C

        def residual_fn(params):
            pred = np.zeros(len(points), dtype=np.complex128)
            for k in range(K):
                xk = params[4*k]
                yk = params[4*k + 1]
                ar = params[4*k + 2]
                ai = params[4*k + 3]
                amp = ar + 1j * ai
                phase_term = np.exp(1j * 2.0 * k0 * (xk * np.cos(theta_arr) + yk * np.sin(theta_arr)))
                pred += amp * phase_term
            diff = pred - target
            return np.concatenate([diff.real, diff.imag])

        try:
            result = scipy.optimize.least_squares(
                residual_fn, x0, method='lm',
                max_nfev=200,
            )
            params = result.x
            refined = []
            for k in range(K):
                sc = initial_centers[k]
                refined.append(ScatteringCenter(
                    x=float(params[4*k]),
                    y=float(params[4*k + 1]),
                    amplitude=complex(params[4*k + 2] + 1j * params[4*k + 3]),
                    freq_dependence=sc.freq_dependence,
                    angular_pattern=sc.angular_pattern,
                    lobe_center_theta=sc.lobe_center_theta,
                    lobe_center_phi=sc.lobe_center_phi,
                    lobe_width_rad=sc.lobe_width_rad,
                    cavity_freq_hz=sc.cavity_freq_hz,
                    cavity_q=sc.cavity_q,
                    label=sc.label,
                ))
            return refined
        except Exception as e:
            warnings.warn(f"LM refinement failed: {e}")
            return initial_centers

    def predict(self, points: list[ObservationPoint]) -> np.ndarray:
        """Evaluate the parametric model at given observation points."""
        if not self._is_fitted or not self.centers:
            return np.zeros(len(points), dtype=np.complex128)
        return _eval_sc_model(self.centers, points)

    def residuals(self, samples: list[DiscrepancySample]) -> list[DiscrepancySample]:
        """Compute residuals: measured discrepancy minus parametric model prediction."""
        points = [s.obs for s in samples]
        pred = self.predict(points)
        return [
            DiscrepancySample(obs=s.obs, residual=s.residual - pred[i])
            for i, s in enumerate(samples)
        ]

    def get_center_count(self) -> int:
        return len(self.centers)

    def to_scattering_centers(self) -> list[ScatteringCenter]:
        return list(self.centers)


def _merge_centers(
    centers: list[ScatteringCenter], distance_threshold: float = 0.05
) -> list[ScatteringCenter]:
    """Simple agglomerative merging of centers close in (x, y)."""
    if not centers:
        return []

    merged = []
    used = [False] * len(centers)

    for i, ci in enumerate(centers):
        if used[i]:
            continue
        group = [ci]
        used[i] = True
        for j, cj in enumerate(centers):
            if used[j] or j == i:
                continue
            dist = np.sqrt((ci.x - cj.x)**2 + (ci.y - cj.y)**2)
            if dist < distance_threshold:
                group.append(cj)
                used[j] = True

        # Average group
        x_avg = np.mean([c.x for c in group])
        y_avg = np.mean([c.y for c in group])
        amp_avg = np.mean([c.amplitude for c in group])
        merged.append(ScatteringCenter(
            x=float(x_avg), y=float(y_avg), amplitude=complex(amp_avg),
            freq_dependence=group[0].freq_dependence,
            angular_pattern=group[0].angular_pattern,
        ))

    return merged
