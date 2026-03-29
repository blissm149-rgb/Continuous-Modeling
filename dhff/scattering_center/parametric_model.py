"""Module 4: Parametric scattering center discrepancy model."""
from __future__ import annotations

import warnings

import numpy as np
import scipy.optimize

from dhff.core.types import DiscrepancySample, ObservationPoint, ScatteringCenter
from .extractor import MatrixPencilExtractor
from .extractor_config import SCExtractorConfig

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
        config: SCExtractorConfig | None = None,
    ):
        self.max_centers = max_centers
        self.amplitude_threshold_db = amplitude_threshold_db
        # Use config if provided; otherwise build one from the individual params
        # so existing callers that pass max_centers/amplitude_threshold_db directly
        # keep working unchanged.
        if config is not None:
            self._cfg = config.effective()
        else:
            self._cfg = SCExtractorConfig(
                amplitude_threshold_db=amplitude_threshold_db,
            ).effective()
        self.centers: list[ScatteringCenter] = []
        self._is_fitted = False

    def fit(
        self,
        samples: list[DiscrepancySample],
        freq_range_hz: tuple[float, float],
    ) -> None:
        """Fit the parametric model to observed discrepancy data."""
        if len(samples) < self._cfg.min_samples_to_fit:
            self.centers = []
            self._is_fitted = True
            return

        extractor = MatrixPencilExtractor(
            n_centers_max=self.max_centers,
            amplitude_threshold_db=self.amplitude_threshold_db,
        )

        # Group samples by angle: auto-scale cluster radius from grid spacing.
        # With a dense 50-angle grid spacing ~0.06 rad we use 0.02 rad;
        # with a sparser grid (e.g. 20 angles, ~0.15 rad spacing) we need
        # a larger radius so nearby angles fall in the same cluster.
        thetas = sorted({s.obs.theta for s in samples})
        if len(thetas) >= 2:
            median_spacing = float(np.median(np.diff(thetas)))
        else:
            median_spacing = 0.1
        angle_tol = max(0.02, median_spacing * 0.6)

        angle_clusters: dict[int, list[DiscrepancySample]] = {}
        angles_list = []  # (theta, phi) for each cluster

        for s in samples:
            theta, phi = s.obs.theta, s.obs.phi
            found_cluster = -1
            for ci, (ct, cp) in enumerate(angles_list):
                if abs(theta - ct) < angle_tol and abs(phi - cp) < angle_tol:
                    found_cluster = ci
                    break
            if found_cluster == -1:
                found_cluster = len(angles_list)
                angles_list.append((theta, phi))
                angle_clusters[found_cluster] = []
            angle_clusters[found_cluster].append(s)

        # Minimum samples per cluster for Matrix Pencil: at least 4, ideally 10.
        min_cluster = max(4, min(10, len(samples) // max(len(angle_clusters), 1)))

        # Extract from frequency sweeps; each angle cluster gives range projections
        # r = x·cos(θ) + y·sin(θ) stored as (r, 0).
        # We keep track of (θ, extracted_centers) so we can triangulate later.
        per_angle_extractions: list[tuple[float, list[ScatteringCenter]]] = []
        for ci, cluster_samples in angle_clusters.items():
            if len(cluster_samples) < min_cluster:
                continue
            theta_k, _phi_k = angles_list[ci]
            cluster_sorted = sorted(cluster_samples, key=lambda s: s.obs.freq_hz)
            freq_arr = np.array([s.obs.freq_hz for s in cluster_sorted])
            res_arr = np.array([s.residual for s in cluster_sorted])
            extracted = extractor.extract_1d(freq_arr, res_arr)
            if extracted:
                per_angle_extractions.append((theta_k, extracted))

        # ------------------------------------------------------------------
        # First try the spectral/phase approach (more robust for cavities):
        # 1. Find f0 from the peak of mean |discrepancy| vs frequency.
        # 2. At f ≈ f0, fit (x, y) using the measured phase vs angle.
        # ------------------------------------------------------------------
        spectral_centers = _extract_by_spectral_peak(
            samples, max_centers=self.max_centers,
            min_peak_ratio=self._cfg.min_peak_ratio,
            grid_step=self._cfg.grid_step_m,
            grid_half_extent=self._cfg.grid_half_extent_m,
            bandwidth_hz=self._cfg.spectral_bandwidth_hz,
            max_range=self._cfg.max_range_m,
        )

        if not per_angle_extractions and not spectral_centers:
            self.centers = []
            self._is_fitted = True
            return

        # Build the list of all extracted centers (with y=0) for the initial
        # per-angle extraction pass and subsequent LM refinement.
        all_extracted = []
        for _theta_k, centers in per_angle_extractions:
            all_extracted.extend(centers)

        # ------------------------------------------------------------------
        # Triangulate 2D positions from per-angle range projections.
        # ------------------------------------------------------------------
        triangulated: list[ScatteringCenter] = []
        if len(per_angle_extractions) >= 3:
            triangulated = _triangulate_centers(per_angle_extractions)

        # Combine: prefer spectral centers (more accurate for cavities) and
        # triangulated centers over raw y=0 per-angle results.
        combined_raw = spectral_centers + triangulated + all_extracted
        if not combined_raw:
            self.centers = []
            self._is_fitted = True
            return

        merged_centers = _merge_centers(combined_raw, distance_threshold=self._cfg.merge_distance_m)

        # Keep only up to max_centers
        merged_centers = merged_centers[:self.max_centers]

        # Nonlinear refinement with LM
        if len(merged_centers) > 0 and len(samples) >= self._cfg.min_samples_to_fit:
            merged_centers = self._refine_with_lm(merged_centers, samples)

        # Fit frequency dependence model for each center
        if len(merged_centers) > 0 and len(samples) >= self._cfg.min_samples_to_fit:
            merged_centers = [self._fit_freq_dependence(sc, samples) for sc in merged_centers]

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
                max_nfev=self._cfg.lm_max_nfev,
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

    def _fit_freq_dependence(
        self, center: ScatteringCenter, samples: list[DiscrepancySample]
    ) -> ScatteringCenter:
        """Fit the best frequency dependence model to a scattering center.

        Demodulates samples by the center's position phase, then fits amplitude
        vs frequency to specular, edge, cavity_resonant, and creeping models.
        """
        import scipy.optimize as sopt

        # Demodulate each sample by the center's position phase to isolate freq signature
        demod_by_freq: dict[float, list[complex]] = {}
        for s in samples:
            p = s.obs
            k0 = 2.0 * np.pi * p.freq_hz / _C
            phase = np.exp(1j * 2.0 * k0 * (center.x * np.cos(p.theta) + center.y * np.sin(p.theta)))
            # Demodulate: remove position phase
            demod = s.residual * np.conj(phase)
            f = round(p.freq_hz, -6)  # group by MHz
            if f not in demod_by_freq:
                demod_by_freq[f] = []
            demod_by_freq[f].append(demod)

        if len(demod_by_freq) < 5:
            return center

        freqs = np.array(sorted(demod_by_freq.keys()), dtype=np.float64)
        amps = np.array([np.mean(np.abs(demod_by_freq[f])) for f in freqs], dtype=np.float64)

        if np.max(amps) < 1e-12:
            return center

        # Normalise amplitudes for fitting
        amps_norm = amps / (np.max(amps) + 1e-30)
        f_ref = freqs[len(freqs) // 2]

        best_model = "specular"
        best_residual = np.inf
        best_params: dict = {}

        # --- Specular: flat ---
        resid_spec = np.mean((amps_norm - 1.0) ** 2)
        if resid_spec < best_residual:
            best_residual = resid_spec
            best_model = "specular"
            best_params = {}

        # --- Edge: f^-0.5 ---
        edge_pred = (freqs / f_ref) ** (-0.5)
        edge_pred /= (np.max(edge_pred) + 1e-30)
        resid_edge = np.mean((amps_norm - edge_pred) ** 2)
        if resid_edge < best_residual:
            best_residual = resid_edge
            best_model = "edge"
            best_params = {}

        # --- Cavity resonant: Lorentzian ---
        def lorentzian_amp(f, f0, Q):
            denom = np.sqrt(1.0 + Q ** 2 * (f / f0 - f0 / f) ** 2)
            return 1.0 / denom

        # Initial guess: f0 = freq of peak amplitude, Q = 20
        peak_idx = np.argmax(amps)
        f0_init = freqs[peak_idx]
        try:
            popt, _ = sopt.curve_fit(
                lorentzian_amp, freqs, amps_norm,
                p0=[f0_init, 20.0],
                bounds=([freqs[0] * 0.5, 1.0], [freqs[-1] * 2.0, 200.0]),
                maxfev=500,
            )
            cav_pred = lorentzian_amp(freqs, *popt)
            resid_cav = np.mean((amps_norm - cav_pred) ** 2)
            if resid_cav < best_residual:
                best_residual = resid_cav
                best_model = "cavity_resonant"
                best_params = {"cavity_freq_hz": float(popt[0]), "cavity_q": float(popt[1])}
        except Exception:
            pass

        # --- Creeping: exp decay ---
        def creeping_amp(f, alpha_L):
            return np.exp(-alpha_L * f / _C)

        try:
            popt_cr, _ = sopt.curve_fit(
                creeping_amp, freqs, amps_norm,
                p0=[0.1], bounds=([0.0], [10.0]),
                maxfev=200,
            )
            cr_pred = creeping_amp(freqs, *popt_cr)
            cr_pred /= (np.max(cr_pred) + 1e-30)
            resid_cr = np.mean((amps_norm - cr_pred) ** 2)
            if resid_cr < best_residual:
                best_residual = resid_cr
                best_model = "creeping"
                best_params = {}
        except Exception:
            pass

        # Refit complex amplitude with selected freq_dep model using least-squares
        # Build Vandermonde-like column: phase_term * F(freq) for each sample
        new_cavity_freq_hz = best_params.get("cavity_freq_hz", center.cavity_freq_hz)
        new_cavity_q = best_params.get("cavity_q", center.cavity_q)

        try:
            A_col = np.zeros(len(samples), dtype=np.complex128)
            for i, s in enumerate(samples):
                p = s.obs
                k0 = 2.0 * np.pi * p.freq_hz / _C
                phase = np.exp(1j * 2.0 * k0 * (center.x * np.cos(p.theta) + center.y * np.sin(p.theta)))
                if best_model == "specular":
                    F = 1.0 + 0j
                elif best_model == "edge":
                    F = complex((p.freq_hz / _F_REF) ** (-0.5))
                elif best_model == "cavity_resonant":
                    F = 1.0 / (1.0 + 1j * new_cavity_q * (p.freq_hz / new_cavity_freq_hz - new_cavity_freq_hz / p.freq_hz))
                else:  # creeping
                    F = complex(np.exp(-0.5 * p.freq_hz * center.lobe_width_rad / _C))
                A_col[i] = phase * F

            # Least-squares: solve A_col * amp = residuals (complex)
            A_mat = np.column_stack([A_col.real, -A_col.imag, A_col.imag, A_col.real])
            b_vec = np.concatenate([
                np.array([s.residual for s in samples]).real,
                np.array([s.residual for s in samples]).imag,
            ])
            # Simple approach: lstsq on [A.real, -A.imag; A.imag, A.real] * [amp_r, amp_i]
            A2 = np.column_stack([A_col.real, -A_col.imag])
            A2 = np.vstack([
                np.column_stack([A_col.real, -A_col.imag]),
                np.column_stack([A_col.imag, A_col.real]),
            ])
            res_vals = np.array([s.residual for s in samples])
            b2 = np.concatenate([res_vals.real, res_vals.imag])
            result_amp, _, _, _ = np.linalg.lstsq(A2, b2, rcond=None)
            new_amplitude = complex(result_amp[0] + 1j * result_amp[1])
        except Exception:
            new_amplitude = center.amplitude

        return ScatteringCenter(
            x=center.x,
            y=center.y,
            amplitude=new_amplitude,
            freq_dependence=best_model,
            angular_pattern=center.angular_pattern,
            lobe_center_theta=center.lobe_center_theta,
            lobe_center_phi=center.lobe_center_phi,
            lobe_width_rad=center.lobe_width_rad,
            cavity_freq_hz=new_cavity_freq_hz,
            cavity_q=new_cavity_q,
            label=center.label,
        )

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


def _extract_by_spectral_peak(
    samples: list[DiscrepancySample],
    max_centers: int = 5,
    min_peak_ratio: float = 1.5,
    grid_step: float = 0.05,
    grid_half_extent: float = 0.6,
    bandwidth_hz: float = 1.5e9,
    max_range: float = 2.0,
) -> list[ScatteringCenter]:
    """Extract scattering centers using spectral peak detection + position grid search.

    Algorithm:
    1. Bin by frequency; find peak frequency (cavity resonance indicator).
    2. Collect discrepancy samples near the peak frequency.
    3. Grid-search over (x, y) positions to find the best-fitting location.
       For each (x_g, y_g): compute optimal amplitude A via linear least-squares,
       then evaluate residual E(x_g, y_g).
    4. Refine the best grid point using scipy LM.
    """
    import scipy.optimize as sopt

    if not samples:
        return []

    # Bin by frequency
    freq_bin: dict[float, list[complex]] = {}
    for s in samples:
        f_key = round(s.obs.freq_hz, -7)  # 10 MHz bins
        freq_bin.setdefault(f_key, []).append(s.residual)

    if len(freq_bin) < 4:
        return []

    sorted_freqs = sorted(freq_bin.keys())
    freq_arr = np.array(sorted_freqs, dtype=np.float64)
    mag_arr = np.array([np.mean(np.abs(freq_bin[f])) for f in sorted_freqs])

    if np.max(mag_arr) < 1e-12:
        return []

    # Find the dominant peak frequency
    from scipy.signal import find_peaks
    noise_floor = np.median(mag_arr)
    min_height = noise_floor * min_peak_ratio
    peaks_idx, _ = find_peaks(mag_arr, height=min_height, prominence=noise_floor * 0.5)
    if len(peaks_idx) == 0:
        peaks_idx = np.array([int(np.argmax(mag_arr))])

    centers = []
    for pk_idx in peaks_idx[:max_centers]:
        f0_approx = float(freq_arr[pk_idx])

        # Use all samples within ±bandwidth_hz of the peak as the "characterisation" set
        peak_samples = [s for s in samples
                        if abs(s.obs.freq_hz - f0_approx) <= bandwidth_hz]
        if len(peak_samples) < 3:
            peak_samples = samples  # fallback: use all

        thetas_ps = np.array([s.obs.theta for s in peak_samples])
        freqs_ps = np.array([s.obs.freq_hz for s in peak_samples])
        vals_ps = np.array([s.residual for s in peak_samples])
        k0_ps = 2.0 * np.pi * freqs_ps / _C

        def _cost_at_xy(x_g, y_g):
            """Linear optimal amplitude + residual for a candidate position."""
            phase = np.exp(1j * 2.0 * k0_ps
                           * (x_g * np.cos(thetas_ps) + y_g * np.sin(thetas_ps)))
            # Optimal amplitude: A = conj(phase) @ vals / (phase @ conj(phase))
            denom = float(np.vdot(phase, phase).real) + 1e-30
            A_opt = np.vdot(phase, vals_ps) / denom
            pred = A_opt * phase
            return float(np.mean(np.abs(vals_ps - pred) ** 2)), A_opt

        # Grid search over (x, y)
        xs = np.arange(-grid_half_extent, grid_half_extent + 1e-9, grid_step)
        ys = np.arange(-grid_half_extent, grid_half_extent + 1e-9, grid_step)
        best_cost = np.inf
        best_xy = (0.0, 0.0)
        best_A = 0j

        for x_g in xs:
            for y_g in ys:
                c, A_opt = _cost_at_xy(x_g, y_g)
                if c < best_cost:
                    best_cost = c
                    best_xy = (x_g, y_g)
                    best_A = A_opt

        x_init, y_init = best_xy

        # Refine with LM
        def residual_fn(params):
            xk, yk, ar, ai = params
            amp = ar + 1j * ai
            phase = np.exp(1j * 2.0 * k0_ps * (xk * np.cos(thetas_ps) + yk * np.sin(thetas_ps)))
            diff = amp * phase - vals_ps
            return np.concatenate([diff.real, diff.imag])

        try:
            res = sopt.least_squares(
                residual_fn,
                [x_init, y_init, best_A.real, best_A.imag],
                method='lm', max_nfev=300,
            )
            x_c, y_c = float(res.x[0]), float(res.x[1])
            amp_c = complex(res.x[2] + 1j * res.x[3])
        except Exception:
            x_c, y_c = x_init, y_init
            amp_c = best_A

        if np.sqrt(x_c ** 2 + y_c ** 2) > max_range:
            continue

        centers.append(ScatteringCenter(
            x=x_c, y=y_c, amplitude=amp_c,
            freq_dependence="specular", angular_pattern="isotropic",
        ))

    return centers


def _triangulate_centers(
    per_angle_extractions: list[tuple[float, list[ScatteringCenter]]],
    max_range: float = 1.5,
) -> list[ScatteringCenter]:
    """Triangulate 2D (x, y) scattering center positions from per-angle range projections.

    Each 1D extraction at angle θ gives r = x·cos(θ) + y·sin(θ).  With
    observations from ≥3 well-separated angles we solve the overdetermined
    linear system [cos(θ_i), sin(θ_i)] @ [x, y]^T = r_i via least-squares.

    For multi-center scenarios we group the top-N range projections per angle by
    amplitude and try all plausible combinations.  For each combination we check
    consistency (small residual across all angles) and only keep consistent ones.
    """
    # Collect the dominant (highest amplitude) center from each angle.
    angle_dominant: list[tuple[float, float, complex]] = []  # (θ, r, amp)
    for theta_k, centers in per_angle_extractions:
        if not centers:
            continue
        best = max(centers, key=lambda c: abs(c.amplitude))
        angle_dominant.append((theta_k, float(best.x), best.amplitude))

    if len(angle_dominant) < 3:
        return []

    # Overdetermined least-squares: minimise sum of (cos(θ)*x + sin(θ)*y - r)^2
    A = np.array([[np.cos(t), np.sin(t)] for t, _r, _a in angle_dominant])
    b = np.array([r for _t, r, _a in angle_dominant])
    xy, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    x_c, y_c = float(xy[0]), float(xy[1])

    # Reject if unreasonably far from origin
    if np.sqrt(x_c ** 2 + y_c ** 2) > max_range:
        return []

    # Average amplitude
    amp_avg = complex(np.mean([a for _t, _r, a in angle_dominant]))

    return [ScatteringCenter(
        x=x_c, y=y_c, amplitude=amp_avg,
        freq_dependence="specular", angular_pattern="isotropic",
    )]
