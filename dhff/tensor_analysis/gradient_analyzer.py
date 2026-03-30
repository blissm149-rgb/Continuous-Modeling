"""Gradient-based amplitude and phase sensitivity analyzer.

Improvements over v1
--------------------
* Gaussian pre-smoothing of |S| before finite differences eliminates noise
  amplification on coarse grids (sigma ~ 4% of each axis).
* Geodesic-correct elevation gradient: multiplied by 1/cos(el) so that a
  feature at el=70° does not appear 2.9× more sensitive than the same
  feature at el=0° — a pure numerical artefact of the non-uniform
  solid-angle element.
* Full 3D phase curvature: adds d²∠S/daz² and d²∠S/del² alongside the
  existing d²∠S/df², catching creeping-wave angular dispersion and dihedral
  lobe oscillations.
* Data-driven amplitude/phase blend: weight of phase curvature scales with
  local |S| (via tanh), suppressing unreliable phase estimates near nulls.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage

from ._utils import robust_scale


class GradientAnalyzer:
    """Compute amplitude and phase-curvature sensitivity from a complex RCS tensor."""

    def __init__(
        self,
        smooth_fraction: float = 0.04,
        phase_weight_alpha: float = 0.4,
        geodesic_correction: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        smooth_fraction     : Gaussian sigma as a fraction of each axis length
                              (e.g. 0.04 → sigma = max(0.5, 0.04 * N_axis)).
        phase_weight_alpha  : Maximum weight given to the phase curvature
                              component (0 = amplitude only, 1 = phase only).
        geodesic_correction : If True, multiply the elevation gradient by
                              1/cos(el) to correct for solid-angle compression.
        """
        self._smooth_frac  = smooth_fraction
        self._alpha        = phase_weight_alpha
        self._geodesic     = geodesic_correction

    def compute(
        self,
        tensor: np.ndarray,   # (N_az, N_el, N_freq), complex128
        az_rad:  np.ndarray,  # (N_az,)
        el_rad:  np.ndarray,  # (N_el,)
        freq_hz: np.ndarray,  # (N_freq,)
    ) -> dict[str, np.ndarray]:
        """Return dict with 'amplitude_gradient' and 'phase_curvature',
        both shape (N_az, N_el, N_freq), dtype float64.
        """
        N_az, N_el, N_freq = tensor.shape

        # ── Pre-smooth amplitude to reduce finite-difference noise ────────────
        sa = max(0.5, N_az   * self._smooth_frac)
        se = max(0.5, N_el   * self._smooth_frac)
        sf = max(0.5, N_freq * self._smooth_frac)
        amp = scipy.ndimage.gaussian_filter(
            np.abs(tensor).astype(float), sigma=[sa, se, sf]
        )

        # ── Amplitude gradients (single call) ────────────────────────────────
        grads = np.gradient(amp, az_rad, el_rad, freq_hz)
        d_daz, d_del, d_df = grads[0], grads[1], grads[2]

        # Geodesic correction: 1/cos(el) converts per-radian-el to per-radian-arc
        if self._geodesic and el_rad.size > 1:
            cos_el = np.cos(el_rad)[None, :, None]  # (1, N_el, 1)
            d_del  = d_del / np.maximum(np.abs(cos_el), 0.05)

        grad_amp = np.sqrt(d_daz ** 2 + d_del ** 2 + d_df ** 2)

        # ── Full 3D phase curvature ───────────────────────────────────────────
        # Unwrap along frequency axis first (longest axis, most reliable),
        # then along azimuth (second most reliable for typical grids).
        phase = np.unwrap(np.angle(tensor.astype(complex)), axis=2)
        if N_az > 2:
            phase = np.unwrap(phase, axis=0)

        dphase_df  = np.gradient(phase, freq_hz, axis=2)
        d2phi_df2  = np.abs(np.gradient(dphase_df, freq_hz, axis=2))

        dphase_daz = np.gradient(phase, az_rad, axis=0)
        d2phi_daz2 = np.abs(np.gradient(dphase_daz, az_rad, axis=0))

        if N_el > 2:
            dphase_del = np.gradient(phase, el_rad, axis=1)
            d2phi_del2 = np.abs(np.gradient(dphase_del, el_rad, axis=1))
        else:
            d2phi_del2 = np.zeros_like(d2phi_df2)

        phase_curv = (
            robust_scale(d2phi_df2)
            + robust_scale(d2phi_daz2)
            + robust_scale(d2phi_del2)
        )

        # ── Data-driven amplitude/phase blend ────────────────────────────────
        # Phase is unreliable near nulls (|S| ≈ 0).  Weight phase curvature by
        # local |S| normalised to the median — tanh saturates at 1 above ~2×median.
        amp_orig = np.abs(tensor).astype(float)
        med_amp  = float(np.median(amp_orig)) + 1e-30
        phase_rel = np.tanh(amp_orig / med_amp)   # in (0, 1)
        w_phase = self._alpha * phase_rel
        w_amp   = 1.0 - w_phase

        combined = w_amp * robust_scale(grad_amp) + w_phase * robust_scale(phase_curv)

        return {
            "amplitude_gradient": grad_amp,
            "phase_curvature":    phase_curv,
            "combined":           combined,
        }
