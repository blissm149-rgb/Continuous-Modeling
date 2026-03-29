"""Gradient-based amplitude and phase sensitivity analyzer.

Computes finite-difference gradients of |S| and ∠S across all three
tensor dimensions (azimuth, elevation, frequency).

High amplitude gradient → lobe edges, resonance flanks — regions where
a small geometric error causes a large RCS change.

High phase curvature (d²∠S/df²) → dispersive / resonant features
with non-linear group delay — hard to model accurately.
"""
from __future__ import annotations

import numpy as np


class GradientAnalyzer:
    """Compute amplitude and phase-curvature sensitivity from a complex RCS tensor."""

    def compute(
        self,
        tensor: np.ndarray,   # (N_az, N_el, N_freq), complex128
        az_rad:  np.ndarray,  # (N_az,)
        el_rad:  np.ndarray,  # (N_el,)
        freq_hz: np.ndarray,  # (N_freq,)
    ) -> dict[str, np.ndarray]:
        """Return dict with 'amplitude_gradient' and 'phase_curvature' arrays,
        both shape (N_az, N_el, N_freq), dtype float64.
        """
        amp = np.abs(tensor)

        # ── Amplitude gradients along each axis ─────────────────────────────
        d_daz = np.gradient(amp, az_rad,  axis=0)
        d_del = np.gradient(amp, el_rad,  axis=1) if el_rad.size > 1 else np.zeros_like(amp)
        d_df  = np.gradient(amp, freq_hz, axis=2)

        grad_amp = np.sqrt(d_daz ** 2 + d_del ** 2 + d_df ** 2)

        # ── Phase curvature along frequency (d²φ/df²) ───────────────────────
        # Unwrap along frequency axis to avoid 2π discontinuities.
        phase = np.unwrap(np.angle(tensor), axis=2)
        dphase_df  = np.gradient(phase,     freq_hz, axis=2)
        phase_curv = np.abs(np.gradient(dphase_df, freq_hz, axis=2))

        return {
            "amplitude_gradient": grad_amp,
            "phase_curvature":    phase_curv,
        }
