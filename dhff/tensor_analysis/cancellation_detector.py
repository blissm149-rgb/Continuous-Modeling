"""Near-amplitude-null (destructive-interference) detector.

Improvements over v1
--------------------
* Resolution-adaptive window: scales to ~12%/20%/10% of each axis so the
  "local" neighbourhood is always physically meaningful regardless of whether
  the tensor is 11×5×10 or 180×36×100.
* Noise-floor-gated null depth: estimates the local noise floor as
  (local_mean - n_sigma * local_std) and scores against that rather than
  against local_mean.  Distinguishes true cancellation nulls from uniformly-
  dark low-amplitude regions (which should NOT score high).
* Null-bandwidth component: narrow nulls (high d²|S|/df² at the null) score
  higher because a small geometry error shifts them dramatically.

Score ranges
------------
Both null_depth and bandwidth_score are in [0, 1].  The combined output is
0.6 * null_depth + 0.4 * bandwidth_score, clipped to [0, 1].
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage

from ._utils import adaptive_odd_window, robust_scale


class CancellationDetector:
    """Detect near-null amplitude points in a complex RCS tensor."""

    def __init__(
        self,
        window: tuple[int, int, int] | None = None,
        window_fraction: tuple[float, float, float] = (0.12, 0.20, 0.10),
        n_sigma_noise: float = 1.0,
        bandwidth_weight: float = 0.4,
    ) -> None:
        """
        Parameters
        ----------
        window          : Fixed (az, el, freq) window sizes.  If None (default),
                          window sizes are computed adaptively from the tensor shape.
        window_fraction : Target fraction of each axis used to compute the adaptive
                          window when *window* is None.
        n_sigma_noise   : Local noise-floor = local_mean − n_sigma × local_std.
                          Higher values tighten the null criterion.
        bandwidth_weight: Weight of the null-bandwidth component (0 = depth only).
        """
        self._fixed_window   = window
        self._wfrac          = window_fraction
        self._n_sigma        = n_sigma_noise
        self._bw_weight      = bandwidth_weight

    def _get_window(self, shape: tuple[int, int, int]) -> tuple[int, int, int]:
        if self._fixed_window is not None:
            return self._fixed_window
        N_az, N_el, N_freq = shape
        return (
            adaptive_odd_window(N_az,   self._wfrac[0]),
            adaptive_odd_window(N_el,   self._wfrac[1]),
            adaptive_odd_window(N_freq, self._wfrac[2]),
        )

    def compute(self, tensor: np.ndarray) -> np.ndarray:
        """Return score array shape (N_az, N_el, N_freq) in [0, 1].

        Score is high where |S| forms a deep, narrow null relative to the
        local amplitude neighbourhood.
        """
        amp   = np.abs(tensor).astype(float)
        win   = self._get_window(amp.shape)
        kw    = dict(mode="nearest")

        local_mean = scipy.ndimage.uniform_filter(amp, size=win, **kw) + 1e-30

        # Local std via E[X²] - E[X]² (avoid a second filter pass on amp²)
        local_mean_sq = scipy.ndimage.uniform_filter(amp ** 2, size=win, **kw)
        local_var     = np.maximum(local_mean_sq - local_mean ** 2, 0.0)
        local_std     = np.sqrt(local_var)

        # Noise floor: local_mean lowered by n_sigma * std
        noise_floor   = np.maximum(local_mean - self._n_sigma * local_std, 1e-30)

        # ── Null depth ────────────────────────────────────────────────────────
        null_depth = np.clip(1.0 - amp / noise_floor, 0.0, 1.0)

        # ── Null bandwidth (sharpness) ────────────────────────────────────────
        # High d²|S|/df² at a null → narrow null → more geometry-sensitive.
        # Use the smoothed amplitude to avoid double-noise-amplification.
        amp_smooth   = scipy.ndimage.gaussian_filter(amp, sigma=[0.5, 0.5, 1.0])
        # Compute along freq axis only (the most reliable axis for bandwidth)
        freq_indices = np.arange(amp.shape[2], dtype=float)
        d_df         = np.gradient(amp_smooth, freq_indices, axis=2)
        d2_df2       = np.abs(np.gradient(d_df, freq_indices, axis=2))
        bw_score     = null_depth * robust_scale(d2_df2)

        combined = (1.0 - self._bw_weight) * null_depth + self._bw_weight * bw_score
        return np.clip(combined, 0.0, 1.0).astype(float)
