"""ISAR complexity analyzer.

Improvements over v1
--------------------
* Selectable window (default 'taylor', −30 dB sidelobes) instead of hardcoded
  Hamming — better main-lobe shape for closely-spaced scatterers.
* Adaptive zero-padding: max(4, next power-of-2 ≥ max(N_az, N_freq)) instead
  of fixed 2×.  Gives at least 4-point interpolation of the ISAR peak.
* Three complementary slice metrics instead of one:
    sidelobe_ratio  — data-driven floor percentile (adapts to scene complexity)
    entropy_score   — information-theoretic scatterer count proxy
    spread_score    — normalised 2nd moment of ISAR power around dominant peak
* Structural 2D score broadcast: within each elevation the score is modulated
  by the local ISAR power, so az/freq cells near ISAR peaks score higher.
* Vectorised batched 2D FFT: all elevation slices processed in a single
  NumPy call rather than a Python loop.
"""
from __future__ import annotations

import numpy as np
import scipy.signal

from ._utils import robust_scale


def _next_pow2_ge(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


class ISARAnalyzer:
    """Compute per-elevation ISAR complexity scores."""

    def __init__(
        self,
        window_name: str = "taylor",
        min_pad_factor: int = 4,
        entropy_weight: float = 0.4,
        spread_weight: float = 0.2,
    ) -> None:
        """
        Parameters
        ----------
        window_name    : Window type accepted by scipy.signal.get_window.
        min_pad_factor : Minimum zero-padding multiplier for the 2D IFFT.
        entropy_weight : Contribution of ISAR spatial entropy to the slice score.
        spread_weight  : Contribution of centroid spread to the slice score.
        """
        self._win_name     = window_name
        self._min_pad      = min_pad_factor
        self._ent_w        = entropy_weight
        self._spr_w        = spread_weight
        self._sidelobe_w   = 1.0 - entropy_weight - spread_weight

    def _make_window(self, N_az: int, N_freq: int) -> np.ndarray:
        """Build a separable 2D window for the ISAR slice."""
        try:
            win_az   = scipy.signal.get_window(self._win_name, N_az,   fftbins=False)
            win_freq = scipy.signal.get_window(self._win_name, N_freq, fftbins=False)
        except Exception:
            win_az   = np.hamming(N_az)
            win_freq = np.hamming(N_freq)
        return np.outer(win_az, win_freq)

    def _pad_factor(self, N_az: int, N_freq: int) -> int:
        bigger = max(N_az, N_freq)
        return max(self._min_pad, _next_pow2_ge(bigger) // max(bigger, 1))

    def _slice_metrics(self, isar: np.ndarray) -> tuple[float, float, float]:
        """Compute (sidelobe_ratio, entropy_score, spread_score) for an ISAR image."""
        total = float(isar.sum()) + 1e-30
        rows, cols = isar.shape

        # ── Sidelobe ratio (data-driven floor percentile) ─────────────────────
        flat = isar.ravel()
        n_sc = max(1, int(np.sum(flat > 0.05 * float(flat.max()))))
        floor_pct  = np.clip(100.0 * (1.0 - 1.0 / (n_sc + 1)), 50.0, 90.0)
        floor_val  = float(np.percentile(flat, floor_pct))
        peak_val   = float(flat.max())
        sidelobe_r = float(np.clip(floor_val / (peak_val + 1e-30), 0.0, 1.0))

        # ── Spatial entropy ────────────────────────────────────────────────────
        p = isar / total
        max_ent = np.log2(float(isar.size))
        entropy = float(-np.sum(p * np.log2(p + 1e-30))) / (max_ent + 1e-30)

        # ── Centroid spread ────────────────────────────────────────────────────
        ri, ci = np.unravel_index(int(isar.argmax()), isar.shape)
        di = (np.arange(rows) - ri).astype(float)
        dj = (np.arange(cols) - ci).astype(float)
        DI, DJ = np.meshgrid(di, dj, indexing="ij")
        dist2  = DI ** 2 + DJ ** 2
        spread = float(np.sum(isar * dist2) / total)
        spread_norm = float(np.tanh(spread / (rows * cols)))

        return sidelobe_r, float(np.clip(entropy, 0.0, 1.0)), spread_norm

    def compute_slice(
        self,
        tensor_2d: np.ndarray,  # (N_az, N_freq), complex128
    ) -> tuple[float, np.ndarray]:
        """Return (slice_score, isar_power_image) for one elevation slice.

        slice_score is a weighted combination of sidelobe_ratio, entropy_score,
        and spread_score.  isar_power_image has shape (pad*N_az, pad*N_freq).
        """
        N_az, N_freq = tensor_2d.shape
        pad   = self._pad_factor(N_az, N_freq)
        win   = self._make_window(N_az, N_freq)
        S_win = tensor_2d * win
        isar  = np.abs(np.fft.fftshift(
            np.fft.ifft2(S_win, s=(pad * N_az, pad * N_freq))
        )) ** 2

        sr, ent, spr = self._slice_metrics(isar)
        score = (
            self._sidelobe_w * sr
            + self._ent_w    * ent
            + self._spr_w    * spr
        )
        return float(np.clip(score, 0.0, 1.0)), isar

    def compute(
        self,
        tensor: np.ndarray,   # (N_az, N_el, N_freq)
        az_rad:  np.ndarray,
        el_rad:  np.ndarray,
        freq_hz: np.ndarray,
    ) -> np.ndarray:
        """Return score array shape (N_az, N_el, N_freq).

        Each elevation gets a slice score that is then modulated by the local
        ISAR power (so az/freq cells near scatterer peaks score higher within
        an elevation), rather than being broadcast uniformly.
        """
        N_az, N_el, N_freq = tensor.shape
        pad = self._pad_factor(N_az, N_freq)
        win = self._make_window(N_az, N_freq)   # (N_az, N_freq)

        # ── Vectorised batched IFFT across all elevations ─────────────────────
        # Shape: (N_el, N_az, N_freq)
        T_elev = tensor.transpose(1, 0, 2)
        T_win  = T_elev * win[None, :, :]          # broadcast window
        T_pad  = np.fft.ifft2(T_win, s=(pad * N_az, pad * N_freq))
        ISAR   = np.abs(np.fft.fftshift(T_pad, axes=(1, 2))) ** 2
        # ISAR shape: (N_el, pad*N_az, pad*N_freq)

        scores = np.empty(tensor.shape, dtype=float)
        for j in range(N_el):
            isar_j = ISAR[j]
            sr, ent, spr = self._slice_metrics(isar_j)
            slice_score  = float(np.clip(
                self._sidelobe_w * sr + self._ent_w * ent + self._spr_w * spr,
                0.0, 1.0,
            ))

            # Down-sample ISAR back to original (N_az, N_freq) grid and
            # modulate: cells near ISAR peaks score higher within the slice.
            isar_orig = isar_j[::pad, ::pad][:N_az, :N_freq]
            local_wt  = isar_orig / (float(isar_orig.max()) + 1e-30)
            # Blend global slice score (50%) + local modulation (50%)
            scores[:, j, :] = np.clip(
                slice_score * (0.5 + 0.5 * local_wt), 0.0, 1.0
            )

        return scores
