"""ISAR sidelobe floor analyzer.

For each elevation slice, computes a 2D IFFT (Hamming-windowed) over
the (azimuth × frequency) plane to produce an ISAR image in the
(cross-range × range) domain.

The ratio of the 75th-percentile power to the peak power —
the "sidelobe floor ratio" — quantifies multi-scatterer interference:

  • One dominant scatterer  → clean isolated peak → floor ratio ≈ 0
  • Many comparable scatterers interfering → elevated floor → ratio → 1

High floor ratio ↔ complex interference ↔ small geometry errors cause
large, unpredictable RCS changes.  These are the most discriminating
measurement points.
"""
from __future__ import annotations

import numpy as np


class ISARAnalyzer:
    """Compute the per-elevation ISAR sidelobe floor score."""

    def compute_slice(
        self,
        tensor_2d: np.ndarray,  # (N_az, N_freq), complex128
    ) -> tuple[float, np.ndarray]:
        """Return (sidelobe_ratio, isar_power_image) for one elevation slice.

        sidelobe_ratio: float in [0, 1]; higher → more interference.
        isar_power_image: 2D float array, shape (2*N_az, 2*N_freq).
        """
        N_az, N_freq = tensor_2d.shape

        # Separable Hamming window to reduce range/cross-range sidelobes
        win = np.outer(np.hamming(N_az), np.hamming(N_freq))
        S_win = tensor_2d * win

        # Zero-padded 2D IFFT → ISAR image (power)
        isar = np.abs(np.fft.fftshift(
            np.fft.ifft2(S_win, s=(2 * N_az, 2 * N_freq))
        )) ** 2

        peak_power    = float(np.max(isar))
        sidelobe_floor = float(np.percentile(isar, 75))
        sidelobe_ratio = sidelobe_floor / (peak_power + 1e-30)

        return float(np.clip(sidelobe_ratio, 0.0, 1.0)), isar

    def compute(
        self,
        tensor: np.ndarray,  # (N_az, N_el, N_freq)
        az_rad:  np.ndarray,
        el_rad:  np.ndarray,
        freq_hz: np.ndarray,
    ) -> np.ndarray:
        """Return score array shape (N_az, N_el, N_freq).

        Each elevation slice gets a uniform score equal to its
        sidelobe floor ratio (broadcast across az and freq dims).
        """
        scores = np.empty(tensor.shape, dtype=float)
        for j in range(len(el_rad)):
            ratio, _ = self.compute_slice(tensor[:, j, :])
            scores[:, j, :] = ratio
        return scores
