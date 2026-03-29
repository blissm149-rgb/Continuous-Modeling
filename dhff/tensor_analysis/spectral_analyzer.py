"""Spectral variance and resonance-peak analyzer.

At each (azimuth, elevation) pixel, the frequency profile |S[az, el, :]|²
is a power spectrum.

Spectral variance (variance of the PSD across frequency) distinguishes:
  • Flat spectrum  → broadband specular feature → easy to model (low variance)
  • Peaky spectrum → cavity resonances / frequency-selective coatings (high variance)

Resonance count (number of peaks in |S[az, el, :]|) catches narrow
high-Q features that are spectrally sparse but individually critical.

Both quantities are broadcast across the frequency axis so the output
has the same (N_az, N_el, N_freq) shape as the other analyzers.
"""
from __future__ import annotations

import numpy as np
import scipy.signal


class SpectralAnalyzer:
    """Spectral variance and resonance peak count per (az, el) pixel."""

    def __init__(self, min_peak_prominence: float = 0.05):
        """
        Parameters
        ----------
        min_peak_prominence:
            Minimum prominence (relative to max |S| over full tensor) for
            a frequency peak to count as a resonance.
        """
        self._prom_frac = min_peak_prominence

    def compute(
        self,
        tensor: np.ndarray,   # (N_az, N_el, N_freq), complex128
        freq_hz: np.ndarray,  # (N_freq,)
    ) -> dict[str, np.ndarray]:
        """Return dict with 'spectral_variance' and 'resonance_count',
        both shape (N_az, N_el, N_freq).
        """
        N_az, N_el, N_freq = tensor.shape
        psd = np.abs(tensor) ** 2

        # ── Spectral variance (high for peaked/resonant, low for flat) ───────
        spectral_var_2d = np.var(psd, axis=2)   # (N_az, N_el)

        # ── Resonance (peak) count ───────────────────────────────────────────
        global_max = float(np.max(np.abs(tensor))) + 1e-30
        height_thresh = self._prom_frac * global_max
        prom_thresh   = self._prom_frac * global_max

        n_peaks_2d = np.zeros((N_az, N_el), dtype=float)
        for i in range(N_az):
            for j in range(N_el):
                peaks, _ = scipy.signal.find_peaks(
                    np.abs(tensor[i, j, :]),
                    height=height_thresh,
                    prominence=prom_thresh,
                )
                n_peaks_2d[i, j] = float(len(peaks))

        # Broadcast to (N_az, N_el, N_freq)
        ones = np.ones((1, 1, N_freq), dtype=float)
        var_3d    = spectral_var_2d[:, :, None] * ones
        n_peaks_3d = n_peaks_2d[:, :, None]     * ones

        return {
            "spectral_variance": var_3d,
            "resonance_count":   n_peaks_3d,
        }
