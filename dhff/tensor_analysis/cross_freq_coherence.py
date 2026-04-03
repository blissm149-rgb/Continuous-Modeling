"""Cross-Frequency Coherence Analyzer (Phase 3B).

Detects two independent failure modes that are invisible to the other five
analyzers:

1. **Range-bin drift** — the dominant ISAR range peak shifts anomalously
   as a function of frequency, beyond what the estimated target extent
   would physically permit.  Indicates dispersive or frequency-selective
   features (coatings, cavities, resonant edges) whose sim representation
   fails to capture the correct group-delay slope.

2. **Inter-frequency angular decoherence** — adjacent frequency slices
   become locally incoherent in the angular domain.  A broadband specular
   return stays angularly coherent across a small frequency step; a narrow-
   band resonance or a regime-transition zone does not.

Both components operate entirely in the amplitude domain — no phase
unwrapping is required, avoiding catastrophic failures at nulls.

Usage
-----
    from dhff.tensor_analysis.cross_freq_coherence import CrossFreqCoherenceAnalyzer

    score = CrossFreqCoherenceAnalyzer().compute(tensor, az, el, freq)
    # score: (N_az, N_el, N_freq) float in [0, 1]
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage
import scipy.signal

from ._utils import robust_scale

# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _estimate_target_extent(
    tensor:    np.ndarray,
    freq_axis: np.ndarray,
    c:         float = 3e8,
    subsample: int   = 10,
) -> float:
    """Estimate the target's range extent from IFFT range profiles (metres).

    Averages over a sparse angular subsample for robustness.  Returns a
    fallback of 1.0 m if no valid profiles are found.
    """
    n_az, n_el, n_freq = tensor.shape
    df = freq_axis[1] - freq_axis[0] if n_freq > 1 else 1.0

    extents: list[float] = []
    for i in range(0, n_az, max(1, n_az // subsample)):
        for j in range(0, n_el, max(1, n_el // subsample)):
            rp = np.abs(np.fft.ifft(tensor[i, j, :]))
            noise = float(np.median(rp)) * 3.0
            above = np.where(rp > noise)[0]
            if len(above) >= 2:
                extent_bins = int(above[-1] - above[0])
                extent_m    = extent_bins * c / (2.0 * n_freq * max(df, 1.0))
                extents.append(extent_m)

    return float(np.median(extents)) if extents else 1.0


def _range_drift_score(
    tensor:          np.ndarray,
    noise_floor:     float,
    stft_win:        int,
    max_range_extent: float,
) -> np.ndarray:
    """Compute per-voxel range-bin drift anomaly score.

    For each (az, el) pixel, compute the short-time Fourier transform along
    the frequency axis and track the peak range-bin across frames.  The drift
    rate is compared against the physical extent bound.

    Returns array of shape (N_az, N_el, N_freq).
    """
    n_az, n_el, n_freq = tensor.shape
    score = np.zeros((n_az, n_el, n_freq), dtype=float)
    overlap = stft_win // 2

    for i in range(n_az):
        for j in range(n_el):
            h = tensor[i, j, :]

            # Short-time Fourier transform of the amplitude along frequency axis.
            # Using amplitude (not complex) keeps us in the magnitude domain and
            # avoids scipy's one-sided/two-sided complexity for complex inputs.
            _, _, Zxx = scipy.signal.stft(
                np.abs(h), nperseg=stft_win, noverlap=overlap, window="hann"
            )
            stft_amp = np.abs(Zxx)                   # (n_range, n_frames)
            n_frames = stft_amp.shape[1]

            # Track peak range-bin per frame, gated by noise floor
            peak_range = np.full(n_frames, np.nan)
            for frame in range(n_frames):
                if stft_amp[:, frame].max() > noise_floor:
                    peak_range[frame] = float(np.argmax(stft_amp[:, frame]))

            valid = ~np.isnan(peak_range)
            if valid.sum() < 3:
                continue

            valid_idx = np.where(valid)[0]
            pr_valid  = peak_range[valid_idx]
            drifts    = np.abs(
                np.diff(pr_valid) / np.maximum(np.diff(valid_idx.astype(float)), 1.0)
            )
            drift_anomaly = np.maximum(drifts - max_range_extent, 0.0)

            # Map frame drift back to frequency bins
            for k_frame, d in zip(valid_idx[:-1], drift_anomaly):
                f_start = k_frame * overlap
                f_end   = min(f_start + stft_win, n_freq)
                score[i, j, f_start:f_end] = np.maximum(
                    score[i, j, f_start:f_end], d
                )

    return score


def _angular_coherence_score(
    tensor:      np.ndarray,
    noise_floor: float,
    coh_win:     int,
) -> np.ndarray:
    """Compute per-voxel inter-frequency angular decoherence score.

    For each adjacent (freq, freq+1) pair, compute the local complex
    coherence over a spatial (az × el) window.  Low coherence = high score.

    Operates entirely on amplitudes and complex cross-products — no phase
    unwrapping.  Amplitude-gated to avoid false positives at nulls.

    Returns array of shape (N_az, N_el, N_freq).
    """
    n_az, n_el, n_freq = tensor.shape
    score = np.zeros((n_az, n_el, n_freq), dtype=float)

    for k in range(n_freq - 1):
        P_curr = tensor[:, :, k]
        P_next = tensor[:, :, k + 1]

        gate = (
            (np.abs(P_curr) > noise_floor)
            & (np.abs(P_next) > noise_floor)
        )

        # Local complex coherence over spatial window of size coh_win × coh_win
        cross_real = scipy.ndimage.uniform_filter(
            np.real(P_curr * np.conj(P_next)), size=coh_win
        )
        cross_imag = scipy.ndimage.uniform_filter(
            np.imag(P_curr * np.conj(P_next)), size=coh_win
        )
        auto_curr = scipy.ndimage.uniform_filter(np.abs(P_curr) ** 2, size=coh_win)
        auto_next = scipy.ndimage.uniform_filter(np.abs(P_next) ** 2, size=coh_win)

        coh = np.abs(cross_real + 1j * cross_imag) / np.sqrt(
            auto_curr * auto_next + 1e-30
        )
        decoherence = np.where(gate, 1.0 - np.clip(coh, 0.0, 1.0), 0.0)
        score[:, :, k] = decoherence

    # Last freq bin: copy from k-1
    if n_freq > 1:
        score[:, :, -1] = score[:, :, -2]

    return score


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

class CrossFreqCoherenceAnalyzer:
    """Detect range-drift and angular decoherence across the frequency axis."""

    def __init__(
        self,
        stft_window:        int   = 24,
        coherence_window:   int   = 5,
        drift_weight:       float = 0.4,
        noise_floor_pct:    float = 10.0,
        speed_of_light:     float = 3e8,
    ) -> None:
        """
        Parameters
        ----------
        stft_window      : Number of frequency bins per STFT frame.
        coherence_window : Spatial (az×el) window size for coherence estimate.
        drift_weight     : Weight of drift component vs (1-weight) decoherence.
        noise_floor_pct  : Amplitude percentile below which voxels are gated out.
        speed_of_light   : m/s.
        """
        self._stft_win   = int(stft_window)
        self._coh_win    = int(coherence_window)
        self._alpha      = float(drift_weight)
        self._nf_pct     = float(noise_floor_pct)
        self._c          = float(speed_of_light)

    def compute(
        self,
        tensor:   np.ndarray,
        az_rad:   np.ndarray,
        el_rad:   np.ndarray,
        freq_hz:  np.ndarray,
    ) -> np.ndarray:
        """Return cross-frequency coherence sensitivity score.

        Parameters
        ----------
        tensor  : (N_az, N_el, N_freq) complex128
        az_rad  : (N_az,) azimuth in radians
        el_rad  : (N_el,) elevation in radians
        freq_hz : (N_freq,) frequency in Hz

        Returns
        -------
        score : (N_az, N_el, N_freq) float in [0, 1] after robust scaling
        """
        amp         = np.abs(tensor)
        noise_floor = float(np.percentile(amp, self._nf_pct))

        # Estimate physical range extent for drift bound
        max_range_m = _estimate_target_extent(tensor, freq_hz, c=self._c)

        # Guard: ensure STFT window fits the frequency axis
        n_freq   = tensor.shape[2]
        stft_win = min(self._stft_win, max(4, n_freq // 2))

        drift_s = _range_drift_score(tensor, noise_floor, stft_win, max_range_m)
        coh_s   = _angular_coherence_score(tensor, noise_floor, self._coh_win)

        combined = (
            self._alpha       * robust_scale(drift_s)
            + (1.0 - self._alpha) * robust_scale(coh_s)
        )
        return robust_scale(combined)
