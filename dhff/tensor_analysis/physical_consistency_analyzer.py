"""Physical consistency analyzer (new in v2).

Provides two sub-signals orthogonal to the four existing analyzers:

Sub-analyzer A — Group Delay Anomaly
    Computes the group delay τ = -(1/2π) d∠S/df.  For a physical target,
    |τ| must be ≤ 2 × r_target / c (time-of-flight to the farthest surface).
    Violations flag dispersive resonances (cavities, coatings) where the group
    delay exceeds what free-space propagation can explain.

Sub-analyzer B — Angular Coherence Drop
    At each frequency, computes the normalised cross-correlation between
    adjacent azimuth slices: xcorr = Re⟨S[i-1,:,:], S[i,:,:]⟩ / (‖·‖‖·‖).
    A sudden drop indicates the field pattern changes character between look
    angles (e.g., entering a creeping-wave shadow zone or a dihedral lobe
    edge).  This is distinct from the amplitude gradient because it measures
    joint amplitude + phase coherence.

Both scores are in [0, 1].  Combined: 0.6 * gd_score + 0.4 * xcorr_score.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage

from ._utils import robust_scale


class PhysicalConsistencyAnalyzer:
    """Detect group-delay anomalies and angular coherence drops."""

    def __init__(
        self,
        gd_tau_margin: float = 2.0,
        isar_spread_percentile: float = 90.0,
        xcorr_smooth_az: int = 3,
        group_delay_weight: float = 0.6,
    ) -> None:
        """
        Parameters
        ----------
        gd_tau_margin          : Safety factor on the estimated physical group-
                                 delay bound (τ_max = margin × τ_estimated).
        isar_spread_percentile : Percentile of the ISAR power used to estimate
                                 the target's physical extent.
        xcorr_smooth_az        : Smoothing window (samples) applied to the
                                 coherence-drop map along the azimuth axis.
        group_delay_weight     : Weight of the group-delay component in the
                                 combined score (xcorr weight = 1 - this).
        """
        self._margin   = gd_tau_margin
        self._isar_pct = isar_spread_percentile
        self._xcorr_sm = xcorr_smooth_az
        self._gd_w     = group_delay_weight

    def _estimate_target_extent_m(self, tensor: np.ndarray, freq_hz: np.ndarray) -> float:
        """Rough estimate of the target's physical extent from its range profile.

        Uses the 1D IFFT (range profile) of |S| averaged across angles and
        takes the RMS spread of the power profile above the median.
        """
        c = 3e8
        N_freq = len(freq_hz)
        # Average amplitude spectrum across all (az, el)
        amp_mean = np.mean(np.abs(tensor), axis=(0, 1))   # (N_freq,)
        win      = np.hamming(N_freq)
        rp       = np.abs(np.fft.ifft(amp_mean * win, n=4 * N_freq)) ** 2
        # Range axis (metres)
        bw_hz    = freq_hz[-1] - freq_hz[0]
        range_res = c / (2.0 * bw_hz) if bw_hz > 0 else 0.1
        r_axis   = np.arange(len(rp)) * range_res

        # Keep only the first half (positive ranges) and above the median
        half     = len(rp) // 2
        rp_half  = rp[:half]
        r_half   = r_axis[:half]
        threshold = float(np.percentile(rp_half, self._isar_pct))
        mask     = rp_half >= threshold
        if not np.any(mask):
            return 0.5   # fallback: 0.5 m

        r_spread_rms = float(np.sqrt(np.mean(r_half[mask] ** 2)))
        return max(0.05, r_spread_rms)

    def compute(
        self,
        tensor: np.ndarray,   # (N_az, N_el, N_freq), complex128
        az_rad:  np.ndarray,
        el_rad:  np.ndarray,
        freq_hz: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Return dict with 'group_delay_anomaly', 'coherence_drop', and
        'combined', all shape (N_az, N_el, N_freq).
        """
        c = 3e8
        N_az, N_el, N_freq = tensor.shape

        # ── Sub-analyzer A: Group Delay Anomaly ───────────────────────────────
        phase     = np.unwrap(np.angle(tensor.astype(complex)), axis=2)
        dphase_df = np.gradient(phase, freq_hz, axis=2)
        group_delay = -dphase_df / (2.0 * np.pi)   # seconds

        # Estimate physical target extent → maximum plausible τ
        r_extent  = self._estimate_target_extent_m(tensor, freq_hz)
        tau_max   = self._margin * 2.0 * r_extent / c   # e.g. 2×0.5 m / 3e8 ≈ 3.3 ns

        gd_excess = np.maximum(np.abs(group_delay) - tau_max, 0.0)
        gd_score  = np.tanh(gd_excess / (tau_max + 1e-30))   # in [0, 1)

        # ── Sub-analyzer B: Angular Coherence Drop ────────────────────────────
        xcorr_drop = np.zeros_like(gd_score)
        if N_az > 1:
            s1 = tensor[:-1, :, :]   # (N_az-1, N_el, N_freq)
            s2 = tensor[1:,  :, :]

            # Dot product per (el, freq) strip
            dot    = np.real(np.sum(np.conj(s1) * s2, axis=0))   # (N_el, N_freq)
            norm1  = np.sqrt(np.sum(np.abs(s1) ** 2, axis=0)) + 1e-30
            norm2  = np.sqrt(np.sum(np.abs(s2) ** 2, axis=0)) + 1e-30
            xcorr  = np.clip(dot / (norm1 * norm2), -1.0, 1.0)   # (N_el, N_freq)
            drop_2d = np.clip(1.0 - xcorr, 0.0, 1.0)             # (N_el, N_freq)

            # Map the (N_az-1) inter-slice drops back to N_az voxels
            # by assigning each drop[i] to az index i+1 (the leading edge)
            xcorr_drop[1:, :, :] = drop_2d[None, :, :] * np.ones((N_az - 1, 1, 1))
            xcorr_drop[0,  :, :] = xcorr_drop[1, :, :]  # copy first

            # Smooth along azimuth to reduce edge noise
            if self._xcorr_sm > 1:
                xcorr_drop = scipy.ndimage.uniform_filter(
                    xcorr_drop,
                    size=(self._xcorr_sm, 1, 1),
                    mode="nearest",
                )

        combined = (
            self._gd_w       * robust_scale(gd_score)
            + (1.0 - self._gd_w) * robust_scale(xcorr_drop)
        )

        return {
            "group_delay_anomaly": gd_score,
            "coherence_drop":      xcorr_drop,
            "combined":            np.clip(combined, 0.0, 1.0),
        }
