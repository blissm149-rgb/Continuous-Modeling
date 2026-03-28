"""Module 4: Scattering center extraction via Matrix Pencil and 2D IFFT."""
from __future__ import annotations

import warnings

import numpy as np
import scipy.ndimage
import scipy.signal

from dhff.core.types import ScatteringCenter

_C = 299792458.0


class MatrixPencilExtractor:
    """Extract scattering centers via the Matrix Pencil Method."""

    def __init__(
        self,
        n_centers_max: int = 20,
        amplitude_threshold_db: float = -30.0,
        pencil_parameter_ratio: float = 1.0 / 3.0,
        sv_threshold: float = 0.01,
    ):
        self.n_centers_max = n_centers_max
        self.amplitude_threshold_db = amplitude_threshold_db
        self.pencil_parameter_ratio = pencil_parameter_ratio
        self.sv_threshold = sv_threshold

    def extract_1d(
        self, freq_hz: np.ndarray, rcs_values: np.ndarray
    ) -> list[ScatteringCenter]:
        """Extract scattering centers from frequency-domain data at a single angle."""
        freq_hz = np.asarray(freq_hz, dtype=np.float64)
        rcs_values = np.asarray(rcs_values, dtype=np.complex128)
        N = len(freq_hz)

        if N < 4:
            return []

        # Check uniform spacing within 0.1%
        diffs = np.diff(freq_hz)
        df = diffs[0]
        if df == 0:
            return []
        if np.max(np.abs(diffs - df)) / np.abs(df) > 0.001:
            warnings.warn("Frequency array is not uniformly spaced")

        L = max(2, int(N * self.pencil_parameter_ratio))
        L = min(L, N - 2)

        # Build Hankel matrices
        rows = N - L
        cols = L
        if rows < 1 or cols < 1:
            return []

        Y0 = np.zeros((rows, cols), dtype=np.complex128)
        Y1 = np.zeros((rows, cols), dtype=np.complex128)
        for i in range(rows):
            for j in range(cols):
                Y0[i, j] = rcs_values[i + j]
                Y1[i, j] = rcs_values[i + j + 1]

        # SVD
        U, S, Vh = np.linalg.svd(Y0, full_matrices=False)

        # Determine model order
        thresh = S[0] * self.sv_threshold
        K = int(np.sum(S > thresh))
        K = min(K, self.n_centers_max, len(S))
        if K == 0:
            return []

        # Truncate to rank K
        U_k = U[:, :K]
        S_k = S[:K]
        Vh_k = Vh[:K, :]

        # Compute poles
        try:
            Z = np.diag(1.0 / (S_k + 1e-30)) @ U_k.conj().T @ Y1 @ Vh_k.conj().T
            poles = np.linalg.eigvals(Z)
        except np.linalg.LinAlgError:
            return []

        # Convert poles to range
        # sigma(f) = sum_k a_k * exp(j * 2 * (2*pi*f/c) * r_k)
        # pole z_k = exp(j * 2 * (2*pi*df/c) * r_k)
        # angle(z_k) = 2 * (2*pi*df/c) * r_k → r_k = angle(z_k) * c / (4*pi*df)
        r_k = np.angle(poles) * _C / (4.0 * np.pi * np.abs(df) + 1e-30)

        # Build Vandermonde matrix for amplitude estimation
        n_vec = np.arange(N)
        V = np.outer(poles ** 0, np.ones(N)).T  # placeholder
        V = np.zeros((N, K), dtype=np.complex128)
        for k in range(K):
            V[:, k] = poles[k] ** n_vec

        # Solve least-squares
        try:
            a_k, _, _, _ = np.linalg.lstsq(V, rcs_values, rcond=None)
        except np.linalg.LinAlgError:
            return []

        # Filter by amplitude threshold
        amp_mags = np.abs(a_k)
        if len(amp_mags) == 0:
            return []
        max_amp = np.max(amp_mags)
        thresh_linear = max_amp * 10.0 ** (self.amplitude_threshold_db / 20.0)

        centers = []
        for k in range(K):
            if amp_mags[k] >= thresh_linear:
                centers.append(ScatteringCenter(
                    x=float(r_k[k].real),
                    y=0.0,
                    amplitude=complex(a_k[k]),
                    freq_dependence="specular",
                    angular_pattern="isotropic",
                ))
        return centers

    def extract_2d(
        self,
        theta_values: np.ndarray,
        freq_values: np.ndarray,
        rcs_matrix: np.ndarray,
    ) -> list[ScatteringCenter]:
        """Extract 2D scattering centers from a theta-frequency grid using 2D IFFT."""
        theta_values = np.asarray(theta_values, dtype=np.float64)
        freq_values = np.asarray(freq_values, dtype=np.float64)
        rcs_matrix = np.asarray(rcs_matrix, dtype=np.complex128)
        N_theta, N_freq = rcs_matrix.shape

        # Apply Hamming window
        win_theta = np.hamming(N_theta)
        win_freq = np.hamming(N_freq)
        windowed = rcs_matrix * np.outer(win_theta, win_freq)

        # Zero-pad to next power of 2
        def next_pow2(n):
            return 1 << (n - 1).bit_length()

        Nf_pad = next_pow2(N_freq)
        Nt_pad = next_pow2(N_theta)

        padded = np.zeros((Nt_pad, Nf_pad), dtype=np.complex128)
        padded[:N_theta, :N_freq] = windowed

        # 2D IFFT
        isar = np.fft.ifft2(padded)
        isar_shifted = np.fft.fftshift(isar)

        # Compute axes
        df = (freq_values[-1] - freq_values[0]) / max(N_freq - 1, 1)
        dtheta = (theta_values[-1] - theta_values[0]) / max(N_theta - 1, 1)
        lambda_center = _C / np.mean(freq_values)

        # Downrange axis
        r_axis = np.fft.fftshift(np.fft.fftfreq(Nf_pad, d=1.0)) * _C / (2.0 * df * Nf_pad + 1e-30)

        # Crossrange axis
        cr_axis = np.fft.fftshift(np.fft.fftfreq(Nt_pad, d=1.0)) * lambda_center / (2.0 * dtheta * Nt_pad + 1e-30)

        img_mag = np.abs(isar_shifted)
        max_val = np.max(img_mag) + 1e-30
        thresh = max_val * 10.0 ** (self.amplitude_threshold_db / 20.0)

        # Peak detection with 3x3 neighborhood
        local_max = scipy.ndimage.maximum_filter(img_mag, size=3)
        peaks = (img_mag == local_max) & (img_mag > thresh)
        peak_indices = np.argwhere(peaks)

        centers = []
        for idx in peak_indices:
            it, ifr = idx[0], idx[1]

            # Parabolic interpolation for sub-pixel accuracy
            t_refined = _parabolic_peak(img_mag[:, ifr], it)
            f_refined = _parabolic_peak(img_mag[it, :], ifr)

            t_coord = np.interp(t_refined, np.arange(Nt_pad), cr_axis)
            f_coord = np.interp(f_refined, np.arange(Nf_pad), r_axis)

            amp = isar_shifted[it, ifr]
            centers.append(ScatteringCenter(
                x=float(f_coord),
                y=float(t_coord),
                amplitude=complex(amp),
                freq_dependence="specular",
                angular_pattern="isotropic",
            ))

        return centers


def _parabolic_peak(arr: np.ndarray, idx: int) -> float:
    """Parabolic interpolation around a peak."""
    if idx <= 0 or idx >= len(arr) - 1:
        return float(idx)
    y0 = np.abs(arr[idx - 1])
    y1 = np.abs(arr[idx])
    y2 = np.abs(arr[idx + 1])
    denom = 2.0 * (2.0 * y1 - y0 - y2)
    if abs(denom) < 1e-30:
        return float(idx)
    return idx + (y0 - y2) / denom
