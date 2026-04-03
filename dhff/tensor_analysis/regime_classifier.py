"""Scattering-regime classifier for per-frequency analyzer weight adaptation.

Classifies each frequency bin into Rayleigh, resonance (Mie), or optical
regime based on the effective ka estimated from null-spacing in the tensor.
Returns a per-frequency weight matrix that the TensorSensitivityMap can use
in place of global fixed weights.

Usage
-----
    from dhff.tensor_analysis.regime_classifier import RegimeClassifier

    wts, labels, confidence = RegimeClassifier().classify(
        tensor, freq_hz, n_methods=5
    )
    # wts: (n_methods, N_freq) — already normalised per frequency bin
"""
from __future__ import annotations

import numpy as np
import scipy.signal

# ──────────────────────────────────────────────────────────────────────────────
# Per-regime weight tables
# Order: [Gradient, ISAR, Spectral, Cancellation, Physical, CrossFreq(optional)]
# Minimum floor of 0.05 on every entry prevents catastrophic suppression from
# misclassification.  Rows are renormalised to 1 inside classify().
# ──────────────────────────────────────────────────────────────────────────────

_W_RAYLEIGH  = np.array([0.08, 0.08, 0.12, 0.12, 0.45, 0.15])
_W_RESONANCE = np.array([0.18, 0.13, 0.32, 0.18, 0.08, 0.11])
_W_OPTICAL   = np.array([0.33, 0.23, 0.12, 0.13, 0.08, 0.11])

# Global defaults (5-method, matching TensorSensitivityMap.DEFAULT_WEIGHTS order)
_GLOBAL_WEIGHTS_5 = np.array([0.30, 0.18, 0.22, 0.17, 0.13])

_SPEED_OF_LIGHT = 3e8


def _estimate_ka_from_nulls(
    tensor: np.ndarray,
    freq_axis: np.ndarray,
    subsample_az: int = 30,
    subsample_el: int = 10,
) -> tuple[np.ndarray, float]:
    """Estimate effective ka per frequency from null spacing in the power spectrum.

    Returns
    -------
    ka         : (N_freq,) effective ka at each frequency
    confidence : scalar confidence in [0, 1] (higher = more nulls found)
    """
    n_az, n_el, n_freq = tensor.shape
    power = np.abs(tensor) ** 2

    az_step = max(1, n_az // subsample_az)
    el_step = max(1, n_el // subsample_el)

    null_spacings: list[float] = []
    null_counts:   list[int]   = []

    for i in range(0, n_az, az_step):
        for j in range(0, n_el, el_step):
            spectrum = power[i, j, :]
            null_idx = scipy.signal.argrelmin(spectrum, order=2)[0]
            if len(null_idx) >= 2:
                spacings = np.diff(freq_axis[null_idx])
                null_spacings.append(float(np.median(spacings)))
                null_counts.append(len(null_idx))

    if len(null_spacings) == 0:
        return _estimate_ka_by_complexity(tensor, freq_axis), 0.0

    median_null_spacing = float(np.median(null_spacings))
    median_null_count   = float(np.median(null_counts))

    # Effective target dimension from null spacing: a = c / (2 * Δf_null)
    a_eff = _SPEED_OF_LIGHT / (2.0 * max(median_null_spacing, 1.0))
    ka    = 2.0 * np.pi * a_eff * freq_axis / _SPEED_OF_LIGHT

    # Confidence: sigmoid gated on median null count (need ≥ 3 nulls to trust)
    confidence = float(1.0 / (1.0 + np.exp(-(median_null_count - 3.0) / 2.0)))

    return ka, confidence


def _estimate_ka_by_complexity(
    tensor: np.ndarray,
    freq_axis: np.ndarray,
) -> np.ndarray:
    """Fallback ka estimate from spectral complexity (variance of power spectrum).

    When no nulls are detectable, use the spectral variance as a proxy for
    ka — higher variance typically indicates the resonance / optical regime.
    Returns a flat ka profile in the Mie transition zone (ka ≈ 3).
    """
    # Flat fallback: sit at the Rayleigh–Mie boundary (ka = 1) so global weights
    # apply via the confidence blend rather than forcing a wrong regime.
    return np.ones(len(freq_axis), dtype=float)


class RegimeClassifier:
    """Classify scattering regime and return per-frequency analyzer weights."""

    def __init__(
        self,
        transition_width: float = 0.5,
        speed_of_light:   float = _SPEED_OF_LIGHT,
    ) -> None:
        """
        Parameters
        ----------
        transition_width : width of regime boundaries in log10(ka) units.
                           Larger = softer transitions.
        speed_of_light   : m/s (override for unit-testing).
        """
        self._delta = float(transition_width)
        self._c     = float(speed_of_light)

    def classify(
        self,
        tensor:    np.ndarray,
        freq_axis: np.ndarray,
        n_methods: int = 5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Classify regime and compute per-frequency weight matrix.

        Parameters
        ----------
        tensor    : (N_az, N_el, N_freq) complex128
        freq_axis : (N_freq,) Hz
        n_methods : number of analyzers (5 or 6)

        Returns
        -------
        weights    : (n_methods, N_freq) float — normalised weights per frequency
        labels     : (N_freq,) str array — "rayleigh", "resonance", or "optical"
        confidence : (N_freq,) float in [0, 1]
        """
        ka, conf_scalar = _estimate_ka_from_nulls(tensor, freq_axis)

        # Sigmoid regime membership functions in log10(ka)
        d = self._delta
        log_ka = np.log10(np.maximum(ka, 1e-12))

        w_rayleigh  = 1.0 / (1.0 + np.exp((log_ka - np.log10(1.0))  / d))
        w_optical   = 1.0 / (1.0 + np.exp(-(log_ka - np.log10(10.0)) / d))
        w_resonance = np.maximum(1.0 - w_rayleigh - w_optical, 0.0)

        # Normalise regime memberships to sum to 1
        denom = w_rayleigh + w_resonance + w_optical + 1e-30
        w_rayleigh  /= denom
        w_resonance /= denom
        w_optical   /= denom

        # Trim weight tables to n_methods columns
        wr = _W_RAYLEIGH[:n_methods].copy()
        wm = _W_RESONANCE[:n_methods].copy()
        wo = _W_OPTICAL[:n_methods].copy()
        # Renormalise each row to sum to 1
        wr /= wr.sum(); wm /= wm.sum(); wo /= wo.sum()

        # Per-frequency blended weights: (N_freq, n_methods)
        regime_wts = (
            w_rayleigh[:, None]  * wr[None, :]
            + w_resonance[:, None] * wm[None, :]
            + w_optical[:, None]   * wo[None, :]
        )

        # Confidence blend: fall back to global weights when estimate is uncertain.
        # When n_methods > len(_GLOBAL_WEIGHTS_5) (e.g., cross_freq = 6th method),
        # pad with a small uniform weight and renormalise.
        n_global = len(_GLOBAL_WEIGHTS_5)
        if n_methods <= n_global:
            gw = _GLOBAL_WEIGHTS_5[:n_methods].copy()
        else:
            extra = np.full(n_methods - n_global, 0.05)
            gw = np.concatenate([_GLOBAL_WEIGHTS_5, extra])
        gw /= gw.sum()
        confidence_vec = np.full(len(freq_axis), conf_scalar)

        blended = (
            confidence_vec[:, None] * regime_wts
            + (1.0 - confidence_vec[:, None]) * gw[None, :]
        )

        # Renormalise to sum to 1 per frequency
        blended /= blended.sum(axis=1, keepdims=True)

        # Regime labels for diagnostics
        labels = np.where(
            w_rayleigh > w_optical,
            np.where(w_rayleigh > w_resonance, "rayleigh", "resonance"),
            np.where(w_optical > w_resonance,  "optical",  "resonance"),
        )

        return blended.T, labels, confidence_vec  # (n_methods, N_freq), (N_freq,), (N_freq,)
