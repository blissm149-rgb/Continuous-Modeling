"""Sequential Measurement Planner for TensorSensitivityMap.

Transforms a static sensitivity heat-map into an ordered measurement list
where each successive point maximises marginal information gain given the
points already selected.  This prevents redundant measurements in spatially
or spectrally correlated regions.

The planner is a SEPARATE MODULE that consumes the sensitivity map — it does
not alter the sensitivity computation itself.

Usage
-----
    from dhff.tensor_analysis.measurement_planner import plan_measurements

    selected, gains = plan_measurements(
        sensitivity_map, az_axis, el_axis, freq_axis,
        config={"planner_budget": 50},
    )
    # selected[0] is the single most valuable point.
    # selected[k] is the most valuable GIVEN selected[0..k-1] are taken.
    # gains[k] is the marginal information gain of selected[k].
"""
from __future__ import annotations

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Default configuration
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG: dict = {
    "planner_budget":               100,
    "planner_sensitivity_threshold_pct": 70,   # top-X% of map = candidates
    "planner_noise_ratio":          0.01,
}


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _estimate_lengthscales(score_map: np.ndarray) -> tuple[float, float, float]:
    """Estimate the autocorrelation 1/e lengthscale along each axis (in cells).

    Uses FFT-based autocorrelation averaged over the other two axes.  Falls
    back to N/4 if the autocorrelation never drops below 1/e.

    Returns
    -------
    (ell_az, ell_el, ell_freq)  — lengthscales in index (cell) units
    """
    lengthscales: list[float] = []
    for axis in range(3):
        other = tuple(i for i in range(3) if i != axis)
        profile = score_map.mean(axis=other).astype(float)
        profile -= profile.mean()

        # FFT autocorrelation
        F   = np.fft.fft(profile)
        acf = np.real(np.fft.ifft(np.abs(F) ** 2))
        denom = acf[0]
        if denom < 1e-30:
            lengthscales.append(float(len(profile)) / 4.0)
            continue
        acf /= denom

        below = np.where(acf < 1.0 / np.e)[0]
        if len(below) > 0:
            lengthscales.append(max(float(below[0]), 2.0))
        else:
            lengthscales.append(float(len(profile)) / 4.0)

    return float(lengthscales[0]), float(lengthscales[1]), float(lengthscales[2])


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def plan_measurements(
    sensitivity_map: np.ndarray,
    az_axis:         np.ndarray,
    el_axis:         np.ndarray,
    freq_axis:       np.ndarray,
    config:          dict | None = None,
) -> tuple[list[tuple[int, int, int]], list[float]]:
    """Select measurement points that are high-sensitivity and mutually non-redundant.

    Uses a greedy sequential algorithm with exponential correlation suppression.
    Points in highly correlated regions of the sensitivity map suppress each
    other's remaining value so the planner spreads across the full observation
    space rather than clustering around a single hot-spot.

    Parameters
    ----------
    sensitivity_map : (N_az, N_el, N_freq) float in [0, 1]
    az_axis         : (N_az,) azimuth in radians
    el_axis         : (N_el,) elevation in radians
    freq_axis       : (N_freq,) frequency in Hz
    config          : override dict (merged with DEFAULT_CONFIG)

    Returns
    -------
    selected : list of (az_idx, el_idx, freq_idx) tuples ordered by marginal
               information gain (most valuable first)
    gains    : corresponding marginal gain values; gains[k] is the sensitivity
               value of selected[k] after suppression from all prior selections
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    budget    = int(cfg["planner_budget"])
    thr_pct   = float(cfg["planner_sensitivity_threshold_pct"])

    # ── Step 1: Build candidate set (top-X% of sensitivity map) ──────────────
    threshold        = float(np.percentile(sensitivity_map, thr_pct))
    candidate_mask   = sensitivity_map > threshold
    candidate_indices = np.argwhere(candidate_mask)           # (M, 3)
    candidate_scores  = sensitivity_map[candidate_mask].copy()  # (M,)

    if len(candidate_indices) == 0:
        # Degenerate case: everything scores identically → take from full map
        candidate_indices = np.argwhere(np.ones(sensitivity_map.shape, dtype=bool))
        candidate_scores  = sensitivity_map.ravel().copy()

    # ── Step 2: Estimate correlation lengthscales ─────────────────────────────
    ell_az, ell_el, ell_freq = _estimate_lengthscales(sensitivity_map)

    # ── Step 3: Normalise coordinates by lengthscale for distance calculation ─
    coords = candidate_indices.astype(np.float64).copy()
    coords[:, 0] /= max(ell_az,   1e-6)
    coords[:, 1] /= max(ell_el,   1e-6)
    coords[:, 2] /= max(ell_freq, 1e-6)

    # ── Step 4: Greedy sequential selection with correlation suppression ───────
    remaining_value = candidate_scores.copy()
    selected: list[tuple[int, int, int]] = []
    gains:    list[float]                = []

    n_select = min(budget, len(candidate_indices))

    for _ in range(n_select):
        best_idx = int(np.argmax(remaining_value))
        gain     = float(remaining_value[best_idx])
        if gain < 1e-12:
            break

        selected.append(tuple(int(x) for x in candidate_indices[best_idx]))
        gains.append(gain)

        # Suppress nearby candidates: exponential decay in normalised distance
        diff = coords - coords[best_idx]                  # (M, 3)
        dist = np.sqrt((diff ** 2).sum(axis=1))           # (M,)
        correlation      = np.exp(-dist)                  # 1 at selected, →0 far away
        remaining_value *= (1.0 - correlation)
        remaining_value  = np.maximum(remaining_value, 0.0)

    return selected, gains
