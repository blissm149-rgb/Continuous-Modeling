"""Shared utility helpers for tensor sensitivity analyzers.

All functions are pure numpy, dependency-free, and importable from
any analyzer module without circular imports.
"""
from __future__ import annotations

import numpy as np


def adaptive_odd_window(N: int, fraction: float, min_w: int = 3) -> int:
    """Return an odd window size that is ~fraction of axis length N.

    Parameters
    ----------
    N        : axis length
    fraction : target fraction (e.g. 0.12 means ~12% of N)
    min_w    : smallest allowable window (must be ≥ 1)
    """
    w = max(min_w, int(round(fraction * N)))
    return w if w % 2 == 1 else w + 1


def robust_scale(arr: np.ndarray, percentile: float = 98.0) -> np.ndarray:
    """Scale arr so that its ``percentile``-th value maps to 1.

    Equivalent to winsorising at *percentile* then dividing by that value.
    Returns zeros if the reference value is effectively zero.
    """
    ref = float(np.percentile(arr, percentile))
    if ref < 1e-30:
        return np.zeros_like(arr, dtype=float)
    return np.clip(arr.astype(float) / ref, 0.0, 1.0)


def fast_row_percentile(arr_2d: np.ndarray, q: float) -> np.ndarray:
    """Per-row percentile using np.partition (O(N) per row).

    Parameters
    ----------
    arr_2d : shape (M, N)
    q      : percentile in [0, 100]

    Returns
    -------
    shape (M,) — the q-th percentile of each row
    """
    N = arr_2d.shape[1]
    k = int(np.clip(np.round(q / 100.0 * (N - 1)), 0, N - 1))
    partitioned = np.partition(arr_2d, k, axis=1)
    return partitioned[:, k]
