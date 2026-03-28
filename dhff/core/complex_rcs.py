from __future__ import annotations

import numpy as np


def complex_to_mag_phase(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mag = np.abs(values)
    phase = np.angle(values)
    return mag, phase


def mag_phase_to_complex(mag: np.ndarray, phase: np.ndarray) -> np.ndarray:
    return (mag * np.exp(1j * phase)).astype(np.complex128)


def dbsm_to_linear(dbsm: np.ndarray) -> np.ndarray:
    return 10.0 ** (dbsm / 10.0)


def linear_to_dbsm(linear: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.abs(linear), 1e-30))


def unwrap_phase_2d(phase: np.ndarray, axis: int = 0) -> np.ndarray:
    """2D phase unwrapping along specified axis."""
    return np.unwrap(phase, axis=axis)
