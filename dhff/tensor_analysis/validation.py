"""Perturbation ensemble validator for TensorSensitivityMap.

Generates physically-motivated tensor perturbations (phase noise, angular
jitter, amplitude scaling) and measures where those perturbations cause the
largest RCS changes.  The correlation between a sensitivity map and the
resulting perturbation variance is the primary quality metric used to gate
every subsequent improvement.

Usage
-----
    from dhff.tensor_analysis.validation import validate_sensitivity

    correlation, pert_var = validate_sensitivity(
        tensor, az_axis, el_axis, freq_axis,
        sensitivity_map=tsm.get_combined_score_grid(),
    )
    print(f"Pearson r = {correlation:.3f}")

A correlation above ~0.30 on realistic tensors is a meaningful result.
Lift of >= 0.02 is required before enabling any new improvement.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage

# ──────────────────────────────────────────────────────────────────────────────
# Default configuration
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG: dict = {
    "validation_n_perturbations": 50,
    "perturb_phase_std":          0.10,   # radians
    "perturb_angle_std":          0.50,   # cells (fractional shift)
    "perturb_amp_std":            0.05,   # fractional RCS amplitude
    # Smooth random-field correlation length for amplitude perturbations (cells)
    "perturb_amp_corr_length":    20,
    # Random seed (None = non-deterministic)
    "validation_seed":            0,
}


# ──────────────────────────────────────────────────────────────────────────────
# Internal perturbation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _perturb_phase(tensor: np.ndarray, rng: np.random.Generator,
                   config: dict) -> np.ndarray:
    """Add per-cell i.i.d. phase noise — simulates small position shifts."""
    sigma = float(config["perturb_phase_std"])
    noise = rng.normal(0.0, sigma, size=tensor.shape)
    return tensor * np.exp(1j * noise)


def _perturb_angular_jitter(tensor: np.ndarray, rng: np.random.Generator,
                             config: dict) -> np.ndarray:
    """Shift tensor by a fractional-pixel amount along az and el axes.

    Uses scipy.ndimage.shift with reflect padding to avoid edge artefacts.
    Only the real and imaginary parts are shifted separately (complex shift
    is not natively supported by ndimage).
    """
    sigma = float(config["perturb_angle_std"])
    daz = float(rng.normal(0.0, sigma))
    del_ = float(rng.normal(0.0, sigma))
    shift = [daz, del_, 0.0]   # no shift along frequency
    real_shifted = scipy.ndimage.shift(tensor.real, shift, mode="reflect")
    imag_shifted = scipy.ndimage.shift(tensor.imag, shift, mode="reflect")
    return real_shifted + 1j * imag_shifted


def _perturb_amplitude_scaling(tensor: np.ndarray, rng: np.random.Generator,
                                config: dict) -> np.ndarray:
    """Multiply by a spatially-smooth random field — simulates material uncertainty.

    The smooth field is produced by Gaussian-filtering white noise, giving a
    field whose correlation length is ``perturb_amp_corr_length`` cells.
    """
    sigma_amp = float(config["perturb_amp_std"])
    corr      = float(config["perturb_amp_corr_length"])
    white     = rng.standard_normal(tensor.shape)
    smooth    = scipy.ndimage.gaussian_filter(white, sigma=corr / 3.0)
    # Normalise so the RMS of the smooth field is ~1
    smooth   /= (float(np.std(smooth)) + 1e-30)
    scale_field = 1.0 + sigma_amp * smooth
    return tensor * scale_field


_PERTURB_FNS = [
    _perturb_phase,
    _perturb_angular_jitter,
    _perturb_amplitude_scaling,
]


def apply_perturbation(tensor: np.ndarray, rng: np.random.Generator,
                       config: dict) -> np.ndarray:
    """Apply one randomly-chosen perturbation type to *tensor*.

    The perturbation type is chosen uniformly at random so that the ensemble
    variance averages over all three physical failure modes.
    """
    fn = _PERTURB_FNS[int(rng.integers(len(_PERTURB_FNS)))]
    return fn(tensor, rng, config)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def validate_sensitivity(
    tensor:          np.ndarray,
    az_axis:         np.ndarray,
    el_axis:         np.ndarray,
    freq_axis:       np.ndarray,
    sensitivity_map: np.ndarray,
    config:          dict | None = None,
) -> tuple[float, np.ndarray]:
    """Measure how well *sensitivity_map* predicts perturbation impact.

    Generates ``config["validation_n_perturbations"]`` physically-motivated
    tensor perturbations and accumulates the squared RCS difference at each
    voxel.  The Pearson correlation between this perturbation variance and the
    provided sensitivity map is returned as the primary quality metric.

    Parameters
    ----------
    tensor           : (N_az, N_el, N_freq) complex128 — base simulation tensor
    az_axis          : (N_az,) azimuth grid in radians
    el_axis          : (N_el,) elevation grid in radians
    freq_axis        : (N_freq,) frequency grid in Hz
    sensitivity_map  : (N_az, N_el, N_freq) float — scores in [0, 1]
    config           : override dict (merged with DEFAULT_CONFIG)

    Returns
    -------
    correlation         : Pearson r between sensitivity_map and perturbation
                          variance (scalar float)
    perturbation_variance : (N_az, N_el, N_freq) float — ground-truth sensitivity
                            from the ensemble; useful for diagnostic plots
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    n_pert = int(cfg["validation_n_perturbations"])
    seed   = cfg.get("validation_seed")
    rng    = np.random.default_rng(seed)

    perturbation_variance = np.zeros(tensor.shape, dtype=np.float64)

    for _ in range(n_pert):
        perturbed = apply_perturbation(tensor, rng, cfg)
        delta     = np.abs(perturbed - tensor)
        perturbation_variance += delta ** 2

    perturbation_variance /= n_pert

    # Pearson correlation between flattened arrays
    s_flat = sensitivity_map.ravel().astype(float)
    v_flat = perturbation_variance.ravel()

    # Mask out degenerate voxels (zero variance everywhere)
    valid  = np.isfinite(s_flat) & np.isfinite(v_flat)
    if valid.sum() < 2 or v_flat[valid].std() < 1e-30 or s_flat[valid].std() < 1e-30:
        return 0.0, perturbation_variance

    corr_matrix = np.corrcoef(s_flat[valid], v_flat[valid])
    correlation  = float(corr_matrix[0, 1])

    if not np.isfinite(correlation):
        correlation = 0.0

    return correlation, perturbation_variance


def compare_sensitivity(
    tensor:    np.ndarray,
    az_axis:   np.ndarray,
    el_axis:   np.ndarray,
    freq_axis: np.ndarray,
    map_before: np.ndarray,
    map_after:  np.ndarray,
    config:    dict | None = None,
) -> dict:
    """Measure the lift from one sensitivity map to another.

    Runs the perturbation ensemble ONCE (shared for both maps) and computes
    the correlation of each map against the common perturbation variance.

    Returns
    -------
    dict with keys:
        "r_before"             : correlation of map_before
        "r_after"              : correlation of map_after
        "lift"                 : r_after - r_before
        "perturbation_variance": shared ensemble variance array
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    n_pert = int(cfg["validation_n_perturbations"])
    seed   = cfg.get("validation_seed")
    rng    = np.random.default_rng(seed)

    pert_var = np.zeros(tensor.shape, dtype=np.float64)
    for _ in range(n_pert):
        p = apply_perturbation(tensor, rng, cfg)
        pert_var += np.abs(p - tensor) ** 2
    pert_var /= n_pert

    def _corr(score_map):
        s = score_map.ravel().astype(float)
        v = pert_var.ravel()
        valid = np.isfinite(s) & np.isfinite(v)
        if valid.sum() < 2 or v[valid].std() < 1e-30 or s[valid].std() < 1e-30:
            return 0.0
        c = float(np.corrcoef(s[valid], v[valid])[0, 1])
        return c if np.isfinite(c) else 0.0

    r_before = _corr(map_before)
    r_after  = _corr(map_after)

    return {
        "r_before":              r_before,
        "r_after":               r_after,
        "lift":                  r_after - r_before,
        "perturbation_variance": pert_var,
    }
