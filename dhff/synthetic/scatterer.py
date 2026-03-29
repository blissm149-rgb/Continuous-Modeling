"""Module 2: Ground-truth synthetic scatterer."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from dhff.core.types import ComplexRCS, ObservationPoint, ScatteringCenter

_C = 299792458.0
_F_REF = 10e9


# ---------------------------------------------------------------------------
# Frequency dependence models
# ---------------------------------------------------------------------------

def freq_dep_specular(f: float | np.ndarray, f_ref: float = _F_REF) -> complex | np.ndarray:
    """Specular: constant amplitude."""
    if np.isscalar(f):
        return 1.0 + 0j
    return np.ones_like(np.asarray(f), dtype=np.complex128)


def freq_dep_edge(f: float | np.ndarray, f_ref: float = _F_REF) -> complex | np.ndarray:
    """Edge diffraction: f^(-0.5) dependence."""
    fa = np.asarray(f, dtype=np.float64)
    result = (fa / f_ref) ** (-0.5)
    return result.astype(np.complex128) if fa.ndim > 0 else complex(result)


def freq_dep_cavity(
    f: float | np.ndarray, f0: float, Q: float
) -> complex | np.ndarray:
    """Lorentzian resonance for cavity, normalised so peak = 1.0 at f = f0.

    The base_amplitude of the ScatteringFeature is therefore the *peak* amplitude
    of the cavity. Off-resonance the function rolls off as 1/Q per normalised
    detuning, giving 3-dB bandwidth = f0/Q.
    """
    fa = np.asarray(f, dtype=np.float64)
    result = 1.0 / (1.0 + 1j * Q * (fa / f0 - f0 / fa))
    return result if fa.ndim > 0 else complex(result)


def freq_dep_creeping(
    f: float | np.ndarray,
    L: float,
    c: float = _C,
    alpha: float = 0.5,
) -> complex | np.ndarray:
    """Creeping wave: exponential decay with frequency."""
    fa = np.asarray(f, dtype=np.float64)
    result = np.exp(-alpha * fa * L / c).astype(np.complex128)
    return result if fa.ndim > 0 else complex(result)


# ---------------------------------------------------------------------------
# Angular pattern models
# ---------------------------------------------------------------------------

def _great_circle_dist(theta1, phi1, theta2, phi2):
    """Great-circle distance, scalar or array."""
    cos_d = (
        np.sin(theta1) * np.sin(theta2)
        + np.cos(theta1) * np.cos(theta2) * np.cos(phi1 - phi2)
    )
    cos_d = np.clip(cos_d, -1.0, 1.0)
    return np.arccos(cos_d)


def angular_gain_isotropic(theta, phi, center_theta, center_phi, width):
    return np.ones_like(np.asarray(theta, dtype=np.float64))


def angular_gain_specular_lobe(theta, phi, center_theta, center_phi, width):
    d = _great_circle_dist(theta, phi, center_theta, center_phi)
    sigma = width / 2.3548
    return np.exp(-(d**2) / (2.0 * sigma**2 + 1e-30))


def angular_gain_broad_lobe(theta, phi, center_theta, center_phi, width):
    d = _great_circle_dist(theta, phi, center_theta, center_phi)
    result = np.cos(d / 2.0) ** 2
    result = np.where(d < np.pi, result, 0.0)
    return result


def angular_gain_narrow_lobe(theta, phi, center_theta, center_phi, width):
    d = _great_circle_dist(theta, phi, center_theta, center_phi)
    x = d / (width + 1e-30)
    return np.sinc(x) ** 2  # np.sinc uses normalized sinc, so sinc(x) = sin(pi*x)/(pi*x)


_FREQ_DEP_FUNCS = {
    "specular": lambda f, feat: freq_dep_specular(f),
    "edge": lambda f, feat: freq_dep_edge(f),
    "cavity_resonant": lambda f, feat: freq_dep_cavity(f, feat.cavity_freq_hz, feat.cavity_q),
    "creeping": lambda f, feat: freq_dep_creeping(f, feat.lobe_width_rad),  # use width as L approx
}

_ANG_PATTERN_FUNCS = {
    "isotropic": angular_gain_isotropic,
    "specular_lobe": angular_gain_specular_lobe,
    "broad_lobe": angular_gain_broad_lobe,
    "narrow_lobe": angular_gain_narrow_lobe,
}


@dataclass
class ScatteringFeature:
    """A single physical scattering feature (edge, specular, cavity, etc.)."""
    x: float               # downrange (m)
    y: float               # crossrange (m)
    base_amplitude: complex # amplitude at reference frequency
    freq_dependence: str    # "specular", "edge", "cavity_resonant", "creeping"
    angular_pattern: str    # "isotropic", "specular_lobe", "broad_lobe", "narrow_lobe"
    z: float = 0.0            # out-of-plane position — scaffold for 3D; physics still 2D
    lobe_center_theta: float = 0.0   # radians
    lobe_center_phi: float = 0.0     # radians
    lobe_width_rad: float = 0.5      # 3dB beamwidth
    cavity_freq_hz: float = 0.0      # resonant frequency
    cavity_q: float = 50.0           # quality factor
    label: str = ""
    geometry_source: str = ""
    position_uncertainty_m: float = 0.0
    amplitude_uncertainty_db: float = 0.0
    freq_param_uncertainty: float = 0.0  # fractional uncertainty in f0/Q


class SyntheticScatterer:
    """A collection of scattering features defining a complete target."""

    def __init__(self, features: list[ScatteringFeature], characteristic_length: float):
        self.features = features
        self.characteristic_length = characteristic_length

    def compute_rcs(self, points: list[ObservationPoint]) -> ComplexRCS:
        """Compute exact complex RCS at given observation points."""
        N = len(points)
        values = np.zeros(N, dtype=np.complex128)

        theta_arr = np.array([p.theta for p in points], dtype=np.float64)
        phi_arr = np.array([p.phi for p in points], dtype=np.float64)
        freq_arr = np.array([p.freq_hz for p in points], dtype=np.float64)

        k0 = 2.0 * np.pi * freq_arr / _C  # wavenumber

        for feat in self.features:
            # Angular gain
            ang_func = _ANG_PATTERN_FUNCS.get(feat.angular_pattern, angular_gain_isotropic)
            G = ang_func(theta_arr, phi_arr, feat.lobe_center_theta, feat.lobe_center_phi, feat.lobe_width_rad)

            # Frequency dependence
            freq_func = _FREQ_DEP_FUNCS.get(feat.freq_dependence, lambda f, ft: np.ones_like(f, dtype=np.complex128))
            F = freq_func(freq_arr, feat)

            # Phase from position: 2D projection (z=0 plane).
            # TODO(3D): add feat.z * cos(elevation) term here once 3D measurement
            # geometry (roll_rad, full phi variation) is implemented.
            phase_term = np.exp(1j * 2.0 * k0 * (feat.x * np.cos(theta_arr) + feat.y * np.sin(theta_arr)))

            values += feat.base_amplitude * G * F * phase_term

        return ComplexRCS(observation_points=list(points), values=values)

    def get_scattering_centers(self) -> list[ScatteringCenter]:
        """Return ScatteringCenter objects from the feature list."""
        centers = []
        for feat in self.features:
            centers.append(ScatteringCenter(
                x=feat.x,
                y=feat.y,
                z=feat.z,
                amplitude=feat.base_amplitude,
                freq_dependence=feat.freq_dependence,
                angular_pattern=feat.angular_pattern,
                lobe_center_theta=feat.lobe_center_theta,
                lobe_center_phi=feat.lobe_center_phi,
                lobe_width_rad=feat.lobe_width_rad,
                cavity_freq_hz=feat.cavity_freq_hz,
                cavity_q=feat.cavity_q,
                label=feat.label,
                geometry_source=feat.geometry_source,
                position_uncertainty_m=feat.position_uncertainty_m,
            ))
        return centers
