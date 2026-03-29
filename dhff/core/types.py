from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np


class AngleUnit(Enum):
    DEGREES = auto()
    RADIANS = auto()


@dataclass(frozen=True)
class AspectAngle:
    """Single observation angle."""
    theta: float      # elevation, radians internally
    phi: float        # azimuth, radians internally
    roll_rad: float = 0.0  # roll angle — scaffold for 3D; physics still 2D


@dataclass(frozen=True)
class FrequencyPoint:
    """Single frequency."""
    freq_hz: float

    @property
    def wavelength(self) -> float:
        return 299792458.0 / self.freq_hz


@dataclass(frozen=True)
class ObservationPoint:
    """A single point in aspect-frequency space."""
    theta: float   # radians
    phi: float     # radians
    freq_hz: float


@dataclass
class ComplexRCS:
    """Complex-valued RCS at one or more observation points.

    Stores I + jQ (linear, NOT dB). Phase is preserved.
    magnitude is |sigma| in m^2. Phase is in radians.
    """
    observation_points: list[ObservationPoint]
    values: np.ndarray  # complex128, shape (N,)

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=np.complex128)
        assert len(self.observation_points) == len(self.values)

    @property
    def magnitude_dbsm(self) -> np.ndarray:
        return 10.0 * np.log10(np.abs(self.values) + 1e-30)

    @property
    def phase_rad(self) -> np.ndarray:
        return np.angle(self.values)


@dataclass
class ScatteringCenter:
    """A single scattering center in the ISAR image domain."""
    x: float           # downrange position (meters)
    y: float           # crossrange position (meters)
    z: float = 0.0     # out-of-plane position — scaffold for 3D; physics still 2D
    amplitude: complex = 0j  # complex amplitude
    freq_dependence: str = "specular"
    angular_pattern: str = "isotropic"
    lobe_center_theta: float = 0.0
    lobe_center_phi: float = 0.0
    lobe_width_rad: float = 0.5
    cavity_freq_hz: float = 0.0
    cavity_q: float = 50.0
    label: str = ""
    geometry_source: str = ""
    position_uncertainty_m: float = 0.0


class AnomalyType(Enum):
    UNMATCHED_MEASUREMENT = auto()   # Real feature missing from simulation
    UNMATCHED_SIMULATION = auto()    # Simulation artifact not in reality
    POSITION_SHIFT = auto()          # Matched but wrong location
    AMPLITUDE_DISCREPANCY = auto()   # Matched location, wrong strength


@dataclass
class ScatteringCenterAnomaly:
    """A detected disagreement between sim and meas scattering centers."""
    anomaly_type: AnomalyType
    meas_center: ScatteringCenter | None
    sim_center: ScatteringCenter | None
    position_error_m: float = 0.0
    amplitude_error_db: float = 0.0


@dataclass
class DiscrepancySample:
    """Measured discrepancy at a single observation point."""
    obs: ObservationPoint
    residual: complex    # meas - sim (complex)


@dataclass
class MeasurementPlan:
    """Ordered list of observation points to measure next."""
    points: list[ObservationPoint]
    scores: list[float]   # acquisition function value at each point
    rationale: list[str]  # human-readable reason for each selection
