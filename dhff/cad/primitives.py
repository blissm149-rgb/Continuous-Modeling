"""CAD geometry primitive dataclasses.

These represent the types of geometric features that CAD tools (CATIA, SolidWorks)
and EM simulators (FEKO, CST, HFSS) expose for scattering parameter estimation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_C = 299_792_458.0  # speed of light (m/s)


@dataclass(frozen=True)
class FlatPanel:
    """A flat rectangular panel — primary specular reflector.

    Maps to Physical Optics (PO) scattering model.
    """
    x: float
    y: float
    width_m: float
    height_m: float
    normal_theta_rad: float        # surface normal direction (elevation)
    normal_phi_rad: float = 0.0    # surface normal direction (azimuth)
    label: str = ""
    manufacturing_tolerance_m: float = 0.001


@dataclass(frozen=True)
class EdgeSegment:
    """A sharp edge or corner — diffracts energy broadly.

    Maps to GTD/UTD edge diffraction model.
    """
    x: float
    y: float
    length_m: float
    edge_theta_rad: float                         # edge orientation
    interior_half_angle_rad: float = math.pi / 2  # wedge half-angle
    label: str = ""
    manufacturing_tolerance_m: float = 0.001


@dataclass(frozen=True)
class CavityVolume:
    """A cavity opening — produces resonant scattering.

    Maps to cavity resonance (TE101 dominant mode) model.
    Highest uncertainty of all primitive types.

    cavity_q_override: if set, bypasses the radiation-loss Q formula and uses
        this value directly (useful when measured or estimated Q is available).
    """
    x: float
    y: float
    interior_dim_a_m: float   # longer transverse dimension (m)
    interior_dim_b_m: float   # shorter transverse dimension (m)
    depth_m: float
    aperture_area_m2: float
    label: str = ""
    manufacturing_tolerance_m: float = 0.001
    cavity_q_override: float | None = None


@dataclass(frozen=True)
class ConvexSurface:
    """A convex curved surface — supports creeping wave propagation.

    Maps to creeping wave / Fock function model.
    """
    x: float
    y: float
    radius_m: float
    arc_length_m: float
    surface_theta_rad: float  # surface orientation
    label: str = ""
    manufacturing_tolerance_m: float = 0.001


# Union type for type hints
CadPrimitive = FlatPanel | EdgeSegment | CavityVolume | ConvexSurface
