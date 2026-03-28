from __future__ import annotations

import math

import numpy as np

from .types import AspectAngle, ObservationPoint


def deg2rad(theta_deg: float, phi_deg: float) -> tuple[float, float]:
    return math.radians(theta_deg), math.radians(phi_deg)


def rad2deg(theta_rad: float, phi_rad: float) -> tuple[float, float]:
    return math.degrees(theta_rad), math.degrees(phi_rad)


def angular_distance(a1: AspectAngle, a2: AspectAngle) -> float:
    """Great-circle distance on the unit sphere between two aspect angles. Returns radians."""
    cos_d = (
        math.sin(a1.theta) * math.sin(a2.theta)
        + math.cos(a1.theta) * math.cos(a2.theta) * math.cos(a1.phi - a2.phi)
    )
    cos_d = max(-1.0, min(1.0, cos_d))
    return math.acos(cos_d)


def angular_distance_points(p1: ObservationPoint, p2: ObservationPoint) -> float:
    """Great-circle distance between two observation points (ignoring frequency)."""
    a1 = AspectAngle(theta=p1.theta, phi=p1.phi)
    a2 = AspectAngle(theta=p2.theta, phi=p2.phi)
    return angular_distance(a1, a2)


def make_observation_grid(
    theta_range: tuple[float, float],
    phi_range: tuple[float, float],
    freq_range: tuple[float, float],
    n_theta: int,
    n_phi: int,
    n_freq: int,
) -> list[ObservationPoint]:
    """Create a regular grid in aspect-frequency space."""
    thetas = np.linspace(theta_range[0], theta_range[1], n_theta)
    phis = np.linspace(phi_range[0], phi_range[1], n_phi)
    freqs = np.linspace(freq_range[0], freq_range[1], n_freq)
    points = []
    for t in thetas:
        for p in phis:
            for f in freqs:
                points.append(ObservationPoint(theta=float(t), phi=float(p), freq_hz=float(f)))
    return points


def make_frequency_sweep(
    theta: float,
    phi: float,
    freq_range: tuple[float, float],
    n_freq: int,
) -> list[ObservationPoint]:
    """Create a frequency sweep at a fixed angle."""
    freqs = np.linspace(freq_range[0], freq_range[1], n_freq)
    return [ObservationPoint(theta=theta, phi=phi, freq_hz=float(f)) for f in freqs]


def make_angular_sweep(
    theta_range: tuple[float, float],
    phi: float,
    freq_hz: float,
    n_theta: int,
) -> list[ObservationPoint]:
    """Create an angular sweep at a fixed frequency and phi."""
    thetas = np.linspace(theta_range[0], theta_range[1], n_theta)
    return [ObservationPoint(theta=float(t), phi=phi, freq_hz=freq_hz) for t in thetas]


def observation_points_to_array(points: list[ObservationPoint]) -> np.ndarray:
    """Convert list of ObservationPoints to (N, 3) array [theta, phi, freq_hz]."""
    if not points:
        return np.zeros((0, 3))
    arr = np.array([[p.theta, p.phi, p.freq_hz] for p in points], dtype=np.float64)
    return arr
