"""Tests for Module 1: Core Data Types."""
import math

import numpy as np
import pytest

from dhff.core import (
    AspectAngle, ObservationPoint, ComplexRCS,
    angular_distance, angular_distance_points,
    make_observation_grid, make_frequency_sweep, make_angular_sweep,
    complex_to_mag_phase, mag_phase_to_complex,
    dbsm_to_linear, linear_to_dbsm,
)


def test_complex_mag_phase_roundtrip():
    rng = np.random.default_rng(42)
    values = rng.standard_normal(50) + 1j * rng.standard_normal(50)
    mag, phase = complex_to_mag_phase(values)
    recovered = mag_phase_to_complex(mag, phase)
    np.testing.assert_allclose(recovered.real, values.real, atol=1e-12)
    np.testing.assert_allclose(recovered.imag, values.imag, atol=1e-12)


def test_dbsm_linear_roundtrip():
    dbsm_orig = np.array([-30.0, -10.0, 0.0, 10.0, 20.0])
    linear = dbsm_to_linear(dbsm_orig)
    dbsm_recovered = linear_to_dbsm(linear)
    np.testing.assert_allclose(dbsm_recovered, dbsm_orig, atol=1e-10)


def test_angular_distance_symmetric():
    a1 = AspectAngle(theta=0.5, phi=0.3)
    a2 = AspectAngle(theta=1.2, phi=-0.4)
    assert abs(angular_distance(a1, a2) - angular_distance(a2, a1)) < 1e-12


def test_angular_distance_zero_for_identical():
    a = AspectAngle(theta=1.0, phi=0.5)
    assert angular_distance(a, a) < 1e-12


def test_angular_distance_pi_for_antipodal():
    a1 = AspectAngle(theta=0.0, phi=0.0)
    a2 = AspectAngle(theta=math.pi, phi=0.0)
    d = angular_distance(a1, a2)
    assert abs(d - math.pi) < 1e-10


def test_observation_grid_count():
    points = make_observation_grid(
        theta_range=(0.1, 1.5), phi_range=(0.0, 0.0),
        freq_range=(8e9, 12e9), n_theta=5, n_phi=1, n_freq=4,
    )
    assert len(points) == 5 * 1 * 4


def test_frequency_sweep_same_angle():
    theta, phi = 0.7, 0.0
    points = make_frequency_sweep(theta, phi, (8e9, 12e9), 20)
    assert len(points) == 20
    for p in points:
        assert abs(p.theta - theta) < 1e-12
        assert abs(p.phi - phi) < 1e-12


def test_angular_sweep_same_freq():
    freq = 10e9
    points = make_angular_sweep((0.1, 1.5), 0.0, freq, 15)
    assert len(points) == 15
    for p in points:
        assert abs(p.freq_hz - freq) < 1e-6


def test_complex_rcs_properties():
    pts = [ObservationPoint(0.5, 0.0, 10e9), ObservationPoint(1.0, 0.0, 10e9)]
    vals = np.array([1.0 + 0j, 0.5 + 0.5j], dtype=np.complex128)
    rcs = ComplexRCS(observation_points=pts, values=vals)
    assert rcs.magnitude_dbsm.shape == (2,)
    assert rcs.phase_rad.shape == (2,)
