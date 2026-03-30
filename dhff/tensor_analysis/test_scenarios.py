"""Physically realistic tensor test scenarios for the sensitivity analyzer suite.

All scenarios are fully vectorised (no Python loops), work from 11×3×10 to
180×36×100 tensors, and use only numpy + standard physics formulas.

Each factory function returns a complex128 ndarray of shape
(N_az, N_el, N_freq).

Available scenarios
-------------------
TensorScenarioFactory.point_scatterer      — single analytic scatterer
TensorScenarioFactory.two_scatterers       — two-point interference
TensorScenarioFactory.extended_scatterer   — PO flat-plate sinc pattern
TensorScenarioFactory.dihedral             — double-bounce + two single-bounce
TensorScenarioFactory.cavity_on_background — Lorentzian + specular background
TensorScenarioFactory.creeping_wave        — Keller GTD creeping return
TensorScenarioFactory.fss_coating          — Fabry-Perot multi-layer coating
TensorScenarioFactory.add_noise            — add complex Gaussian noise
"""
from __future__ import annotations

import numpy as np


class TensorScenarioFactory:
    """Factory for physically realistic complex RCS test tensors."""

    _C = 3e8   # speed of light (m/s)

    # ------------------------------------------------------------------
    # Default coordinate grids (used when callers pass None)
    # ------------------------------------------------------------------

    @staticmethod
    def default_az(N: int = 21) -> np.ndarray:
        return np.linspace(0.1, np.pi - 0.1, N)

    @staticmethod
    def default_el(N: int = 5) -> np.ndarray:
        return np.linspace(-0.25, 0.25, N)

    @staticmethod
    def default_freq(N: int = 20) -> np.ndarray:
        return np.linspace(8e9, 12e9, N)

    # ------------------------------------------------------------------
    # Scenario constructors
    # ------------------------------------------------------------------

    @classmethod
    def point_scatterer(
        cls,
        x_m: float = 0.2,
        y_m: float = 0.0,
        amplitude: complex = 1.0 + 0j,
        az: np.ndarray | None = None,
        el: np.ndarray | None = None,
        freq: np.ndarray | None = None,
    ) -> np.ndarray:
        """Single analytic point scatterer at position (x, y) metres.

        S(az, el, f) = amplitude × exp(2j k₀ (x cos(az) + y sin(az)))
        """
        az   = az   if az   is not None else cls.default_az()
        el   = el   if el   is not None else cls.default_el()
        freq = freq if freq is not None else cls.default_freq()

        k0 = 2.0 * np.pi * freq[None, None, :] / cls._C
        az3 = az[:, None, None]
        phase = 2.0 * k0 * (x_m * np.cos(az3) + y_m * np.sin(az3))
        return amplitude * np.ones((len(el),))[None, :, None] * np.exp(1j * phase)

    @classmethod
    def two_scatterers(
        cls,
        x1: float = 0.2, y1: float = 0.0,
        x2: float = -0.1, y2: float = 0.15,
        amp1: complex = 1.0 + 0j,
        amp2: complex = 0.8 + 0j,
        az: np.ndarray | None = None,
        el: np.ndarray | None = None,
        freq: np.ndarray | None = None,
    ) -> np.ndarray:
        """Two-point scatterer interference tensor."""
        az   = az   if az   is not None else cls.default_az()
        el   = el   if el   is not None else cls.default_el()
        freq = freq if freq is not None else cls.default_freq()

        S1 = cls.point_scatterer(x1, y1, amp1, az, el, freq)
        S2 = cls.point_scatterer(x2, y2, amp2, az, el, freq)
        return S1 + S2

    @classmethod
    def extended_scatterer(
        cls,
        L_m: float = 0.3,
        az_boresight: float | None = None,
        r_ctr: float = 0.3,
        amplitude: float = 1.0,
        az: np.ndarray | None = None,
        el: np.ndarray | None = None,
        freq: np.ndarray | None = None,
    ) -> np.ndarray:
        """Physical Optics flat-plate of length L (sinc-pattern RCS).

        S = A · cos(el) · sinc(k L sin(az - az_boresight) / π) · exp(2jk r_ctr)
        """
        az   = az   if az   is not None else cls.default_az()
        el   = el   if el   is not None else cls.default_el()
        freq = freq if freq is not None else cls.default_freq()
        if az_boresight is None:
            az_boresight = float(np.mean(az))

        k    = 2.0 * np.pi * freq[None, None, :] / cls._C
        az3  = az[:, None, None]
        el3  = el[None, :, None]
        sinc_arg = k * L_m * np.sin(az3 - az_boresight)
        return (
            amplitude
            * np.cos(el3)
            * np.sinc(sinc_arg / np.pi)
            * np.exp(2j * k * r_ctr)
        )

    @classmethod
    def dihedral(
        cls,
        opening_half_angle: float = np.pi / 4,
        L_m: float = 0.15,
        r_dihedral: float = 0.3,
        az: np.ndarray | None = None,
        el: np.ndarray | None = None,
        freq: np.ndarray | None = None,
    ) -> np.ndarray:
        """Dihedral corner reflector: two PO single-bounce returns + double bounce.

        The double-bounce amplitude is 1.5× the single-bounce amplitude.
        """
        az   = az   if az   is not None else cls.default_az()
        el   = el   if el   is not None else cls.default_el()
        freq = freq if freq is not None else cls.default_freq()

        az3 = az[:, None, None]
        el3 = el[None, :, None]
        k   = 2.0 * np.pi * freq[None, None, :] / cls._C

        az1 = az3 - opening_half_angle
        az2 = az3 + opening_half_angle

        S1 = (np.cos(el3)
              * np.sinc(k * L_m * np.sin(az1) / np.pi)
              * np.exp(2j * k * r_dihedral * np.cos(az1)))
        S2 = (np.cos(el3)
              * np.sinc(k * L_m * np.sin(az2) / np.pi)
              * np.exp(2j * k * r_dihedral * np.cos(az2)))
        # Double-bounce: peaks at az ≈ 0 (looking into corner)
        S_dbl = (1.5 * np.cos(el3) ** 2
                 * np.sinc(k * L_m * np.sin(az3) / np.pi)
                 * np.exp(4j * k * r_dihedral * np.cos(opening_half_angle)))
        return S1 + S2 + S_dbl

    @classmethod
    def cavity_on_background(
        cls,
        f0_hz: float = 10e9,
        Q: float = 60.0,
        cavity_amp: float = 0.3,
        specular_amp: float = 1.0,
        r_spec: float = 0.0,
        r_cav: float = 0.2,
        az: np.ndarray | None = None,
        el: np.ndarray | None = None,
        freq: np.ndarray | None = None,
    ) -> np.ndarray:
        """Lorentzian cavity return on top of a broadband specular background.

        S = A_spec · cos(el) · exp(2jk r_spec)
          + A_cav  · L(f, f₀, Q) · exp(2jk r_cav)

        L(f) = 1 / (1 + jQ(f/f₀ − f₀/f))
        """
        az   = az   if az   is not None else cls.default_az()
        el   = el   if el   is not None else cls.default_el()
        freq = freq if freq is not None else cls.default_freq()

        k   = 2.0 * np.pi * freq[None, None, :] / cls._C
        el3 = el[None, :, None]
        freq3 = freq[None, None, :]

        lor = 1.0 / (1.0 + 1j * Q * (freq3 / f0_hz - f0_hz / freq3))
        S_spec = specular_amp * np.cos(el3) * np.exp(2j * k * r_spec)
        S_cav  = cavity_amp   * lor         * np.exp(2j * k * r_cav)
        return S_spec + S_cav

    @classmethod
    def creeping_wave(
        cls,
        a_sphere_m: float = 0.05,
        specular_amp: float = 1.0,
        creeping_amp_0: float = 0.3,
        az: np.ndarray | None = None,
        el: np.ndarray | None = None,
        freq: np.ndarray | None = None,
    ) -> np.ndarray:
        """Creeping wave on a sphere of radius a (Keller GTD approximation).

        S = A_spec · cos(el) · exp(2jk a)
          + A_creep · cos(el) · exp(-α f πa/c) · exp(2jk(a + πa))
        """
        az   = az   if az   is not None else cls.default_az()
        el   = el   if el   is not None else cls.default_el()
        freq = freq if freq is not None else cls.default_freq()

        k    = 2.0 * np.pi * freq[None, None, :] / cls._C
        el3  = el[None, :, None]
        f3   = freq[None, None, :]

        alpha   = 0.5 * f3 * a_sphere_m / cls._C
        delta_r = np.pi * a_sphere_m

        S_spec  = specular_amp   * np.cos(el3) * np.exp(2j * k * a_sphere_m)
        S_creep = (creeping_amp_0 * np.cos(el3)
                   * np.exp(-alpha)
                   * np.exp(2j * k * (a_sphere_m + delta_r)))
        return S_spec + S_creep

    @classmethod
    def fss_coating(
        cls,
        d_m: float = 0.01,
        n_refrac: float = 2.0,
        r0_m: float = 0.3,
        base_amp: float = 1.0,
        r_coeff: float = 0.3,
        az: np.ndarray | None = None,
        el: np.ndarray | None = None,
        freq: np.ndarray | None = None,
    ) -> np.ndarray:
        """Multi-layer Fabry-Perot coating (frequency-selective surface).

        T(f) = (1 − r²) / (1 − r² · exp(2jδ))
        δ(f) = 2π f · 2 n d / c

        For d=1 cm, n=2: period ≈ 7.5 GHz → one full FSS cycle in 8–12 GHz.
        """
        az   = az   if az   is not None else cls.default_az()
        el   = el   if el   is not None else cls.default_el()
        freq = freq if freq is not None else cls.default_freq()

        k    = 2.0 * np.pi * freq[None, None, :] / cls._C
        el3  = el[None, :, None]
        f3   = freq[None, None, :]

        delta_phi = 2.0 * np.pi * f3 * 2.0 * n_refrac * d_m / cls._C
        T_fss     = (1.0 - r_coeff ** 2) / (1.0 - r_coeff ** 2 * np.exp(2j * delta_phi))
        return base_amp * np.cos(el3) * T_fss * np.exp(2j * k * r0_m)

    # ------------------------------------------------------------------
    # Noise utility
    # ------------------------------------------------------------------

    @staticmethod
    def add_noise(
        tensor: np.ndarray,
        snr_db: float = 25.0,
        seed: int = 42,
    ) -> np.ndarray:
        """Add complex Gaussian noise at the specified SNR level."""
        rng          = np.random.default_rng(seed)
        sig_power    = float(np.mean(np.abs(tensor) ** 2))
        noise_power  = sig_power / (10.0 ** (snr_db / 10.0))
        noise        = (rng.standard_normal(tensor.shape)
                        + 1j * rng.standard_normal(tensor.shape))
        noise       *= np.sqrt(noise_power / 2.0)
        return tensor + noise
