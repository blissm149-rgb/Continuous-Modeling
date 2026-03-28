"""Module 5: Random Fourier Feature approximation for fast acquisition."""
from __future__ import annotations

import warnings

import numpy as np

from dhff.core.types import DiscrepancySample, ObservationPoint


class RandomFourierFeatureApproximation:
    """Approximate the residual GP posterior using Random Fourier Features."""

    def __init__(self, n_features: int = 500, seed: int = 42):
        self.n_features = n_features
        self.seed = seed
        self._W = None
        self._b = None
        self._alpha = None
        self._A_inv = None
        self._sigma2 = None
        self._freq_range_hz = None
        self._is_fitted = False

    def _normalize_freq(self, freq_hz: np.ndarray) -> np.ndarray:
        if self._freq_range_hz is None:
            return freq_hz
        f_min, f_max = self._freq_range_hz
        return (freq_hz - f_min) / (f_max - f_min + 1e-30)

    def _phi(self, X: np.ndarray) -> np.ndarray:
        """Compute RFF feature matrix: (N, D)."""
        z = X @ self._W.T + self._b[np.newaxis, :]  # (N, D)
        scale = np.sqrt(2.0 * self._output_scale / self.n_features)
        return scale * np.cos(z)

    def fit_from_gp(self, gp, samples: list[DiscrepancySample]) -> None:
        """Build the RFF approximation from a fitted GP."""
        if not gp._is_fitted or gp.model is None:
            self._is_fitted = False
            return

        try:
            import torch
            self._freq_range_hz = gp.freq_range_hz

            # Extract hyperparameters
            ls = gp.model.covar_module.data_covar_module.base_kernel.lengthscale.detach().numpy().flatten()
            outputscale = float(gp.model.covar_module.data_covar_module.outputscale.detach().numpy())
            noise = float(gp.likelihood.noise.detach().numpy().item())

            self._output_scale = outputscale
            self._sigma2 = noise + 1e-6
            D = self.n_features
            rng = np.random.default_rng(self.seed)

            # Sample W from spectral density of RBF (Matern 5/2 approximated by RBF for RFF)
            ls_safe = np.maximum(ls, 1e-6)
            self._W = rng.standard_normal((D, 3)) / (ls_safe + 1e-30)
            self._b = rng.uniform(0, 2 * np.pi, D)

            # Build feature matrix for training data
            X = np.array([[s.obs.theta, s.obs.phi, s.obs.freq_hz] for s in samples], dtype=np.float64)
            X[:, 2] = self._normalize_freq(X[:, 2])
            Phi = self._phi(X)  # (N, D)

            # Average real and imag targets
            y_real = np.array([s.residual.real for s in samples])
            y_imag = np.array([s.residual.imag for s in samples])
            y = (y_real + y_imag) / 2.0  # simplified scalar target for variance

            # Ridge regression
            N = len(samples)
            A = Phi.T @ Phi + self._sigma2 * np.eye(D)
            try:
                self._A_inv = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                self._A_inv = np.linalg.pinv(A)
            self._alpha = self._A_inv @ Phi.T @ y

            self._is_fitted = True
        except Exception as e:
            warnings.warn(f"RFF fit_from_gp failed: {e}")
            self._is_fitted = False

    def predict_variance(self, points: list[ObservationPoint]) -> np.ndarray:
        """Fast variance prediction for acquisition."""
        N = len(points)
        if not self._is_fitted:
            return np.ones(N)

        X = np.array([[p.theta, p.phi, p.freq_hz] for p in points], dtype=np.float64)
        X[:, 2] = self._normalize_freq(X[:, 2])
        Phi = self._phi(X)  # (N, D)

        # Var(f(x)) ≈ sigma^2 * (1 + phi(x)^T @ A_inv @ phi(x))
        qf = np.einsum('nd,de,ne->n', Phi, self._A_inv, Phi)  # (N,)
        variance = self._sigma2 * (1.0 + qf)
        return np.maximum(variance, 0.0)
