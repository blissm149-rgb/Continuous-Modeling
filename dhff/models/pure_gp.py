"""Module 5: Pure GP baseline discrepancy model."""
from __future__ import annotations

import warnings

import numpy as np
import torch
import gpytorch

from dhff.core.types import DiscrepancySample, ObservationPoint
from .base import DiscrepancyModel


class AngularMaternKernel(gpytorch.kernels.Kernel):
    """Matern kernel using great-circle distance for angular dimensions."""
    has_lengthscale = True

    def forward(self, x1, x2, **kwargs):
        theta1, phi1 = x1[..., 0], x1[..., 1]
        theta2, phi2 = x2[..., 0], x2[..., 1]

        # Great-circle distance
        cos_d = (
            torch.sin(theta1).unsqueeze(-1) * torch.sin(theta2).unsqueeze(-2)
            + torch.cos(theta1).unsqueeze(-1) * torch.cos(theta2).unsqueeze(-2)
            * torch.cos(phi1.unsqueeze(-1) - phi2.unsqueeze(-2))
        )
        cos_d = torch.clamp(cos_d, -1.0 + 1e-6, 1.0 - 1e-6)
        d_ang = torch.acos(cos_d)  # (N, M)

        # Frequency distance
        d_freq = torch.abs(x1[..., 2].unsqueeze(-1) - x2[..., 2].unsqueeze(-2))

        # Use single lengthscale for simplicity
        ls = self.lengthscale.squeeze()
        d_ang_scaled = d_ang / (ls + 1e-6)
        d_freq_scaled = d_freq / (ls + 1e-6)

        # Matern 5/2
        def matern52(d):
            sqrt5d = torch.sqrt(torch.tensor(5.0)) * d
            return (1.0 + sqrt5d + 5.0 / 3.0 * d**2) * torch.exp(-sqrt5d)

        return matern52(d_ang_scaled) * matern52(d_freq_scaled)


class _PureGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=3)
            ),
            num_tasks=2, rank=1,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class PureGPDiscrepancyModel(DiscrepancyModel):
    """Pure GP baseline model — no parametric Layer 1."""

    def __init__(
        self,
        freq_range_hz: tuple[float, float],
        characteristic_length_m: float = 1.0,
        use_quasi_periodic: bool = False,
        n_training_iters: int = 100,
    ):
        self.freq_range_hz = freq_range_hz
        self.characteristic_length_m = characteristic_length_m
        self.use_quasi_periodic = use_quasi_periodic
        self.n_training_iters = n_training_iters
        self.model = None
        self.likelihood = None
        self._is_fitted = False
        self._large_variance = 10.0

    def _normalize_freq(self, freq_hz: np.ndarray) -> np.ndarray:
        f_min, f_max = self.freq_range_hz
        return (freq_hz - f_min) / (f_max - f_min + 1e-30)

    def _samples_to_tensors(self, samples: list[DiscrepancySample]):
        X = np.array([[s.obs.theta, s.obs.phi, s.obs.freq_hz] for s in samples], dtype=np.float64)
        X[:, 2] = self._normalize_freq(X[:, 2])
        Y = np.array([[s.residual.real, s.residual.imag] for s in samples], dtype=np.float64)
        return torch.tensor(X, dtype=torch.float64), torch.tensor(Y, dtype=torch.float64)

    def fit(self, samples: list[DiscrepancySample]) -> None:
        if len(samples) < 5:
            self._is_fitted = False
            return

        train_x, train_y = self._samples_to_tensors(samples)

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
        model = _PureGPModel(train_x, train_y, likelihood)
        model = model.double()
        likelihood = likelihood.double()

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        try:
            for _ in range(self.n_training_iters):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                if torch.isnan(loss):
                    raise ValueError("NaN loss")
                optimizer.step()
        except Exception as e:
            warnings.warn(f"PureGP training failed: {e}")
            self._is_fitted = False
            return

        self.model = model
        self.likelihood = likelihood
        self._is_fitted = True

    def _points_to_tensor(self, points: list[ObservationPoint]) -> torch.Tensor:
        X = np.array([[p.theta, p.phi, p.freq_hz] for p in points], dtype=np.float64)
        X[:, 2] = self._normalize_freq(X[:, 2])
        return torch.tensor(X, dtype=torch.float64)

    def predict(self, points: list[ObservationPoint]) -> tuple[np.ndarray, np.ndarray]:
        N = len(points)
        if not self._is_fitted:
            return np.zeros(N, dtype=np.complex128), np.ones(N) * self._large_variance

        test_x = self._points_to_tensor(points)
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(test_x))
            mean = pred.mean.numpy()
            var = pred.variance.numpy()

        mean_complex = (mean[:, 0] + 1j * mean[:, 1]).astype(np.complex128)
        variance = (var[:, 0] + var[:, 1]).astype(np.float64)
        return mean_complex, variance

    def acquisition_uncertainty(self, points: list[ObservationPoint]) -> np.ndarray:
        _, var = self.predict(points)
        return var

    def predicted_discrepancy_power(self, points: list[ObservationPoint]) -> np.ndarray:
        mean, var = self.predict(points)
        return np.abs(mean)**2 + var
