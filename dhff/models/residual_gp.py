"""Module 5: Residual GP fitted to the parametric model residuals."""
from __future__ import annotations

import warnings

import numpy as np
import torch
import gpytorch

from dhff.core.types import DiscrepancySample, ObservationPoint


class _MultitaskGPModel(gpytorch.models.ExactGP):
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


class ResidualGP:
    """GP fitted to residuals after the parametric SC model."""

    def __init__(
        self,
        freq_range_hz: tuple[float, float],
        n_training_iters: int = 80,
        learning_rate: float = 0.1,
        seed: int | None = None,
    ):
        self.freq_range_hz = freq_range_hz
        self.n_training_iters = n_training_iters
        self.learning_rate = learning_rate
        self._seed = seed
        self.model = None
        self.likelihood = None
        self._is_fitted = False
        self._train_x = None
        self._large_variance = 10.0

    def _normalize_freq(self, freq_hz: np.ndarray) -> np.ndarray:
        f_min, f_max = self.freq_range_hz
        return (freq_hz - f_min) / (f_max - f_min + 1e-30)

    def _samples_to_tensors(self, samples: list[DiscrepancySample]):
        X = np.array([[s.obs.theta, s.obs.phi, s.obs.freq_hz] for s in samples], dtype=np.float64)
        X[:, 2] = self._normalize_freq(X[:, 2])
        Y = np.array([[s.residual.real, s.residual.imag] for s in samples], dtype=np.float64)
        return torch.tensor(X, dtype=torch.float64), torch.tensor(Y, dtype=torch.float64)

    def fit(self, residual_samples: list[DiscrepancySample]) -> None:
        """Fit the GP to residual discrepancy samples."""
        if len(residual_samples) < 5:
            self._is_fitted = False
            return

        train_x, train_y = self._samples_to_tensors(residual_samples)
        self._train_x = train_x

        if self._seed is not None:
            torch.manual_seed(self._seed)

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
        model = _MultitaskGPModel(train_x, train_y, likelihood)

        model = model.double()
        likelihood = likelihood.double()

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        try:
            for _ in range(self.n_training_iters):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                if torch.isnan(loss):
                    raise ValueError("NaN loss during GP training")
                optimizer.step()
        except Exception as e:
            warnings.warn(f"GP training issue: {e}. Using prior-only model.")
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
        """Predict residual at new points."""
        N = len(points)
        if not self._is_fitted:
            return np.zeros(N, dtype=np.complex128), np.ones(N) * self._large_variance

        test_x = self._points_to_tensor(points)
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(test_x))
            mean = pred.mean.numpy()  # (N, 2)
            var = pred.variance.numpy()  # (N, 2)

        mean_complex = mean[:, 0] + 1j * mean[:, 1]
        variance = var[:, 0] + var[:, 1]
        return mean_complex.astype(np.complex128), variance.astype(np.float64)

    def fast_variance(self, points: list[ObservationPoint]) -> np.ndarray:
        """Compute only the variance."""
        _, var = self.predict(points)
        return var
