"""Module 5: Hybrid Parametric SC + Residual GP discrepancy model."""
from __future__ import annotations

import numpy as np

from dhff.core.types import DiscrepancySample, ObservationPoint, ScatteringCenter
from .base import DiscrepancyModel
from .residual_gp import ResidualGP
from .rff_approximation import RandomFourierFeatureApproximation
from dhff.scattering_center.parametric_model import ParametricSCModel


class HybridDiscrepancyModel(DiscrepancyModel):
    """Two-layer hybrid discrepancy model: SC + Residual GP."""

    def __init__(
        self,
        freq_range_hz: tuple[float, float],
        max_sc_centers: int = 15,
        n_ensemble: int = 5,
        rff_features: int = 500,
        gp_training_iters: int = 80,
    ):
        self.freq_range_hz = freq_range_hz
        self.max_sc_centers = max_sc_centers
        self.n_ensemble = n_ensemble
        self.rff_features = rff_features
        self.gp_training_iters = gp_training_iters

        self.sc_model = ParametricSCModel(max_centers=max_sc_centers)
        self.sc_ensemble: list[ParametricSCModel] = []
        self.residual_gp = ResidualGP(freq_range_hz=freq_range_hz, n_training_iters=gp_training_iters)
        self.rff = RandomFourierFeatureApproximation(n_features=rff_features)
        self._is_fitted = False
        self._samples: list[DiscrepancySample] = []

    def fit(self, samples: list[DiscrepancySample]) -> None:
        """Fit the full hybrid model."""
        self._samples = samples

        # 1. Fit primary SC model
        self.sc_model.fit(samples, self.freq_range_hz)

        # 2. Check SC model quality: only use it if it explains variance
        sc_useful = False
        if self.sc_model.get_center_count() > 0 and len(samples) > 0:
            pts = [s.obs for s in samples]
            sc_pred = self.sc_model.predict(pts)
            disc_vals = np.array([s.residual for s in samples])
            sc_resids = disc_vals - sc_pred
            ss_before = float(np.mean(np.abs(disc_vals) ** 2))
            ss_after = float(np.mean(np.abs(sc_resids) ** 2))
            sc_r2 = 1.0 - ss_after / (ss_before + 1e-30)
            if sc_r2 >= 0.20:  # SC explains at least 20% of variance
                sc_useful = True
            else:
                # SC model is not helping; discard its centers
                self.sc_model.centers = []

        # 3. Build SC ensemble (only if SC is useful)
        rng = np.random.default_rng(99)
        self.sc_ensemble = []
        if sc_useful:
            for i in range(self.n_ensemble):
                mc_delta = int(rng.integers(-2, 3))
                thresh_delta = float(rng.uniform(-3, 3))
                mc = max(1, self.max_sc_centers + mc_delta)
                thresh = self.sc_model.amplitude_threshold_db + thresh_delta

                # Bootstrap: subsample 80%
                if len(samples) >= 10:
                    n_sub = max(10, int(0.8 * len(samples)))
                    idx = rng.choice(len(samples), size=n_sub, replace=False)
                    sub_samples = [samples[j] for j in idx]
                else:
                    sub_samples = samples

                m = ParametricSCModel(max_centers=mc, amplitude_threshold_db=thresh)
                m.fit(sub_samples, self.freq_range_hz)
                self.sc_ensemble.append(m)

        # 4. Compute residuals from primary SC model (or use original samples if SC not useful)
        residual_samples = self.sc_model.residuals(samples) if sc_useful else samples

        # 5. Fit residual GP
        self.residual_gp.fit(residual_samples)

        # 6. Build RFF approximation
        if self.residual_gp._is_fitted:
            self.rff.fit_from_gp(self.residual_gp, residual_samples)

        self._is_fitted = True

    def predict(self, points: list[ObservationPoint]) -> tuple[np.ndarray, np.ndarray]:
        """Predict discrepancy using both layers.

        If the SC model found no centers, we skip adding the GP correction to avoid
        adding noise from a GP trained on insufficiently-characterised residuals.
        """
        sc_mean = self.sc_model.predict(points)
        sc_count = self.sc_model.get_center_count()

        # Only use GP correction if SC model has at least 1 confirmed center and
        # the residual GP was successfully fitted.
        if sc_count > 0 and self.residual_gp._is_fitted:
            gp_mean, gp_var = self.residual_gp.predict(points)
        else:
            gp_mean = np.zeros(len(points), dtype=np.complex128)
            gp_var = np.ones(len(points)) * self.residual_gp._large_variance

        mean = sc_mean + gp_mean

        # SC ensemble variance
        ens_var = self._sc_ensemble_variance(points)
        variance = ens_var + np.maximum(gp_var, 0.0)

        return mean.astype(np.complex128), variance.astype(np.float64)

    def _sc_ensemble_variance(self, points: list[ObservationPoint]) -> np.ndarray:
        """Compute variance across the SC ensemble."""
        if not self.sc_ensemble:
            return np.zeros(len(points))
        preds = np.array([m.predict(points) for m in self.sc_ensemble])  # (n_ens, N)
        return np.var(np.abs(preds), axis=0).astype(np.float64)

    def acquisition_uncertainty(self, points: list[ObservationPoint]) -> np.ndarray:
        """Fast uncertainty: RFF + SC ensemble variance."""
        rff_var = self.rff.predict_variance(points) if self.rff._is_fitted else np.ones(len(points))
        ens_var = self._sc_ensemble_variance(points)
        return np.maximum(rff_var + ens_var, 0.0)

    def predicted_discrepancy_power(self, points: list[ObservationPoint]) -> np.ndarray:
        """E[|delta|^2] = |E[delta]|^2 + Var[delta]."""
        mean, var = self.predict(points)
        return np.abs(mean)**2 + var

    def get_parametric_centers(self) -> list[ScatteringCenter]:
        return self.sc_model.to_scattering_centers()
