"""Module 3: CAD feature extraction and susceptibility prediction."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dhff.core.types import AspectAngle, ObservationPoint
from dhff.core.coordinate_system import angular_distance


@dataclass
class GeometricFeaturePrior:
    """A geometric feature extracted from the CAD model with its predicted
    region of influence in aspect-frequency space."""
    feature_type: str
    angular_region_center: AspectAngle
    angular_region_width_rad: float
    freq_range_hz: tuple[float, float]
    uncertainty_score: float       # 0-1
    description: str


class GeometricFeatureAnalyzer:
    """Extract geometric features from a SyntheticScatterer's feature list."""

    UNCERTAINTY_BY_FREQ_DEP = {
        "specular": 0.1,
        "edge": 0.4,
        "cavity_resonant": 0.8,
        "creeping": 0.7,
    }

    UNCERTAINTY_BY_ANGULAR = {
        "isotropic": 0.3,
        "specular_lobe": 0.1,
        "broad_lobe": 0.3,
        "narrow_lobe": 0.5,
    }

    def __init__(self, simulator, freq_range_hz: tuple[float, float] = (8e9, 12e9)):
        self.simulator = simulator
        self.scatterer = simulator.degraded_scatterer
        self.freq_range_hz = freq_range_hz
        self._features: list[GeometricFeaturePrior] | None = None

    def extract_features(self) -> list[GeometricFeaturePrior]:
        """Return a list of geometric feature priors."""
        priors = []
        feature_centers = []

        for feat in self.scatterer.features:
            freq_dep = feat.freq_dependence
            ang_pat = feat.angular_pattern

            # Determine frequency range
            if freq_dep == "cavity_resonant":
                f0 = feat.cavity_freq_hz
                Q = feat.cavity_q
                bw = f0 / (2.0 * Q)
                freq_range = (max(f0 - bw, self.freq_range_hz[0]),
                              min(f0 + bw, self.freq_range_hz[1]))
            else:
                freq_range = self.freq_range_hz

            # Angular region width
            if ang_pat == "isotropic":
                ang_width = 2.0 * math.pi
            else:
                ang_width = feat.lobe_width_rad

            uncertainty = max(
                self.UNCERTAINTY_BY_FREQ_DEP.get(freq_dep, 0.5),
                self.UNCERTAINTY_BY_ANGULAR.get(ang_pat, 0.3),
            )

            center = AspectAngle(theta=feat.lobe_center_theta, phi=feat.lobe_center_phi)
            feature_centers.append((center, ang_width))

            priors.append(GeometricFeaturePrior(
                feature_type=freq_dep,
                angular_region_center=center,
                angular_region_width_rad=ang_width,
                freq_range_hz=freq_range,
                uncertainty_score=uncertainty,
                description=f"Feature {feat.label}: {freq_dep}/{ang_pat}",
            ))

        # Generate gap priors for regions not covered by any feature
        gap_centers = self._find_gap_centers(feature_centers)
        for gc in gap_centers:
            priors.append(GeometricFeaturePrior(
                feature_type="gap",
                angular_region_center=gc,
                angular_region_width_rad=0.3,
                freq_range_hz=self.freq_range_hz,
                uncertainty_score=0.6,
                description=f"Gap region at theta={gc.theta:.2f}, phi={gc.phi:.2f}",
            ))

        self._features = priors
        return priors

    def _find_gap_centers(
        self,
        feature_centers: list[tuple[AspectAngle, float]],
    ) -> list[AspectAngle]:
        """Find angular regions not covered by any feature."""
        gap_candidates = []
        for theta in np.linspace(0.2, math.pi - 0.2, 10):
            candidate = AspectAngle(theta=float(theta), phi=0.0)
            covered = False
            for center, width in feature_centers:
                dist = angular_distance(candidate, center)
                if dist < width / 2.0:
                    covered = True
                    break
            if not covered:
                gap_candidates.append(candidate)
        return gap_candidates

    def predict_susceptibility(
        self, points: list[ObservationPoint]
    ) -> np.ndarray:
        """For each observation point, compute total geometric susceptibility."""
        if self._features is None:
            self.extract_features()

        N = len(points)
        susceptibility = np.zeros(N)

        for feat_prior in self._features:
            center = feat_prior.angular_region_center
            width = feat_prior.angular_region_width_rad + 1e-6

            for i, pt in enumerate(points):
                # Frequency indicator
                in_freq = (feat_prior.freq_range_hz[0] <= pt.freq_hz <= feat_prior.freq_range_hz[1])

                # Angular overlap
                pt_angle = AspectAngle(theta=pt.theta, phi=pt.phi)
                dist = angular_distance(pt_angle, center)
                w = math.exp(-(dist ** 2) / (2.0 * (width / 2.355) ** 2 + 1e-30))

                score = w * feat_prior.uncertainty_score
                if in_freq:
                    score *= 1.5  # boost for in-frequency-range

                if score > susceptibility[i]:
                    susceptibility[i] = score

        return susceptibility
