"""Physics-based feature extractor: converts CAD primitives to ScatteringFeature objects."""
from __future__ import annotations

import math

from dhff.synthetic.scatterer import ScatteringFeature
from .primitives import CadPrimitive, FlatPanel, EdgeSegment, CavityVolume, ConvexSurface

_C = 299_792_458.0  # speed of light (m/s)


class CadFeatureExtractor:
    """Convert a list of CAD geometry primitives into ScatteringFeature objects.

    Physics formulas used:
      FlatPanel   — Physical Optics (PO): sigma = 4*pi*w*h/lambda^2
      EdgeSegment — UTD approximation: amplitude ~ 0.1 * length / sqrt(f_GHz)
      CavityVolume — TE101 dominant mode: f0 = C / (2*sqrt(a^2 + d^2))
      ConvexSurface — Creeping wave: amplitude ~ 0.05 * arc_length / radius

    Uncertainty metadata (amplitude_uncertainty_db, freq_param_uncertainty,
    position_uncertainty_m) is set per primitive type and propagated into the
    GeometricFeatureAnalyzer's D_prior map.
    """

    def __init__(
        self,
        freq_range_hz: tuple[float, float] = (8e9, 12e9),
        f_center: float = 10e9,
        max_amplitude_cap: float = 1e4,
    ) -> None:
        self.freq_range_hz = freq_range_hz
        self.f_center = f_center
        self.max_amplitude_cap = max_amplitude_cap
        self._lambda_center = _C / f_center

    def extract(self, primitives: list[CadPrimitive]) -> list[ScatteringFeature]:
        """Convert each primitive to a ScatteringFeature."""
        features: list[ScatteringFeature] = []
        for prim in primitives:
            if isinstance(prim, FlatPanel):
                features.append(self._panel_to_feature(prim))
            elif isinstance(prim, EdgeSegment):
                features.append(self._edge_to_feature(prim))
            elif isinstance(prim, CavityVolume):
                features.append(self._cavity_to_feature(prim))
            elif isinstance(prim, ConvexSurface):
                features.append(self._surface_to_feature(prim))
            else:
                raise TypeError(f"Unknown CAD primitive type: {type(prim)}")
        return features

    # ------------------------------------------------------------------
    # Per-primitive conversion methods
    # ------------------------------------------------------------------

    def _panel_to_feature(self, p: FlatPanel) -> ScatteringFeature:
        lam = self._lambda_center
        # PO RCS: sigma = 4*pi*w*h / lambda^2
        sigma_po = 4.0 * math.pi * p.width_m * p.height_m / (lam ** 2)
        base_amplitude = min(math.sqrt(sigma_po), self.max_amplitude_cap)

        # Beamwidth from PO: FWHM ~ 0.886 * lambda / max_dimension
        lobe_width = 0.886 * lam / max(p.width_m, p.height_m)
        angular_pattern = "narrow_lobe" if lobe_width < 0.1 else "specular_lobe"

        # PO is accurate for panels — low amplitude uncertainty
        return ScatteringFeature(
            x=p.x,
            y=p.y,
            base_amplitude=complex(base_amplitude),
            freq_dependence="specular",
            angular_pattern=angular_pattern,
            lobe_center_theta=p.normal_theta_rad,
            lobe_center_phi=p.normal_phi_rad,
            lobe_width_rad=lobe_width,
            label=p.label,
            geometry_source="FlatPanel",
            position_uncertainty_m=p.manufacturing_tolerance_m,
            amplitude_uncertainty_db=1.0,
            freq_param_uncertainty=0.0,
        )

    def _edge_to_feature(self, e: EdgeSegment) -> ScatteringFeature:
        f_ghz = self.f_center / 1e9
        # UTD amplitude approximation: scales with length, decreases with frequency
        base_amplitude = 0.1 * e.length_m / math.sqrt(f_ghz)
        base_amplitude = min(base_amplitude, self.max_amplitude_cap)

        return ScatteringFeature(
            x=e.x,
            y=e.y,
            base_amplitude=complex(base_amplitude),
            freq_dependence="edge",
            angular_pattern="broad_lobe",
            lobe_center_theta=e.edge_theta_rad,
            lobe_center_phi=0.0,
            lobe_width_rad=math.pi / 2,
            label=e.label,
            geometry_source="EdgeSegment",
            position_uncertainty_m=e.manufacturing_tolerance_m,
            amplitude_uncertainty_db=3.0,
            freq_param_uncertainty=0.0,
        )

    def _cavity_to_feature(self, c: CavityVolume) -> ScatteringFeature:
        a = c.interior_dim_a_m
        d = c.depth_m
        # TE101 resonant frequency
        f0 = _C / (2.0 * math.sqrt(a ** 2 + d ** 2))

        # Q: use override if provided, otherwise compute from radiation loss
        if c.cavity_q_override is not None:
            cavity_q = float(c.cavity_q_override)
        else:
            volume = a * c.interior_dim_b_m * d
            q_raw = f0 * volume / (_C * max(c.aperture_area_m2, 1e-12))
            cavity_q = float(max(2.0, min(500.0, q_raw)))

        lam = self._lambda_center
        base_amplitude = 0.1 * c.aperture_area_m2 / (lam ** 2)
        base_amplitude = min(max(base_amplitude, 1e-6), self.max_amplitude_cap)

        # Cavities are highest-uncertainty: geometry doesn't capture loading, coupling, etc.
        return ScatteringFeature(
            x=c.x,
            y=c.y,
            base_amplitude=complex(base_amplitude),
            freq_dependence="cavity_resonant",
            angular_pattern="broad_lobe",
            lobe_center_theta=0.0,
            lobe_center_phi=0.0,
            lobe_width_rad=math.pi / 2,
            cavity_freq_hz=f0,
            cavity_q=cavity_q,
            label=c.label,
            geometry_source="CavityVolume",
            position_uncertainty_m=c.manufacturing_tolerance_m,
            amplitude_uncertainty_db=10.0,
            freq_param_uncertainty=0.15,
        )

    def _surface_to_feature(self, s: ConvexSurface) -> ScatteringFeature:
        # Creeping wave amplitude scales with arc_length / radius ratio
        base_amplitude = 0.05 * s.arc_length_m / max(s.radius_m, 1e-9)
        base_amplitude = min(base_amplitude, self.max_amplitude_cap)

        return ScatteringFeature(
            x=s.x,
            y=s.y,
            base_amplitude=complex(base_amplitude),
            freq_dependence="creeping",
            angular_pattern="broad_lobe",
            lobe_center_theta=s.surface_theta_rad,
            lobe_center_phi=0.0,
            lobe_width_rad=math.pi,
            label=s.label,
            geometry_source="ConvexSurface",
            position_uncertainty_m=s.manufacturing_tolerance_m,
            amplitude_uncertainty_db=8.0,
            freq_param_uncertainty=0.0,
        )
