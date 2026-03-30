"""Tensor-based simulation sensitivity analysis.

Derives ranked measurement candidates from a raw (az × el × freq)
complex RCS tensor without requiring CAD geometry labels or a running
EM simulator.

Public API
----------
TensorSensitivityMap
    Drop-in replacement for DiscrepancySusceptibilityMap when a
    pre-computed simulation tensor is available.

GradientAnalyzer, ISARAnalyzer, SpectralAnalyzer, CancellationDetector
    Individual analysis components (use directly for inspection /
    custom weighting).
"""
from .tensor_sensitivity_map         import TensorSensitivityMap
from .gradient_analyzer              import GradientAnalyzer
from .isar_analyzer                  import ISARAnalyzer
from .spectral_analyzer              import SpectralAnalyzer
from .cancellation_detector          import CancellationDetector
from .physical_consistency_analyzer  import PhysicalConsistencyAnalyzer

__all__ = [
    "TensorSensitivityMap",
    "GradientAnalyzer",
    "ISARAnalyzer",
    "SpectralAnalyzer",
    "CancellationDetector",
    "PhysicalConsistencyAnalyzer",
]
