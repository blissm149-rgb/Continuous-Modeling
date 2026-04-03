"""Tensor-based simulation sensitivity analysis.

Derives ranked measurement candidates from a raw (az × el × freq)
complex RCS tensor without requiring CAD geometry labels or a running
EM simulator.

Public API
----------
TensorSensitivityMap
    Drop-in replacement for DiscrepancySusceptibilityMap when a
    pre-computed simulation tensor is available.

GradientAnalyzer, ISARAnalyzer, SpectralAnalyzer, CancellationDetector,
PhysicalConsistencyAnalyzer, CrossFreqCoherenceAnalyzer
    Individual analysis components (use directly for inspection /
    custom weighting).

RegimeClassifier
    Per-frequency scattering-regime classifier for adaptive weights.

plan_measurements
    Sequential measurement planner (greedy, correlation-suppressed).

validate_sensitivity, compare_sensitivity
    Perturbation-ensemble validators that measure map quality.
"""
from .tensor_sensitivity_map         import TensorSensitivityMap
from .gradient_analyzer              import GradientAnalyzer
from .isar_analyzer                  import ISARAnalyzer
from .spectral_analyzer              import SpectralAnalyzer
from .cancellation_detector          import CancellationDetector
from .physical_consistency_analyzer  import PhysicalConsistencyAnalyzer
from .cross_freq_coherence           import CrossFreqCoherenceAnalyzer
from .regime_classifier              import RegimeClassifier
from .measurement_planner            import plan_measurements
from .validation                     import validate_sensitivity, compare_sensitivity

__all__ = [
    "TensorSensitivityMap",
    "GradientAnalyzer",
    "ISARAnalyzer",
    "SpectralAnalyzer",
    "CancellationDetector",
    "PhysicalConsistencyAnalyzer",
    "CrossFreqCoherenceAnalyzer",
    "RegimeClassifier",
    "plan_measurements",
    "validate_sensitivity",
    "compare_sensitivity",
]
