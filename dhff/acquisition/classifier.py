"""Module 6: Discrepancy type classification using Kramers-Kronig test."""
from __future__ import annotations

import numpy as np
import scipy.signal

from dhff.core.types import AnomalyType, DiscrepancySample, ObservationPoint


class KramersKronigConsistencyTest:
    """Test whether the measured discrepancy is physically causal."""

    def __init__(self, tolerance: float = 0.3):
        self.tolerance = tolerance

    def test(
        self, freq_hz: np.ndarray, discrepancy: np.ndarray
    ) -> dict:
        """Run the KK consistency test."""
        freq_hz = np.asarray(freq_hz, dtype=np.float64)
        discrepancy = np.asarray(discrepancy, dtype=np.complex128)

        re_part = discrepancy.real

        # Analytic signal of Re[discrepancy]
        analytic = scipy.signal.hilbert(re_part)
        predicted_im = analytic.imag

        actual_im = discrepancy.imag
        norm = np.linalg.norm(actual_im) + 1e-30
        kk_violation = float(np.linalg.norm(actual_im - predicted_im) / norm)

        is_causal = kk_violation < self.tolerance
        diagnosis = "missing_scatterer" if is_causal else "solver_error"

        return {
            "kk_violation_score": kk_violation,
            "is_causal": is_causal,
            "diagnosis": diagnosis,
            "predicted_imag": predicted_im,
        }


class DiscrepancyTypeClassifier:
    """Combines scattering center anomaly analysis with KK consistency."""

    def __init__(
        self,
        kk_test: KramersKronigConsistencyTest,
        anomaly_classifier,
    ):
        self.kk_test = kk_test
        self.anomaly_classifier = anomaly_classifier

    def classify_all(
        self,
        anomalies,
        discrepancy_samples: list[DiscrepancySample],
        freq_range_hz: tuple[float, float],
        n_freq_for_kk: int = 64,
    ) -> list[dict]:
        """Classify each anomaly with full diagnostics."""
        results = []

        for anomaly in anomalies:
            # Find discrepancy samples near the anomaly's meas center angle
            ref_center = anomaly.meas_center
            if ref_center is not None:
                ref_theta = ref_center.lobe_center_theta
                ref_phi = ref_center.lobe_center_phi
            elif anomaly.sim_center is not None:
                ref_theta = anomaly.sim_center.lobe_center_theta
                ref_phi = anomaly.sim_center.lobe_center_phi
            else:
                ref_theta = None
                ref_phi = None

            kk_result = None
            n_freq_samples = 0
            if ref_theta is not None:
                nearby = [
                    s for s in discrepancy_samples
                    if abs(s.obs.theta - ref_theta) < 0.1
                    and abs(s.obs.phi - ref_phi) < 0.1
                ]
                n_freq_samples = len(nearby)
                if n_freq_samples >= n_freq_for_kk // 2:
                    nearby_sorted = sorted(nearby, key=lambda s: s.obs.freq_hz)
                    freq_arr = np.array([s.obs.freq_hz for s in nearby_sorted])
                    disc_arr = np.array([s.residual for s in nearby_sorted])
                    kk_result = self.kk_test.test(freq_arr, disc_arr)

            # Determine root cause
            root_cause, action, confidence, kk_score, n_used = _determine_root_cause(
                anomaly.anomaly_type, kk_result, n_freq_samples
            )

            results.append({
                "anomaly": anomaly,
                "kk_result": kk_result,
                "root_cause": root_cause,
                "recommended_action": action,
                "confidence": confidence,
                "kk_violation_score": kk_score,
                "n_freq_samples_used": n_used,
            })

        return results


def _compute_confidence(kk_result: dict | None, n_freq_samples: int) -> float:
    """Continuous [0, 1] confidence based on KK margin and sample count."""
    if kk_result is None:
        return 0.5

    violation = kk_result["kk_violation_score"]
    threshold = 0.3
    margin = abs(violation - threshold) / max(threshold, 1e-9)
    boundary_confidence = min(1.0, 0.5 + 0.5 * margin)

    # Full confidence needs >= 32 samples; < 8 → ceiling of 0.6
    sample_factor = min(1.0, max(0.6, n_freq_samples / 32.0))

    return round(boundary_confidence * sample_factor, 2)


def _determine_root_cause(anomaly_type, kk_result, n_freq_samples: int = 0):
    """Map anomaly type + KK result to root cause."""
    is_causal = kk_result["is_causal"] if kk_result is not None else None
    confidence = _compute_confidence(kk_result, n_freq_samples)
    kk_violation_score = kk_result["kk_violation_score"] if kk_result is not None else None

    if anomaly_type == AnomalyType.UNMATCHED_MEASUREMENT:
        if is_causal is None:
            return "missing_scatterer_or_artifact", "Collect more frequency data for KK test", 0.5, kk_violation_score, n_freq_samples
        elif is_causal:
            return "missing_scatterer", "Add feature to CAD model", confidence, kk_violation_score, n_freq_samples
        else:
            return "measurement_artifact", "Verify measurement setup", confidence, kk_violation_score, n_freq_samples

    elif anomaly_type == AnomalyType.POSITION_SHIFT:
        return "cad_geometry_error", "Correct feature position in CAD model", confidence, kk_violation_score, n_freq_samples

    elif anomaly_type == AnomalyType.AMPLITUDE_DISCREPANCY:
        if is_causal is None:
            return "material_or_solver_error", "Check material properties and solver settings", 0.5, kk_violation_score, n_freq_samples
        elif is_causal:
            return "material_coating_error", "Update material properties", confidence, kk_violation_score, n_freq_samples
        else:
            return "solver_accuracy_issue", "Refine mesh or solver", confidence, kk_violation_score, n_freq_samples

    elif anomaly_type == AnomalyType.UNMATCHED_SIMULATION:
        return "simulation_artifact", "Remove feature from CAD model", confidence, kk_violation_score, n_freq_samples

    return "unknown", "Investigate further", 0.3, kk_violation_score, n_freq_samples
