"""Module 4: Anomaly classification for scattering center discrepancies."""
from __future__ import annotations

import math

import numpy as np

from dhff.core.types import (
    AnomalyType, ScatteringCenter, ScatteringCenterAnomaly,
)


class AnomalyClassifier:
    """Classify matched and unmatched scattering centers into anomaly types."""

    def __init__(
        self,
        position_threshold_m: float = 0.1,
        amplitude_threshold_db: float = 3.0,
    ):
        self.position_threshold_m = position_threshold_m
        self.amplitude_threshold_db = amplitude_threshold_db

    def classify(
        self,
        matched: list[tuple[ScatteringCenter, ScatteringCenter]],
        unmatched_sim: list[ScatteringCenter],
        unmatched_meas: list[ScatteringCenter],
    ) -> list[ScatteringCenterAnomaly]:
        anomalies = []

        for sim_c, meas_c in matched:
            pos_err = math.sqrt((sim_c.x - meas_c.x)**2 + (sim_c.y - meas_c.y)**2)
            amp_err = 20.0 * math.log10(
                max(abs(meas_c.amplitude), 1e-30) / max(abs(sim_c.amplitude), 1e-30)
            )

            if pos_err > self.position_threshold_m:
                anomalies.append(ScatteringCenterAnomaly(
                    anomaly_type=AnomalyType.POSITION_SHIFT,
                    meas_center=meas_c,
                    sim_center=sim_c,
                    position_error_m=pos_err,
                    amplitude_error_db=amp_err,
                ))
            elif abs(amp_err) > self.amplitude_threshold_db:
                anomalies.append(ScatteringCenterAnomaly(
                    anomaly_type=AnomalyType.AMPLITUDE_DISCREPANCY,
                    meas_center=meas_c,
                    sim_center=sim_c,
                    position_error_m=pos_err,
                    amplitude_error_db=amp_err,
                ))

        for sc in unmatched_sim:
            anomalies.append(ScatteringCenterAnomaly(
                anomaly_type=AnomalyType.UNMATCHED_SIMULATION,
                meas_center=None,
                sim_center=sc,
            ))

        for mc in unmatched_meas:
            anomalies.append(ScatteringCenterAnomaly(
                anomaly_type=AnomalyType.UNMATCHED_MEASUREMENT,
                meas_center=mc,
                sim_center=None,
            ))

        return anomalies

    def suggest_measurement_strategy(self, anomaly: ScatteringCenterAnomaly) -> dict:
        """Return a measurement strategy recommendation for the given anomaly."""
        full_theta = (0.1, math.pi - 0.1)
        full_freq = None

        if anomaly.anomaly_type == AnomalyType.UNMATCHED_MEASUREMENT:
            mc = anomaly.meas_center
            return {
                "strategy": "Broad angular coverage to image unknown feature",
                "angular_priority": "broad",
                "frequency_priority": "broadband",
                "suggested_theta_range": full_theta,
                "suggested_freq_range": full_freq,
            }
        elif anomaly.anomaly_type == AnomalyType.POSITION_SHIFT:
            mc = anomaly.meas_center
            if mc is not None:
                tc = mc.lobe_center_theta
                theta_range = (max(0.05, tc - 0.2), min(math.pi - 0.05, tc + 0.2))
            else:
                theta_range = full_theta
            return {
                "strategy": "Dense angular sampling near specular direction",
                "angular_priority": "fine",
                "frequency_priority": "broadband",
                "suggested_theta_range": theta_range,
                "suggested_freq_range": full_freq,
            }
        elif anomaly.anomaly_type == AnomalyType.AMPLITUDE_DISCREPANCY:
            sc = anomaly.sim_center
            if sc is not None:
                tc = sc.lobe_center_theta
                hw = sc.lobe_width_rad
                theta_range = (max(0.05, tc - hw), min(math.pi - 0.05, tc + hw))
            else:
                theta_range = full_theta
            return {
                "strategy": "Targeted measurement near lobe center",
                "angular_priority": "targeted",
                "frequency_priority": "narrowband",
                "suggested_theta_range": theta_range,
                "suggested_freq_range": full_freq,
            }
        else:  # UNMATCHED_SIMULATION
            return {
                "strategy": "Low priority — simulation artifact",
                "angular_priority": "broad",
                "frequency_priority": "broadband",
                "suggested_theta_range": full_theta,
                "suggested_freq_range": full_freq,
            }
