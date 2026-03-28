"""Module 6: Sequential measurement planner."""
from __future__ import annotations

import math
import warnings

import numpy as np

from dhff.core.types import (
    DiscrepancySample, MeasurementPlan, ObservationPoint, ScatteringCenterAnomaly,
)
from dhff.models.hybrid_model import HybridDiscrepancyModel
from dhff.models.pure_gp import PureGPDiscrepancyModel
from dhff.models.fused_model import FusedRCSModel
from dhff.scattering_center import (
    MatrixPencilExtractor, ScatteringCenterAssociator, AnomalyClassifier,
)
from .functions import DiscrepancyAcquisition, ScatteringCenterAcquisition
from .classifier import KramersKronigConsistencyTest, DiscrepancyTypeClassifier


class SequentialMeasurementPlanner:
    """Full sequential measurement planner implementing the 4-phase strategy."""

    def __init__(
        self,
        simulator,
        measurement_system,
        susceptibility_map,
        candidate_grid: list[ObservationPoint],
        freq_range_hz: tuple[float, float],
        characteristic_length_m: float = 1.0,
        total_budget: int = 100,
        phase_budgets: tuple[float, float, float, float] = (0.15, 0.35, 0.25, 0.25),
        batch_size: int = 5,
        model_type: str = "hybrid",
    ):
        self.simulator = simulator
        self.measurement_system = measurement_system
        self.susceptibility_map = susceptibility_map
        self.candidate_grid = candidate_grid
        self.freq_range_hz = freq_range_hz
        self.characteristic_length_m = characteristic_length_m
        self.total_budget = total_budget
        self.phase_budgets = phase_budgets
        self.batch_size = batch_size
        self.model_type = model_type

        self.measured_points: list[ObservationPoint] = []
        self.measured_values: list[complex] = []
        self.discrepancy_samples: list[DiscrepancySample] = []
        self.discrepancy_model = None
        self.anomalies: list[ScatteringCenterAnomaly] = []
        self.classifications: list[dict] = []
        self.history: list[dict] = []

        self._anomaly_classifier = AnomalyClassifier()
        self._kk_test = KramersKronigConsistencyTest()
        self._type_classifier = DiscrepancyTypeClassifier(
            kk_test=self._kk_test, anomaly_classifier=self._anomaly_classifier
        )

    def _create_model(self):
        if self.model_type == "hybrid":
            return HybridDiscrepancyModel(
                freq_range_hz=self.freq_range_hz,
                gp_training_iters=50,
            )
        elif self.model_type == "pure_gp":
            return PureGPDiscrepancyModel(
                freq_range_hz=self.freq_range_hz,
                characteristic_length_m=self.characteristic_length_m,
                n_training_iters=80,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _take_measurements(self, plan: MeasurementPlan) -> None:
        for point in plan.points:
            value = self.measurement_system.measure_single(point)
            sim_value = self.simulator.compute_rcs([point]).values[0]
            residual = complex(value - sim_value)
            self.measured_points.append(point)
            self.measured_values.append(value)
            self.discrepancy_samples.append(DiscrepancySample(obs=point, residual=residual))

    def _update_model(self) -> None:
        self.discrepancy_model = self._create_model()
        self.discrepancy_model.fit(self.discrepancy_samples)

    def _run_anomaly_detection(self) -> None:
        try:
            sim_centers = self.simulator.degraded_scatterer.get_scattering_centers()

            if self.model_type == "hybrid" and hasattr(self.discrepancy_model, 'get_parametric_centers'):
                meas_centers = self.discrepancy_model.get_parametric_centers()
            else:
                # Extract from discrepancy samples using MatrixPencil
                meas_centers = self._extract_meas_centers()

            associator = ScatteringCenterAssociator(max_association_distance_m=0.5)
            matched, unmatched_sim, unmatched_meas = associator.associate(sim_centers, meas_centers)
            self.anomalies = self._anomaly_classifier.classify(matched, unmatched_sim, unmatched_meas)
        except Exception as e:
            warnings.warn(f"Anomaly detection failed: {e}")
            self.anomalies = []

    def _extract_meas_centers(self):
        if not self.discrepancy_samples:
            return []
        extractor = MatrixPencilExtractor(n_centers_max=10)
        # Group by angle, extract, return
        angle_groups: dict[tuple, list] = {}
        for s in self.discrepancy_samples:
            key = (round(s.obs.theta, 2), round(s.obs.phi, 2))
            angle_groups.setdefault(key, []).append(s)

        all_centers = []
        for samples in angle_groups.values():
            if len(samples) < 8:
                continue
            sorted_s = sorted(samples, key=lambda s: s.obs.freq_hz)
            freqs = np.array([s.obs.freq_hz for s in sorted_s])
            resids = np.array([s.residual for s in sorted_s])
            centers = extractor.extract_1d(freqs, resids)
            all_centers.extend(centers)
        return all_centers

    def _log_iteration(self, phase: int, batch_idx: int, plan: MeasurementPlan) -> None:
        self.history.append({
            "phase": phase,
            "batch_idx": batch_idx,
            "n_measurements": len(self.measured_points),
            "n_anomalies": len(self.anomalies),
            "selected_points": [(p.theta, p.phi, p.freq_hz) for p in plan.points],
            "scores": list(plan.scores),
        })

    def run_phase1_discovery(self) -> None:
        """Phase 1: D_prior-guided initial measurements."""
        n_phase1 = max(self.batch_size, int(self.total_budget * self.phase_budgets[0]))
        plan = self.susceptibility_map.select_initial_measurements(
            self.candidate_grid, n_phase1
        )
        self._take_measurements(plan)
        self._update_model()
        self._log_iteration(1, 0, plan)

    def run_phase2_anomaly_hunting(self) -> None:
        """Phase 2: Iterative discrepancy exploration."""
        n_phase2 = max(self.batch_size, int(self.total_budget * self.phase_budgets[1]))
        n_batches = max(1, n_phase2 // self.batch_size)

        for batch_idx in range(n_batches):
            self._update_model()
            acq = DiscrepancyAcquisition(
                discrepancy_model=self.discrepancy_model,
                susceptibility_map=self.susceptibility_map,
                lambda_explore=2.0,
                mu_prior=0.3,
            )
            plan = acq.select_batch(self.candidate_grid, self.batch_size)
            self._take_measurements(plan)
            if batch_idx % 2 == 1:
                self._run_anomaly_detection()
            self._log_iteration(2, batch_idx, plan)

    def run_phase3_characterization(self) -> None:
        """Phase 3: Targeted measurements to classify anomalies."""
        n_phase3 = max(self.batch_size, int(self.total_budget * self.phase_budgets[2]))
        self._update_model()
        self._run_anomaly_detection()

        # Generate candidates from anomalies
        sc_acq = ScatteringCenterAcquisition(
            anomalies=self.anomalies,
            anomaly_classifier=self._anomaly_classifier,
            freq_range_hz=self.freq_range_hz,
        )
        sc_candidates_with_rationale = sc_acq.generate_candidates(n_per_anomaly=5)
        sc_candidates = [c[0] for c in sc_candidates_with_rationale]

        # Also add some acquisition-based candidates
        acq = DiscrepancyAcquisition(
            discrepancy_model=self.discrepancy_model,
            susceptibility_map=self.susceptibility_map,
            lambda_explore=1.0,
            mu_prior=0.5,
        )
        acq_plan = acq.select_batch(self.candidate_grid, min(n_phase3, 20))
        all_candidates = sc_candidates + acq_plan.points

        if all_candidates:
            # Take measurements in batches
            n_batches = max(1, n_phase3 // self.batch_size)
            for batch_idx in range(n_batches):
                batch_pts = all_candidates[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
                if not batch_pts:
                    break
                plan = MeasurementPlan(
                    points=batch_pts,
                    scores=[1.0] * len(batch_pts),
                    rationale=["Phase 3 targeted"] * len(batch_pts),
                )
                self._take_measurements(plan)
                self._log_iteration(3, batch_idx, plan)

        # KK classification
        try:
            self.classifications = self._type_classifier.classify_all(
                self.anomalies, self.discrepancy_samples, self.freq_range_hz
            )
        except Exception as e:
            warnings.warn(f"KK classification failed: {e}")
            self.classifications = []

        self._update_model()

    def run_phase4_refinement(self) -> None:
        """Phase 4: Dense exploitation of confirmed discrepancy regions."""
        n_phase4 = max(self.batch_size, int(self.total_budget * self.phase_budgets[3]))
        n_exploit = int(0.8 * n_phase4)
        n_verify = n_phase4 - n_exploit

        # Exploitation: low lambda
        acq_exploit = DiscrepancyAcquisition(
            discrepancy_model=self.discrepancy_model,
            susceptibility_map=self.susceptibility_map,
            lambda_explore=0.2,
            mu_prior=0.0,
        )
        plan_exploit = acq_exploit.select_batch(self.candidate_grid, n_exploit)
        self._take_measurements(plan_exploit)
        self._log_iteration(4, 0, plan_exploit)

        # Verification: from low-acquisition regions
        all_scores = acq_exploit.evaluate(self.candidate_grid)
        threshold = np.percentile(all_scores, 20)
        verify_candidates = [
            self.candidate_grid[i] for i in range(len(self.candidate_grid))
            if all_scores[i] <= threshold
        ]
        if verify_candidates:
            n_actual = min(n_verify, len(verify_candidates))
            verify_plan = MeasurementPlan(
                points=verify_candidates[:n_actual],
                scores=[0.0] * n_actual,
                rationale=["Verification"] * n_actual,
            )
            self._take_measurements(verify_plan)
            self._log_iteration(4, 1, verify_plan)

        self._update_model()

    def run_full_campaign(self) -> dict:
        """Execute all 4 phases sequentially."""
        self.run_phase1_discovery()
        self.run_phase2_anomaly_hunting()
        self.run_phase3_characterization()
        self.run_phase4_refinement()

        # Final anomaly detection
        self._run_anomaly_detection()
        try:
            self.classifications = self._type_classifier.classify_all(
                self.anomalies, self.discrepancy_samples, self.freq_range_hz
            )
        except Exception as e:
            warnings.warn(f"Final classification failed: {e}")

        fused = FusedRCSModel(
            simulator=self.simulator,
            discrepancy_model=self.discrepancy_model,
        )

        parametric_centers = []
        if self.model_type == "hybrid" and hasattr(self.discrepancy_model, 'get_parametric_centers'):
            parametric_centers = self.discrepancy_model.get_parametric_centers()

        return {
            "fused_model": fused,
            "discrepancy_model": self.discrepancy_model,
            "anomalies": self.anomalies,
            "classifications": self.classifications,
            "measurement_history": self.history,
            "total_measurements": len(self.measured_points),
            "parametric_centers": parametric_centers,
        }
