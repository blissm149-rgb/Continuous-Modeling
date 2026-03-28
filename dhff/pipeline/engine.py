"""Module 7: End-to-end DHFF pipeline orchestration."""
from __future__ import annotations

import math
import os
import warnings

import numpy as np

from dhff.core import make_observation_grid
from dhff.synthetic import (
    scenario_simple_missing_feature,
    scenario_shifted_and_amplitude,
    scenario_complex_target,
)
from dhff.discrepancy_prior import (
    EnsembleDisagreement, GeometricFeatureAnalyzer, DiscrepancySusceptibilityMap,
)
from dhff.acquisition import SequentialMeasurementPlanner
from dhff.models import HybridDiscrepancyModel, PureGPDiscrepancyModel, FusedRCSModel


_SCENARIOS = {
    "simple_missing_feature": scenario_simple_missing_feature,
    "shifted_and_amplitude": scenario_shifted_and_amplitude,
    "complex_target": scenario_complex_target,
}


class DHFFEngine:
    """Discrepancy-Hunting Fusion Framework — top-level entry point."""

    def __init__(
        self,
        scenario_name: str = "simple_missing_feature",
        total_measurement_budget: int = 100,
        candidate_grid_density: int = 50,
        n_freq_candidates: int = 40,
        model_type: str = "hybrid",
    ):
        self.scenario_name = scenario_name
        self.total_budget = total_measurement_budget
        self.grid_density = candidate_grid_density
        self.n_freq = n_freq_candidates
        self.model_type = model_type

        self.ground_truth = None
        self.simulator = None
        self.measurement_system = None
        self.candidate_grid = None
        self.susceptibility_map = None
        self.freq_range_hz = (8e9, 12e9)
        self._is_setup = False

    def setup(self) -> None:
        """Initialize everything."""
        scenario_fn = _SCENARIOS.get(self.scenario_name)
        if scenario_fn is None:
            raise ValueError(f"Unknown scenario: {self.scenario_name}. "
                             f"Available: {list(_SCENARIOS.keys())}")

        self.ground_truth, self.simulator, self.measurement_system = scenario_fn()

        self.candidate_grid = make_observation_grid(
            theta_range=(0.1, math.pi - 0.1),
            phi_range=(0.0, 0.0),
            freq_range=self.freq_range_hz,
            n_theta=self.grid_density,
            n_phi=1,
            n_freq=self.n_freq,
        )

        ensemble = EnsembleDisagreement(self.simulator)
        geometric = GeometricFeatureAnalyzer(self.simulator, freq_range_hz=self.freq_range_hz)
        self.susceptibility_map = DiscrepancySusceptibilityMap(ensemble, geometric)
        self._is_setup = True

    def run(self) -> dict:
        """Run full DHFF pipeline."""
        if not self._is_setup:
            self.setup()

        planner = SequentialMeasurementPlanner(
            simulator=self.simulator,
            measurement_system=self.measurement_system,
            susceptibility_map=self.susceptibility_map,
            candidate_grid=self.candidate_grid,
            freq_range_hz=self.freq_range_hz,
            characteristic_length_m=self.ground_truth.characteristic_length,
            total_budget=self.total_budget,
            model_type=self.model_type,
        )

        campaign_results = planner.run_full_campaign()
        fused_model = campaign_results["fused_model"]

        # Evaluation grid (separate from candidates)
        eval_grid = make_observation_grid(
            theta_range=(0.2, math.pi - 0.2),
            phi_range=(0.0, 0.0),
            freq_range=self.freq_range_hz,
            n_theta=20, n_phi=1, n_freq=20,
        )

        try:
            error_metrics = fused_model.error_vs_ground_truth(self.ground_truth, eval_grid)
        except Exception as e:
            warnings.warn(f"Error computing metrics: {e}")
            error_metrics = {"complex_nmse": 1.0, "sim_only_nmse": 1.0}

        sim_nmse = error_metrics.get("sim_only_nmse", 1.0)
        fused_nmse = error_metrics.get("complex_nmse", 1.0)
        improvement = sim_nmse / (fused_nmse + 1e-30)

        return {
            "fused_model": fused_model,
            "ground_truth": self.ground_truth,
            "anomalies_detected": campaign_results["anomalies"],
            "anomalies_classified": campaign_results["classifications"],
            "error_metrics": error_metrics,
            "sim_only_metrics": {"nmse": sim_nmse},
            "improvement_factor": float(improvement),
            "measurement_efficiency": float(improvement / max(planner.total_budget, 1)),
            "parametric_centers_found": len(campaign_results["parametric_centers"]),
            "history": campaign_results["measurement_history"],
            "total_measurements": campaign_results["total_measurements"],
        }

    def run_comparison(self) -> dict:
        """Run DHFF (hybrid), DHFF (pure_gp), and naive uniform baseline."""
        if not self._is_setup:
            self.setup()

        results = {}

        # 1. DHFF hybrid
        try:
            hybrid_engine = DHFFEngine(
                scenario_name=self.scenario_name,
                total_measurement_budget=self.total_budget,
                candidate_grid_density=self.grid_density,
                n_freq_candidates=self.n_freq,
                model_type="hybrid",
            )
            hybrid_engine.setup()
            results["dhff_hybrid"] = hybrid_engine.run()
        except Exception as e:
            warnings.warn(f"Hybrid run failed: {e}")
            results["dhff_hybrid"] = {"error": str(e), "error_metrics": {"complex_nmse": 1.0}}

        # 2. DHFF pure GP
        try:
            gp_engine = DHFFEngine(
                scenario_name=self.scenario_name,
                total_measurement_budget=self.total_budget,
                candidate_grid_density=self.grid_density,
                n_freq_candidates=self.n_freq,
                model_type="pure_gp",
            )
            gp_engine.setup()
            results["dhff_pure_gp"] = gp_engine.run()
        except Exception as e:
            warnings.warn(f"Pure GP run failed: {e}")
            results["dhff_pure_gp"] = {"error": str(e), "error_metrics": {"complex_nmse": 1.0}}

        # 3. Uniform baseline
        try:
            results["uniform_baseline"] = self._run_uniform_baseline()
        except Exception as e:
            warnings.warn(f"Uniform baseline failed: {e}")
            results["uniform_baseline"] = {"error": str(e), "error_metrics": {"complex_nmse": 1.0}}

        # Build comparison table
        table = {}
        for method, res in results.items():
            metrics = res.get("error_metrics", {})
            table[method] = {
                "nmse": metrics.get("complex_nmse", 1.0),
                "anomalies_found": len(res.get("anomalies_detected", [])),
                "coverage_68": metrics.get("coverage_68", 0.0),
            }
        results["comparison_table"] = table
        return results

    def _run_uniform_baseline(self) -> dict:
        """Uniform grid sampling baseline."""
        from dhff.core import DiscrepancySample

        # Uniform measurements across the candidate grid
        n = min(self.total_budget, len(self.candidate_grid))
        step = max(1, len(self.candidate_grid) // n)
        selected_pts = self.candidate_grid[::step][:n]

        samples = []
        for pt in selected_pts:
            val = self.measurement_system.measure_single(pt)
            sim_val = self.simulator.compute_rcs([pt]).values[0]
            samples.append(DiscrepancySample(obs=pt, residual=complex(val - sim_val)))

        model = HybridDiscrepancyModel(freq_range_hz=self.freq_range_hz, gp_training_iters=50)
        model.fit(samples)

        fused = FusedRCSModel(simulator=self.simulator, discrepancy_model=model)

        eval_grid = make_observation_grid(
            theta_range=(0.2, math.pi - 0.2),
            phi_range=(0.0, 0.0),
            freq_range=self.freq_range_hz,
            n_theta=20, n_phi=1, n_freq=20,
        )

        try:
            metrics = fused.error_vs_ground_truth(self.ground_truth, eval_grid)
        except Exception as e:
            warnings.warn(f"Uniform baseline metrics failed: {e}")
            metrics = {"complex_nmse": 1.0, "sim_only_nmse": 1.0}

        return {
            "fused_model": fused,
            "ground_truth": self.ground_truth,
            "anomalies_detected": [],
            "anomalies_classified": [],
            "error_metrics": metrics,
            "total_measurements": len(samples),
        }

    def generate_report(self, results: dict, output_dir: str = "./results") -> None:
        """Generate visualization outputs."""
        os.makedirs(output_dir, exist_ok=True)
        from dhff.visualization import plots

        eval_grid = make_observation_grid(
            theta_range=(0.2, math.pi - 0.2),
            phi_range=(0.0, 0.0),
            freq_range=self.freq_range_hz,
            n_theta=20, n_phi=1, n_freq=20,
        )

        fused_model = results.get("fused_model")
        ground_truth = results.get("ground_truth")

        if fused_model and ground_truth:
            try:
                gt_rcs = ground_truth.compute_rcs(eval_grid)
                sim_rcs = self.simulator.compute_rcs(eval_grid)
                fused_rcs, uncertainty = fused_model.predict(eval_grid)

                from dhff.core.coordinate_system import observation_points_to_array
                x_vals = observation_points_to_array(eval_grid)[:, 0]

                fig = plots.plot_rcs_comparison(
                    ground_truth=gt_rcs, simulation=sim_rcs, fused=fused_rcs,
                    uncertainty=uncertainty, x_values=x_vals,
                )
                fig.savefig(os.path.join(output_dir, "rcs_comparison.png"), dpi=100)
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception as e:
                warnings.warn(f"RCS comparison plot failed: {e}")

        # Write summary
        metrics = results.get("error_metrics", {})
        with open(os.path.join(output_dir, "summary.txt"), "w") as f:
            f.write(f"DHFF Summary\n")
            f.write(f"============\n")
            f.write(f"Scenario: {self.scenario_name}\n")
            f.write(f"Total measurements: {results.get('total_measurements', 'N/A')}\n")
            f.write(f"Complex NMSE: {metrics.get('complex_nmse', 'N/A'):.4f}\n")
            f.write(f"Sim-only NMSE: {metrics.get('sim_only_nmse', 'N/A'):.4f}\n")
            f.write(f"Improvement: {results.get('improvement_factor', 'N/A'):.2f}x\n")
            f.write(f"Anomalies detected: {len(results.get('anomalies_detected', []))}\n")
            f.write(f"Parametric centers found: {results.get('parametric_centers_found', 'N/A')}\n")
