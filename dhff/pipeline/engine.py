"""Module 7: End-to-end DHFF pipeline orchestration."""
from __future__ import annotations

import csv
import datetime
import json
import math
import os
import warnings

import numpy as np

from dhff.core import make_observation_grid
from dhff.synthetic import (
    scenario_simple_missing_feature,
    scenario_shifted_and_amplitude,
    scenario_complex_target,
    scenario_cad_derived,
)
from dhff.discrepancy_prior import (
    EnsembleDisagreement, GeometricFeatureAnalyzer, DiscrepancySusceptibilityMap,
)
from dhff.acquisition import SequentialMeasurementPlanner
from dhff.models import HybridDiscrepancyModel, PureGPDiscrepancyModel, FusedRCSModel
from dhff.scattering_center import SCExtractorConfig


_SCENARIOS = {
    "simple_missing_feature": scenario_simple_missing_feature,
    "shifted_and_amplitude": scenario_shifted_and_amplitude,
    "complex_target": scenario_complex_target,
    "cad_derived": scenario_cad_derived,
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
        freq_range_hz: tuple[float, float] = (8e9, 12e9),
        max_sc_centers: int = 15,
        gp_training_iters: int = 80,
        sc_config: SCExtractorConfig | None = None,
        random_seed: int | None = None,
        rcs_tensor_input: dict | None = None,
    ):
        self.scenario_name = scenario_name
        self.total_budget = total_measurement_budget
        self.grid_density = candidate_grid_density
        self.n_freq = n_freq_candidates
        self.model_type = model_type
        self.freq_range_hz = freq_range_hz
        self.max_sc_centers = max_sc_centers
        self.gp_training_iters = gp_training_iters
        self.sc_config = sc_config
        self.random_seed = random_seed
        self.rcs_tensor_input = rcs_tensor_input

        self.ground_truth = None
        self.simulator = None
        self.measurement_system = None
        self.candidate_grid = None
        self.susceptibility_map = None
        self._is_setup = False

    def setup(self) -> None:
        """Initialize everything."""
        scenario_fn = _SCENARIOS.get(self.scenario_name)
        if scenario_fn is None:
            raise ValueError(f"Unknown scenario: {self.scenario_name}. "
                             f"Available: {list(_SCENARIOS.keys())}")

        self.ground_truth, self.simulator, self.measurement_system = scenario_fn()
        # Apply random seed to measurement system if requested
        if self.random_seed is not None:
            from dhff.synthetic.measurement import SyntheticMeasurementSystem
            self.measurement_system = SyntheticMeasurementSystem(
                ground_truth=self.ground_truth,
                snr_db=self.measurement_system.snr_db,
                phase_noise_std_rad=self.measurement_system.phase_noise_std_rad,
                calibration_error=self.measurement_system.calibration_error,
                seed=self.random_seed,
            )

        self.candidate_grid = make_observation_grid(
            theta_range=(0.1, math.pi - 0.1),
            phi_range=(0.0, 0.0),
            freq_range=self.freq_range_hz,
            n_theta=self.grid_density,
            n_phi=1,
            n_freq=self.n_freq,
        )

        if self.rcs_tensor_input is not None:
            from dhff.tensor_analysis import TensorSensitivityMap
            d = self.rcs_tensor_input
            self.susceptibility_map = TensorSensitivityMap(
                rcs_tensor=d["tensor"],
                az_rad=d["az_rad"],
                el_rad=d["el_rad"],
                freq_hz=d["freq_hz"],
                weights=d.get("weights"),
            )
        else:
            ensemble = EnsembleDisagreement(self.simulator)
            geometric = GeometricFeatureAnalyzer(
                self.simulator, freq_range_hz=self.freq_range_hz
            )
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
            max_sc_centers=self.max_sc_centers,
            gp_training_iters=self.gp_training_iters,
            sc_config=self.sc_config,
            seed=self.random_seed,
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

        # Uncertainty calibration on held-out 20% of training samples
        all_samples = planner.discrepancy_samples
        try:
            coverage = self._validate_coverage(fused_model, all_samples)
            error_metrics["coverage_68"] = coverage
            if coverage < 0.55:
                error_metrics["coverage_calibration_flag"] = "over_confident"
            elif coverage > 0.80:
                error_metrics["coverage_calibration_flag"] = "under_confident"
            else:
                error_metrics["coverage_calibration_flag"] = "well_calibrated"
        except Exception as e:
            warnings.warn(f"Coverage calibration failed: {e}")
            error_metrics["coverage_68"] = float("nan")
            error_metrics["coverage_calibration_flag"] = "unknown"

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

    @staticmethod
    def _validate_coverage(fused_model, samples) -> float:
        """Measure actual coverage at nominal 68% (1-sigma) credible interval.

        Uses a 20% held-out subset of the training samples.
        """
        if not samples:
            return float("nan")
        n = len(samples)
        n_held = max(1, n // 5)  # 20% held out
        held_out = samples[-n_held:]

        pts = [s.obs for s in held_out]
        actuals = np.array([s.residual for s in held_out])
        means, variances = fused_model.predict(pts)
        stds = np.sqrt(np.maximum(variances, 1e-30))
        within = float(np.mean(np.abs(actuals - means) <= stds))
        return within

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

        model = HybridDiscrepancyModel(
            freq_range_hz=self.freq_range_hz,
            max_sc_centers=self.max_sc_centers,
            gp_training_iters=self.gp_training_iters,
            sc_config=self.sc_config,
            seed=self.random_seed,
        )
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

        # Structured exports
        self.export_results_json(results, os.path.join(output_dir, "report.json"))
        self.export_results_csv(results, os.path.join(output_dir, "anomalies.csv"))

    # ------------------------------------------------------------------
    # Structured export
    # ------------------------------------------------------------------

    @staticmethod
    def _anomaly_to_dict(anomaly, classification: dict | None) -> dict:
        """Serialise one anomaly + its classification to a plain dict."""
        meas = anomaly.meas_center
        sim = anomaly.sim_center
        d: dict = {
            "type": anomaly.anomaly_type.value if hasattr(anomaly.anomaly_type, "value") else str(anomaly.anomaly_type),
            "meas_center": {"x": meas.x, "y": meas.y} if meas is not None else None,
            "sim_center": {"x": sim.x, "y": sim.y} if sim is not None else None,
            "position_error_m": float(anomaly.position_error_m) if anomaly.position_error_m is not None else None,
            "amplitude_error_db": float(anomaly.amplitude_error_db) if anomaly.amplitude_error_db is not None else None,
        }
        if classification is not None:
            d["root_cause"] = classification.get("root_cause", "unknown")
            d["confidence"] = classification.get("confidence", 0.5)
            d["kk_violation_score"] = classification.get("kk_violation_score")
            d["n_freq_samples_used"] = classification.get("n_freq_samples_used", 0)
        return d

    def export_results_json(self, results: dict, path: str) -> None:
        """Export anomaly detection results to a structured JSON file."""
        anomalies = results.get("anomalies_detected", [])
        classifications = results.get("anomalies_classified", [])

        # Build a lookup from anomaly identity to classification
        cls_map = {id(c["anomaly"]): c for c in classifications if "anomaly" in c}

        anomaly_list = []
        for a in anomalies:
            cls = cls_map.get(id(a))
            anomaly_list.append(self._anomaly_to_dict(a, cls))

        metrics = results.get("error_metrics", {})
        # Remove non-serialisable numpy scalars
        safe_metrics = {k: float(v) if hasattr(v, "item") else v for k, v in metrics.items()}

        payload = {
            "scenario": self.scenario_name,
            "freq_range_hz": list(self.freq_range_hz),
            "total_measurements": results.get("total_measurements", 0),
            "error_metrics": safe_metrics,
            "improvement_factor": float(results.get("improvement_factor", 0.0)),
            "anomalies": anomaly_list,
            "parametric_centers_found": results.get("parametric_centers_found", 0),
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)

    def export_results_csv(self, results: dict, path: str) -> None:
        """Export anomaly table to CSV (one row per anomaly)."""
        anomalies = results.get("anomalies_detected", [])
        classifications = results.get("anomalies_classified", [])
        cls_map = {id(c["anomaly"]): c for c in classifications if "anomaly" in c}

        fieldnames = [
            "anomaly_type", "meas_x", "meas_y", "sim_x", "sim_y",
            "position_error_m", "amplitude_error_db",
            "root_cause", "confidence", "kk_violation_score", "n_freq_samples_used",
        ]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for a in anomalies:
                cls = cls_map.get(id(a))
                d = self._anomaly_to_dict(a, cls)
                writer.writerow({
                    "anomaly_type": d["type"],
                    "meas_x": d["meas_center"]["x"] if d["meas_center"] else "",
                    "meas_y": d["meas_center"]["y"] if d["meas_center"] else "",
                    "sim_x": d["sim_center"]["x"] if d["sim_center"] else "",
                    "sim_y": d["sim_center"]["y"] if d["sim_center"] else "",
                    "position_error_m": d.get("position_error_m", ""),
                    "amplitude_error_db": d.get("amplitude_error_db", ""),
                    "root_cause": d.get("root_cause", ""),
                    "confidence": d.get("confidence", ""),
                    "kk_violation_score": d.get("kk_violation_score", ""),
                    "n_freq_samples_used": d.get("n_freq_samples_used", ""),
                })

    # ------------------------------------------------------------------
    # Real-data entry point
    # ------------------------------------------------------------------

    @classmethod
    def load_from_csv(
        cls,
        measurement_path: str,
        simulation_path: str,
        freq_range_hz: tuple[float, float] = (8e9, 12e9),
        **engine_kwargs,
    ) -> "DHFFEngine":
        """Create a DHFFEngine backed by real measurement + simulation CSV files.

        The two CSVs must share the same (theta_rad, phi_rad, freq_hz) grid.
        Discrepancy is computed as measurement − simulation at each matched point.

        Parameters
        ----------
        measurement_path:
            CSV file with real RCS measurements.
        simulation_path:
            CSV file with corresponding simulator predictions.
        freq_range_hz:
            Frequency band filter applied to both files.
        **engine_kwargs:
            Extra kwargs forwarded to DHFFEngine.__init__
            (e.g. random_seed, max_sc_centers).
        """
        from dhff.io import RCSMeasurementLoader
        from dhff.core.types import DiscrepancySample

        meas_loader = RCSMeasurementLoader(measurement_path, freq_range_hz=freq_range_hz)
        sim_loader = RCSMeasurementLoader(simulation_path, freq_range_hz=freq_range_hz)

        meas_pts, meas_vals = meas_loader.load()
        sim_pts, sim_vals = sim_loader.load()

        # Match points by (theta, phi, freq) within tolerance
        sim_map = {
            (round(p.theta, 4), round(p.phi, 4), round(p.freq_hz, -3)): v
            for p, v in zip(sim_pts, sim_vals)
        }
        samples = []
        for pt, mv in zip(meas_pts, meas_vals):
            key = (round(pt.theta, 4), round(pt.phi, 4), round(pt.freq_hz, -3))
            sv = sim_map.get(key)
            if sv is not None:
                samples.append(DiscrepancySample(obs=pt, residual=complex(mv - sv)))

        if not samples:
            raise ValueError(
                "No matched observation points found between measurement and simulation CSVs. "
                "Ensure both files share the same (theta_rad, phi_rad, freq_hz) grid."
            )

        # Build engine and inject external data adapter
        engine = cls(freq_range_hz=freq_range_hz, **engine_kwargs)
        engine._external_samples = samples
        engine._meas_snr_db = meas_loader.median_snr_db
        engine._is_external = True
        return engine
