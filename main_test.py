"""
DHFF v2 — Example end-to-end run.

Click "Run" in VSCode (or `python main_test.py`) to execute a complete
discrepancy-hunting pipeline on the simple missing-feature scenario.
"""

import math
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Frequency array is not uniformly spaced")

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Build the engine (small budget so it finishes in < 30 s)
# ──────────────────────────────────────────────────────────────────────────────
from dhff.pipeline import DHFFEngine

print("=" * 60)
print("DHFF v2 — Discrepancy-Hunting Fusion Framework")
print("=" * 60)
print()

engine = DHFFEngine(
    scenario_name="simple_missing_feature",
    total_measurement_budget=60,   # enough for all 4 phases to do useful work
    candidate_grid_density=25,     # 25 theta × 1 phi
    n_freq_candidates=25,          # × 25 freq = 625 candidates
    model_type="hybrid",
)

print("[1/4]  Setting up scenario …")
engine.setup()
print(f"       Ground truth features  : {len(engine.ground_truth.features)}")
print(f"       Simulator features     : {len(engine.simulator.degraded_scatterer.features)}")
print(f"       Candidate grid size    : {len(engine.candidate_grid)} points")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Run the 4-phase campaign
# ──────────────────────────────────────────────────────────────────────────────
print("[2/4]  Running measurement campaign (4 phases) …")
results = engine.run()
print(f"       Total measurements taken : {results['total_measurements']}")
print(f"       Phases logged            : {len(results['history'])}")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Report error metrics
# ──────────────────────────────────────────────────────────────────────────────
print("[3/4]  Error metrics vs ground truth:")
metrics = results["error_metrics"]
sim_nmse   = metrics.get("sim_only_nmse", float("nan"))
fused_nmse = metrics.get("complex_nmse",  float("nan"))
improv     = results.get("improvement_factor", float("nan"))

print(f"       Sim-only  complex NMSE  : {sim_nmse:.4f}")
print(f"       Fused     complex NMSE  : {fused_nmse:.4f}")
print(f"       Improvement factor      : {improv:.2f}×")
print(f"       Coverage @ 68 %%        : {metrics.get('coverage_68', float('nan')):.2f}")
print(f"       Phase angle RMSE (rad)  : {metrics.get('phase_rmse_rad', float('nan')):.4f}")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 4.  Anomaly summary
# ──────────────────────────────────────────────────────────────────────────────
print("[4/4]  Anomaly detection summary:")
anomalies = results["anomalies_detected"]
classifications = results["anomalies_classified"]
print(f"       Anomalies detected      : {len(anomalies)}")
for anom in anomalies:
    meas_pos = (
        f"({anom.meas_center.x:.3f}, {anom.meas_center.y:.3f})"
        if anom.meas_center else "—"
    )
    sim_pos = (
        f"({anom.sim_center.x:.3f}, {anom.sim_center.y:.3f})"
        if anom.sim_center else "—"
    )
    print(f"         • {anom.anomaly_type.name:<28}  meas={meas_pos}  sim={sim_pos}")

for clf in classifications:
    anom = clf["anomaly"]
    print(f"           root_cause={clf['root_cause']}  "
          f"confidence={clf['confidence']:.1f}  "
          f"action='{clf['recommended_action']}'")

pc = results["parametric_centers_found"]
print(f"       Parametric SC centers   : {pc}")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 5.  Quick sanity assertions (will raise AssertionError if something is wrong)
# ──────────────────────────────────────────────────────────────────────────────
assert results["total_measurements"] > 0, "No measurements were taken"
assert "fused_model" in results,          "fused_model missing from results"
assert 0.0 <= metrics.get("coverage_68", 0.0) <= 1.0, "Coverage out of range"
if not math.isnan(fused_nmse) and not math.isnan(sim_nmse):
    assert fused_nmse <= sim_nmse * 3.0, (
        f"Fused NMSE ({fused_nmse:.4f}) is unreasonably worse than sim ({sim_nmse:.4f})"
    )

print("All sanity checks passed ✓")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 6.  (Optional) Save diagnostic plots to ./results/
# ──────────────────────────────────────────────────────────────────────────────
try:
    engine.generate_report(results, output_dir="./results")
    print("Plots saved to ./results/  (rcs_comparison.png, summary.txt)")
except Exception as exc:
    print(f"(Plot generation skipped: {exc})")

print()
print("Done.")
