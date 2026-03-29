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

# ──────────────────────────────────────────────────────────────────────────────
# 7.  CAD-derived scenario — demonstrates geometry-primitive pipeline
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("CAD Geometry Primitives — cad_derived scenario")
print("=" * 60)
print()

from dhff.cad import CadFeatureExtractor, FlatPanel, EdgeSegment, CavityVolume, ConvexSurface

_primitives = [
    FlatPanel(x=0.0, y=0.0, width_m=0.004, height_m=0.004,
              normal_theta_rad=math.pi / 2, label="panel_main"),
    FlatPanel(x=0.3, y=0.1, width_m=0.003, height_m=0.003,
              normal_theta_rad=math.pi / 3, label="panel_side"),
    EdgeSegment(x=-0.2, y=0.15, length_m=0.08,
                edge_theta_rad=2 * math.pi / 3, label="leading_edge"),
    CavityVolume(x=0.25, y=-0.1,
                 interior_dim_a_m=0.012, interior_dim_b_m=0.008, depth_m=0.009,
                 aperture_area_m2=0.0040, cavity_q_override=15.0, label="inlet_cavity"),
    ConvexSurface(x=0.1, y=-0.2, radius_m=0.08, arc_length_m=0.12,
                  surface_theta_rad=math.pi / 4, label="nose_surface"),
]
_extractor = CadFeatureExtractor(freq_range_hz=(8e9, 12e9), f_center=10e9)
_features  = _extractor.extract(_primitives)

print("[CAD-1]  Geometry primitives → ScatteringFeatures:")
panels   = [f for f in _features if f.geometry_source == "FlatPanel"]
edges    = [f for f in _features if f.geometry_source == "EdgeSegment"]
cavities = [f for f in _features if f.geometry_source == "CavityVolume"]
convex   = [f for f in _features if f.geometry_source == "ConvexSurface"]
print(f"         Panels   : {len(panels)}"
      + ("".join(f"  {p.label}(w={_primitives[i].width_m*1000:.0f}mm)"
                  for i, p in enumerate(panels)) if panels else ""))
print(f"         Edges    : {len(edges)}"
      + ("".join(f"  {e.label}(L={_primitives[len(panels)+i].length_m*1000:.0f}mm)"
                  for i, e in enumerate(edges)) if edges else ""))
for cv in cavities:
    print(f"         Cavities : {len(cavities)}  {cv.label}"
          f"  f₀={cv.cavity_freq_hz/1e9:.2f} GHz  Q={cv.cavity_q:.1f}"
          f"  unc=±{cv.amplitude_uncertainty_db:.0f} dB")
print(f"         Convex   : {len(convex)}"
      + ("".join(f"  {s.label}" for s in convex) if convex else ""))
print()

print("[CAD-2]  Running DHFF pipeline on cad_derived scenario …")
cad_engine = DHFFEngine(
    scenario_name="cad_derived",
    total_measurement_budget=60,
    candidate_grid_density=25,
    n_freq_candidates=25,
    model_type="hybrid",
)
cad_engine.setup()
cad_results = cad_engine.run()

cad_metrics = cad_results["error_metrics"]
cad_sim_nmse   = cad_metrics.get("sim_only_nmse", float("nan"))
cad_fused_nmse = cad_metrics.get("complex_nmse",  float("nan"))
cad_improv     = cad_results.get("improvement_factor", float("nan"))

print(f"         Sim-only  NMSE  : {cad_sim_nmse:.4f}")
print(f"         Fused     NMSE  : {cad_fused_nmse:.4f}")
print(f"         Improvement     : {cad_improv:.2f}×")
print(f"         SC centers found: {cad_results['parametric_centers_found']}")

cad_anomalies = cad_results["anomalies_detected"]
print(f"         Anomalies found : {len(cad_anomalies)}")
for anom in cad_anomalies:
    meas_pos = (f"({anom.meas_center.x:.3f}, {anom.meas_center.y:.3f})"
                if anom.meas_center else "—")
    print(f"           • {anom.anomaly_type.name:<26}  meas={meas_pos}")
print()

assert not math.isnan(cad_fused_nmse), "cad_derived: fused NMSE is NaN"
assert not math.isnan(cad_sim_nmse),   "cad_derived: sim-only NMSE is NaN"

print("CAD scenario sanity checks passed ✓")
print()
print("Done.")
