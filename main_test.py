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
_cov_val = metrics.get("coverage_68", float("nan"))
assert math.isnan(_cov_val) or (0.0 <= _cov_val <= 1.0), "Coverage out of range"
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

# ──────────────────────────────────────────────────────────────────────────────
# 8.  Reproducibility demo
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("Section 8 — Reproducibility (random_seed=123)")
print("=" * 70)

_repro_kwargs = dict(
    scenario_name="simple_missing_feature",
    total_measurement_budget=25,
    candidate_grid_density=8,
    n_freq_candidates=8,
    random_seed=123,
)

_r_a = DHFFEngine(**_repro_kwargs).run()
_r_b = DHFFEngine(**_repro_kwargs).run()

nmse_a = _r_a["error_metrics"]["complex_nmse"]
nmse_b = _r_b["error_metrics"]["complex_nmse"]
print(f"  Run A  complex_nmse = {nmse_a:.6f}")
print(f"  Run B  complex_nmse = {nmse_b:.6f}")
print(f"  Match  : {'YES ✓' if abs(nmse_a - nmse_b) < 1e-9 else 'NO (unexpected variability)'}")

# Without seed — may differ
_r_no_seed = DHFFEngine(
    scenario_name="simple_missing_feature",
    total_measurement_budget=25,
    candidate_grid_density=8,
    n_freq_candidates=8,
).run()
print(f"  No-seed run nmse   = {_r_no_seed['error_metrics']['complex_nmse']:.6f}  (may differ)")
print()

assert abs(nmse_a - nmse_b) < 1e-9, "Seeded runs produced different results!"
print("Reproducibility sanity checks passed ✓")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 9.  Real-data CSV round-trip
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("Section 9 — CSV I/O round-trip")
print("=" * 70)

import tempfile, os
from dhff.io import RCSMeasurementLoader
from dhff.core import make_observation_grid

# Write the first section-1 run's ground-truth RCS to a CSV
_eval_pts = make_observation_grid(
    theta_range=(0.3, math.pi - 0.3),
    phi_range=(0.0, 0.0),
    freq_range=(8e9, 12e9),
    n_theta=6, n_phi=1, n_freq=6,
)
_gt_rcs = results["ground_truth"].compute_rcs(_eval_pts)

_csv_tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
_csv_path = _csv_tmp.name
_csv_tmp.close()

try:
    RCSMeasurementLoader.write_csv(_csv_path, _eval_pts, _gt_rcs.values, snr_db=28.0)

    _loader = RCSMeasurementLoader(_csv_path)
    _loaded_pts, _loaded_vals = _loader.load()

    print(f"  Written : {len(_eval_pts)} observation points → {_csv_path}")
    print(f"  Reloaded: {len(_loaded_pts)} observation points")
    print(f"  Median SNR from CSV: {_loader.median_snr_db:.1f} dB")
    print(f"  First 3 rows:")
    for i in range(min(3, len(_loaded_pts))):
        p = _loaded_pts[i]
        v = _loaded_vals[i]
        print(f"    θ={p.theta:.3f} φ={p.phi:.2f} f={p.freq_hz/1e9:.2f} GHz  "
              f"RCS={v.real:+.4f}{v.imag:+.4f}j")
    print()

    assert len(_loaded_pts) == len(_eval_pts), "Row count mismatch"
    np.testing.assert_allclose(_loaded_vals, _gt_rcs.values, rtol=1e-5)
    assert abs(_loader.median_snr_db - 28.0) < 0.1
    print("CSV I/O sanity checks passed ✓")
finally:
    os.unlink(_csv_path)
print()

# ──────────────────────────────────────────────────────────────────────────────
# 10.  JSON export
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("Section 10 — JSON export")
print("=" * 70)

import json as _json

_json_path = os.path.join("./results", "report.json")
engine.export_results_json(results, _json_path)

with open(_json_path) as _jfh:
    _jdata = _json.load(_jfh)

print(f"  Written : {_json_path}")
print(f"  Scenario: {_jdata['scenario']}")
print(f"  Total measurements : {_jdata['total_measurements']}")
print(f"  Improvement factor : {_jdata['improvement_factor']:.2f}×")
print(f"  Anomalies in JSON  : {len(_jdata['anomalies'])}")
print()
if _jdata["anomalies"]:
    print("  Anomaly table:")
    for _a in _jdata["anomalies"]:
        print(f"    type={_a['type']:<28}  root_cause={_a.get('root_cause','—')}")
print()

for _key in ("scenario", "freq_range_hz", "total_measurements",
             "error_metrics", "improvement_factor", "anomalies", "timestamp"):
    assert _key in _jdata, f"Key '{_key}' missing from JSON"
print("JSON export sanity checks passed ✓")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 11.  Confidence scores
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("Section 11 — Anomaly confidence scores")
print("=" * 70)

_classifications = results.get("anomalies_classified", [])
if _classifications:
    print(f"  {'Type':<28}  {'Root cause':<28}  {'Conf':>5}  {'KK score':>8}  {'N freq':>6}")
    print(f"  {'-'*28}  {'-'*28}  {'-'*5}  {'-'*8}  {'-'*6}")
    for _cls in _classifications:
        _anom  = _cls["anomaly"]
        _type  = _anom.anomaly_type.name if hasattr(_anom.anomaly_type, "name") else str(_anom.anomaly_type)
        _rc    = _cls.get("root_cause", "—")
        _conf  = _cls.get("confidence", 0.5)
        _kk    = _cls.get("kk_violation_score")
        _nf    = _cls.get("n_freq_samples_used", 0)
        _kk_s  = f"{_kk:.3f}" if _kk is not None else "  n/a"
        print(f"  {_type:<28}  {_rc:<28}  {_conf:>5.2f}  {_kk_s:>8}  {_nf:>6}")
else:
    print("  (no anomalies classified — try a larger budget)")

print()

# Verify coverage calibration is present
_cov = results["error_metrics"].get("coverage_68")
_flag = results["error_metrics"].get("coverage_calibration_flag", "not set")
print(f"  Coverage 68%: {_cov:.3f}  → {_flag}")
print()

assert _cov is not None, "coverage_68 key missing from error_metrics"
assert _flag in ("well_calibrated", "over_confident", "under_confident", "unknown")
print("Confidence / calibration sanity checks passed ✓")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 12.  Tensor-based sensitivity analysis (simulation-only path)
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("Section 12 — Tensor-Based Sensitivity Analysis")
print("=" * 70)
print("  Building (21 az × 5 el × 20 freq) RCS tensor from cad_derived scenario …")

from dhff.tensor_analysis import TensorSensitivityMap
from dhff.core import make_observation_grid
from dhff.synthetic import scenario_cad_derived

# Re-use the cad_derived ground truth to build a synthetic 3D tensor
_cad_gt, _, _ = scenario_cad_derived()

_az_grid   = np.linspace(0.15, math.pi - 0.15, 21)
_el_grid   = np.linspace(-0.25, 0.25, 5)
_freq_grid = np.linspace(8e9, 12e9, 20)

# Populate tensor by calling the ground-truth scatterer over the 3D grid
_tensor = np.zeros((len(_az_grid), len(_el_grid), len(_freq_grid)), dtype=complex)
for _i, _az in enumerate(_az_grid):
    for _j, _el in enumerate(_el_grid):
        for _k, _f in enumerate(_freq_grid):
            from dhff.core.types import ObservationPoint as _OP
            _pts = [_OP(theta=_el, phi=_az, freq_hz=_f)]
            _rcs = _cad_gt.compute_rcs(_pts)
            _tensor[_i, _j, _k] = _rcs.values[0]

print(f"  Tensor shape   : {_tensor.shape}  dtype={_tensor.dtype}")
print(f"  Amplitude range: {np.abs(_tensor).min():.4f} – {np.abs(_tensor).max():.4f}")
print()

# Build sensitivity map (no geometry labels required)
_tsm = TensorSensitivityMap(_tensor, _az_grid, _el_grid, _freq_grid)

# Per-method raw scores
_method_scores = _tsm.get_per_method_scores()
print(f"  Per-method mean scores:")
for _mname, _mscore in sorted(_method_scores.items()):
    print(f"    {_mname:<15}: {_mscore.mean():.4f}  max={_mscore.max():.4f}")
print()

# Top 5 sensitive points
_top = _tsm.get_top_points(n=5)
print(f"  Top-5 most sensitive observation points:")
print(f"  {'Rank':<5} {'Az (°)':<8} {'El (°)':<8} {'Freq (GHz)':<12} {'Score':<7} {'Driver'}")
print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*12} {'-'*7} {'-'*15}")
for _rank, (_az_r, _el_r, _f_hz, _s) in enumerate(_top, 1):
    # Identify which method contributes most at this grid point
    _i_az = int(np.argmin(np.abs(_az_grid - _az_r)))
    _i_el = int(np.argmin(np.abs(_el_grid - _el_r)))
    _i_f  = int(np.argmin(np.abs(_freq_grid - _f_hz)))
    _driver = max(_method_scores.items(), key=lambda kv: kv[1][_i_az, _i_el, _i_f])[0]
    print(f"  {_rank:<5} {math.degrees(_az_r):<8.1f} {math.degrees(_el_r):<8.1f} "
          f"{_f_hz/1e9:<12.2f} {_s:<7.3f} {_driver}")
print()

# ISAR quick-look for el_idx=2 (el≈0°)
_isar_img, _cr_axis, _range_axis = _tsm.get_isar_image(el_idx=2)
_isar_peak  = float(np.max(_isar_img))
_isar_floor = float(np.percentile(_isar_img, 75))
print(f"  ISAR image (el≈0°):")
print(f"    Peak power   : {_isar_peak:.4f}")
print(f"    Floor (p75)  : {_isar_floor:.4f}")
print(f"    Sidelobe ratio: {_isar_floor/(_isar_peak+1e-30):.3f}  "
      f"({'complex scene' if _isar_floor/_isar_peak > 0.2 else 'clean dominant peak'})")
print()

# Sanity checks
assert _tsm.get_combined_score_grid().shape == _tensor.shape
assert len(_top) == 5
assert all(0.0 <= s <= 1.0 for _, _, _, s in _top)

print("Tensor sensitivity sanity checks passed ✓")
print()
print("Done.")
