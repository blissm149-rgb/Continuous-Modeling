"""Head-to-head comparison: traditional metadata-driven D_prior vs tensor-based sensitivity map.

Both runs use the same scenario, the same random seed, and the same measurement budget.
The only difference is HOW Phase 1 measurement points are chosen and how the D_prior
is computed throughout the campaign.

Traditional path: EnsembleDisagreement (multi-solver variance) +
                  GeometricFeatureAnalyzer (CAD metadata uncertainty)

Tensor path:      TensorSensitivityMap built from the simulator output tensor
                  (GradientAnalyzer + ISARAnalyzer + SpectralAnalyzer + CancellationDetector)
"""
import math
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from dhff.core.types import ObservationPoint
from dhff.pipeline import DHFFEngine
from dhff.synthetic import scenario_simple_missing_feature, scenario_cad_derived

# ──────────────────────────────────────────────────────────────────
# Helper: build simulator tensor from a scenario
# ──────────────────────────────────────────────────────────────────

def build_sim_tensor(simulator, az_grid, el_grid, freq_grid):
    """Evaluate the imperfect simulator over a 3-D (az x el x freq) grid."""
    T = np.zeros((len(az_grid), len(el_grid), len(freq_grid)), dtype=complex)
    for i, az in enumerate(az_grid):
        for j, el in enumerate(el_grid):
            pts = [ObservationPoint(theta=el, phi=az, freq_hz=f) for f in freq_grid]
            rcs = simulator.compute_rcs(pts)
            T[i, j, :] = rcs.values
    return T


def run_comparison(scenario_name, budget, seed, grid_density, n_freq,
                   az_pts=21, el_pts=5, freq_pts=20):
    print(f"\n{'='*65}")
    print(f"  Scenario : {scenario_name}")
    print(f"  Budget   : {budget} measurements   seed={seed}")
    print(f"  Tensor   : {az_pts} az × {el_pts} el × {freq_pts} freq")
    print(f"{'='*65}")

    # Build tensor from the simulator (imperfect — as in real use)
    if scenario_name == "simple_missing_feature":
        _, simulator, _ = scenario_simple_missing_feature()
    else:
        _, simulator, _ = scenario_cad_derived()

    az_grid   = np.linspace(0.1,  math.pi - 0.1, az_pts)
    el_grid   = np.linspace(-0.25, 0.25,          el_pts)
    freq_grid = np.linspace(8e9,   12e9,           freq_pts)

    print("  Building simulator tensor …", flush=True)
    tensor = build_sim_tensor(simulator, az_grid, el_grid, freq_grid)
    print(f"  Amplitude range: {np.abs(tensor).min():.4f} – {np.abs(tensor).max():.4f}")

    common = dict(
        scenario_name=scenario_name,
        total_measurement_budget=budget,
        candidate_grid_density=grid_density,
        n_freq_candidates=n_freq,
        random_seed=seed,
    )

    # ── Run 1: Traditional D_prior ───────────────────────────────
    print("\n  [1/2] Traditional D_prior (geometry + ensemble) …", flush=True)
    r_trad = DHFFEngine(**common).run()

    # ── Run 2: Tensor sensitivity map ────────────────────────────
    print("  [2/2] Tensor sensitivity map …", flush=True)
    r_tensor = DHFFEngine(
        **common,
        rcs_tensor_input=dict(
            tensor=tensor, az_rad=az_grid, el_rad=el_grid, freq_hz=freq_grid
        ),
    ).run()

    # ── Extract metrics ─────────────────────────────────────────
    def metrics(r):
        m = r["error_metrics"]
        return {
            "sim_nmse":    m.get("sim_only_nmse", float("nan")),
            "fused_nmse":  m.get("complex_nmse",  float("nan")),
            "improv":      r.get("improvement_factor", float("nan")),
            "coverage":    m.get("coverage_68",    float("nan")),
            "cov_flag":    m.get("coverage_calibration_flag", "—"),
            "anomalies":   len(r.get("anomalies_detected", [])),
            "sc_centers":  r.get("parametric_centers_found", 0),
            "n_meas":      r.get("total_measurements", 0),
        }

    mt = metrics(r_trad)
    mn = metrics(r_tensor)

    # ── Print table ─────────────────────────────────────────────
    print()
    col_w = 22
    fmt = f"  {{:<{col_w}}} {{:>12}} {{:>12}}"
    print(fmt.format("Metric", "Traditional", "Tensor"))
    print("  " + "-" * (col_w + 26))
    def row(label, key, fmt_str=".4f"):
        vt = mt[key]; vn = mn[key]
        vs = f"{vt:{fmt_str}}" if isinstance(vt, float) else str(vt)
        vns = f"{vn:{fmt_str}}" if isinstance(vn, float) else str(vn)
        print(fmt.format(label, vs, vns))

    row("Sim-only NMSE",   "sim_nmse")
    row("Fused NMSE",      "fused_nmse")
    row("Improvement",     "improv",    ".2f")
    row("Coverage 68%",    "coverage",  ".3f")
    row("Coverage flag",   "cov_flag",  "s")
    row("Anomalies found", "anomalies", "d")
    row("SC centers",      "sc_centers","d")
    row("Measurements",    "n_meas",    "d")

    delta_nmse = mn["fused_nmse"] - mt["fused_nmse"]
    winner = "TENSOR" if delta_nmse < 0 else ("TRADITIONAL" if delta_nmse > 0.001 else "TIE")
    print(f"\n  ΔNMSE (tensor − trad) = {delta_nmse:+.4f}  → winner: {winner}")

    return mt, mn


# ── Run for both scenarios ────────────────────────────────────────

results = {}

results["simple_missing_feature"] = run_comparison(
    scenario_name="simple_missing_feature",
    budget=60,
    seed=42,
    grid_density=25,
    n_freq=25,
    az_pts=21, el_pts=5, freq_pts=20,
)

results["cad_derived"] = run_comparison(
    scenario_name="cad_derived",
    budget=60,
    seed=42,
    grid_density=25,
    n_freq=25,
    az_pts=21, el_pts=5, freq_pts=20,
)

# ── Summary ──────────────────────────────────────────────────────
print("\n\n" + "="*65)
print("  SUMMARY — Traditional D_prior  vs  Tensor Sensitivity Map")
print("="*65)
print(f"  {'Scenario':<28} {'Trad NMSE':>10} {'Tensor NMSE':>12} {'Winner':>12}")
print(f"  {'-'*28} {'-'*10} {'-'*12} {'-'*12}")
for sname, (mt, mn) in results.items():
    delta = mn["fused_nmse"] - mt["fused_nmse"]
    winner = "tensor" if delta < -0.001 else ("trad" if delta > 0.001 else "tie")
    print(f"  {sname:<28} {mt['fused_nmse']:>10.4f} {mn['fused_nmse']:>12.4f} {winner:>12}")
print()
