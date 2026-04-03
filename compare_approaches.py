"""Head-to-head comparison: traditional D_prior vs base TSM vs enhanced TSM.

Three variants run on the same scenario, seed, and measurement budget:
  Traditional : EnsembleDisagreement + GeometricFeatureAnalyzer
  Tensor base : TensorSensitivityMap — 5 methods, fixed weights
  Tensor enh  : TensorSensitivityMap — 6 methods (cross-freq), regime weights,
                disagreement bonus (beta=0.3)

Validation: compare_sensitivity() measures Pearson r between each sensitivity
map and the empirical perturbation variance — a direct indicator of how well
the map tracks real physical sensitivity rather than numerical artefacts.
"""
import math
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from dhff.core.types import ObservationPoint
from dhff.pipeline import DHFFEngine
from dhff.synthetic import scenario_simple_missing_feature, scenario_cad_derived
from dhff.tensor_analysis import TensorSensitivityMap
from dhff.tensor_analysis.validation import compare_sensitivity

# ──────────────────────────────────────────────────────────────────
SEEDS    = [0, 7, 13, 17, 21, 42, 99]
BUDGET   = 60
GRID_D   = 25
N_FREQ_C = 25
AZ_PTS   = 21
EL_PTS   = 5
FREQ_PTS = 20
ENH_BETA = 0.3      # disagreement_beta for enhanced variant
VAL_PERTS = 30      # perturbations for compare_sensitivity


def _get_scenario(name):
    if name == "simple_missing_feature":
        return scenario_simple_missing_feature()
    return scenario_cad_derived()


def build_sim_tensor(simulator, az_grid, el_grid, freq_grid):
    T = np.zeros((len(az_grid), len(el_grid), len(freq_grid)), dtype=complex)
    for i, az in enumerate(az_grid):
        for j, el in enumerate(el_grid):
            pts = [ObservationPoint(theta=el, phi=az, freq_hz=f) for f in freq_grid]
            T[i, j, :] = simulator.compute_rcs(pts).values
    return T


def run_one(scenario_name, seed):
    """Run all three variants for one (scenario, seed) pair.

    Returns dict with keys: trad, base, enh, val
    where trad/base/enh are metrics dicts and val is the validation dict.
    """
    _, simulator, _ = _get_scenario(scenario_name)
    az_grid   = np.linspace(0.1,  math.pi - 0.1, AZ_PTS)
    el_grid   = np.linspace(-0.25, 0.25,          EL_PTS)
    freq_grid = np.linspace(8e9,   12e9,           FREQ_PTS)

    tensor = build_sim_tensor(simulator, az_grid, el_grid, freq_grid)

    common = dict(
        scenario_name=scenario_name,
        total_measurement_budget=BUDGET,
        candidate_grid_density=GRID_D,
        n_freq_candidates=N_FREQ_C,
        random_seed=seed,
    )

    # ── Variant 1: Traditional ──────────────────────────────────────
    r_trad = DHFFEngine(**common).run()

    # ── Variant 2: Tensor base (5-method, no Phase 2-3 flags) ───────
    r_base = DHFFEngine(
        **common,
        rcs_tensor_input=dict(
            tensor=tensor, az_rad=az_grid, el_rad=el_grid, freq_hz=freq_grid,
        ),
    ).run()

    # ── Variant 3: Tensor enhanced (Phase 2-3 all on) ───────────────
    r_enh = DHFFEngine(
        **common,
        rcs_tensor_input=dict(
            tensor=tensor, az_rad=az_grid, el_rad=el_grid, freq_hz=freq_grid,
            use_cross_freq=True,
            use_regime_weights=True,
            disagreement_beta=ENH_BETA,
        ),
    ).run()

    # ── Validation: Pearson r for both tensor maps ──────────────────
    tsm_base = TensorSensitivityMap(tensor, az_grid, el_grid, freq_grid)
    tsm_enh  = TensorSensitivityMap(
        tensor, az_grid, el_grid, freq_grid,
        use_cross_freq=True, use_regime_weights=True, disagreement_beta=ENH_BETA,
    )
    val = compare_sensitivity(
        tensor, az_grid, el_grid, freq_grid,
        tsm_base.get_combined_score_grid(),
        tsm_enh.get_combined_score_grid(),
        config={"validation_n_perturbations": VAL_PERTS, "validation_seed": seed},
    )

    def _m(r):
        m = r["error_metrics"]
        return {
            "sim_nmse":  m.get("sim_only_nmse", float("nan")),
            "fused_nmse": m.get("complex_nmse",  float("nan")),
            "improv":    r.get("improvement_factor", float("nan")),
            "coverage":  m.get("coverage_68",    float("nan")),
            "cov_flag":  m.get("coverage_calibration_flag", "—"),
            "anomalies": len(r.get("anomalies_detected", [])),
        }

    return {
        "trad": _m(r_trad),
        "base": _m(r_base),
        "enh":  _m(r_enh),
        "val":  val,   # keys: r_before, r_after, lift
    }


def run_scenario(scenario_name):
    print(f"\n{'='*72}")
    print(f"  SCENARIO: {scenario_name}   budget={BUDGET}   seeds={SEEDS}")
    print(f"  Tensor: {AZ_PTS}az × {EL_PTS}el × {FREQ_PTS}freq | "
          f"enhanced: cross_freq=True, regime_wts=True, beta={ENH_BETA}")
    print(f"{'='*72}")

    rows = []
    for seed in SEEDS:
        print(f"\n  --- seed={seed} ---", flush=True)
        result = run_one(scenario_name, seed)
        rows.append((seed, result))
        t = result["trad"]
        b = result["base"]
        e = result["enh"]
        v = result["val"]
        print(f"  seed={seed:3d}  "
              f"trad={t['improv']:6.2f}×  "
              f"base={b['improv']:6.2f}×  "
              f"enh={e['improv']:6.2f}×  "
              f"r_base={v['r_before']:.3f}  r_enh={v['r_after']:.3f}  "
              f"lift={v['lift']:+.3f}")

    return rows


def print_summary(scenario_name, rows):
    print(f"\n\n{'─'*92}")
    print(f"  {scenario_name}  —  improvement factor (sim_only_NMSE / fused_NMSE)")
    print(f"{'─'*92}")
    hdr = (f"  {'Seed':>4}  {'Trad':>8}  {'Base TSM':>10}  {'Enh TSM':>10}"
           f"  {'r_base':>7}  {'r_enh':>6}  {'lift':>6}  {'winner'}")
    print(hdr)
    print(f"  {'─'*4}  {'─'*8}  {'─'*10}  {'─'*10}"
          f"  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*8}")

    improv_trad, improv_base, improv_enh = [], [], []
    r_bases, r_enhs, lifts = [], [], []

    for seed, res in rows:
        t, b, e, v = res["trad"], res["base"], res["enh"], res["val"]
        improv_trad.append(t["improv"]); improv_base.append(b["improv"])
        improv_enh.append(e["improv"])
        r_bases.append(v["r_before"]); r_enhs.append(v["r_after"])
        lifts.append(v["lift"])

        # winner among tensor variants
        if e["improv"] > b["improv"] + 0.1:
            tensor_winner = "enh ▲"
        elif b["improv"] > e["improv"] + 0.1:
            tensor_winner = "base ▲"
        else:
            tensor_winner = "tie"

        print(f"  {seed:>4}  {t['improv']:>7.2f}×  {b['improv']:>9.2f}×  "
              f"{e['improv']:>9.2f}×  "
              f"{v['r_before']:>7.3f}  {v['r_after']:>6.3f}  "
              f"{v['lift']:>+6.3f}  {tensor_winner}")

    def _nanmean(vals):
        clean = [v for v in vals if not math.isnan(v)]
        return sum(clean) / len(clean) if clean else float("nan")

    mt = _nanmean(improv_trad)
    mb = _nanmean(improv_base)
    me = _nanmean(improv_enh)
    mr = _nanmean(r_bases)
    mr2 = _nanmean(r_enhs)
    ml = _nanmean(lifts)

    print(f"  {'─'*4}  {'─'*8}  {'─'*10}  {'─'*10}"
          f"  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*8}")
    print(f"  {'Mean':>4}  {mt:>7.2f}×  {mb:>9.2f}×  {me:>9.2f}×  "
          f"{mr:>7.3f}  {mr2:>6.3f}  {ml:>+6.3f}")

    best_tensor = "enhanced" if me > mb + 0.05 else ("base" if mb > me + 0.05 else "tied")
    print(f"\n  Best tensor variant: {best_tensor}  "
          f"(base mean {mb:.2f}× vs enhanced mean {me:.2f}×)")
    print(f"  Validation: mean r_base={mr:.3f}  r_enh={mr2:.3f}  lift={ml:+.3f}")


def main():
    all_rows = {}
    for scenario in ["simple_missing_feature", "cad_derived"]:
        rows = run_scenario(scenario)
        all_rows[scenario] = rows
        print_summary(scenario, rows)

    # Machine-readable summary for README
    print("\n\n" + "="*72)
    print("  MACHINE-READABLE SUMMARY (copy into README tables)")
    print("="*72)
    for scenario, rows in all_rows.items():
        print(f"\n### {scenario}")
        print(f"| Seed | Trad | Base TSM | Enh TSM | r_base | r_enh | lift |")
        print(f"|-----:|-----:|---------:|--------:|-------:|------:|-----:|")
        improv_t, improv_b, improv_e = [], [], []
        for seed, res in rows:
            t, b, e, v = res["trad"], res["base"], res["enh"], res["val"]
            improv_t.append(t["improv"]); improv_b.append(b["improv"])
            improv_e.append(e["improv"])
            print(f"| {seed} | {t['improv']:.2f}× | {b['improv']:.2f}× | "
                  f"{e['improv']:.2f}× | {v['r_before']:.3f} | "
                  f"{v['r_after']:.3f} | {v['lift']:+.3f} |")
        clean_t = [v for v in improv_t if not math.isnan(v)]
        clean_b = [v for v in improv_b if not math.isnan(v)]
        clean_e = [v for v in improv_e if not math.isnan(v)]
        mt = sum(clean_t)/len(clean_t) if clean_t else float("nan")
        mb = sum(clean_b)/len(clean_b) if clean_b else float("nan")
        me = sum(clean_e)/len(clean_e) if clean_e else float("nan")
        print(f"| **Mean** | **{mt:.2f}×** | **{mb:.2f}×** | **{me:.2f}×** | | | |")


if __name__ == "__main__":
    main()
