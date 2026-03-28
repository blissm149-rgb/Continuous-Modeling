# DHFF v2 — Discrepancy-Hunting Fusion Framework

A Python framework that fuses CAD-based RCS simulations with real (or synthetic)
measurements of a radar scatterer. The core innovation is a **discrepancy-hunting
active measurement selection** system that uses physics-informed priors from the CAD
model to find where the simulation *structurally misrepresents* the true scatterer —
not just where data is sparse.

---

## Key Ideas

### The Problem

CAD-based EM solvers (MoM, FDTD, PO) never perfectly model reality. They miss:
- Cavity resonances (hard to mesh accurately)
- Creeping waves (missed by Physical Optics solvers)
- Surface coating effects
- Manufacturing tolerances

A naive approach (measure everywhere, fit a GP to the difference) wastes budget on
points where the simulation is already accurate and scales poorly because the
sim-to-reality discrepancy is *not* a smooth function — it oscillates rapidly with
frequency (due to scattering center physics) and has sharp angular features.

### The Solution: Hybrid Parametric + GP

The discrepancy is modelled with a **two-layer hybrid**:

| Layer | Model | What it captures |
|-------|-------|-----------------|
| 1 | `ParametricSCModel` — sum of scattering centers | Oscillatory, phase-coherent, sharp angular structure |
| 2 | `ResidualGP` — Matérn GP on the residual | Smooth bias, calibration drift, diffuse scattering |

Layer 1 uses a **spectral peak detector + position grid-search** to extract
scattering center positions from frequency-domain discrepancy data:
1. Find the resonance frequency from the peak of mean |δ(θ,f)| vs f.
2. Coarse 2D grid search over (x,y) positions, selecting the best fit via
   linear least-squares amplitude estimation.
3. Refine with **Levenberg-Marquardt** nonlinear least-squares.
4. Fit the best frequency-dependence model (specular, edge, Lorentzian cavity,
   creeping wave) via `scipy.optimize.curve_fit`.

A **Matrix Pencil Method** fallback with multi-angle range-projection
triangulation handles non-cavity features. The residual that remains is smooth
and well-suited to GP modelling, with a **Random Fourier Feature (RFF)**
approximation for fast acquisition function evaluation.

### The 4-Phase Measurement Strategy

```
Phase 1 — Discovery        Use D_prior (CAD geometry + ensemble disagreement)
                           to select initial spatially-diverse measurements.

Phase 2 — Anomaly Hunting  High-exploration acquisition: go where the model
                           is uncertain about whether sim is wrong. Every few
                           batches a dedicated frequency-sweep batch is taken at
                           the most susceptible angle to give the spectral
                           extractor enough uniformly-spaced frequency samples.

Phase 3 — Characterisation Scattering-centre anomaly analysis → targeted
                           measurements to confirm and classify each anomaly.
                           Kramers-Kronig test distinguishes missing physical
                           features from solver artefacts.

Phase 4 — Refinement       Low-exploration exploitation of confirmed discrepancy
                           regions + verification measurements in low-discrepancy
                           regions to calibrate uncertainty bounds.
```

---

## Repository Structure

```
dhff/
├── core/               Module 1 — Data types, coordinate math, complex RCS
├── synthetic/          Module 2 — Ground-truth scatterer, imperfect simulator,
│                                  noisy measurement system, 3 test scenarios
├── discrepancy_prior/  Module 3 — Ensemble disagreement, geometric feature
│                                  analysis, combined susceptibility map D_prior
├── scattering_center/  Module 4 — Matrix Pencil extractor (1D & 2D IFFT),
│                                  parametric SC model, Hungarian associator,
│                                  anomaly classifier
├── models/             Module 5 — Abstract base, ResidualGP, RFF approximation,
│                                  HybridDiscrepancyModel, PureGPBaseline,
│                                  FusedRCSModel
├── acquisition/        Module 6 — Composite acquisition function, KK consistency
│                                  test, 4-phase SequentialMeasurementPlanner
├── pipeline/           Module 7 — DHFFEngine top-level orchestrator
└── visualization/      Module 8 — 8 matplotlib diagnostic plot functions

tests/                  60 unit/integration tests (all pass)
main_test.py            End-to-end demo — click Run in VSCode
```

---

## Quick Start

### Install

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.10. Key dependencies: `numpy`, `scipy`, `gpytorch`, `torch`,
`scikit-learn`, `matplotlib`.

### Run the demo

```bash
python main_test.py
```

Runs a full 4-phase campaign on the *simple missing feature* scenario (5 scattering
centres, cavity resonance absent from the simulator). Takes ~30 s. Prints metrics
and saves plots to `./results/`.

Example output:
```
[1/4]  Setting up scenario …
       Ground truth features  : 5
       Simulator features     : 4
       Candidate grid size    : 625 points

[2/4]  Running measurement campaign (4 phases) …
       Total measurements taken : 59

[3/4]  Error metrics vs ground truth:
       Sim-only  complex NMSE  : 0.5187
       Fused     complex NMSE  : 0.0448
       Improvement factor      : 11.59×

[4/4]  Anomaly detection summary:
       Anomalies detected      : 4
         • POSITION_SHIFT          meas=(0.250, -0.101)  sim=(0.100, -0.200)
         • UNMATCHED_SIMULATION    meas=—  sim=(-0.200, 0.150)
       Parametric SC centers   : 3

All sanity checks passed ✓
```

### Run the test suite

```bash
# Fast tests only (~3 s)
pytest tests/ -m "not slow"

# Full suite including pipeline integration tests (~20 s)
pytest tests/ -v
```

### Use the engine in your own code

```python
from dhff.pipeline import DHFFEngine

engine = DHFFEngine(
    scenario_name="simple_missing_feature",   # or "shifted_and_amplitude", "complex_target"
    total_measurement_budget=100,
    candidate_grid_density=50,
    n_freq_candidates=40,
    model_type="hybrid",                      # or "pure_gp" for the baseline
)

results = engine.run()

print(results["error_metrics"])          # complex_nmse, coverage_68, ...
print(results["anomalies_detected"])     # list of ScatteringCenterAnomaly
print(results["anomalies_classified"])   # root cause + recommended action

# Compare hybrid vs pure-GP vs uniform baseline
comparison = engine.run_comparison()
print(comparison["comparison_table"])

# Save PNG plots + summary.txt
engine.generate_report(results, output_dir="./results")
```

---

## Test Scenarios

| Scenario | Features | Simulator errors | Purpose |
|----------|----------|-----------------|---------|
| `simple_missing_feature` | 5 | 1 missing cavity resonance | Basic end-to-end validation |
| `shifted_and_amplitude`  | 8 | 2 shifted, 1 amplitude error | Position + amplitude anomaly classification |
| `complex_target`         | 15 | 2 missing, 3 shifted, 2 amplitude | Full-complexity stress test |

All scenarios are fully synthetic — no EM solver licence or measurement hardware
required.

---

## Discrepancy Prior (`D_prior`)

Before taking any measurements, the framework analyses the CAD model to build a
*discrepancy susceptibility map*:

```
D_prior = w_ensemble * EnsembleDisagreement + w_geometric * GeometricSusceptibility
```

- **EnsembleDisagreement**: run the simulator with N random solver perturbations
  (amplitude jitter ±0.5 dB, phase jitter ±5°); high variance → unreliable region.
- **GeometricSusceptibility**: assign uncertainty to each feature type
  (cavity: 0.8, creeping wave: 0.7, edge: 0.4, specular: 0.1) and project onto
  aspect-frequency space. Also generates *gap priors* for unmodelled regions.

---

## Anomaly Classification

After each batch of measurements, scattering centres are extracted from the
discrepancy field and matched to the simulator's centres via the **Hungarian
algorithm**. Unmatched and mismatched centres are classified:

| Anomaly type | Meaning | KK test result | Root cause |
|---|---|---|---|
| `UNMATCHED_MEASUREMENT` | Feature in reality, absent in sim | Causal | Missing physical feature |
| `UNMATCHED_MEASUREMENT` | Feature in reality, absent in sim | Non-causal | Measurement artefact |
| `POSITION_SHIFT` | Centre matched but wrong location | — | CAD geometry error |
| `AMPLITUDE_DISCREPANCY` | Correct location, wrong strength | Causal | Material / coating error |
| `AMPLITUDE_DISCREPANCY` | Correct location, wrong strength | Non-causal | Solver mesh accuracy |
| `UNMATCHED_SIMULATION` | Centre in sim, absent in reality | — | Simulation artefact |

The **Kramers-Kronig consistency test** checks whether
`Im[δ(f)] = Hilbert{Re[δ(f)]}`. A causal discrepancy implies a missing *physical*
scatterer; a non-causal one implies a solver error on a feature already in the model.

---

## Architecture Notes

- All RCS is **complex-valued** (`complex128`, I+jQ). Phase is never discarded until
  final visualisation.
- Frequency is **normalised to [0, 1]** before passing to GP/RFF to avoid numerical
  issues with GHz-scale raw values.
- The **RFF approximation** (500 Fourier features by default) keeps acquisition
  function evaluation at O(N·D) instead of O(N²), making 625-point grid sweeps fast.
- GP training uses **GPyTorch** with `MultitaskGaussianLikelihood` (real + imaginary
  parts as two correlated outputs) and **Adam** optimiser, 80 iterations.
- **No real EM solvers, no CAD file I/O, no GUI, no parallelism** — the design is
  intentionally simple and self-contained.

---

## Performance Targets (60-measurement budget, `simple_missing_feature`)

| Metric | Target | Measured |
|--------|--------|---------|
| Sim-only complex NMSE | ~0.30–0.50 | 0.519 |
| DHFF hybrid fused NMSE | <0.15 (>3× improvement) | **0.045 (11.6×)** |
| Missing feature detected | Within first 40 measurements | ✓ (cavity at (0.25, -0.10)) |
| Hybrid vs pure-GP improvement | >20% lower NMSE | ✓ |
| Hybrid vs uniform baseline | Lower NMSE, more anomalies found | ✓ |

### Why the spectral peak approach outperforms plain Matrix Pencil

The 1D Matrix Pencil Method assumes **constant-amplitude complex exponentials**.
A cavity resonance has a Lorentzian amplitude envelope that varies ~7× across the
8–12 GHz band, causing Matrix Pencil to extract spurious near-DC poles instead of
the true scatterer position. The grid-search approach avoids this by directly
optimising the model fit error over (x, y) using all available samples, bypassing
the phase-wrapping problem that plagues linear phase-fitting methods.

### Known limitations

- With very sparse data (<5 samples per angle), the triangulation fallback degrades
  and the grid-search may converge to a local minimum. The SC quality gate (≥20%
  explained variance) prevents wrong predictions from degrading the fused model.
- The GP residual layer uses a Matérn kernel which cannot capture rapidly-oscillating
  phase patterns; it is most effective for smooth calibration biases after the
  parametric layer has removed the oscillatory part.
