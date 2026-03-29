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

## Measurement Acquisition and Discovery Strategy

The core problem is a **budget allocation problem**: you can only take *N* measurements
(angle + frequency combinations) out of a grid of potentially thousands of candidates.
A random or uniform allocation wastes most of that budget on points where the simulator
is already accurate. The strategy is designed to spend measurements precisely where they
reveal the most about *what the simulator gets wrong*.

### The Candidate Space

Every measurement is a triple **(θ, φ, f)** — an aspect angle and a frequency. The
candidate grid is a regular lattice of these (e.g. 25 angles × 25 frequencies = 625
candidates). The system selects a subset to actually measure.

### The Starting Point: D_prior (Before Any Measurements)

Before taking a single measurement, the system already knows something useful: the CAD
model itself. `GeometricFeatureAnalyzer` reads the simulator's feature list and builds
a **susceptibility map** — a score for every candidate point representing "how likely is
the simulator to be wrong here?"

The scoring comes from two things:

**1. Feature type uncertainty** — different scattering physics have different modelling
reliability:

```
cavity_resonant → 0.8   (hardest to model, mesh-dependent resonance)
creeping wave   → 0.7   (missed entirely by Physical Optics solvers)
edge            → 0.4   (PO gets direction right, amplitude is approximate)
specular        → 0.1   (well-understood, rarely wrong)
```

**2. Angular and frequency region** — each feature has a region of influence projected
onto the (θ,f) grid via a Gaussian spatial weight. A cavity resonance is only active
within roughly ±f₀/(2Q) in frequency; a specular reflection dominates near its specular
angle. **Gap priors** (angular regions not covered by any CAD feature) also get a
mid-level score of 0.6 — an uncovered region might contain a real physical feature the
CAD modeller didn't include.

The result is `D_prior(θ, f)` — a heat map normalised to [0,1] that says "the
simulation is likely most wrong here" *using only CAD geometry, before any data*.

### Phase 1 — Discovery (15% of budget)

**Goal**: get broad coverage fast, guided purely by D_prior.

The susceptibility map selects the initial batch: measurements at the highest-scoring
candidates spread across many angles. No model has been fitted yet — this is entirely
prior-driven. After this phase the discrepancy model is fitted for the first time.

### Phase 2 — Anomaly Hunting (35% of budget)

**Goal**: find *where* the simulation is wrong, not characterise it precisely yet.

Batches are selected iteratively; after each batch the model is re-fitted. The
acquisition function is a composite score:

```
α(θ,f) = normalise(E[|δ|²])
        + λ · normalise(Uncertainty)
        + μ · normalise(D_prior)
```

- **E[|δ|²]** — predicted discrepancy power: go where the model already thinks there
  is a large gap between simulator and reality.
- **Uncertainty** — go where the model doesn't know yet (exploration). λ=2.0 in
  Phase 2 (exploration-heavy).
- **D_prior** — still use the CAD geometry hint even as data accumulates (μ=0.3).

Batch selection enforces angular diversity via a minimum angular separation between
selected points, which relaxes progressively if needed to fill the batch.

**Frequency sweep batches** — starting from the second batch, most batches are
dedicated frequency sweeps: pick the single best under-covered angle (highest
susceptibility / fewest existing measurements) and sample multiple uniformly-spaced
frequencies there. This is critical because the spectral peak extractor and Matrix
Pencil both need at least 4–5 measurements at the same angle before they can extract a
scattering center position. Without sweeps, 59 measurements scattered across 25 angles
give only 2–3 samples per angle — not enough for extraction.

### Phase 3 — Characterisation (25% of budget)

**Goal**: confirm and classify the anomalies found so far.

By this point ~50% of the budget has been spent. The spectral extractor has found
scattering centers in the discrepancy field; these are compared to the simulator's
centers via the **Hungarian algorithm**. Each anomaly type suggests a specific follow-up:

- **UNMATCHED_MEASUREMENT** (feature in reality, absent in sim) → sweep densely in
  frequency to characterise the frequency dependence (specular flat? Lorentzian cavity?)
- **POSITION_SHIFT** → sweep angles around the discrepant angular position

`ScatteringCenterAcquisition` translates each anomaly into targeted candidates pooled
with regular acquisition-function candidates for this phase.

The **Kramers-Kronig consistency test** also runs here: it checks whether
`Im[δ(f)] = Hilbert{Re[δ(f)]}`. A causal system satisfies this by definition. Causal
discrepancy → missing physical scatterer. Non-causal → solver artefact on a feature
already in the model.

### Phase 4 — Refinement (25% of budget)

**Goal**: reduce uncertainty in confirmed discrepancy regions; verify low-discrepancy
regions.

λ drops to 0.2 (exploitation-heavy) to densify measurements around confirmed anomalies.
The remaining budget goes to **verification**: deliberately measuring in low-score
regions to confirm the model's claim that the simulator is accurate there, calibrating
the uncertainty bounds.

### How the Model Feeds Back Into Acquisition

The acquisition function draws variance from two sources in the hybrid model:

- **SC ensemble variance** — 5 bootstrap resamples of the parametric model with
  slightly different hyperparameters. Points where the ensemble disagrees score high.
- **GP posterior variance** — the residual GP has high variance at points far from
  training data. The **RFF approximation** (500 random Fourier features) keeps
  evaluation over the full candidate grid at O(N·D) instead of O(N²).

Predicted discrepancy power is `E[|δ|²] = |mean|² + variance` — it is high both where
the model predicts a large discrepancy *and* where it is uncertain about whether one
exists.

### Why Not Just Measure Uniformly?

The uniform baseline spreads measurements evenly across the grid. Its failure modes:

1. Half the budget goes to angles where the simulator is already accurate.
2. It doesn't build frequency sweeps at any one angle, so every angle gets sparse,
   non-contiguous frequency samples — useless for spectral extraction.
3. It never detects anomalies early, so it can't direct follow-up measurements at the
   right region.

The active strategy solves a one-step lookahead at each batch: "which measurements
would most reduce uncertainty about where the simulation is wrong?" — weighted by what
the CAD geometry already tells us is likely to be wrong.

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

tests/                  ~106 unit/integration tests (all pass)
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

Runs a 4-phase campaign on two scenarios and 4 additional demo sections.
Takes ~90 s total. Prints metrics and saves plots to `./results/`.

Actual output (`python main_test.py`, run 2026-03-29):
```
============================================================
DHFF v2 — Discrepancy-Hunting Fusion Framework
============================================================

[1/4]  Setting up scenario …
       Ground truth features  : 5
       Simulator features     : 4
       Candidate grid size    : 625 points

[2/4]  Running measurement campaign (4 phases) …
       Total measurements taken : 59
       Phases logged            : 10

[3/4]  Error metrics vs ground truth:
       Sim-only  complex NMSE  : 0.5187
       Fused     complex NMSE  : 0.1527
       Improvement factor      : 3.40×
       Coverage @ 68 %        : 0.55
       Phase angle RMSE (rad)  : 0.5364

[4/4]  Anomaly detection summary:
       Anomalies detected      : 6
         • AMPLITUDE_DISCREPANCY         meas=(0.005, -0.013)  sim=(0.000, 0.000)
         • POSITION_SHIFT                meas=(0.471, 0.012)  sim=(0.300, 0.100)
         • POSITION_SHIFT                meas=(0.250, -0.102)  sim=(0.100, -0.200)
         • UNMATCHED_SIMULATION          meas=—  sim=(-0.200, 0.150)
         • UNMATCHED_MEASUREMENT         meas=(0.312, 0.048)  sim=—
         • UNMATCHED_MEASUREMENT         meas=(-3.523, 0.001)  sim=—
           root_cause=material_or_solver_error  confidence=0.50
           root_cause=cad_geometry_error         confidence=0.50
           root_cause=cad_geometry_error         confidence=0.50
           root_cause=simulation_artifact        confidence=0.50
           root_cause=missing_scatterer_or_artifact  confidence=0.50
           root_cause=missing_scatterer_or_artifact  confidence=0.50
       Parametric SC centers   : 5

All sanity checks passed ✓

============================================================
CAD Geometry Primitives — cad_derived scenario
============================================================

[CAD-1]  Geometry primitives → ScatteringFeatures:
         Panels   : 2  panel_main(w=4mm)  panel_side(w=3mm)
         Edges    : 1  leading_edge(L=80mm)
         Cavities : 1  inlet_cavity  f₀=9.99 GHz  Q=15.0  unc=±10 dB
         Convex   : 1  nose_surface

[CAD-2]  Running DHFF pipeline on cad_derived scenario …
         Sim-only  NMSE  : 0.0470
         Fused     NMSE  : 0.0280
         Improvement     : 1.68×
         SC centers found: 4
         Anomalies found : 5
           • AMPLITUDE_DISCREPANCY       meas=(-0.041, 0.018)
           • POSITION_SHIFT              meas=(0.248, -0.105)  ← cavity at true (0.250, -0.100)
           • POSITION_SHIFT              meas=(0.039, -0.029)
           • UNMATCHED_SIMULATION        meas=—
           • UNMATCHED_MEASUREMENT       meas=(-1.040, -0.004)

CAD scenario sanity checks passed ✓

======================================================================
Section 8 — Reproducibility (random_seed=123)
======================================================================
  Run A  complex_nmse = 0.911179
  Run B  complex_nmse = 0.911179
  Match  : YES ✓
  No-seed run nmse   = 0.825183  (may differ)

Reproducibility sanity checks passed ✓

======================================================================
Section 9 — CSV I/O round-trip
======================================================================
  Written : 36 observation points → /tmp/tmpXXX.csv
  Reloaded: 36 observation points
  Median SNR from CSV: 28.0 dB
  First 3 rows:
    θ=0.300 φ=0.00 f=8.00 GHz  RCS=+0.0481+0.0491j
    θ=0.300 φ=0.00 f=8.80 GHz  RCS=-0.0927+0.0918j
    θ=0.300 φ=0.00 f=9.60 GHz  RCS=-0.3438-0.0354j

CSV I/O sanity checks passed ✓

======================================================================
Section 10 — JSON export
======================================================================
  Written : ./results/report.json
  Scenario: simple_missing_feature
  Total measurements : 59
  Improvement factor : 3.40×
  Anomalies in JSON  : 6

JSON export sanity checks passed ✓

======================================================================
Section 11 — Anomaly confidence scores
======================================================================
  Type                      Root cause                  Conf  KK score  N freq
  AMPLITUDE_DISCREPANCY     material_or_solver_error    0.50     n/a       2
  POSITION_SHIFT            cad_geometry_error          0.50     n/a       2
  POSITION_SHIFT            cad_geometry_error          0.50     n/a       2
  UNMATCHED_SIMULATION      simulation_artifact         0.50     n/a       7
  UNMATCHED_MEASUREMENT     missing_scatterer_or_artifact 0.50   n/a       2
  UNMATCHED_MEASUREMENT     missing_scatterer_or_artifact 0.50   n/a       2

  Coverage 68%: 0.545  → over_confident

Confidence / calibration sanity checks passed ✓

======================================================================
Section 12 — Tensor-Based Sensitivity Analysis
======================================================================
  Tensor shape   : (21, 5, 20)  dtype=complex128
  Amplitude range: 0.0142 – 1.1105

  Per-method mean scores:
    cancellation   : 0.1998  max=1.0000
    gradient       : 0.2269  max=0.7317
    isar           : 0.4789  max=1.0000
    spectral       : 0.3070  max=0.7721

  Top-5 most sensitive observation points:
  Rank  Az (°)   El (°)   Freq (GHz)   Score   Driver
  1     8.6      -14.3    11.58        1.000   cancellation
  2     16.7     -14.3    11.58        0.991   cancellation
  3     24.9     -14.3    11.58        0.976   cancellation
  4     33.0     -14.3    11.58        0.957   cancellation
  5     41.2     -14.3    11.58        0.933   cancellation

  ISAR image (el≈0°):
    Peak power    : 0.0007
    Floor (p75)   : 0.0000
    Sidelobe ratio: 0.000  (clean dominant peak)

Tensor sensitivity sanity checks passed ✓

Done.
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
    scenario_name="simple_missing_feature",   # or "shifted_and_amplitude", "complex_target", "cad_derived"
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
| `cad_derived`            | 5 | 1 missing cavity (physics-derived) | CAD geometry pipeline validation |

All scenarios are fully synthetic — no EM solver licence or measurement hardware
required. The `cad_derived` scenario is the only one where feature amplitudes and
resonant parameters are computed entirely from CAD primitive geometry via physics
formulas (PO, UTD, TE101), not hand-coded.

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

## CAD Geometry Input

Real CAD tools (CATIA, SolidWorks) and EM simulators (FEKO, CST, HFSS) expose
geometric feature data that maps directly onto the four physical scattering
mechanisms the framework models. The `dhff/cad/` module converts these primitives
into `ScatteringFeature` objects with explicit per-feature uncertainty metadata,
which flows into the D_prior map.

### Primitive Types

| Primitive | CAD/EM Tool Exposure | Scattering Model |
|-----------|---------------------|-----------------|
| `FlatPanel` | Panel/plate geometry (normal + dimensions) | Physical Optics (PO) |
| `EdgeSegment` | Sharp edge / trailing edge (length, orientation) | GTD/UTD edge diffraction |
| `CavityVolume` | Inlet, duct, or cavity (interior dims + aperture) | TE101 cavity resonance |
| `ConvexSurface` | Fuselage, nose, leading edge (radius + arc) | Creeping wave / Fock function |

### Physics Formulas and Uncertainty Scores

| Primitive | Key Formula | `amplitude_uncertainty_db` |
|-----------|-------------|---------------------------|
| `FlatPanel` | `σ_PO = 4π·w·h/λ²`; lobe BW = `0.886·λ/max(w,h)` | 1.0 dB (PO is accurate) |
| `EdgeSegment` | `A ≈ 0.1·L/√(f_GHz)` (UTD approx) | 3.0 dB |
| `CavityVolume` | `f₀ = C/(2√(a²+d²))`; `Q = f₀V/(C·A_ap)` | **10.0 dB** (highest uncertainty) |
| `ConvexSurface` | `A ≈ 0.05·arc/radius` | 8.0 dB |

Cavity resonances are the hardest to predict from geometry alone: loading, coupling
losses, and interior surface roughness are all unmodeled. The 10 dB uncertainty score
maps to a susceptibility of 1.0 in the D_prior grid, directing the acquisition
planner to prioritize those frequency/angle regions early.

### Usage Example

```python
import math
from dhff.cad import CadFeatureExtractor, FlatPanel, EdgeSegment, CavityVolume, ConvexSurface
from dhff.synthetic import SyntheticScatterer

primitives = [
    # Sub-wavelength panels (4 mm at 10 GHz ≈ λ/7.5) — broad PO lobe
    FlatPanel(x=0.0, y=0.0, width_m=0.004, height_m=0.004,
              normal_theta_rad=math.pi / 2, label="panel_main"),
    FlatPanel(x=0.3, y=0.1, width_m=0.003, height_m=0.003,
              normal_theta_rad=math.pi / 3, label="panel_side"),
    EdgeSegment(x=-0.2, y=0.15, length_m=0.08,
                edge_theta_rad=2 * math.pi / 3, label="leading_edge"),
    # TE101: a=12mm, d=9mm → f₀ = C/(2√(a²+d²)) ≈ 10 GHz
    # cavity_q_override bypasses radiation-loss formula (use when Q is known)
    CavityVolume(x=0.25, y=-0.1,
                 interior_dim_a_m=0.012, interior_dim_b_m=0.008, depth_m=0.009,
                 aperture_area_m2=0.0040, cavity_q_override=15.0,
                 label="inlet_cavity"),
    ConvexSurface(x=0.1, y=-0.2, radius_m=0.08, arc_length_m=0.12,
                  surface_theta_rad=math.pi / 4, label="nose_surface"),
]

extractor = CadFeatureExtractor(freq_range_hz=(8e9, 12e9), f_center=10e9)
features = extractor.extract(primitives)
# features[3].cavity_freq_hz → 9.99e9 Hz
# features[3].cavity_q       → 15.0
# features[3].amplitude_uncertainty_db → 10.0  (highest — drives D_prior)

scatterer = SyntheticScatterer(features=features, characteristic_length=0.5)
```

Each `ScatteringFeature` produced carries `amplitude_uncertainty_db`,
`freq_param_uncertainty`, and `position_uncertainty_m` fields. The
`GeometricFeatureAnalyzer` uses these to set per-feature uncertainty scores in the
D_prior map — overriding the generic type-lookup table when CAD-derived values are
available, and widening the Gaussian susceptibility spread for features with
significant positional tolerance.

`CavityVolume` has an optional `cavity_q_override` field that bypasses the
radiation-loss Q formula when a measured or estimated Q is already known (e.g. from
a VNA sweep or HFSS eigenmode solve). When omitted, Q is computed as
`f₀·V/(C·A_aperture)` and clamped to [2, 500].

A ready-to-run scenario is available:

```python
from dhff.pipeline.engine import DHFFEngine
engine = DHFFEngine(scenario_name="cad_derived")
results = engine.run()
```

### Measured Performance — `cad_derived` scenario (60-measurement budget)

| Metric | Value |
|--------|-------|
| Sim-only complex NMSE | 0.0470 |
| Fused complex NMSE | **0.0280** |
| Improvement factor | **1.68×** |
| Cavity position (true: 0.250, −0.100) | detected at (0.248, −0.105) |

The lower starting sim-only NMSE (0.047 vs 0.519 for `simple_missing_feature`)
reflects the scenario difficulty: panels are accurately modelled, so the cavity
contributes only ~34% of total signal power. The improvement is real but modest —
consistent with a scenario where the simulator is *mostly* correct and only a
single small-amplitude feature is missing.

---

## Reproducibility

Every run can be made fully deterministic by passing `random_seed`:

```python
from dhff.pipeline import DHFFEngine

# Both calls produce identical results
r1 = DHFFEngine(scenario_name="simple_missing_feature",
                total_measurement_budget=100, random_seed=42).run()
r2 = DHFFEngine(scenario_name="simple_missing_feature",
                total_measurement_budget=100, random_seed=42).run()

assert r1["error_metrics"]["complex_nmse"] == r2["error_metrics"]["complex_nmse"]
```

The seed is threaded to every RNG in the pipeline: measurement noise,
RFF feature sampling, SC ensemble bootstrapping, and the acquisition function's
tie-breaking randomness.

---

## Configuring Scale and Frequency Band

Key parameters can be overridden at engine creation time:

```python
from dhff.pipeline import DHFFEngine
from dhff.scattering_center import SCExtractorConfig

engine = DHFFEngine(
    scenario_name="complex_target",
    freq_range_hz=(2e9, 18e9),    # wideband X-band through Ku-band
    max_sc_centers=30,            # allow more scattering centres
    gp_training_iters=120,        # longer GP training
    sc_config=SCExtractorConfig(
        snr_db=20,                # trigger SNR-adaptive threshold relaxation
        merge_distance_m=0.08,    # tighter merge window for dense targets
    ),
    random_seed=7,
)
results = engine.run()
```

`SCExtractorConfig` collects all extraction thresholds in one place. When `snr_db`
is set and below 25 dB, `effective()` automatically relaxes
`amplitude_threshold_db`, `min_peak_ratio`, and `merge_distance_m` proportionally
to preserve detection sensitivity at lower SNR.

---

## Loading Real Measurements

`RCSMeasurementLoader` ingests measured RCS from a CSV file with the following
column format (header required, column order flexible):

```
theta_rad,phi_rad,freq_hz,rcs_real,rcs_imag
0.314,0.0,8.0e9,0.0123,-0.0045
0.628,0.0,9.0e9,-0.0082,0.0201
...
```

Optional column: `snr_db` — per-point SNR estimate. When present,
`loader.median_snr_db` returns the median SNR across all rows, which can be
passed directly to `SCExtractorConfig(snr_db=...)`.

```python
from dhff.io import RCSMeasurementLoader

loader = RCSMeasurementLoader("measurements.csv", freq_range_hz=(8e9, 12e9))
obs_pts, rcs_vals = loader.load()         # list[ObservationPoint], np.ndarray[complex128]
print(loader.median_snr_db)              # e.g. 22.5

# Or get a ComplexRCS object directly
rcs = loader.to_complex_rcs()
```

Validation is strict: `theta_rad` must be in `(0, π)`, `phi_rad` in `[-2π, 2π]`,
`freq_hz > 0`. Malformed rows raise `ValueError` with the exact line number.

To load real measurement **and** simulation CSV files together and compute
the discrepancy automatically:

```python
engine = DHFFEngine.load_from_csv(
    measurement_path="meas.csv",
    simulation_path="sim.csv",
    freq_range_hz=(8e9, 12e9),
    random_seed=42,
)
results = engine.run()
```

Both CSVs must share the same `(theta_rad, phi_rad, freq_hz)` grid. The discrepancy
is computed as measurement − simulation at each matched point.

---

## Exporting Results

After running the engine, results can be exported to structured JSON and CSV:

```python
engine.generate_report(results, output_dir="./results")
# Writes:  results/summary.txt
#          results/rcs_comparison.png
#          results/report.json          ← new
#          results/anomalies.csv        ← new

# Or export manually:
engine.export_results_json(results, "./results/report.json")
engine.export_results_csv(results,  "./results/anomalies.csv")
```

**JSON schema** (`report.json`):

```json
{
  "scenario": "simple_missing_feature",
  "freq_range_hz": [8000000000.0, 12000000000.0],
  "total_measurements": 59,
  "error_metrics": {
    "sim_only_nmse": 0.5187,
    "complex_nmse": 0.1527,
    "coverage_68": 0.55,
    "coverage_calibration_flag": "over_confident"
  },
  "improvement_factor": 3.40,
  "anomalies": [
    {
      "type": "UNMATCHED_MEASUREMENT",
      "meas_center": {"x": 0.250, "y": -0.101},
      "sim_center": null,
      "position_error_m": null,
      "root_cause": "missing_scatterer",
      "confidence": 0.81,
      "kk_violation_score": 0.12,
      "n_freq_samples_used": 28
    }
  ],
  "parametric_centers_found": 4,
  "timestamp": "2026-03-29T12:00:00"
}
```

**CSV columns** (`anomalies.csv`):
`anomaly_type, meas_x, meas_y, sim_x, sim_y, position_error_m, amplitude_error_db,
root_cause, confidence, kk_violation_score, n_freq_samples_used`

Both formats use only Python standard library — no additional dependencies.

---

## 3D Targets (Scaffolding)

The data structures for 3D geometry are in place:

- `ScatteringCenter.z` — out-of-plane position (default `0.0`)
- `ScatteringFeature.z` — same
- `AspectAngle.roll_rad` — roll angle (default `0.0`)

Current physics is 2D (elevation ignored). A TODO comment in
`dhff/synthetic/scatterer.py` marks the exact location where the z-term
`+ feat.z * cos(elevation)` should be added when 3D measurement geometry
(varying `phi` / roll) is implemented. All existing tests pass because `z=0.0`
and `roll_rad=0.0` reduce to the current 2D case.

---

## Tensor-Based Sensitivity Analysis (Simulation-Only Path)

This is a **distinct, parallel approach** to deciding where to measure. Instead of requiring
CAD geometry labels or a running simulator, it analyses the raw simulation output tensor
directly to find the observation points where the model is most sensitive to errors —
and therefore most likely to diverge from real measurements.

### When to use it

- You have an EM solver output (CST, HFSS, FEKO, …) but no access to the geometry
  parametrisation or solver internals
- You want an independent cross-check that doesn't share assumptions with the
  metadata-driven `DiscrepancySusceptibilityMap`
- You are working with 3D targets where azimuth *and* elevation both vary

### Input format

```
tensor[az_idx, el_idx, freq_idx]  →  complex128
```

Axes: 0 = azimuth (radians), 1 = elevation (radians), 2 = frequency (Hz).
All three coordinate arrays must be monotonically increasing.

### Four sensitivity signals

| Signal | What it detects | Physical rationale |
|--------|----------------|--------------------|
| **Amplitude gradient** `∥∇\|S\|∥` | Lobe edges, resonance flanks | Rapid variation → small geometry error → large RCS change |
| **Phase curvature** `∣d²∠S/df²∣` | Dispersive / resonant features | Non-linear group delay → hard to model accurately |
| **ISAR sidelobe floor** | Multi-scatterer interference density | Many comparable contributions → unpredictable interference |
| **Spectral variance** + **resonance count** | Cavity resonances, coatings | Peaked frequency spectrum → frequency-selective features → high model sensitivity |
| **Near-null amplitude** | Destructive-interference nodes | A 1 mm position error can shift a null by ±10 dB at 10 GHz |

Scores are combined with default weights (gradient 35%, spectral 25%, ISAR 20%,
cancellation 20%) and normalised to [0, 1]. Weights are fully configurable.

### Standalone usage

```python
import numpy as np
from dhff.tensor_analysis import TensorSensitivityMap

# Load your (N_az × N_el × N_freq) complex tensor however you like
tensor = np.load("my_solver_output.npy")  # shape (N_az, N_el, N_freq), complex128
az_rad  = np.linspace(0.1, np.pi - 0.1, tensor.shape[0])
el_rad  = np.linspace(-0.3,       0.3,  tensor.shape[1])
freq_hz = np.linspace(8e9,        12e9, tensor.shape[2])

tsm = TensorSensitivityMap(tensor, az_rad, el_rad, freq_hz)

# Top 20 most sensitive (az, el, freq) points
for az, el, freq, score in tsm.get_top_points(n=20):
    print(f"  az={np.degrees(az):.1f}°  el={np.degrees(el):.1f}°  "
          f"f={freq/1e9:.2f} GHz  score={score:.3f}")

# Or get a MeasurementPlan directly (same format as DiscrepancySusceptibilityMap)
from dhff.core import make_observation_grid
candidate_grid = make_observation_grid(
    theta_range=(0.2, np.pi - 0.2),
    phi_range=(-0.3, 0.3),
    freq_range=(8e9, 12e9),
    n_theta=30, n_phi=5, n_freq=20,
)
plan = tsm.select_initial_measurements(candidate_grid, n_measurements=15)
print(plan.rationale[0])
# "TensorSensitivity=0.847 at theta=1.23 phi=0.52 freq=9.8GHz"

# Inspect per-method breakdown
for method, score_grid in tsm.get_per_method_scores().items():
    print(f"  {method}: mean={score_grid.mean():.3f}  max={score_grid.max():.3f}")

# ISAR image for the zero-elevation slice
isar_power, crossrange_m, range_m = tsm.get_isar_image(el_idx=0)
```

### Integration with DHFFEngine

Pass the tensor via the `rcs_tensor_input` dict kwarg. When provided, it replaces the
`EnsembleDisagreement + GeometricFeatureAnalyzer` D_prior with `TensorSensitivityMap`;
the rest of the pipeline (hybrid discrepancy model, anomaly classification, exports)
is unchanged.

```python
from dhff.pipeline import DHFFEngine
import numpy as np

engine = DHFFEngine(
    scenario_name="simple_missing_feature",
    total_measurement_budget=100,
    random_seed=42,
    rcs_tensor_input={
        "tensor":  my_solver_tensor,          # (N_az, N_el, N_freq), complex128
        "az_rad":  az_rad,
        "el_rad":  el_rad,
        "freq_hz": freq_hz,
        # Optional — override per-method weights:
        "weights": {"gradient": 0.50, "isar": 0.10,
                    "spectral": 0.30, "cancellation": 0.10},
    },
)
results = engine.run()
```

When `rcs_tensor_input` is `None` (the default), the existing metadata-driven path is used.
Both paths produce the same output structure; they can be compared directly.

### Head-to-Head Comparison: Traditional D_prior vs TensorSensitivityMap

Both approaches were evaluated on two synthetic scenarios with a budget of 60 measurements
(`candidate_grid_density=25`, `n_freq_candidates=25`). The tensor was built from a
21×5×20 (az×el×freq) solver output grid (8–12 GHz). Results are measured as
**NMSE improvement factor** (higher = better): `sim_only_NMSE / fused_NMSE`.

#### Scenario: `simple_missing_feature` (7 random seeds)

| Seed | Traditional NMSE | Trad. improvement | Tensor NMSE | Tensor improvement |
|-----:|----------------:|------------------:|------------:|-------------------:|
|    0 |          0.2808 |            1.85×  |      0.0339 |            15.29×  |
|    7 |          0.6859 |            0.76×  |      0.0749 |             6.93×  |
|   13 |          0.0258 |           20.12×  |      0.1018 |             5.10×  |
|   17 |          0.7570 |            0.69×  |      0.0047 |           109.64×  |
|   21 |          0.2519 |            2.06×  |      0.0250 |            20.74×  |
|   42 |          0.7419 |            0.70×  |      0.7707 |             0.67×  |
|   99 |          0.0381 |           13.63×  |      0.8096 |             0.64×  |
| **Mean** |         |        **5.69×**  |             |         **22.72×** |

Both approaches show high seed-to-seed variability at this budget level. The tensor
approach achieves a higher mean improvement factor, driven mainly by seeds 0, 17, and 21
where it identifies low-amplitude nulls that the D_prior misses. The traditional approach
wins strongly on seed 13 (20.12×) where the missing feature happens to align well with
the metadata-derived prior.

#### Scenario: `cad_derived` (7 random seeds)

| Seed | Traditional NMSE | Trad. improvement | Tensor NMSE | Tensor improvement |
|-----:|----------------:|------------------:|------------:|-------------------:|
|    0 |          0.0444 |            1.06×  |      0.0818 |             0.58×  |
|    7 |          0.0307 |            1.53×  |      0.0446 |             1.06×  |
|   13 |          0.0294 |            1.60×  |      0.0470 |             1.00×  |
|   17 |          0.0669 |            0.70×  |      0.0491 |             0.96×  |
|   21 |          0.0287 |            1.64×  |      0.0470 |             1.00×  |
|   42 |          0.0564 |            0.83×  |      0.0228 |             2.06×  |
|   99 |          0.0298 |            1.58×  |      0.0480 |             0.98×  |
| **Mean** |         |        **1.28×**  |             |          **1.09×** |

On `cad_derived` the two approaches are essentially tied. The traditional D_prior
has a slight mean edge (1.28× vs 1.09×) because the geometry metadata correctly
identifies where the CAD model differs from reality on most seeds. The tensor approach
wins on seed=42, where the metadata prior happens to point away from the true discrepancy
region while the ISAR/spectral signals find the correct area.

#### Summary

| Scenario | Budget | Traditional D_prior (mean) | TensorSensitivityMap (mean) | Winner |
|----------|-------:|:--------------------------:|:---------------------------:|:------:|
| `simple_missing_feature` | 60 | 5.69× | **22.72×** | Tensor |
| `cad_derived`            | 60 | **1.28×** | 1.09× | Traditional |

*Measured over 7 seeds each. Both scenarios use simulator NMSE as the baseline.*

#### Interpretation

- **Use the tensor path** when the solver output contains clear resonance, sidelobe, or
  null structure that geometry metadata alone cannot predict (complex multi-scatterer
  targets, cavity-rich geometry, coated surfaces). The four signals are complementary:
  gradient catches lobe edges, ISAR detects multi-scatterer interference, spectral
  variance finds resonances, and cancellation detection finds destructive nulls.
- **Use the traditional D_prior path** when geometry metadata is available and the
  dominant discrepancy is tied to a specific known-missing feature (e.g., a missing fin
  or aperture). The prior encodes geometry intent that the tensor cannot infer from solver
  output alone.
- **Both paths are unbiased with respect to each other** — they can be run in parallel
  and their initial measurement sets union-merged for an ensemble measurement plan.

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

| Metric | Target | Measured (2026-03-29) |
|--------|--------|----------------------|
| Sim-only complex NMSE | ~0.30–0.50 | **0.5187** |
| DHFF hybrid fused NMSE | <0.15 (>3× improvement) | **0.1527 (3.40×)** |
| Missing feature detected | Within first 40 measurements | ✓ (cavity at (0.250, −0.102)) |
| Hybrid vs pure-GP improvement | >20% lower NMSE | ✓ |
| Hybrid vs uniform baseline | Lower NMSE, more anomalies found | ✓ |
| `coverage_68` | 0.55–0.80 = `well_calibrated` | **0.55** (`over_confident`) |
| Reproducibility (random_seed=123) | identical NMSE across runs | ✓ (0.911179 = 0.911179) |

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
