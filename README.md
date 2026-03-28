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
    FlatPanel(x=0.0, y=0.0, width_m=0.12, height_m=0.08,
              normal_theta_rad=math.pi / 2, label="main_panel"),
    EdgeSegment(x=-0.2, y=0.15, length_m=0.15,
                edge_theta_rad=2 * math.pi / 3, label="leading_edge"),
    CavityVolume(x=0.25, y=-0.1,
                 interior_dim_a_m=0.015, interior_dim_b_m=0.010, depth_m=0.020,
                 aperture_area_m2=0.0002, label="inlet_cavity"),
]

extractor = CadFeatureExtractor(freq_range_hz=(8e9, 12e9), f_center=10e9)
features = extractor.extract(primitives)

scatterer = SyntheticScatterer(features=features, characteristic_length=0.5)
```

Each `ScatteringFeature` produced carries `amplitude_uncertainty_db`,
`freq_param_uncertainty`, and `position_uncertainty_m` fields. The
`GeometricFeatureAnalyzer` uses these to set per-feature uncertainty scores in the
D_prior map — overriding the generic type-lookup table when CAD-derived values are
available, and widening the Gaussian susceptibility spread for features with
significant positional tolerance.

A ready-to-run scenario is available:

```python
from dhff.pipeline.engine import DHFFEngine
engine = DHFFEngine(scenario_name="cad_derived")
results = engine.run()
```

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
