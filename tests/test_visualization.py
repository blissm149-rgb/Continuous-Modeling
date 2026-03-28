"""Tests for Module 8: Visualization."""
import math

import numpy as np
import pytest
from matplotlib.figure import Figure

from dhff.core import ObservationPoint, make_frequency_sweep
from dhff.core.types import ComplexRCS, ScatteringCenter, ScatteringCenterAnomaly, AnomalyType
from dhff.visualization import plots


def make_simple_rcs(n=30):
    pts = make_frequency_sweep(math.pi/2, 0.0, (8e9, 12e9), n)
    vals = np.exp(1j * np.linspace(0, 4 * math.pi, n)) * 0.5 + 0.01
    return ComplexRCS(observation_points=pts, values=vals.astype(np.complex128))


def test_plot_rcs_comparison_returns_figure():
    gt = make_simple_rcs()
    sim = make_simple_rcs()
    fused = make_simple_rcs()
    fig = plots.plot_rcs_comparison(gt, sim, fused)
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_rcs_comparison_with_uncertainty():
    gt = make_simple_rcs()
    sim = make_simple_rcs()
    fused = make_simple_rcs()
    unc = np.ones(30) * 0.01
    x_vals = np.linspace(0, math.pi, 30)
    fig = plots.plot_rcs_comparison(gt, sim, fused, uncertainty=unc, x_values=x_vals)
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_discrepancy_map_returns_figure():
    theta = np.linspace(0.2, math.pi - 0.2, 10)
    freq = np.linspace(8e9, 12e9, 10)
    disc = np.random.default_rng(0).uniform(0, 1, (10, 10))
    fig = plots.plot_discrepancy_map(theta, freq, disc)
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_susceptibility_prior_returns_figure():
    theta = np.linspace(0.2, math.pi - 0.2, 10)
    freq = np.linspace(8e9, 12e9, 10)
    d_prior = np.random.default_rng(1).uniform(0, 1, (10, 10))
    fig = plots.plot_susceptibility_prior(theta, freq, d_prior)
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_acquisition_function_returns_figure():
    theta = np.linspace(0.2, math.pi - 0.2, 10)
    freq = np.linspace(8e9, 12e9, 10)
    acq = np.random.default_rng(2).uniform(0, 1, (10, 10))
    fig = plots.plot_acquisition_function(theta, freq, acq)
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_scattering_centers_returns_figure():
    sim = [ScatteringCenter(x=0.0, y=0.0, amplitude=1.0+0j)]
    meas = [ScatteringCenter(x=0.1, y=0.0, amplitude=0.9+0j)]
    fig = plots.plot_scattering_centers(sim, meas)
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_scattering_centers_with_anomalies():
    sim = [ScatteringCenter(x=0.0, y=0.0, amplitude=1.0+0j)]
    meas = [ScatteringCenter(x=0.15, y=0.0, amplitude=1.0+0j)]
    anomalies = [
        ScatteringCenterAnomaly(
            anomaly_type=AnomalyType.UNMATCHED_MEASUREMENT,
            meas_center=meas[0], sim_center=None,
        ),
    ]
    fig = plots.plot_scattering_centers(sim, meas, anomalies=anomalies)
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_convergence_empty_history():
    fig = plots.plot_convergence([])
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_convergence_with_history():
    history = [
        {"phase": 1, "n_measurements": 5, "n_anomalies": 0},
        {"phase": 2, "n_measurements": 10, "n_anomalies": 1},
        {"phase": 3, "n_measurements": 15, "n_anomalies": 2},
    ]
    fig = plots.plot_convergence(history)
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_kk_consistency_returns_figure():
    freq = np.linspace(8e9, 12e9, 64)
    disc = np.exp(1j * np.linspace(0, 4*math.pi, 64)).astype(np.complex128)
    kk_result = {
        "kk_violation_score": 0.2,
        "is_causal": True,
        "diagnosis": "missing_scatterer",
        "predicted_imag": np.sin(np.linspace(0, 4*math.pi, 64)),
    }
    fig = plots.plot_kk_consistency(freq, disc, kk_result)
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_model_comparison_returns_figure():
    comparison = {
        "comparison_table": {
            "dhff_hybrid": {"nmse": 0.1, "anomalies_found": 2, "coverage_68": 0.7},
            "dhff_pure_gp": {"nmse": 0.2, "anomalies_found": 1, "coverage_68": 0.6},
            "uniform_baseline": {"nmse": 0.3, "anomalies_found": 0, "coverage_68": 0.5},
        }
    }
    fig = plots.plot_model_comparison(comparison)
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)
