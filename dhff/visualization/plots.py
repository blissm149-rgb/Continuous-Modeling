"""Module 8: Diagnostic and result visualization."""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from dhff.core.types import (
    ComplexRCS, ObservationPoint, ScatteringCenter, ScatteringCenterAnomaly, AnomalyType,
)


def plot_rcs_comparison(
    ground_truth: ComplexRCS,
    simulation: ComplexRCS,
    fused: ComplexRCS,
    uncertainty: np.ndarray | None = None,
    x_axis: str = "theta",
    x_values: np.ndarray | None = None,
    title: str = "RCS Comparison",
) -> Figure:
    """2-panel figure: magnitude and phase comparison."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(title)

    N = len(ground_truth.values)
    x = x_values if x_values is not None else np.arange(N)

    gt_mag = ground_truth.magnitude_dbsm
    sim_mag = simulation.magnitude_dbsm
    fused_mag = fused.magnitude_dbsm
    gt_phase = np.degrees(ground_truth.phase_rad)
    sim_phase = np.degrees(simulation.phase_rad)
    fused_phase = np.degrees(fused.phase_rad)

    ax1 = axes[0]
    ax1.plot(x, gt_mag, 'k-', label='Truth', linewidth=1.5)
    ax1.plot(x, sim_mag, 'b--', label='Sim', linewidth=1.5)
    ax1.plot(x, fused_mag, 'r-', label='Fused', linewidth=1.5)
    if uncertainty is not None and len(uncertainty) == N:
        sigma = np.sqrt(np.maximum(uncertainty, 1e-30))
        sigma_db = 10.0 * np.log10(np.maximum(np.abs(fused.values) + sigma, 1e-30)) - fused_mag
        ax1.fill_between(x, fused_mag - sigma_db, fused_mag + sigma_db,
                          alpha=0.2, color='red', label='±1σ')
    ax1.set_ylabel('Magnitude (dBsm)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(x, gt_phase, 'k-', label='Truth')
    ax2.plot(x, sim_phase, 'b--', label='Sim')
    ax2.plot(x, fused_phase, 'r-', label='Fused')
    ax2.set_ylabel('Phase (degrees)')
    ax2.set_xlabel(x_axis)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_discrepancy_map(
    theta_values: np.ndarray,
    freq_values: np.ndarray,
    discrepancy_magnitude: np.ndarray,
    measured_points: list[ObservationPoint] | None = None,
    title: str = "Discrepancy Map",
) -> Figure:
    """2D pcolormesh of |delta| in dB over (theta, freq)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(title)

    disc_db = 10.0 * np.log10(np.maximum(discrepancy_magnitude, 1e-30))
    T, F = np.meshgrid(theta_values, freq_values / 1e9, indexing='ij')
    pcm = ax.pcolormesh(T, F, disc_db, cmap='viridis', shading='auto')
    plt.colorbar(pcm, ax=ax, label='Discrepancy (dB)')

    if measured_points:
        thetas = [p.theta for p in measured_points]
        freqs = [p.freq_hz / 1e9 for p in measured_points]
        ax.scatter(thetas, freqs, c='white', s=20, alpha=0.7, zorder=5)

    ax.set_xlabel('Theta (rad)')
    ax.set_ylabel('Frequency (GHz)')
    fig.tight_layout()
    return fig


def plot_susceptibility_prior(
    theta_values: np.ndarray,
    freq_values: np.ndarray,
    d_prior: np.ndarray,
    selected_points: list[ObservationPoint] | None = None,
    title: str = "Discrepancy Susceptibility Prior",
) -> Figure:
    """2D pcolormesh of D_prior."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(title)

    T, F = np.meshgrid(theta_values, freq_values / 1e9, indexing='ij')
    pcm = ax.pcolormesh(T, F, d_prior, cmap='hot_r', shading='auto', vmin=0, vmax=1)
    plt.colorbar(pcm, ax=ax, label='D_prior')

    if selected_points:
        thetas = [p.theta for p in selected_points]
        freqs = [p.freq_hz / 1e9 for p in selected_points]
        ax.scatter(thetas, freqs, marker='x', c='red', s=60, zorder=5, label='Selected')
        ax.legend()

    ax.set_xlabel('Theta (rad)')
    ax.set_ylabel('Frequency (GHz)')
    fig.tight_layout()
    return fig


def plot_acquisition_function(
    theta_values: np.ndarray,
    freq_values: np.ndarray,
    acquisition_values: np.ndarray,
    next_batch: list[ObservationPoint] | None = None,
    title: str = "Acquisition Function",
) -> Figure:
    """2D pcolormesh of acquisition function."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(title)

    T, F = np.meshgrid(theta_values, freq_values / 1e9, indexing='ij')
    pcm = ax.pcolormesh(T, F, acquisition_values, cmap='plasma', shading='auto')
    plt.colorbar(pcm, ax=ax, label='Acquisition score')

    if next_batch:
        thetas = [p.theta for p in next_batch]
        freqs = [p.freq_hz / 1e9 for p in next_batch]
        ax.scatter(thetas, freqs, marker='*', c='lime', s=100, zorder=5, label='Next batch')
        ax.legend()

    ax.set_xlabel('Theta (rad)')
    ax.set_ylabel('Frequency (GHz)')
    fig.tight_layout()
    return fig


def plot_scattering_centers(
    sim_centers: list[ScatteringCenter],
    meas_centers: list[ScatteringCenter],
    anomalies: list[ScatteringCenterAnomaly] | None = None,
    title: str = "Scattering Center Comparison",
) -> Figure:
    """2D scatter of scattering centers with anomaly annotations."""
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle(title)

    if sim_centers:
        sx = [c.x for c in sim_centers]
        sy = [c.y for c in sim_centers]
        ax.scatter(sx, sy, marker='s', c='blue', s=80, label='Sim centers', zorder=3)

    if meas_centers:
        mx = [c.x for c in meas_centers]
        my = [c.y for c in meas_centers]
        ax.scatter(mx, my, marker='o', c='red', s=80, label='Meas centers', zorder=3)

    if anomalies:
        for anom in anomalies:
            if anom.anomaly_type == AnomalyType.POSITION_SHIFT:
                sc, mc = anom.sim_center, anom.meas_center
                if sc and mc:
                    ax.plot([sc.x, mc.x], [sc.y, mc.y], '--', c='orange', linewidth=1.5)
                    ax.scatter([sc.x, mc.x], [sc.y, mc.y], c='orange', s=120, zorder=4)
            elif anom.anomaly_type == AnomalyType.AMPLITUDE_DISCREPANCY:
                sc, mc = anom.sim_center, anom.meas_center
                if sc and mc:
                    ax.scatter([sc.x, mc.x], [sc.y, mc.y], c='purple', s=120, zorder=4)
            elif anom.anomaly_type == AnomalyType.UNMATCHED_MEASUREMENT:
                mc = anom.meas_center
                if mc:
                    ax.scatter([mc.x], [mc.y], marker='*', c='red', s=200, zorder=5)
                    ax.annotate('MISSING FROM SIM', (mc.x, mc.y),
                                textcoords='offset points', xytext=(5, 5), fontsize=8)
            elif anom.anomaly_type == AnomalyType.UNMATCHED_SIMULATION:
                sc = anom.sim_center
                if sc:
                    ax.scatter([sc.x], [sc.y], marker='x', c='blue', s=200, zorder=5, linewidths=2)
                    ax.annotate('SIM ARTIFACT', (sc.x, sc.y),
                                textcoords='offset points', xytext=(5, 5), fontsize=8)

    ax.set_xlabel('Downrange (m)')
    ax.set_ylabel('Crossrange (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_convergence(
    history: list[dict],
    ground_truth=None,
    simulator=None,
    eval_points: list[ObservationPoint] | None = None,
    title: str = "Convergence",
) -> Figure:
    """Line plot of convergence metrics vs number of measurements."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(title)

    if not history:
        ax.text(0.5, 0.5, 'No history data', ha='center', va='center')
        return fig

    n_meas = [h.get("n_measurements", 0) for h in history]
    n_anom = [h.get("n_anomalies", 0) for h in history]

    ax.plot(n_meas, n_anom, 'b-o', markersize=4, label='Anomalies detected')

    # Annotate phase transitions
    phases = [h.get("phase", 0) for h in history]
    prev_phase = phases[0] if phases else 0
    for i, (nm, ph) in enumerate(zip(n_meas, phases)):
        if ph != prev_phase:
            ax.axvline(nm, linestyle='--', color='gray', alpha=0.7)
            ax.text(nm, ax.get_ylim()[0] if ax.get_ylim()[0] != ax.get_ylim()[1] else 0,
                    f'P{ph}', ha='center', fontsize=8)
            prev_phase = ph

    ax.set_xlabel('Number of measurements')
    ax.set_ylabel('Anomalies detected')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_kk_consistency(
    freq_hz: np.ndarray,
    discrepancy: np.ndarray,
    kk_result: dict,
    title: str = "Kramers-Kronig Consistency",
) -> Figure:
    """2-panel KK consistency figure."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(title)

    freq_ghz = freq_hz / 1e9

    ax1 = axes[0]
    ax1.plot(freq_ghz, discrepancy.real, 'b-', label='Re[delta]')
    ax1.set_ylabel('Re[delta]')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(freq_ghz, discrepancy.imag, 'b-', label='Im[delta] (actual)')
    predicted_im = kk_result.get("predicted_imag", np.zeros_like(discrepancy.imag))
    ax2.plot(freq_ghz, predicted_im, 'r--', label='Im[delta] (predicted by KK)')
    ax2.fill_between(freq_ghz, discrepancy.imag, predicted_im,
                     alpha=0.2, color='red', label='Difference')

    score = kk_result.get("kk_violation_score", 0.0)
    diagnosis = kk_result.get("diagnosis", "unknown")
    ax2.set_title(f"KK violation={score:.3f}, diagnosis={diagnosis}")
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('Im[delta]')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_model_comparison(
    comparison_results: dict,
    title: str = "Method Comparison",
) -> Figure:
    """Grouped bar chart comparing methods."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    fig.suptitle(title)

    table = comparison_results.get("comparison_table", {})
    if not table:
        for ax in axes:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig

    methods = list(table.keys())
    colors = ['steelblue', 'coral', 'mediumseagreen']

    metrics = ["nmse", "anomalies_found", "coverage_68"]
    labels = ["NMSE", "Anomalies Found", "Coverage 68%"]

    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[idx]
        values = [table[m].get(metric, 0) for m in methods]
        bars = ax.bar(methods, values, color=colors[:len(methods)])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        ax.set_title(label)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=8)

    fig.tight_layout()
    return fig
