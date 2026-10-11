"""Generate publication-quality scientific and performance figures.

The script consumes only machine-readable records already written under the
validation directory.  It intentionally does not rerun any solver.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import lzma
from collections import defaultdict
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
MAGENTA = "#CC79A7"
GOLD = "#E69F00"
SKY = "#56B4E9"
GRAY = "#666666"
MODEL_COLORS = {
    "single_exponential": BLUE,
    "two_exponential": ORANGE,
    "analytical_modal": GREEN,
    "gaussian_overlap": MAGENTA,
}
MODEL_LABELS = {
    "single_exponential": "single exponential",
    "two_exponential": "two exponential",
    "analytical_modal": "analytical modes",
    "gaussian_overlap": "Gaussian overlap",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    compressed = path.with_suffix(path.suffix + ".xz")
    if path.exists():
        stream = path.open(newline="", encoding="utf-8")
    elif compressed.exists():
        stream = lzma.open(compressed, mode="rt", newline="", encoding="utf-8")
    else:
        parts = sorted(path.parent.glob(path.name + ".xz.part*"))
        payload = lzma.decompress(b"".join(part.read_bytes() for part in parts))
        stream = io.StringIO(payload.decode("utf-8"), newline="")
    with stream:
        return list(csv.DictReader(stream))


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.13,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )


def finish(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def scientific_uq_figure(directory: Path, output: Path) -> None:
    rows = read_csv(directory / "uq_10000_cpu_results.csv")
    profiles = read_csv(directory / "uq_spatial_width_profile.csv")
    summary = json.loads((directory / "many_query_cpu_summary.json").read_text())

    nominal = float(summary["nominal_heat_W_per_m"])
    heat = np.array([float(row["total_heat_W_per_m"]) for row in rows])
    change = 100.0 * (heat / nominal - 1.0)
    amplitude = 100.0 * np.array([float(row["amplitude_bound"]) for row in rows])
    min_gap = np.minimum(
        [float(row["gap0_min_mm"]) for row in rows],
        [float(row["gap1_min_mm"]) for row in rows],
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.3), constrained_layout=True)
    fig.suptitle("Synthetic spatial gap uncertainty: response and mechanism", fontsize=15)

    ax = axes[0, 0]
    grid = np.linspace(0.001, max(change), 700)
    exceedance = np.array([np.mean(change > value) for value in grid])
    ax.semilogy(grid, np.maximum(exceedance, 1.0 / len(change)), color=BLUE, lw=2.2)
    for threshold, color in zip((0.5, 1.0, 2.0), (SKY, GOLD, ORANGE)):
        probability = np.mean(change > threshold)
        ax.axvline(threshold, color=color, lw=1.2, ls="--")
        ax.plot(threshold, probability, "o", color=color, ms=5)
        ax.annotate(
            f">{threshold:g}%: {probability:.1%}",
            (threshold, probability),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )
    ax.set(
        xlabel="Heat-flow increase from nominal (%)",
        ylabel="Exceedance probability",
        ylim=(8e-5, 1.05),
        title="Tail risk across 10,000 realizations",
    )
    ax.grid(alpha=0.25, which="both")
    panel_label(ax, "a")

    ax = axes[0, 1]
    hb = ax.hexbin(amplitude, change, gridsize=46, mincnt=1, bins="log", cmap="viridis")
    edges = np.linspace(amplitude.min(), amplitude.max(), 13)
    centers = 0.5 * (edges[:-1] + edges[1:])
    medians = []
    p90 = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = change[(amplitude >= low) & (amplitude < high)]
        medians.append(np.median(selected))
        p90.append(np.quantile(selected, 0.9))
    median_line, = ax.plot(centers, medians, color="white", marker="o", ms=3, lw=1.5, label="bin median")
    p90_line, = ax.plot(centers, p90, color="white", ls="--", lw=1.3, label="bin 90th percentile")
    outline = [path_effects.Stroke(linewidth=3.0, foreground="black"), path_effects.Normal()]
    median_line.set_path_effects(outline)
    p90_line.set_path_effects(outline)
    ax.set(
        xlabel="Bound on local gap deviation (%)",
        ylabel="Heat-flow increase (%)",
        title="Severity controls the upper tail",
    )
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.colorbar(hb, ax=ax, label="log count")
    panel_label(ax, "b")

    ax = axes[1, 0]
    hb = ax.hexbin(min_gap, change, gridsize=46, mincnt=1, bins="log", cmap="magma")
    ax.axvspan(9.0, 12.5, color=SKY, alpha=0.13, label="flat-conductance interval")
    ax.axvline(9.0, color=SKY, lw=1.2, ls="--")
    ax.set(
        xlabel="Minimum local gap across both cavities (mm)",
        ylabel="Heat-flow increase (%)",
        title="Finite fields activate the narrow-gap branch",
    )
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.colorbar(hb, ax=ax, label="log count")
    panel_label(ax, "c")

    ax = axes[1, 1]
    x_mm = np.array([1000.0 * float(row["x_m"]) for row in profiles])
    for gap, color, label in (("gap0", BLUE, "gap 1"), ("gap1", ORANGE, "gap 2")):
        mean = np.array([float(row[f"{gap}_mean_mm"]) for row in profiles])
        low = np.array([float(row[f"{gap}_p05_mm"]) for row in profiles])
        high = np.array([float(row[f"{gap}_p95_mm"]) for row in profiles])
        ax.fill_between(x_mm, low, high, color=color, alpha=0.16)
        ax.plot(x_mm, mean, color=color, lw=1.7, label=f"{label}: mean and 5–95%")
    ax.axhline(9.0, color=GRAY, lw=1, ls="--", label="flat-interval bounds")
    ax.axhline(12.5, color=GRAY, lw=1, ls="--")
    ax.axhline(10.0, color="black", lw=1, ls=":", label="nominal 10 mm")
    ax.set(
        xlabel="Position along section (mm)",
        ylabel="Gap width (mm)",
        title="Prescribed spatial-field envelope",
        ylim=(7.4, 12.65),
    )
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower center")
    ax.grid(alpha=0.2)
    panel_label(ax, "d")

    finish(fig, output / "uq-scientific-evidence")


def decay_and_separation_figure(directory: Path, output: Path) -> None:
    penalties = read_csv(directory / "severity_heat_penalties.csv")
    profiles = read_csv(directory / "severity_profiles.csv")
    decay = read_csv(directory / "decay_model_comparison.csv")
    interactions = read_csv(directory / "separation_interaction_predictions.csv")
    scores = read_csv(directory / "separation_prediction_scores.csv")
    mesh = read_csv(directory / "severity_study_mesh_check.csv")

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.4), constrained_layout=True)
    fig.suptitle("Defect severity, spatial decay, and pair interaction", fontsize=15)

    ax = axes[0, 0]
    severity = 100.0 * np.array([float(row["severity"]) for row in penalties])
    penalty = 100.0 * np.array([float(row["relative_penalty"]) for row in penalties])
    ax.plot(severity, penalty, color=BLUE, marker="o", lw=2)
    for x, y in zip(severity, penalty):
        ax.annotate(f"{y:.2f}%", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    ax.set(xlabel="Gap-narrowing severity (%)", ylabel="Heat-flow penalty (%)", title="Nonlinear total penalty")
    ax.grid(alpha=0.25)
    panel_label(ax, "a")

    grouped: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for row in profiles:
        grouped[float(row["severity"])].append((float(row["x_m"]), float(row["delta_q_W_m2"])))
    ax = axes[0, 1]
    for level, color in ((0.2, BLUE), (0.4, GREEN), (0.6, ORANGE)):
        values = np.array(sorted(grouped[level]))
        normalized = values[:, 1] / np.max(np.abs(values[:, 1]))
        ax.plot(1000.0 * values[:, 0], normalized, color=color, lw=1.8, label=f"{level:.0%}")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(120, color=GRAY, lw=1, ls=":", label="defect center")
    ax.set_yscale("symlog", linthresh=2e-4, linscale=0.7)
    ax.set(
        xlabel="Position along section (mm)",
        ylabel="Normalized heat-flux perturbation",
        title="Weak sign-changing return field",
        ylim=(-2e-3, 1.2),
    )
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(alpha=0.2, which="both")
    panel_label(ax, "b")

    ax = axes[0, 2]
    for model in ("single_exponential", "two_exponential", "analytical_modal"):
        selected = sorted((row for row in decay if row["model"] == model), key=lambda row: float(row["severity"]))
        x = 100.0 * np.array([float(row["severity"]) for row in selected])
        y = np.array([float(row["test_relative_l2"]) for row in selected])
        ax.plot(x, y, marker="o", lw=1.8, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    ax.axhspan(0.5, 0.7, color=GRAY, alpha=0.08)
    ax.set(
        xlabel="Gap-narrowing severity (%)",
        ylabel="Blocked-tail relative L2 error",
        title="Two exponentials win, but all extrapolate poorly",
        ylim=(0.48, 0.71),
    )
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    panel_label(ax, "c")

    ax = axes[1, 0]
    for level, color in ((0.2, BLUE), (0.4, GREEN), (0.6, ORANGE)):
        selected = sorted((row for row in interactions if float(row["severity"]) == level), key=lambda row: float(row["separation_m"]))
        x = 1000.0 * np.array([float(row["separation_m"]) for row in selected])
        y = np.array([float(row["normalized_interaction"]) for row in selected])
        ax.plot(x, y, color=color, marker="o", lw=1.8, label=f"{level:.0%}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_yscale("symlog", linthresh=1e-4, linscale=0.7)
    ax.set(
        xlabel="Defect separation (mm)",
        ylabel="Normalized non-additive interaction",
        title="Interaction collapses by 50–80 mm",
    )
    ax.legend(title="severity", frameon=False, fontsize=8)
    ax.grid(alpha=0.2, which="both")
    panel_label(ax, "d")

    ax = axes[1, 1]
    levels = (0.2, 0.4, 0.6)
    models = ("single_exponential", "two_exponential", "analytical_modal", "gaussian_overlap")
    width = 0.18
    base = np.arange(len(levels))
    for index, model in enumerate(models):
        values = []
        for level in levels:
            row = next(row for row in scores if float(row["severity"]) == level and row["model"] == model)
            values.append(float(row["held_out_relative_l2"]))
        ax.bar(base + (index - 1.5) * width, values, width=width, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    ax.set_xticks(base, [f"{level:.0%}" for level in levels])
    ax.set(
        xlabel="Gap-narrowing severity",
        ylabel="Held-out separation relative L2 error",
        title="No universal pair-interaction predictor",
    )
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    panel_label(ax, "e")

    ax = axes[1, 2]
    cells = np.array([int(row["nx"]) * int(row["ny"]) for row in mesh])
    increments = np.array([float(row["defect_increment_W_per_m"]) for row in mesh])
    ax.plot(cells, increments, color=GREEN, marker="o", lw=2)
    for row, x, y in zip(mesh, cells, increments):
        ax.annotate(
            f"{row['nx']}×{row['ny']}",
            (x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set(
        xlabel="Finite-volume cells",
        ylabel="30% defect increment (W m$^{-1}$)",
        title="Defect increment is mesh-converged",
    )
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(alpha=0.25)
    panel_label(ax, "f")

    finish(fig, output / "decay-separation-evidence")


def execution_figure(directory: Path, output: Path) -> None:
    cpu = read_csv(directory / "many_query_cpu_performance.csv")
    gpu = read_csv(directory / "many_query_gpu_performance.csv")
    cpu_summary = json.loads((directory / "many_query_cpu_summary.json").read_text())
    gpu_summary = json.loads((directory / "many_query_gpu_summary.json").read_text())

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.3), constrained_layout=True)
    fig.suptitle("Measured many-query execution path", fontsize=15)

    labels = ["legacy direct", "fixed-CSR direct", "reused-factor PCG", "CPU production", "GPU production"]
    rates = [
        float(next(row for row in cpu if row["stage"] == "legacy_scalar")["samples_per_second"]),
        float(next(row for row in cpu if row["stage"] == "cpu_ensemble" and row["processes"] == "1")["samples_per_second"]),
        float(next(row for row in cpu if row["stage"] == "preconditioner_reuse")["samples_per_second"]),
        float(next(row for row in cpu if row["stage"] == "production_cpu")["samples_per_second"]),
        float(gpu_summary["production_samples_per_second"]),
    ]
    colors = [GRAY, SKY, GREEN, BLUE, ORANGE]
    ax = axes[0, 0]
    y = np.arange(len(labels))
    ax.barh(y, rates, color=colors)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set(xlabel="Throughput (cases s$^{-1}$, log scale)", title="Reuse changes the feasible campaign size")
    for yi, rate in zip(y, rates):
        ax.text(rate * 1.06, yi, f"{rate:,.0f}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.25, which="both")
    panel_label(ax, "a")

    ax = axes[0, 1]
    ensemble = sorted((row for row in cpu if row["stage"] == "cpu_ensemble"), key=lambda row: int(row["processes"]))
    processes = np.array([int(row["processes"]) for row in ensemble])
    speedup = np.array([float(row["speedup_vs_direct_1p"]) for row in ensemble])
    ax.plot(processes, processes, color=GRAY, ls="--", lw=1.3, label="ideal")
    ax.plot(processes, speedup, color=MAGENTA, marker="o", lw=2, label="measured")
    ax.axhline(1, color="black", lw=0.8)
    ax.set(
        xlabel="Worker processes",
        ylabel="Speedup relative to one process",
        title="Process ensembles are rejected on this host",
        xticks=processes,
        ylim=(0, 6.4),
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    panel_label(ax, "b")

    ax = axes[1, 0]
    batch = np.array([int(row["batch_size"]) for row in gpu])
    throughput = np.array([float(row["samples_per_second"]) for row in gpu])
    cpu_rate = float(cpu_summary["performance"][-1]["samples_per_second"])
    ax.semilogx(batch, throughput, color=ORANGE, marker="o", lw=2, label="GPU crossover probe")
    ax.axhline(cpu_rate, color=BLUE, ls="--", lw=1.5, label=f"CPU production: {cpu_rate:.0f} cases/s")
    ax.scatter([gpu_summary["selected_batch_size"]], [gpu_summary["production_samples_per_second"]], s=70, marker="D", color=GREEN, zorder=4, label="10,000-case GPU production")
    ax.set(
        xlabel="Simultaneous GPU systems per batch",
        ylabel="Throughput (cases s$^{-1}$)",
        title="GPU advantage appears only after aggregation",
    )
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25, which="both")
    panel_label(ax, "c")

    ax = axes[1, 1]
    errors = gpu_summary["maximum_relative_error_vs_cpu"]
    quantities = ["total heat", "edge fraction", "peak flux", "surface temp.", "GPU residual"]
    ratios = [
        errors["total_heat_W_per_m"] / 1.5e-4,
        errors["edge_heat_fraction"] / 2.0e-4,
        errors["peak_heat_flux_W_m2"] / 2.0e-4,
        errors["minimum_indoor_surface_K"] / 2.0e-4,
        gpu_summary["maximum_gpu_residual_relative_l2"] / 5.0e-4,
    ]
    x = np.arange(len(quantities))
    ax.bar(x, ratios, color=[BLUE, SKY, ORANGE, GREEN, MAGENTA])
    ax.axhline(1.0, color="black", lw=1.3, ls="--", label="acceptance limit")
    ax.set_xticks(x, quantities, rotation=22, ha="right")
    ax.set(
        ylabel="Observed maximum / declared limit",
        title="Every GPU accuracy gate retains margin",
        ylim=(0, 1.12),
    )
    for xi, value in zip(x, ratios):
        ax.text(xi, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    panel_label(ax, "d")

    finish(fig, output / "many-query-execution-evidence")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("validation_directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    directory = args.validation_directory
    output = args.output_directory or directory.parent / "figures"

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    scientific_uq_figure(directory, output)
    decay_and_separation_figure(directory, output)
    execution_figure(directory, output)


if __name__ == "__main__":
    main()
