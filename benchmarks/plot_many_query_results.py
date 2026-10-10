"""Render the many-query scientific and performance summary figure."""

from __future__ import annotations

import argparse
import csv
import io
import json
import lzma
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def csv_rows(path: Path) -> list[dict[str, str]]:
    compressed = path.with_suffix(path.suffix + ".xz")
    if path.exists():
        stream = path.open(newline="")
    elif compressed.exists():
        stream = lzma.open(compressed, mode="rt", newline="")
    else:
        parts = sorted(path.parent.glob(path.name + ".xz.part*"))
        payload = lzma.decompress(b"".join(part.read_bytes() for part in parts))
        stream = io.StringIO(payload.decode("utf-8"), newline="")
    with stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("validation_directory", type=Path)
    args = parser.parse_args()
    directory = args.validation_directory
    uq = csv_rows(directory / "uq_10000_cpu_results.csv")
    cpu = csv_rows(directory / "many_query_cpu_performance.csv")
    gpu = csv_rows(directory / "many_query_gpu_performance.csv")
    cpu_summary = json.loads((directory / "many_query_cpu_summary.json").read_text())

    heat = np.array([float(row["total_heat_W_per_m"]) for row in uq])
    amplitude = np.array([float(row["amplitude_bound"]) for row in uq])
    linear_error = 100 * np.array([float(row["linearized_relative_error"]) for row in uq])
    relative_heat = 100 * (heat / cpu_summary["nominal_heat_W_per_m"] - 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), constrained_layout=True)
    axes[0, 0].hist(relative_heat, bins=55, color="#2878B5", alpha=0.86)
    axes[0, 0].axvline(0, color="black", lw=1)
    axes[0, 0].set(xlabel="Heat-flow change from nominal (%)", ylabel="Realizations",
                   title="10,000 synthetic spatial fields")
    axes[0, 0].text(0.98, 0.94, "24.47% thermally unchanged",
                    transform=axes[0, 0].transAxes, ha="right", va="top", fontsize=9)

    hb = axes[0, 1].hexbin(amplitude, linear_error, gridsize=42, mincnt=1,
                           cmap="magma", bins="log")
    axes[0, 1].axhline(0, color="white", lw=0.8, alpha=0.8)
    axes[0, 1].set(xlabel="Bound on fractional gap deviation",
                   ylabel="Linear-surrogate error (%)",
                   title="Nominal linearization misses branch activation")
    fig.colorbar(hb, ax=axes[0, 1], label="log count")

    selected_cpu = [row for row in cpu if row["stage"] in
                    ("legacy_scalar", "topology_reuse", "cpu_ensemble",
                     "preconditioner_reuse", "production_cpu")]
    labels = []
    rates = []
    for row in selected_cpu:
        if row["stage"] == "cpu_ensemble" and row["processes"] not in ("1", "6"):
            continue
        label = {
            "legacy_scalar": "legacy",
            "topology_reuse": "CSR reuse",
            "preconditioner_reuse": "reused PCG",
            "production_cpu": "CPU production",
        }.get(row["stage"], f"direct {row['processes']}p")
        labels.append(label); rates.append(float(row["samples_per_second"]))
    axes[1, 0].bar(np.arange(len(rates)), rates, color="#35A16B")
    axes[1, 0].set_xticks(np.arange(len(rates)), labels, rotation=24, ha="right")
    axes[1, 0].set(ylabel="Cases per second", title="Measured CPU paths")
    axes[1, 0].grid(axis="y", alpha=0.25)

    batch = np.array([int(row["batch_size"]) for row in gpu])
    throughput = np.array([float(row["samples_per_second"]) for row in gpu])
    axes[1, 1].semilogx(batch, throughput, marker="o", lw=2, color="#D55E00")
    axes[1, 1].axhline(
        cpu_summary["performance"][-1]["samples_per_second"],
        color="#35A16B", linestyle="--", label="CPU production")
    axes[1, 1].set(xlabel="GPU batch size", ylabel="Cases per second",
                   title="RTX 2080 Ti batching crossover")
    axes[1, 1].grid(alpha=0.25); axes[1, 1].legend()
    fig.savefig(directory / "many_query_summary.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
