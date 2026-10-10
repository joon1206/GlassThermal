"""10,000-realization UQ campaign and staged CPU performance measurements."""

from __future__ import annotations

import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import csv
import json
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy.stats import spearmanr

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from thin_igu.fvm2d import EdgeSection, solve_edge_section  # noqa: E402
from thin_igu.many_query import (  # noqa: E402
    EdgeTopology,
    QueryResult,
    generate_spatial_ensemble,
    solve_chunk_direct,
    solve_chunk_preconditioned,
)


def parallel_run(method: str, processes: int, nx: int, ny: int,
                 widths_0: np.ndarray, widths_1: np.ndarray,
                 rtol: float = 1e-9) -> tuple[list[QueryResult], float, float]:
    indices = np.array_split(np.arange(len(widths_0)), processes)
    if method == "direct":
        worker = solve_chunk_direct
        payloads = [(nx, ny, widths_0[index], widths_1[index]) for index in indices]
    elif method == "preconditioned_cg":
        worker = solve_chunk_preconditioned
        payloads = [(nx, ny, widths_0[index], widths_1[index], rtol) for index in indices]
    else:
        raise ValueError(method)
    start = perf_counter()
    if processes == 1:
        returned = [worker(payloads[0])]
    else:
        with ProcessPoolExecutor(max_workers=processes) as pool:
            returned = [future.result() for future in
                        [pool.submit(worker, payload) for payload in payloads]]
    wall = perf_counter() - start
    results = [item for _, chunk, _ in returned for item in chunk]
    worker_seconds = sum(elapsed for _, _, elapsed in returned)
    return results, wall, worker_seconds


def legacy_result(section: EdgeSection, topology: EdgeTopology,
                  width_0: np.ndarray, width_1: np.ndarray) -> float:
    def field(layer: int, x: np.ndarray) -> np.ndarray:
        if not np.array_equal(x, topology.x):
            raise RuntimeError("reference grid differs from topology grid")
        return width_0 if layer == 0 else width_1
    return solve_edge_section(section, nx=topology.nx, ny=topology.ny,
                              gap_width=field).total_heat_indoor


def sensitivity_kernel(topology: EdgeTopology) -> tuple[float, np.ndarray, float]:
    nominal = np.full(topology.nx, 0.010)
    baseline = topology.solve_direct(nominal, nominal).total_heat_W_per_m
    step = 1e-5
    sensitivity = np.empty((2, topology.nx))
    start = perf_counter()
    for layer in range(2):
        for i in range(topology.nx):
            plus_0, plus_1 = nominal.copy(), nominal.copy()
            minus_0, minus_1 = nominal.copy(), nominal.copy()
            (plus_0 if layer == 0 else plus_1)[i] += step
            (minus_0 if layer == 0 else minus_1)[i] -= step
            q_plus = topology.solve_direct(plus_0, plus_1).total_heat_W_per_m
            q_minus = topology.solve_direct(minus_0, minus_1).total_heat_W_per_m
            sensitivity[layer, i] = (q_plus - q_minus) / (2.0 * step)
    return baseline, sensitivity, perf_counter() - start


def statistics(values: np.ndarray) -> dict[str, object]:
    probabilities = np.array((0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0))
    return {
        "mean": float(values.mean()), "standard_deviation": float(values.std(ddof=1)),
        "quantiles": {f"p{int(probability * 100):02d}": float(value)
                      for probability, value in zip(probabilities,
                                                     np.quantile(values, probabilities))},
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run(output_directory: Path, samples: int = 10_000,
        nx: int = 120, ny: int = 36, seed: int = 20261008) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    section = EdgeSection()
    topology_start = perf_counter()
    topology = EdgeTopology(section, nx=nx, ny=ny)
    topology_seconds = perf_counter() - topology_start
    generation_start = perf_counter()
    ensemble = generate_spatial_ensemble(samples, topology.x, seed=seed)
    generation_seconds = perf_counter() - generation_start

    validation_count = 12
    legacy_start = perf_counter()
    legacy = np.array([legacy_result(section, topology, first, second)
                       for first, second in zip(ensemble.width_gap_0_m[:validation_count],
                                                ensemble.width_gap_1_m[:validation_count])])
    legacy_seconds = perf_counter() - legacy_start
    reused_start = perf_counter()
    reused = [topology.solve_direct(first, second)
              for first, second in zip(ensemble.width_gap_0_m[:validation_count],
                                       ensemble.width_gap_1_m[:validation_count])]
    reused_seconds = perf_counter() - reused_start
    reused_q = np.array([item.total_heat_W_per_m for item in reused])
    topology_reference_error = float(np.max(np.abs(reused_q - legacy) / np.abs(legacy)))

    benchmark_count = min(600, samples)
    performance_rows = [{
        "stage": "legacy_scalar", "method": "original_loop_assembly_direct",
        "processes": 1, "samples": validation_count, "wall_seconds": legacy_seconds,
        "samples_per_second": validation_count / legacy_seconds,
        "speedup_vs_direct_1p": "", "efficiency": "",
    }, {
        "stage": "topology_reuse", "method": "fixed_csr_direct",
        "processes": 1, "samples": validation_count, "wall_seconds": reused_seconds,
        "samples_per_second": validation_count / reused_seconds,
        "speedup_vs_direct_1p": "", "efficiency": "",
    }]

    direct_results = None
    direct_one_process_wall = None
    for processes in (1, 2, 4, 6):
        results, wall, _ = parallel_run(
            "direct", processes, nx, ny,
            ensemble.width_gap_0_m[:benchmark_count],
            ensemble.width_gap_1_m[:benchmark_count])
        if processes == 1:
            direct_results = results
            direct_one_process_wall = wall
        speedup = direct_one_process_wall / wall
        performance_rows.append({
            "stage": "cpu_ensemble", "method": "fixed_csr_direct",
            "processes": processes, "samples": benchmark_count,
            "wall_seconds": wall, "samples_per_second": benchmark_count / wall,
            "speedup_vs_direct_1p": speedup, "efficiency": speedup / processes,
        })

    preconditioner_start = perf_counter()
    preconditioner, initial = topology.nominal_reusable_preconditioner()
    preconditioner_setup_seconds = perf_counter() - preconditioner_start
    pcg_start = perf_counter()
    pcg_results = [topology.solve_preconditioned(first, second, preconditioner,
                                                 initial, rtol=1e-9)
                   for first, second in zip(ensemble.width_gap_0_m[:benchmark_count],
                                            ensemble.width_gap_1_m[:benchmark_count])]
    pcg_wall = perf_counter() - pcg_start
    direct_q = np.array([item.total_heat_W_per_m for item in direct_results])
    pcg_q = np.array([item.total_heat_W_per_m for item in pcg_results])
    pcg_reference_error = float(np.max(np.abs(pcg_q - direct_q) / np.abs(direct_q)))
    performance_rows.append({
        "stage": "preconditioner_reuse", "method": "nominal_factorization_pcg",
        "processes": 1, "samples": benchmark_count, "wall_seconds": pcg_wall,
        "samples_per_second": benchmark_count / pcg_wall,
        "speedup_vs_direct_1p": direct_one_process_wall / pcg_wall,
        "efficiency": "",
    })

    # The staged measurement above decides against process ensembles on this
    # Windows host: spawn/import overhead exceeds direct-solve savings. Run the
    # production CPU reference with the fastest verified serial PCG path.
    production_processes = 1
    production_results, production_wall, production_worker_seconds = parallel_run(
        "preconditioned_cg", production_processes, nx, ny,
        ensemble.width_gap_0_m, ensemble.width_gap_1_m, rtol=1e-9)
    performance_rows.append({
        "stage": "production_cpu", "method": "serial_nominal_factorization_pcg",
        "processes": production_processes, "samples": samples,
        "wall_seconds": production_wall, "samples_per_second": samples / production_wall,
        "speedup_vs_direct_1p": "", "efficiency": "",
    })

    baseline, sensitivity, sensitivity_seconds = sensitivity_kernel(topology)
    delta_0 = ensemble.width_gap_0_m - 0.010
    delta_1 = ensemble.width_gap_1_m - 0.010
    linear_prediction = (baseline + delta_0 @ sensitivity[0]
                         + delta_1 @ sensitivity[1])

    rows = []
    total_heat = np.empty(samples)
    for index, result in enumerate(production_results):
        total_heat[index] = result.total_heat_W_per_m
        rows.append({
            "sample": index,
            "amplitude_bound": ensemble.amplitude[index],
            "gap0_mean_mm": 1000.0 * ensemble.width_gap_0_m[index].mean(),
            "gap0_min_mm": 1000.0 * ensemble.width_gap_0_m[index].min(),
            "gap1_mean_mm": 1000.0 * ensemble.width_gap_1_m[index].mean(),
            "gap1_min_mm": 1000.0 * ensemble.width_gap_1_m[index].min(),
            "total_heat_W_per_m": result.total_heat_W_per_m,
            "area_average_u_W_m2K": result.total_heat_W_per_m / (section.length * 39.0),
            "edge_heat_fraction": result.edge_heat_W_per_m / result.total_heat_W_per_m,
            "peak_heat_flux_W_m2": result.peak_heat_flux_W_m2,
            "minimum_indoor_surface_K": result.minimum_indoor_surface_K,
            "linearized_heat_W_per_m": linear_prediction[index],
            "linearized_relative_error": ((linear_prediction[index]
                                             - result.total_heat_W_per_m)
                                            / result.total_heat_W_per_m),
            "pcg_iterations": result.iterations,
            "residual_relative_l2": result.residual_relative_l2,
        })
    write_rows(output_directory / "uq_10000_cpu_results.csv", rows)
    write_rows(output_directory / "many_query_cpu_performance.csv", performance_rows)
    write_rows(output_directory / "uq_linear_sensitivity.csv", [
        {"x_m": x, "gap0_dQ_dw_W_m2": sensitivity[0, i],
         "gap1_dQ_dw_W_m2": sensitivity[1, i]}
        for i, x in enumerate(topology.x)
    ])
    profile_rows = []
    for i, x in enumerate(topology.x):
        profile_rows.append({
            "x_m": x,
            "gap0_mean_mm": 1000 * ensemble.width_gap_0_m[:, i].mean(),
            "gap0_p05_mm": 1000 * np.quantile(ensemble.width_gap_0_m[:, i], 0.05),
            "gap0_p95_mm": 1000 * np.quantile(ensemble.width_gap_0_m[:, i], 0.95),
            "gap1_mean_mm": 1000 * ensemble.width_gap_1_m[:, i].mean(),
            "gap1_p05_mm": 1000 * np.quantile(ensemble.width_gap_1_m[:, i], 0.05),
            "gap1_p95_mm": 1000 * np.quantile(ensemble.width_gap_1_m[:, i], 0.95),
        })
    write_rows(output_directory / "uq_spatial_width_profile.csv", profile_rows)

    relative_change = total_heat / baseline - 1.0
    linear_error = linear_prediction / total_heat - 1.0
    mean_width_0 = ensemble.width_gap_0_m.mean(axis=1)
    mean_width_1 = ensemble.width_gap_1_m.mean(axis=1)
    summary = {
        "scientific_boundary": {
            "construction": "low-e thin triple with two nominal 10 mm gaps and a generic edge spacer",
            "distribution": "synthetic bounded correlated Gaussian-derived fields; not manufacturing tolerances",
            "samples": samples, "seed": seed,
            "correlation_length_m": ensemble.correlation_length_m,
            "cross_gap_correlation": ensemble.cross_gap_correlation,
            "amplitude_bound_range": [0.05, 0.35],
        },
        "mesh": {"nx": nx, "ny": ny, "unknowns": topology.n},
        "timing": {
            "topology_setup_seconds": topology_seconds,
            "ensemble_generation_seconds": generation_seconds,
            "preconditioner_setup_seconds": preconditioner_setup_seconds,
            "sensitivity_kernel_seconds": sensitivity_seconds,
            "production_cpu_wall_seconds": production_wall,
            "production_cpu_worker_seconds": production_worker_seconds,
        },
        "verification": {
            "topology_max_relative_heat_error_vs_original": topology_reference_error,
            "pcg_max_relative_heat_error_vs_direct": pcg_reference_error,
            "maximum_production_residual_relative_l2": max(
                item.residual_relative_l2 for item in production_results),
            "maximum_production_iterations": max(item.iterations for item in production_results),
            "all_results_finite": bool(np.all(np.isfinite(total_heat))),
        },
        "nominal_heat_W_per_m": baseline,
        "total_heat_W_per_m": statistics(total_heat),
        "relative_heat_change": statistics(relative_change),
        "linear_surrogate_relative_error": statistics(linear_error),
        "exceedance_probability": {
            "abs_heat_change_below_1e_8": float(np.mean(np.abs(relative_change) < 1e-8)),
            "abs_heat_change_gt_0_5_percent": float(np.mean(np.abs(relative_change) > 0.005)),
            "abs_heat_change_gt_1_percent": float(np.mean(np.abs(relative_change) > 0.01)),
            "abs_heat_change_gt_2_percent": float(np.mean(np.abs(relative_change) > 0.02)),
            "abs_linear_error_gt_1_percent": float(np.mean(np.abs(linear_error) > 0.01)),
        },
        "spearman_correlations_with_total_heat": {
            "amplitude_bound": float(spearmanr(ensemble.amplitude, total_heat).statistic),
            "gap0_mean_width": float(spearmanr(mean_width_0, total_heat).statistic),
            "gap1_mean_width": float(spearmanr(mean_width_1, total_heat).statistic),
        },
        "pcg_iterations": statistics(np.array([item.iterations for item in production_results])),
        "performance": performance_rows,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "platform": platform.platform()},
        "acceptance": {
            "at_least_10000_realizations": samples >= 10_000,
            "topology_matches_original_below_1e_10": topology_reference_error < 1e-10,
            "pcg_matches_direct_below_1e_7": pcg_reference_error < 1e-7,
            "production_residual_below_1e_8": max(
                item.residual_relative_l2 for item in production_results) < 1e-8,
        },
    }
    (output_directory / "many_query_cpu_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not all(summary["acceptance"].values()):
        raise SystemExit(1)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, default=PROJECT / "validation")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--nx", type=int, default=120)
    parser.add_argument("--ny", type=int, default=36)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    run(args.output_directory, args.samples, args.nx, args.ny, args.seed)


if __name__ == "__main__":
    main()
