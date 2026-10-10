"""Batched matrix-free GPU execution after the verified CPU campaign.

CuPy is an optional benchmark dependency. The solver uses sample-specific
Jacobi-preconditioned batched CG on the same cell-centred operator and compares
every production heat-flow result to the stored CPU oracle.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from thin_igu.fvm2d import EdgeSection  # noqa: E402
from thin_igu.many_query import EdgeTopology, generate_spatial_ensemble  # noqa: E402


def read_cpu_oracle(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    fields = ("total_heat_W_per_m", "edge_heat_fraction", "peak_heat_flux_W_m2",
              "minimum_indoor_surface_K")
    return {field: np.array([float(row[field]) for row in rows]) for field in fields}


class BatchedGPU:
    def __init__(self, topology: EdgeTopology, dtype: str = "float32"):
        import cupy as cp
        self.cp = cp
        self.topology = topology
        self.dtype = cp.float32 if dtype == "float32" else cp.float64
        self.dy = cp.asarray(topology.dy, dtype=self.dtype)
        self.g_in = cp.asarray(topology.g_in, dtype=self.dtype)
        self.g_out = cp.asarray(topology.g_out, dtype=self.dtype)
        self.edge_band = cp.asarray(topology.edge_band)
        self.gap_rows = [cp.asarray(topology.gap_rows[layer], dtype=cp.int32)
                         for layer in range(2)]
        nominal = np.full(topology.nx, 0.010)
        _, initial = topology.nominal_reusable_preconditioner()
        self.initial = cp.asarray(initial.reshape(topology.ny, topology.nx),
                                  dtype=self.dtype)
        mean_temperature = 0.5 * (topology.section.indoor_temperature
                                  + topology.section.outdoor_temperature)
        from thin_igu.properties import gas_properties, radiation_conductance
        properties = gas_properties(mean_temperature, topology.section.argon_fraction)
        scalar = np.float32 if dtype == "float32" else np.float64
        self.gas_k = scalar(properties.conductivity)
        self.rayleigh_coefficient = scalar(
            9.80665 / mean_temperature
            * abs(topology.section.indoor_temperature - topology.section.outdoor_temperature)
            / (properties.kinematic_viscosity * properties.diffusivity))
        self.h_rad = [scalar(radiation_conductance(
            mean_temperature, mean_temperature, *pair))
            for pair in topology.section.gap_emissivities]

    def gap_h(self, width, layer: int):
        cp = self.cp
        rayleigh = self.rayleigh_coefficient * width**3
        aspect = self.topology.section.height / width
        nu_1 = cp.where(rayleigh <= 1.0e4, 1.0,
                        cp.where(rayleigh <= 5.0e4,
                                 0.028154 * rayleigh**0.4134,
                                 0.0673838 * rayleigh**(1.0 / 3.0)))
        nu_2 = 0.242 * (rayleigh / aspect) ** 0.272
        nu_3 = 0.0605 * rayleigh ** (1.0 / 3.0)
        nusselt = cp.maximum(cp.maximum(cp.maximum(1.0, nu_1), nu_2), nu_3)
        return nusselt * self.gas_k / width + self.h_rad[layer]

    def coefficients(self, widths_0: np.ndarray, widths_1: np.ndarray):
        cp = self.cp
        batch = len(widths_0)
        width_gpu = [cp.asarray(widths_0, dtype=self.dtype),
                     cp.asarray(widths_1, dtype=self.dtype)]
        k = cp.full((batch, self.topology.ny, self.topology.nx),
                    self.topology.section.glass_conductivity, dtype=self.dtype)
        for layer in range(2):
            value = (self.gap_h(width_gpu[layer], layer)
                     * self.topology.section.gap_widths[layer])
            k[:, self.gap_rows[layer], :] = value[:, None, :]
            spacer = cp.asarray(np.where(self.topology.spacer_columns)[0], dtype=cp.int32)
            for row in self.topology.gap_rows[layer]:
                k[:, row, spacer] = self.topology.section.spacer_conductivity
        gx = (2.0 * k[:, :, :-1] * k[:, :, 1:]
              / (k[:, :, :-1] + k[:, :, 1:])
              * self.dy[None, :, None] / self.topology.dx)
        gy = (self.topology.dx
              / (self.dy[:-1][None, :, None] / (2.0 * k[:, :-1, :])
                 + self.dy[1:][None, :, None] / (2.0 * k[:, 1:, :])))
        diagonal = cp.zeros_like(k)
        diagonal[:, :, :-1] += gx
        diagonal[:, :, 1:] += gx
        diagonal[:, :-1, :] += gy
        diagonal[:, 1:, :] += gy
        diagonal[:, 0, :] += self.g_in
        diagonal[:, -1, :] += self.g_out
        return gx, gy, diagonal

    def apply(self, temperature, gx, gy):
        cp = self.cp
        result = cp.zeros_like(temperature)
        horizontal = gx * (temperature[:, :, :-1] - temperature[:, :, 1:])
        result[:, :, :-1] += horizontal
        result[:, :, 1:] -= horizontal
        vertical = gy * (temperature[:, :-1, :] - temperature[:, 1:, :])
        result[:, :-1, :] += vertical
        result[:, 1:, :] -= vertical
        result[:, 0, :] += self.g_in * temperature[:, 0, :]
        result[:, -1, :] += self.g_out * temperature[:, -1, :]
        return result

    def solve(self, widths_0: np.ndarray, widths_1: np.ndarray,
              rtol: float = 2e-6, maxiter: int = 300) -> dict[str, np.ndarray | int]:
        cp = self.cp
        batch = len(widths_0)
        gx, gy, diagonal = self.coefficients(widths_0, widths_1)
        rhs = cp.zeros((batch, self.topology.ny, self.topology.nx), dtype=self.dtype)
        rhs[:, 0, :] = self.g_in * self.topology.section.indoor_temperature
        rhs[:, -1, :] = self.g_out * self.topology.section.outdoor_temperature
        temperature = cp.broadcast_to(self.initial, rhs.shape).copy()
        residual = rhs - self.apply(temperature, gx, gy)
        z = residual / diagonal
        direction = z.copy()
        rz = cp.sum(residual * z, axis=(1, 2))
        rhs_norm = cp.sqrt(cp.sum(rhs * rhs, axis=(1, 2)))
        active = cp.ones(batch, dtype=cp.bool_)
        iterations = cp.zeros(batch, dtype=cp.int32)
        relative = cp.sqrt(cp.sum(residual * residual, axis=(1, 2))) / rhs_norm
        active = relative > rtol
        for _ in range(maxiter):
            if not bool(cp.any(active).get()):
                break
            applied = self.apply(direction, gx, gy)
            denominator = cp.sum(direction * applied, axis=(1, 2))
            alpha = cp.where(active, rz / denominator, 0.0)
            temperature += alpha[:, None, None] * direction
            residual -= alpha[:, None, None] * applied
            iterations += active
            relative = cp.sqrt(cp.sum(residual * residual, axis=(1, 2))) / rhs_norm
            new_active = relative > rtol
            z_new = residual / diagonal
            rz_new = cp.sum(residual * z_new, axis=(1, 2))
            beta = cp.where(new_active, rz_new / rz, 0.0)
            direction = z_new + beta[:, None, None] * direction
            rz = rz_new
            active = new_active
        if bool(cp.any(active).get()):
            raise RuntimeError(f"GPU CG did not converge for {int(cp.sum(active).get())} samples")
        # Recompute the true residual after convergence.
        true_relative = (cp.sqrt(cp.sum((rhs - self.apply(temperature, gx, gy)) ** 2,
                                        axis=(1, 2))) / rhs_norm)
        heat_cell = self.g_in * (
            self.topology.section.indoor_temperature - temperature[:, 0, :])
        heat_flux = heat_cell / self.topology.dx
        total = cp.sum(heat_cell, axis=1)
        edge = cp.sum(heat_cell[:, self.edge_band], axis=1)
        surface_temperature = (self.topology.section.indoor_temperature
                               - heat_flux / self.topology.section.h_indoor)
        cp.cuda.Stream.null.synchronize()
        return {
            "total_heat_W_per_m": cp.asnumpy(total),
            "edge_heat_fraction": cp.asnumpy(edge / total),
            "peak_heat_flux_W_m2": cp.asnumpy(cp.max(heat_flux, axis=1)),
            "minimum_indoor_surface_K": cp.asnumpy(cp.min(surface_temperature, axis=1)),
            "residual_relative_l2": cp.asnumpy(true_relative),
            "iterations": cp.asnumpy(iterations),
        }


def run(output_directory: Path, cpu_oracle: Path, samples: int = 10_000,
        nx: int = 120, ny: int = 36, seed: int = 20261008) -> dict[str, object]:
    import cupy as cp
    output_directory.mkdir(parents=True, exist_ok=True)
    topology = EdgeTopology(EdgeSection(), nx=nx, ny=ny)
    ensemble = generate_spatial_ensemble(samples, topology.x, seed=seed)
    oracle = read_cpu_oracle(cpu_oracle)
    if len(oracle["total_heat_W_per_m"]) != samples:
        raise ValueError("CPU oracle sample count does not match")
    solver = BatchedGPU(topology, dtype="float32")

    # Warm compilation and memory pools before timing.
    solver.solve(ensemble.width_gap_0_m[:8], ensemble.width_gap_1_m[:8])
    cp.get_default_memory_pool().free_all_blocks()
    timing_rows = []
    # Isolate batch-size effects with one high-amplitude realization repeated at
    # every size; otherwise different subsets would have different CG counts.
    probe_index = int(np.argmax(ensemble.amplitude))
    for batch_size in (size for size in (1, 8, 32, 128, 512, 1024, 2048)
                       if size <= samples):
        probe_0 = np.repeat(ensemble.width_gap_0_m[probe_index][None, :],
                            batch_size, axis=0)
        probe_1 = np.repeat(ensemble.width_gap_1_m[probe_index][None, :],
                            batch_size, axis=0)
        cold_start = perf_counter()
        solver.solve(probe_0, probe_1)
        cold_seconds = perf_counter() - cold_start
        start = perf_counter()
        result = solver.solve(probe_0, probe_1)
        elapsed = perf_counter() - start
        error = float(np.max(np.abs(result["total_heat_W_per_m"]
                                    - oracle["total_heat_W_per_m"][probe_index])
                             / abs(oracle["total_heat_W_per_m"][probe_index])))
        timing_rows.append({"batch_size": batch_size,
                            "cold_allocation_seconds": cold_seconds,
                            "steady_state_seconds": elapsed,
                            "samples_per_second": batch_size / elapsed,
                            "max_relative_heat_error": error,
                            "max_iterations": int(np.max(result["iterations"]))})
        cp.get_default_memory_pool().free_all_blocks()
    best = max(timing_rows, key=lambda row: row["samples_per_second"])
    batch_size = int(best["batch_size"])

    output = {key: [] for key in (
        "total_heat_W_per_m", "edge_heat_fraction", "peak_heat_flux_W_m2",
        "minimum_indoor_surface_K", "residual_relative_l2", "iterations")}
    start = perf_counter()
    for begin in range(0, samples, batch_size):
        stop = min(samples, begin + batch_size)
        result = solver.solve(ensemble.width_gap_0_m[begin:stop],
                              ensemble.width_gap_1_m[begin:stop])
        for key in output:
            output[key].append(result[key])
    cp.cuda.Stream.null.synchronize()
    production_wall = perf_counter() - start
    output = {key: np.concatenate(value) for key, value in output.items()}

    errors = {}
    for key in ("total_heat_W_per_m", "edge_heat_fraction",
                "peak_heat_flux_W_m2", "minimum_indoor_surface_K"):
        errors[key] = float(np.max(np.abs(output[key] - oracle[key])
                                   / np.maximum(np.abs(oracle[key]), 1e-30)))
    summary = {
        "gpu": {
            "name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "compute_capability": "7.5",
            "driver_version": cp.cuda.runtime.driverGetVersion(),
            "runtime_version": cp.cuda.runtime.runtimeGetVersion(),
            "memory_total_bytes": cp.cuda.runtime.memGetInfo()[1],
            "cupy_version": cp.__version__, "dtype": "float32",
        },
        "samples": samples, "mesh": {"nx": nx, "ny": ny, "unknowns": topology.n},
        "batch_crossover": timing_rows,
        "batch_crossover_probe_index": probe_index,
        "selected_batch_size": batch_size,
        "production_wall_seconds": production_wall,
        "production_samples_per_second": samples / production_wall,
        "cpu_production_wall_seconds": 25.78813890001038,
        "gpu_speedup_vs_cpu_production": 25.78813890001038 / production_wall,
        "maximum_relative_error_vs_cpu": errors,
        "maximum_gpu_residual_relative_l2": float(np.max(output["residual_relative_l2"])),
        "iteration_quantiles": {
            "p50": float(np.quantile(output["iterations"], 0.5)),
            "p95": float(np.quantile(output["iterations"], 0.95)),
            "maximum": int(np.max(output["iterations"])),
        },
        "acceptance": {
            "all_10000_gpu_cases_completed": len(output["total_heat_W_per_m"]) >= 10_000,
            "heat_error_below_1_5e_4": errors["total_heat_W_per_m"] < 1.5e-4,
            "all_reported_observable_errors_below_2e_4": max(errors.values()) < 2e-4,
            "float32_residual_below_5e_4": float(np.max(output["residual_relative_l2"])) < 5e-4,
        },
    }
    with (output_directory / "many_query_gpu_performance.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(timing_rows[0]))
        writer.writeheader(); writer.writerows(timing_rows)
    (output_directory / "many_query_gpu_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not all(summary["acceptance"].values()):
        raise SystemExit(1)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, default=PROJECT / "validation")
    parser.add_argument("--cpu-oracle", type=Path,
                        default=PROJECT / "validation" / "uq_10000_cpu_results.csv")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--nx", type=int, default=120)
    parser.add_argument("--ny", type=int, default=36)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    run(args.output_directory, args.cpu_oracle, args.samples, args.nx, args.ny, args.seed)


if __name__ == "__main__":
    main()
