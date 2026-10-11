"""Reusable fixed-topology operator for many-query IGU uncertainty studies."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator, cg, factorized, spsolve

from .fvm2d import EdgeSection, _aligned_y_grid, _layer_at
from .properties import gas_properties, radiation_conductance


@dataclass(frozen=True)
class QueryResult:
    total_heat_W_per_m: float
    edge_heat_W_per_m: float
    peak_heat_flux_W_m2: float
    minimum_indoor_surface_K: float
    residual_relative_l2: float
    iterations: int


@dataclass(frozen=True)
class SpatialEnsemble:
    width_gap_0_m: np.ndarray
    width_gap_1_m: np.ndarray
    amplitude: np.ndarray
    seed: int
    correlation_length_m: float
    cross_gap_correlation: float


class EdgeTopology:
    """Fixed CSR graph with sample-dependent conductance values."""

    def __init__(self, section: EdgeSection, nx: int = 120, ny: int = 36,
                 edge_band_m: float = 0.063):
        self.section = section
        self.nx = nx
        self.ny = ny
        self.dx = section.length / nx
        self.x = (np.arange(nx) + 0.5) * self.dx
        self.y, self.dy = _aligned_y_grid(section, ny)
        self.n = nx * ny
        self.edge_band = self.x < edge_band_m
        self.spacer_columns = self.x <= section.spacer_width
        self.gap_rows = {0: [], 1: []}
        for j, yc in enumerate(self.y):
            kind, layer = _layer_at(section, yc)
            if kind == "gap":
                self.gap_rows[layer].append(j)

        horizontal_p = []
        horizontal_q = []
        for j in range(ny):
            base = j * nx
            horizontal_p.extend(base + np.arange(nx - 1))
            horizontal_q.extend(base + np.arange(1, nx))
        vertical_p = np.arange((ny - 1) * nx)
        vertical_q = vertical_p + nx
        self.edge_p = np.concatenate((np.asarray(horizontal_p), vertical_p)).astype(np.int32)
        self.edge_q = np.concatenate((np.asarray(horizontal_q), vertical_q)).astype(np.int32)
        self.horizontal_count = ny * (nx - 1)

        rows = np.concatenate((np.arange(self.n), self.edge_p, self.edge_q))
        cols = np.concatenate((np.arange(self.n), self.edge_q, self.edge_p))
        pattern = coo_matrix((np.ones(len(rows)), (rows, cols)),
                             shape=(self.n, self.n)).tocsr()
        pattern.sort_indices()
        self.indptr = pattern.indptr.copy()
        self.indices = pattern.indices.copy()

        def position(row: int, column: int) -> int:
            start, stop = self.indptr[row], self.indptr[row + 1]
            local = np.searchsorted(self.indices[start:stop], column)
            if local == stop - start or self.indices[start + local] != column:
                raise RuntimeError("CSR topology lookup failed")
            return int(start + local)

        self.diagonal_position = np.array([position(i, i) for i in range(self.n)])
        self.pq_position = np.array([position(int(p), int(q))
                                     for p, q in zip(self.edge_p, self.edge_q)])
        self.qp_position = np.array([position(int(q), int(p))
                                     for p, q in zip(self.edge_p, self.edge_q)])

        k_bottom = np.full(nx, section.glass_conductivity)
        k_top = np.full(nx, section.glass_conductivity)
        self.g_in = self.dx / (self.dy[0] / (2.0 * k_bottom) + 1.0 / section.h_indoor)
        self.g_out = self.dx / (self.dy[-1] / (2.0 * k_top) + 1.0 / section.h_outdoor)
        self.boundary_diagonal = np.zeros(self.n)
        self.boundary_diagonal[:nx] = self.g_in
        self.boundary_diagonal[-nx:] = self.g_out
        self.rhs = np.zeros(self.n)
        self.rhs[:nx] = self.g_in * section.indoor_temperature
        self.rhs[-nx:] = self.g_out * section.outdoor_temperature

    def gap_conductance(self, width: np.ndarray, emissivities: tuple[float, float]) -> np.ndarray:
        """Vectorized equivalent gas+radiation conductance matching properties.py."""
        section = self.section
        mean_temperature = 0.5 * (section.indoor_temperature + section.outdoor_temperature)
        properties = gas_properties(mean_temperature, section.argon_fraction)
        beta = 1.0 / mean_temperature
        delta_temperature = abs(section.indoor_temperature - section.outdoor_temperature)
        rayleigh = (9.80665 * beta * delta_temperature * width**3 /
                    (properties.kinematic_viscosity * properties.diffusivity))
        aspect = section.height / width
        nu_1 = np.where(rayleigh <= 1.0e4, 1.0,
                        np.where(rayleigh <= 5.0e4,
                                 0.028154 * rayleigh**0.4134,
                                 0.0673838 * rayleigh**(1.0 / 3.0)))
        nu_2 = 0.242 * (rayleigh / aspect) ** 0.272
        nu_3 = 0.0605 * rayleigh ** (1.0 / 3.0)
        nusselt = np.maximum.reduce((np.ones_like(width), nu_1, nu_2, nu_3))
        h_gas = nusselt * properties.conductivity / width
        h_rad = radiation_conductance(mean_temperature, mean_temperature, *emissivities)
        return h_gas + h_rad

    def conductivity(self, width_gap_0: np.ndarray, width_gap_1: np.ndarray) -> np.ndarray:
        if width_gap_0.shape != (self.nx,) or width_gap_1.shape != (self.nx,):
            raise ValueError("gap widths must match the topology x-grid")
        if np.any(width_gap_0 <= 0) or np.any(width_gap_1 <= 0):
            raise ValueError("gap widths must be positive")
        k = np.full((self.ny, self.nx), self.section.glass_conductivity)
        for layer, width in enumerate((width_gap_0, width_gap_1)):
            effective_k = (self.gap_conductance(width, self.section.gap_emissivities[layer])
                           * self.section.gap_widths[layer])
            k[self.gap_rows[layer], :] = effective_k
        k[:, self.spacer_columns] = np.where(
            np.isin(np.arange(self.ny)[:, None], self.gap_rows[0] + self.gap_rows[1]),
            self.section.spacer_conductivity, k[:, self.spacer_columns])
        return k

    def assemble(self, width_gap_0: np.ndarray,
                 width_gap_1: np.ndarray) -> tuple[csr_matrix, np.ndarray]:
        k = self.conductivity(width_gap_0, width_gap_1)
        horizontal = (2.0 * k[:, :-1] * k[:, 1:] / (k[:, :-1] + k[:, 1:])
                      * self.dy[:, None] / self.dx).ravel()
        vertical = (self.dx / (self.dy[:-1, None] / (2.0 * k[:-1, :])
                               + self.dy[1:, None] / (2.0 * k[1:, :]))).ravel()
        conductance = np.concatenate((horizontal, vertical))
        diagonal = (np.bincount(self.edge_p, weights=conductance, minlength=self.n)
                    + np.bincount(self.edge_q, weights=conductance, minlength=self.n)
                    + self.boundary_diagonal)
        data = np.zeros_like(self.indices, dtype=float)
        data[self.pq_position] = -conductance
        data[self.qp_position] = -conductance
        data[self.diagonal_position] = diagonal
        return csr_matrix((data, self.indices, self.indptr), shape=(self.n, self.n),
                          copy=False), k

    def observables(self, temperature: np.ndarray, matrix: csr_matrix,
                    iterations: int = 0) -> QueryResult:
        bottom = temperature[:self.nx]
        heat_cell = self.g_in * (self.section.indoor_temperature - bottom)
        heat_flux = heat_cell / self.dx
        surface_temperature = self.section.indoor_temperature - heat_flux / self.section.h_indoor
        residual = matrix @ temperature - self.rhs
        return QueryResult(
            total_heat_W_per_m=float(heat_cell.sum()),
            edge_heat_W_per_m=float(heat_cell[self.edge_band].sum()),
            peak_heat_flux_W_m2=float(heat_flux.max()),
            minimum_indoor_surface_K=float(surface_temperature.min()),
            residual_relative_l2=float(np.linalg.norm(residual) / np.linalg.norm(self.rhs)),
            iterations=iterations,
        )

    def solve_direct(self, width_gap_0: np.ndarray,
                     width_gap_1: np.ndarray) -> QueryResult:
        matrix, _ = self.assemble(width_gap_0, width_gap_1)
        temperature = spsolve(matrix, self.rhs)
        return self.observables(temperature, matrix)

    def nominal_reusable_preconditioner(self):
        nominal = np.full(self.nx, self.section.gap_widths[0])
        matrix, _ = self.assemble(nominal, nominal)
        solve = factorized(matrix.tocsc())
        operator = LinearOperator(matrix.shape, matvec=solve, dtype=float)
        initial = solve(self.rhs)
        return operator, initial

    def solve_preconditioned(self, width_gap_0: np.ndarray,
                             width_gap_1: np.ndarray,
                             preconditioner: LinearOperator,
                             initial: np.ndarray,
                             rtol: float = 1e-9,
                             maxiter: int = 100) -> QueryResult:
        matrix, _ = self.assemble(width_gap_0, width_gap_1)
        count = 0

        def callback(_: np.ndarray) -> None:
            nonlocal count
            count += 1

        temperature, info = cg(matrix, self.rhs, x0=initial, M=preconditioner,
                               rtol=rtol, atol=0.0, maxiter=maxiter,
                               callback=callback)
        if info != 0:
            raise RuntimeError(f"preconditioned CG failed with info={info}")
        return self.observables(temperature, matrix, count)


def generate_spatial_ensemble(n_samples: int, x: np.ndarray, seed: int = 20261008,
                              correlation_length_m: float = 0.018,
                              cross_gap_correlation: float = 0.35) -> SpatialEnsemble:
    """Generate bounded, smooth synthetic gap fields; not manufacturing data."""
    if n_samples < 1 or not 0 <= cross_gap_correlation < 1:
        raise ValueError("invalid ensemble dimensions or correlation")
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    distance = x[:, None] - x[None, :]
    covariance = np.exp(-0.5 * (distance / correlation_length_m) ** 2)
    cholesky = np.linalg.cholesky(covariance + 1e-10 * np.eye(len(x)))
    independent_0 = rng.standard_normal((n_samples, len(x))) @ cholesky.T
    independent_1 = rng.standard_normal((n_samples, len(x))) @ cholesky.T
    field_1 = (cross_gap_correlation * independent_0
               + np.sqrt(1.0 - cross_gap_correlation**2) * independent_1)
    amplitude = rng.uniform(0.05, 0.35, n_samples)
    nominal = 0.010
    width_0 = nominal * (1.0 + amplitude[:, None] * np.tanh(independent_0))
    width_1 = nominal * (1.0 + amplitude[:, None] * np.tanh(field_1))
    return SpatialEnsemble(width_0, width_1, amplitude, seed,
                           correlation_length_m, cross_gap_correlation)


def solve_chunk_direct(payload: tuple[int, int, np.ndarray, np.ndarray]) -> tuple[int, list[QueryResult], float]:
    """Process-safe chunk worker used by the CPU ensemble benchmark."""
    nx, ny, widths_0, widths_1 = payload
    topology = EdgeTopology(EdgeSection(), nx=nx, ny=ny)
    start = perf_counter()
    results = [topology.solve_direct(first, second)
               for first, second in zip(widths_0, widths_1)]
    return len(results), results, perf_counter() - start


def solve_chunk_preconditioned(
    payload: tuple[int, int, np.ndarray, np.ndarray, float]
) -> tuple[int, list[QueryResult], float]:
    """Process-safe worker with one nominal factorization reused per chunk."""
    nx, ny, widths_0, widths_1, rtol = payload
    topology = EdgeTopology(EdgeSection(), nx=nx, ny=ny)
    preconditioner, initial = topology.nominal_reusable_preconditioner()
    start = perf_counter()
    results = [topology.solve_preconditioned(first, second, preconditioner,
                                              initial, rtol=rtol)
               for first, second in zip(widths_0, widths_1)]
    return len(results), results, perf_counter() - start
