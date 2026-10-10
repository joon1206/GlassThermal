"""Conservative cell-centred finite-volume solver for an IGU edge section.

The mesh spans distance from the sealed edge (x) and through-thickness (y).
Gas/radiation are represented as an equivalent local conductivity. This Level-2
model captures lateral conduction and edge bridges, but not resolved gas flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .properties import gap_conductance, radiation_conductance


@dataclass(frozen=True)
class EdgeSection:
    length: float = 0.30
    glass_thicknesses: tuple[float, float, float] = (0.003, 0.0007, 0.003)
    gap_widths: tuple[float, float] = (0.010, 0.010)
    glass_conductivity: float = 1.0
    spacer_width: float = 0.012
    spacer_conductivity: float = 0.15
    argon_fraction: float = 0.90
    gap_emissivities: tuple[tuple[float, float], tuple[float, float]] = (
        (0.84, 0.04),
        (0.04, 0.84),
    )
    indoor_temperature: float = 294.15
    outdoor_temperature: float = 255.15
    h_indoor: float = 7.7
    h_outdoor: float = 25.0
    height: float = 1.0

    @property
    def thickness(self) -> float:
        return sum(self.glass_thicknesses) + sum(self.gap_widths)


@dataclass(frozen=True)
class EdgeResult:
    temperature: np.ndarray
    x: np.ndarray
    y: np.ndarray
    heat_flux_indoor: np.ndarray
    heat_flux_outdoor: np.ndarray
    total_heat_indoor: float
    total_heat_outdoor: float
    energy_imbalance: float
    conductivity: np.ndarray


def _layer_at(section: EdgeSection, y: float) -> tuple[str, int]:
    cursor = 0.0
    for i in range(3):
        cursor += section.glass_thicknesses[i]
        if y <= cursor + 1e-14:
            return "glass", i
        if i < 2:
            cursor += section.gap_widths[i]
            if y <= cursor + 1e-14:
                return "gap", i
    return "glass", 2


def _aligned_y_grid(section: EdgeSection, target_cells: int) -> tuple[np.ndarray, np.ndarray]:
    """Return cell centres and widths with every material interface on a face."""

    thicknesses = np.asarray([
        section.glass_thicknesses[0], section.gap_widths[0],
        section.glass_thicknesses[1], section.gap_widths[1],
        section.glass_thicknesses[2],
    ])
    if target_cells < 2 * len(thicknesses):
        raise ValueError("ny must be at least 10 to resolve every layer")
    counts = np.maximum(2, np.floor(target_cells * thicknesses / thicknesses.sum()).astype(int))
    while counts.sum() < target_cells:
        counts[np.argmax(thicknesses / counts)] += 1
    while counts.sum() > target_cells:
        candidates = np.where(counts > 2)[0]
        if len(candidates) == 0:
            raise ValueError("could not allocate the requested layer-aligned grid")
        # Remove a cell from the currently most over-resolved layer.
        choice = candidates[np.argmin(thicknesses[candidates] / counts[candidates])]
        counts[choice] -= 1
    widths = np.concatenate([
        np.full(count, thickness / count)
        for thickness, count in zip(thicknesses, counts)
    ])
    faces = np.concatenate(([0.0], np.cumsum(widths)))
    centres = 0.5 * (faces[:-1] + faces[1:])
    return centres, widths


def conductivity_field(
    section: EdgeSection,
    x: np.ndarray,
    y: np.ndarray,
    gap_width: Callable[[int, np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Build the conductivity map for fixed-topology defect studies."""

    k = np.empty((len(y), len(x)), dtype=float)
    mean_t = 0.5 * (section.indoor_temperature + section.outdoor_temperature)
    # Initial conductance uses the total temperature difference. Nonlinear 1-D
    # solves provide the high-accuracy center model; this 2-D operator is linear.
    for j, yc in enumerate(y):
        kind, layer = _layer_at(section, yc)
        if kind == "glass":
            k[j, :] = section.glass_conductivity
            continue
        nominal = section.gap_widths[layer]
        widths = (
            np.full_like(x, nominal)
            if gap_width is None
            else np.asarray(gap_width(layer, x), dtype=float)
        )
        if widths.shape != x.shape or np.any(widths <= 0.0):
            raise ValueError("gap-width field must be positive and match x")
        h_values = np.empty_like(x)
        eps_l, eps_r = section.gap_emissivities[layer]
        for i, width in enumerate(widths):
            h_gas, _, _ = gap_conductance(
                section.indoor_temperature,
                section.outdoor_temperature,
                float(width),
                section.height,
                section.argon_fraction,
            )
            h_values[i] = h_gas + radiation_conductance(mean_t, mean_t, eps_l, eps_r)
        # Geometry is fixed at nominal width. Preserve the desired local
        # through-gap conductance by k_mesh / nominal = h(width).
        k[j, :] = h_values * nominal
        k[j, x <= section.spacer_width] = section.spacer_conductivity
    return k


def solve_edge_section(
    section: EdgeSection,
    nx: int = 160,
    ny: int = 48,
    gap_width: Callable[[int, np.ndarray], np.ndarray] | None = None,
) -> EdgeResult:
    """Solve the 2-D section with Robin faces and adiabatic lateral boundaries."""

    if nx < 4 or ny < 10:
        raise ValueError("nx must be at least four and ny at least ten")
    dx = section.length / nx
    x = (np.arange(nx) + 0.5) * dx
    y, dy = _aligned_y_grid(section, ny)
    k = conductivity_field(section, x, y, gap_width)
    n = nx * ny
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros(n)

    def add(r: int, c: int, value: float) -> None:
        rows.append(r); cols.append(c); data.append(value)

    for j in range(ny):
        for i in range(nx):
            p = j * nx + i
            ap = 0.0
            if i > 0:
                g = 2.0 * k[j, i] * k[j, i - 1] / (k[j, i] + k[j, i - 1]) * dy[j] / dx
                add(p, p - 1, -g); ap += g
            if i < nx - 1:
                g = 2.0 * k[j, i] * k[j, i + 1] / (k[j, i] + k[j, i + 1]) * dy[j] / dx
                add(p, p + 1, -g); ap += g
            if j > 0:
                g = dx / (dy[j] / (2.0 * k[j, i]) + dy[j - 1] / (2.0 * k[j - 1, i]))
                add(p, p - nx, -g); ap += g
            else:
                g = dx / (dy[j] / (2.0 * k[j, i]) + 1.0 / section.h_indoor)
                ap += g; rhs[p] += g * section.indoor_temperature
            if j < ny - 1:
                g = dx / (dy[j] / (2.0 * k[j, i]) + dy[j + 1] / (2.0 * k[j + 1, i]))
                add(p, p + nx, -g); ap += g
            else:
                g = dx / (dy[j] / (2.0 * k[j, i]) + 1.0 / section.h_outdoor)
                ap += g; rhs[p] += g * section.outdoor_temperature
            add(p, p, ap)

    matrix = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    temperature = spsolve(matrix, rhs).reshape(ny, nx)
    g_in = dx / (dy[0] / (2.0 * k[0, :]) + 1.0 / section.h_indoor)
    g_out = dx / (dy[-1] / (2.0 * k[-1, :]) + 1.0 / section.h_outdoor)
    q_in_cell = g_in * (section.indoor_temperature - temperature[0, :])
    q_out_cell = g_out * (temperature[-1, :] - section.outdoor_temperature)
    q_in = q_in_cell / dx
    q_out = q_out_cell / dx
    total_in, total_out = float(q_in_cell.sum()), float(q_out_cell.sum())
    scale = max(abs(total_in), abs(total_out), 1e-30)
    return EdgeResult(
        temperature=temperature,
        x=x,
        y=y,
        heat_flux_indoor=q_in,
        heat_flux_outdoor=q_out,
        total_heat_indoor=total_in,
        total_heat_outdoor=total_out,
        energy_imbalance=abs(total_in - total_out) / scale,
        conductivity=k,
    )
