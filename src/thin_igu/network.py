"""Nonlinear one-dimensional glazing thermal network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import root

from .properties import gap_conductance, radiation_flux


@dataclass(frozen=True)
class Solid:
    thickness: float
    conductivity: float
    name: str = "solid"

    def resistance(self) -> float:
        if self.thickness <= 0.0 or self.conductivity <= 0.0:
            raise ValueError("solid thickness and conductivity must be positive")
        return self.thickness / self.conductivity


@dataclass(frozen=True)
class Gap:
    width: float
    height: float = 1.0
    argon_fraction: float = 0.9
    emissivity_left: float = 0.84
    emissivity_right: float = 0.04
    name: str = "gap"


@dataclass(frozen=True)
class StackResult:
    heat_flux: float
    interface_temperatures: np.ndarray
    gap_diagnostics: tuple[dict[str, float], ...]
    residual_inf: float


def _element_flux(element: Solid | Gap, t_left: float, t_right: float) -> tuple[float, dict[str, float] | None]:
    if isinstance(element, Solid):
        return (t_left - t_right) / element.resistance(), None
    h_gas, nu, ra = gap_conductance(
        t_left, t_right, element.width, element.height, element.argon_fraction
    )
    q_gas = h_gas * (t_left - t_right)
    q_rad = radiation_flux(
        t_left, t_right, element.emissivity_left, element.emissivity_right
    )
    return q_gas + q_rad, {
        "h_gas": h_gas,
        "h_radiation": q_rad / (t_left - t_right) if t_left != t_right else 0.0,
        "nusselt": nu,
        "rayleigh": ra,
    }


def solve_stack(
    elements: Sequence[Solid | Gap],
    indoor_temperature: float,
    outdoor_temperature: float,
    h_indoor: float,
    h_outdoor: float,
) -> StackResult:
    """Solve a steady nonlinear series stack including surface films.

    Temperatures are ordered from indoor air to outdoor air; positive heat flux
    therefore denotes heat loss from indoors when indoor temperature is higher.
    """

    if not elements:
        raise ValueError("at least one element is required")
    if h_indoor <= 0.0 or h_outdoor <= 0.0:
        raise ValueError("film coefficients must be positive")
    resistances = [1.0 / h_indoor]
    for item in elements:
        if isinstance(item, Solid):
            resistances.append(item.resistance())
        else:
            # Stable initial estimate; nonlinear gap physics is used in residual.
            resistances.append(item.width / 0.02)
    resistances.append(1.0 / h_outdoor)
    total_r = sum(resistances)
    q0 = (indoor_temperature - outdoor_temperature) / total_r
    drops = np.cumsum(np.asarray(resistances[:-1])) * q0
    guess = indoor_temperature - drops

    def residual(temperatures: np.ndarray) -> np.ndarray:
        nodes = np.concatenate(([indoor_temperature], temperatures, [outdoor_temperature]))
        fluxes = [h_indoor * (nodes[0] - nodes[1])]
        for i, element in enumerate(elements):
            q, _ = _element_flux(element, nodes[i + 1], nodes[i + 2])
            fluxes.append(q)
        fluxes.append(h_outdoor * (nodes[-2] - nodes[-1]))
        return np.diff(np.asarray(fluxes))

    solved = root(residual, guess, method="hybr", options={"xtol": 1e-11})
    if not solved.success:
        raise RuntimeError(f"nonlinear stack solve failed: {solved.message}")
    nodes = np.concatenate(([indoor_temperature], solved.x, [outdoor_temperature]))
    heat_flux = h_indoor * (nodes[0] - nodes[1])
    diagnostics: list[dict[str, float]] = []
    for i, element in enumerate(elements):
        _, diag = _element_flux(element, nodes[i + 1], nodes[i + 2])
        if diag is not None:
            diagnostics.append(diag)
    return StackResult(
        heat_flux=float(heat_flux),
        interface_temperatures=solved.x,
        gap_diagnostics=tuple(diagnostics),
        residual_inf=float(np.linalg.norm(residual(solved.x), ord=np.inf)),
    )
