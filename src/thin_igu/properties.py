"""Public, generic material and gas properties used by the reference model.

Values are representative near room temperature, not proprietary product data.
The gas-mixture model is deliberately transparent and will be replaced or
validated against ISO 15099/pyWinCalc before quantitative claims are made.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GasProperties:
    conductivity: float
    density: float
    dynamic_viscosity: float
    heat_capacity: float

    @property
    def kinematic_viscosity(self) -> float:
        return self.dynamic_viscosity / self.density

    @property
    def diffusivity(self) -> float:
        return self.conductivity / (self.density * self.heat_capacity)

    @property
    def prandtl(self) -> float:
        return self.kinematic_viscosity / self.diffusivity


_REFERENCE = {
    "air": GasProperties(0.02624, 1.177, 1.846e-5, 1007.0),
    "argon": GasProperties(0.01772, 1.622, 2.230e-5, 520.3),
}


def gas_properties(temperature: float, argon_fraction: float) -> GasProperties:
    """Return approximate air/argon mixture properties at *temperature* [K].

    A mole-fraction interpolation is used at 300 K, with first-order temperature
    scaling. This is adequate for solver verification, but not an ISO 15099 gas
    property implementation. ``argon_fraction`` must be in [0, 1].
    """

    if not 0.0 <= argon_fraction <= 1.0:
        raise ValueError("argon_fraction must lie in [0, 1]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    x = argon_fraction
    a, r = _REFERENCE["air"], _REFERENCE["argon"]
    # Ideal-gas density and mild power-law transport scaling near 300 K.
    tr = temperature / 300.0
    blend = lambda va, vr: (1.0 - x) * va + x * vr
    return GasProperties(
        conductivity=blend(a.conductivity, r.conductivity) * tr**0.76,
        density=blend(a.density, r.density) / tr,
        dynamic_viscosity=blend(a.dynamic_viscosity, r.dynamic_viscosity) * tr**0.70,
        heat_capacity=blend(a.heat_capacity, r.heat_capacity),
    )


def cavity_nusselt_vertical(
    hot_temperature: float,
    cold_temperature: float,
    gap: float,
    height: float,
    argon_fraction: float,
) -> tuple[float, float]:
    """Return ``(Nu, Ra)`` for a tall, vertical sealed cavity.

    The correlation is the commonly used piecewise vertical-cavity form reported
    in the WINDOW/ISO-15099 technical lineage. The conduction lower bound is
    enforced. It is used here as an auditable baseline pending direct pyWinCalc
    comparison across the intended regime.
    """

    if gap <= 0.0 or height <= 0.0:
        raise ValueError("gap and height must be positive")
    tm = 0.5 * (hot_temperature + cold_temperature)
    p = gas_properties(tm, argon_fraction)
    beta = 1.0 / tm
    ra = 9.80665 * beta * abs(hot_temperature - cold_temperature) * gap**3 / (
        p.kinematic_viscosity * p.diffusivity
    )
    aspect = height / gap
    if ra <= 1.0e4:
        nu_1 = 1.0
    elif ra <= 5.0e4:
        nu_1 = 0.028154 * ra**0.4134
    else:
        nu_1 = 0.0673838 * ra ** (1.0 / 3.0)
    nu_2 = 0.242 * (ra / aspect) ** 0.272 if ra > 0.0 else 0.0
    nu_3 = 0.0605 * ra ** (1.0 / 3.0) if ra > 0.0 else 0.0
    return max(1.0, nu_1, nu_2, nu_3), ra


def gap_conductance(
    hot_temperature: float,
    cold_temperature: float,
    gap: float,
    height: float,
    argon_fraction: float,
) -> tuple[float, float, float]:
    """Return gas conductance [W/m2/K], Nusselt number, and Rayleigh number."""

    tm = 0.5 * (hot_temperature + cold_temperature)
    p = gas_properties(tm, argon_fraction)
    nu, ra = cavity_nusselt_vertical(
        hot_temperature, cold_temperature, gap, height, argon_fraction
    )
    return nu * p.conductivity / gap, nu, ra


def radiation_conductance(t1: float, t2: float, eps1: float, eps2: float) -> float:
    """Exact secant linearization for two infinite diffuse-gray parallel plates."""

    if not (0.0 < eps1 <= 1.0 and 0.0 < eps2 <= 1.0):
        raise ValueError("emissivities must lie in (0, 1]")
    sigma = 5.670374419e-8
    denominator = 1.0 / eps1 + 1.0 / eps2 - 1.0
    return sigma * (t1 + t2) * (t1 * t1 + t2 * t2) / denominator


def radiation_flux(t1: float, t2: float, eps1: float, eps2: float) -> float:
    """Net long-wave heat flux from surface 1 to surface 2 [W/m2]."""

    return radiation_conductance(t1, t2, eps1, eps2) * (t1 - t2)
