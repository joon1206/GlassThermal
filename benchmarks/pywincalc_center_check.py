"""Compare the local network to pyWinCalc 3.3.1 for a public clear triple."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

import pywincalc

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from thin_igu.network import Gap, Solid, solve_stack  # noqa: E402


def run(product_directory: Path) -> dict[str, object]:
    glass = pywincalc.parse_optics_file(str(product_directory / "CLEAR_3.DAT"))
    gas = pywincalc.create_gas([
        [0.9, pywincalc.PredefinedGasType.ARGON],
        [0.1, pywincalc.PredefinedGasType.AIR],
    ])
    gaps = [pywincalc.Layers.gap(thickness=0.0127, gas=gas) for _ in range(2)]
    model = pywincalc.BoundaryConditionsCoefficientModelType.H_PRESCRIBED
    direction = pywincalc.AirHorizontalDirection.NONE
    indoor = pywincalc.Environment(
        air_temperature=294.15, pressure=101325.0, convection_coefficient=7.7,
        coefficient_model=model, radiation_temperature=294.15, emissivity=1.0,
        air_speed=0.0, air_direction=direction, direct_solar_radiation=0.0)
    outdoor = pywincalc.Environment(
        air_temperature=255.15, pressure=101325.0, convection_coefficient=25.0,
        coefficient_model=model, radiation_temperature=255.15, emissivity=1.0,
        air_speed=0.0, air_direction=direction, direct_solar_radiation=0.0)
    system = pywincalc.GlazingSystem(
        solid_layers=[glass, glass, glass], gap_layers=gaps,
        environment=pywincalc.Environments(outdoor, indoor))

    stack = []
    thickness = float(glass.thickness) / 1000.0  # optics file stores mm
    for pane in range(3):
        stack.append(Solid(thickness, float(glass.conductivity)))
        if pane < 2:
            stack.append(Gap(
                0.0127, argon_fraction=0.9,
                emissivity_left=float(glass.emissivity_back),
                emissivity_right=float(glass.emissivity_front)))
    local = solve_stack(stack, 294.15, 255.15, 7.7, 25.0)
    local_u, pywincalc_u = local.heat_flux / 39.0, float(system.u())
    return {
        "case": "official CLEAR_3.DAT; 3 panes; two 12.7 mm gaps; 90% Ar/10% air",
        "pywincalc_version": importlib.metadata.version("pywincalc"),
        "pywincalc_u_W_m2K": pywincalc_u,
        "local_u_W_m2K": local_u,
        "relative_difference": (local_u - pywincalc_u) / pywincalc_u,
        "pywincalc_layer_temperatures_K": [
            float(x) for x in system.layer_temperatures(pywincalc.TarcogSystemType.U)],
        "pywincalc_gap_effective_conductivities_W_mK": [
            float(x) for x in system.gap_layers_effective_conductivities(
                pywincalc.TarcogSystemType.U)],
        "local_interface_temperatures_K": local.interface_temperatures.tolist(),
        "local_gap_diagnostics": [
            {key: float(value) for key, value in item.items()}
            for item in local.gap_diagnostics],
        "interpretation": "One component case; not ISO certification.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("product_directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(run(args.product_directory), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
