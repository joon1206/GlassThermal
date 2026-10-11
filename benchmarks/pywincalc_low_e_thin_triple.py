"""Low-e thin-triple comparison against pyWinCalc 3.3.1.

The optical/thermal products are public files in LBNL-ETA/pyWinCalc. The
outer and inner LOW-E_5 panes place the low-e surfaces toward their cavities;
the public CLEAR_3 product is copied and assigned a 0.7 mm center thickness.
"""

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


def environments() -> pywincalc.Environments:
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
    return pywincalc.Environments(outdoor, indoor)


def run_case(product_directory: Path, center_thickness_mm: float) -> dict[str, object]:
    outer = pywincalc.parse_optics_file(str(product_directory / "LOW-E_5.LOF"))
    center = pywincalc.parse_optics_file(str(product_directory / "CLEAR_3.DAT"))
    inner = pywincalc.parse_optics_file(str(product_directory / "LOW-E_5.LOF"))
    center.thickness = center_thickness_mm
    gas = pywincalc.create_gas([
        [0.9, pywincalc.PredefinedGasType.ARGON],
        [0.1, pywincalc.PredefinedGasType.AIR],
    ])
    gaps = [pywincalc.Layers.gap(thickness=0.010, gas=gas) for _ in range(2)]
    system = pywincalc.GlazingSystem(
        solid_layers=[outer, center, inner], gap_layers=gaps,
        environment=environments())
    # pyWinCalc orders layers outside-to-inside. LOW-E_5 is coated on its
    # front, so flip only the outdoor lite to put coatings on surfaces 2 and 5.
    system.flip_layer(0, True)

    emissivity_low_e = float(outer.emissivity_front)
    emissivity_clear = float(center.emissivity_front)
    outer_thickness = float(outer.thickness) / 1000.0
    stack = [
        Solid(outer_thickness, float(inner.conductivity), "indoor low-e lite"),
        Gap(0.010, argon_fraction=0.9,
            emissivity_left=emissivity_low_e, emissivity_right=emissivity_clear),
        Solid(center_thickness_mm / 1000.0, float(center.conductivity), "center lite"),
        Gap(0.010, argon_fraction=0.9,
            emissivity_left=emissivity_clear, emissivity_right=emissivity_low_e),
        Solid(outer_thickness, float(outer.conductivity), "outdoor low-e lite"),
    ]
    local = solve_stack(stack, 294.15, 255.15, 7.7, 25.0)
    pywincalc_u = float(system.u())
    local_u = local.heat_flux / 39.0
    return {
        "center_thickness_mm": center_thickness_mm,
        "pywincalc_u_W_m2K": pywincalc_u,
        "local_u_W_m2K": local_u,
        "relative_difference": (local_u - pywincalc_u) / pywincalc_u,
        "pywincalc_layer_temperatures_K_outside_to_inside": [
            float(value) for value in system.layer_temperatures(pywincalc.TarcogSystemType.U)],
        "local_interface_temperatures_K_inside_to_outside": local.interface_temperatures.tolist(),
        "pywincalc_gap_effective_conductivities_W_mK": [
            float(value) for value in system.gap_layers_effective_conductivities(
                pywincalc.TarcogSystemType.U)],
        "local_gap_effective_conductivities_W_mK_inside_to_outside": [
            0.010 * (float(item["h_gas"]) + float(item["h_radiation"]))
            for item in local.gap_diagnostics],
    }


def run(product_directory: Path) -> dict[str, object]:
    thin = run_case(product_directory, 0.7)
    conventional = run_case(product_directory, 3.048)
    return {
        "case": "low-e triple; surfaces 2 and 5 low-e; two 10 mm 90% Ar gaps",
        "source": "LBNL-ETA/pyWinCalc public LOW-E_5.LOF and CLEAR_3.DAT products",
        "source_url": "https://github.com/LBNL-ETA/pyWinCalc/tree/main/examples/products",
        "pywincalc_version": importlib.metadata.version("pywincalc"),
        "thin_center_case": thin,
        "conventional_center_case": conventional,
        "thin_center_u_change_vs_conventional_pywincalc": (
            thin["pywincalc_u_W_m2K"] / conventional["pywincalc_u_W_m2K"] - 1.0),
        "acceptance": {
            "thin_case_difference_below_2_percent": abs(thin["relative_difference"]) < 0.02,
            "conventional_case_difference_below_2_percent": abs(conventional["relative_difference"]) < 0.02,
        },
        "interpretation": "Thermal-only component validation, not product certification.",
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
