"""Mesh/domain robustness study for the first localized gap defect."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from thin_igu.experiments import fit_decay_length, localized_gap_defect  # noqa: E402
from thin_igu.fvm2d import EdgeSection, solve_edge_section  # noqa: E402


def evaluate(length: float, nx: int, ny: int) -> dict[str, float]:
    section = EdgeSection(length=length)
    reference = solve_edge_section(section, nx=nx, ny=ny)
    defect = solve_edge_section(section, nx=nx, ny=ny, gap_width=localized_gap_defect)
    delta = defect.heat_flux_indoor - reference.heat_flux_indoor
    fit = fit_decay_length(reference.x, delta, 0.080)
    return {
        "length_m": length,
        "nx": nx,
        "ny": ny,
        "reference_heat_W_per_m": reference.total_heat_indoor,
        "defect_penalty_W_per_m": defect.total_heat_indoor - reference.total_heat_indoor,
        "relative_penalty": (defect.total_heat_indoor - reference.total_heat_indoor) / reference.total_heat_indoor,
        "energy_imbalance": max(reference.energy_imbalance, defect.energy_imbalance),
        **fit,
    }


def main() -> None:
    mesh_rows = [evaluate(0.30, nx, ny) for nx, ny in [(40, 20), (80, 40), (160, 80)]]
    domain_rows = [evaluate(length, nx, 48) for length, nx in [(0.15, 80), (0.30, 160), (0.60, 320)]]
    output = PROJECT / "validation"
    output.mkdir(exist_ok=True)
    for name, rows in [("mesh_convergence", mesh_rows), ("domain_sensitivity", domain_rows)]:
        with (output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    summary = {
        "mesh_rows": mesh_rows,
        "domain_rows": domain_rows,
        "finest_mesh_relative_change_from_medium": abs(
            mesh_rows[-1]["defect_penalty_W_per_m"] - mesh_rows[-2]["defect_penalty_W_per_m"]
        ) / abs(mesh_rows[-1]["defect_penalty_W_per_m"]),
        "domain_ell_relative_range": (
            max(row["ell_m"] for row in domain_rows) - min(row["ell_m"] for row in domain_rows)
        ) / sum(row["ell_m"] for row in domain_rows) * len(domain_rows),
    }
    (output / "convergence_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    markdown = f"""# Generated convergence summary

Regenerate with `python benchmarks/convergence_study.py`.

- Medium-to-fine defect-penalty change: {100 * summary['finest_mesh_relative_change_from_medium']:.4f}%
- Domain-study decay-length relative range: {100 * summary['domain_ell_relative_range']:.4f}%
- Finest-mesh defect penalty: {mesh_rows[-1]['defect_penalty_W_per_m']:.8f} W/m
- Finest-mesh fitted length: {1000 * mesh_rows[-1]['ell_m']:.4f} mm
- Finest-mesh log-space fit R-squared: {mesh_rows[-1]['r2_log']:.6f}

These values verify numerical stability for this case only. They do not establish
a universal decay law or manufacturing threshold.
"""
    (output / "convergence_summary.md").write_text(markdown)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
