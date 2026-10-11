"""Small deterministic experiments; outputs are data, not publication claims."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import curve_fit

from .decay import pane_decay_lengths
from .fvm2d import EdgeSection, solve_edge_section
from .network import Gap, Solid, solve_stack


def thin_triple_stack() -> list[Solid | Gap]:
    return [
        Solid(0.003, 1.0, "outer glass"),
        Gap(0.010, argon_fraction=0.90, emissivity_left=0.84, emissivity_right=0.04),
        Solid(0.0007, 1.0, "thin center glass"),
        Gap(0.010, argon_fraction=0.90, emissivity_left=0.04, emissivity_right=0.84),
        Solid(0.003, 1.0, "inner glass"),
    ]


def localized_gap_defect(layer: int, x: np.ndarray) -> np.ndarray:
    """Synthetic 30% narrowing centred 80 mm from the edge (sigma=12 mm)."""

    nominal = 0.010
    if layer != 0:
        return np.full_like(x, nominal)
    return nominal * (1.0 - 0.30 * np.exp(-0.5 * ((x - 0.080) / 0.012) ** 2))


def fit_decay_length(x: np.ndarray, perturbation: np.ndarray, defect_center: float) -> dict[str, float]:
    """Fit an exponential only on the edge-facing tail and report fit quality."""

    mask = (x > 0.015) & (x < defect_center - 0.018)
    distance = defect_center - x[mask]
    magnitude = np.abs(perturbation[mask])
    floor = max(magnitude.max() * 1e-8, 1e-14)
    keep = magnitude > floor
    distance, magnitude = distance[keep], magnitude[keep]
    if len(distance) < 5:
        return {"ell_m": float("nan"), "r2_log": float("nan")}

    def model(d: np.ndarray, amplitude: float, ell: float) -> np.ndarray:
        return amplitude * np.exp(-d / ell)

    pars, _ = curve_fit(model, distance, magnitude, p0=(magnitude.max(), 0.02), bounds=(0, np.inf))
    prediction = model(distance, *pars)
    log_y, log_p = np.log(magnitude), np.log(np.maximum(prediction, floor))
    ss_res = float(np.sum((log_y - log_p) ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    return {"ell_m": float(pars[1]), "r2_log": 1.0 - ss_res / ss_tot if ss_tot else float("nan")}


def _gaussian_gap_field(centres: tuple[float, ...], severity: float = 0.30, sigma: float = 0.012):
    def field(layer: int, x: np.ndarray) -> np.ndarray:
        nominal = 0.010
        if layer != 0:
            return np.full_like(x, nominal)
        shape = sum(np.exp(-0.5 * ((x - centre) / sigma) ** 2) for centre in centres)
        return nominal * (1.0 - severity * shape)
    return field


def pair_interaction_sweep(section: EdgeSection, reference_heat: float) -> list[dict[str, float]]:
    """Compute same-family interaction versus separation at fixed midpoint."""

    rows: list[dict[str, float]] = []
    midpoint = 0.15
    for separation in (0.018, 0.030, 0.050, 0.080, 0.120):
        a, b = midpoint - separation / 2.0, midpoint + separation / 2.0
        qa = solve_edge_section(section, gap_width=_gaussian_gap_field((a,))).total_heat_indoor
        qb = solve_edge_section(section, gap_width=_gaussian_gap_field((b,))).total_heat_indoor
        qab = solve_edge_section(section, gap_width=_gaussian_gap_field((a, b))).total_heat_indoor
        interaction = qab - qa - qb + reference_heat
        denominator = abs(qa - reference_heat) + abs(qb - reference_heat) + 1e-15
        rows.append({
            "separation_m": separation,
            "q_a_W_per_m": qa,
            "q_b_W_per_m": qb,
            "q_ab_W_per_m": qab,
            "interaction_W_per_m": interaction,
            "normalized_interaction": interaction / denominator,
        })
    return rows


def run(output_dir: Path | None = None) -> dict[str, float]:
    output_dir = output_dir or Path(__file__).resolve().parents[2] / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    stack = solve_stack(thin_triple_stack(), 294.15, 255.15, 7.7, 25.0)
    section = EdgeSection()
    reference = solve_edge_section(section)
    defect = solve_edge_section(section, gap_width=localized_gap_defect)
    delta_q = defect.heat_flux_indoor - reference.heat_flux_indoor
    fit = fit_decay_length(reference.x, delta_q, 0.080)
    gap_h = np.array([
        diagnostic["h_gas"] + diagnostic["h_radiation"]
        for diagnostic in stack.gap_diagnostics
    ])
    modes = pane_decay_lengths(
        np.array(section.glass_thicknesses) * section.glass_conductivity,
        gap_h,
        section.h_indoor,
        section.h_outdoor,
    )
    interactions = pair_interaction_sweep(section, reference.total_heat_indoor)
    with (output_dir / "localized_gap_profile.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["x_m", "q_reference_W_m2", "q_defect_W_m2", "delta_q_W_m2"])
        writer.writerows(zip(reference.x, reference.heat_flux_indoor, defect.heat_flux_indoor, delta_q))
    with (output_dir / "pair_interaction.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(interactions[0]))
        writer.writeheader(); writer.writerows(interactions)
    summary = {
        "center_u_W_m2K": stack.heat_flux / (294.15 - 255.15),
        "reference_heat_W_per_m": reference.total_heat_indoor,
        "defect_heat_W_per_m": defect.total_heat_indoor,
        "relative_defect_penalty": (defect.total_heat_indoor - reference.total_heat_indoor) / reference.total_heat_indoor,
        "reference_energy_imbalance": reference.energy_imbalance,
        "coupled_sheet_decay_modes_m": modes.tolist(),
        "max_abs_normalized_pair_interaction": max(
            abs(row["normalized_interaction"]) for row in interactions),
        **fit,
    }
    try:
        revision = subprocess.run(
            ["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[4],
            check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    metadata = {
        "git_revision": revision,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unreported",
        "gpu_used": False,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "mesh": {"nx": 160, "ny": 48},
        "solver": "scipy.sparse.linalg.spsolve",
        "random_seed": None,
        "synthetic_defect": {"family": "gaussian gap narrowing", "amplitude": 0.30, "center_m": 0.080, "sigma_m": 0.012},
        "summary": summary,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
