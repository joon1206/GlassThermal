"""Mesh-controlled severity/separation campaign for localized gap defects."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from thin_igu.decay import pane_decay_lengths  # noqa: E402
from thin_igu.decay_models import fit_decay_models, normalized_decay_shape  # noqa: E402
from thin_igu.experiments import thin_triple_stack  # noqa: E402
from thin_igu.fvm2d import EdgeSection, solve_edge_section  # noqa: E402
from thin_igu.network import solve_stack  # noqa: E402


# Below roughly 15% narrowing, the selected vertical-cavity correlation crosses
# its conduction/convection branch and nearly cancels the imposed perturbation.
# Start at 20% so every fitted case is above the numerical/model signal floor.
SEVERITIES = (0.20, 0.30, 0.40, 0.50, 0.60)
PAIR_SEVERITIES = (0.20, 0.40, 0.60)
SEPARATIONS = np.array((0.018, 0.030, 0.050, 0.080, 0.120))
DEFECT_CENTER = 0.120
PAIR_MIDPOINT = 0.150
SIGMA = 0.012


def gap_field(centres: tuple[float, ...], severity: float):
    def field(layer: int, x: np.ndarray) -> np.ndarray:
        nominal = 0.010
        if layer != 0:
            return np.full_like(x, nominal)
        shape = sum(np.exp(-0.5 * ((x - centre) / SIGMA) ** 2)
                    for centre in centres)
        return nominal * (1.0 - severity * shape)
    return field


def modal_lengths(section: EdgeSection) -> np.ndarray:
    stack = solve_stack(thin_triple_stack(), section.indoor_temperature,
                        section.outdoor_temperature, section.h_indoor,
                        section.h_outdoor)
    gap_h = np.array([item["h_gas"] + item["h_radiation"]
                      for item in stack.gap_diagnostics])
    return pane_decay_lengths(
        np.asarray(section.glass_thicknesses) * section.glass_conductivity,
        gap_h, section.h_indoor, section.h_outdoor)


def fit_profile(x: np.ndarray, delta_q: np.ndarray,
                lengths: np.ndarray) -> list[dict[str, object]]:
    # Use the interior-facing tail. The opposite tail eventually feels the
    # adiabatic sealed-edge boundary and changes sign, violating the half-space
    # modal assumption being tested here.
    mask = (x > DEFECT_CENTER + 0.020) & (x < 0.240)
    distance = x[mask] - DEFECT_CENTER
    magnitude = delta_q[mask]
    keep = magnitude > max(float(magnitude.max()) * 1e-4, 1e-12)
    return fit_decay_models(distance[keep], magnitude[keep], lengths)


def model_row(severity: float, model: dict[str, object]) -> dict[str, object]:
    return {
        "severity": severity,
        "model": model["model"],
        "length_1_m": model["lengths_m"][0],
        "length_2_m": model["lengths_m"][1] if len(model["lengths_m"]) > 1 else "",
        "length_3_m": model["lengths_m"][2] if len(model["lengths_m"]) > 2 else "",
        "amplitude_1_W_m2": model["amplitudes_W_m2"][0],
        "amplitude_2_W_m2": model["amplitudes_W_m2"][1] if len(model["amplitudes_W_m2"]) > 1 else "",
        "amplitude_3_W_m2": model["amplitudes_W_m2"][2] if len(model["amplitudes_W_m2"]) > 2 else "",
        "parameter_count": model["parameter_count"],
        "train_count": model["train_count"],
        "test_count": model["test_count"],
        "train_relative_l2": model["train_relative_l2"],
        "test_relative_l2": model["test_relative_l2"],
        "train_log_rmse": model["train_log_rmse"],
        "test_log_rmse": model["test_log_rmse"],
    }


def mesh_check(section: EdgeSection) -> list[dict[str, float]]:
    rows = []
    for nx, ny in ((120, 36), (160, 48), (240, 72)):
        reference = solve_edge_section(section, nx=nx, ny=ny)
        defect = solve_edge_section(
            section, nx=nx, ny=ny,
            gap_width=gap_field((DEFECT_CENTER,), 0.30))
        penalty = (defect.total_heat_indoor - reference.total_heat_indoor)
        rows.append({"nx": nx, "ny": ny,
                     "reference_heat_W_per_m": reference.total_heat_indoor,
                     "defect_increment_W_per_m": penalty,
                     "relative_defect_penalty": penalty / reference.total_heat_indoor,
                     "energy_imbalance": max(reference.energy_imbalance,
                                             defect.energy_imbalance)})
    return rows


def run(output_directory: Path, nx: int = 160, ny: int = 48) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    section = EdgeSection()
    lengths = modal_lengths(section)
    reference = solve_edge_section(section, nx=nx, ny=ny)
    all_fits: dict[float, list[dict[str, object]]] = {}
    fit_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, float]] = []
    heat_penalties = []

    for severity in SEVERITIES:
        result = solve_edge_section(
            section, nx=nx, ny=ny,
            gap_width=gap_field((DEFECT_CENTER,), severity))
        delta_q = result.heat_flux_indoor - reference.heat_flux_indoor
        fits = fit_profile(reference.x, delta_q, lengths)
        all_fits[severity] = fits
        fit_rows.extend(model_row(severity, model) for model in fits)
        increment = result.total_heat_indoor - reference.total_heat_indoor
        heat_penalties.append({"severity": severity,
                               "increment_W_per_m": increment,
                               "relative_penalty": increment / reference.total_heat_indoor})
        for x, perturbation in zip(reference.x, delta_q):
            profile_rows.append({"severity": severity, "x_m": x,
                                 "delta_q_W_m2": perturbation})

    separation_rows: list[dict[str, float]] = []
    separation_scores: list[dict[str, object]] = []
    for severity in PAIR_SEVERITIES:
        interactions = []
        pair_records = []
        for separation in SEPARATIONS:
            a = PAIR_MIDPOINT - separation / 2.0
            b = PAIR_MIDPOINT + separation / 2.0
            qa = solve_edge_section(section, nx=nx, ny=ny,
                                    gap_width=gap_field((a,), severity)).total_heat_indoor
            qb = solve_edge_section(section, nx=nx, ny=ny,
                                    gap_width=gap_field((b,), severity)).total_heat_indoor
            qab = solve_edge_section(section, nx=nx, ny=ny,
                                     gap_width=gap_field((a, b), severity)).total_heat_indoor
            interaction = qab - qa - qb + reference.total_heat_indoor
            denominator = abs(qa - reference.total_heat_indoor) + abs(qb - reference.total_heat_indoor)
            normalized = interaction / denominator
            interactions.append(normalized)
            pair_records.append({
                "severity": severity, "separation_m": separation,
                "q_a_W_per_m": qa, "q_b_W_per_m": qb, "q_ab_W_per_m": qab,
                "interaction_W_per_m": interaction,
                "normalized_interaction": normalized,
                "gaussian_overlap": float(np.exp(-separation**2 / (4.0 * SIGMA**2))),
            })

        observed = np.asarray(interactions)
        train = np.arange(len(SEPARATIONS)) % 2 == 0
        test = ~train
        models = {item["model"]: item for item in all_fits[severity]}
        shapes = {name: normalized_decay_shape(SEPARATIONS, model)
                  for name, model in models.items()}
        shapes["gaussian_overlap"] = np.exp(-SEPARATIONS**2 / (4.0 * SIGMA**2))
        for name, shape in shapes.items():
            coefficient = float(np.dot(shape[train], observed[train]) /
                                np.dot(shape[train], shape[train]))
            prediction = coefficient * shape
            denominator = max(float(np.linalg.norm(observed[test])), 1e-15)
            separation_scores.append({
                "severity": severity, "model": name,
                "calibration_scale": coefficient,
                "held_out_relative_l2": float(np.linalg.norm(
                    prediction[test] - observed[test]) / denominator),
            })
            for record, value in zip(pair_records, prediction):
                record[f"prediction_{name}"] = float(value)
        separation_rows.extend(pair_records)

    meshes = mesh_check(section)
    mesh_change = abs(meshes[-1]["defect_increment_W_per_m"]
                      - meshes[-2]["defect_increment_W_per_m"]) / abs(
                          meshes[-1]["defect_increment_W_per_m"])

    def write_csv(name: str, rows: list[dict[str, object]]) -> None:
        with (output_directory / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    write_csv("decay_model_comparison.csv", fit_rows)
    write_csv("severity_profiles.csv", profile_rows)
    write_csv("separation_interaction_predictions.csv", separation_rows)
    write_csv("separation_prediction_scores.csv", separation_scores)
    write_csv("severity_heat_penalties.csv", heat_penalties)
    write_csv("severity_study_mesh_check.csv", meshes)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    for model in ("single_exponential", "two_exponential", "analytical_modal"):
        rows = [row for row in fit_rows if row["model"] == model]
        axes[0].plot([row["severity"] for row in rows],
                     [row["test_relative_l2"] for row in rows], marker="o", label=model)
    axes[0].set(xlabel="Gap-narrowing severity", ylabel="Held-out relative L2 error",
                title="Spatial-tail prediction")
    axes[0].grid(alpha=0.25); axes[0].legend(fontsize=8)
    for severity in PAIR_SEVERITIES:
        rows = [row for row in separation_rows if row["severity"] == severity]
        axes[1].semilogy([row["separation_m"] * 1000 for row in rows],
                         np.maximum(np.abs([row["normalized_interaction"] for row in rows]), 1e-8),
                         marker="o", label=f"severity {severity:.1f}")
    axes[1].set(xlabel="Defect separation (mm)", ylabel="|Normalized interaction|",
                title="Non-additivity versus separation")
    axes[1].grid(alpha=0.25); axes[1].legend(fontsize=8)
    fig.savefig(output_directory / "severity_separation_summary.png", dpi=180)
    plt.close(fig)

    winner_counts = {name: 0 for name in
                     ("single_exponential", "two_exponential", "analytical_modal")}
    for severity in SEVERITIES:
        winner = min(all_fits[severity], key=lambda item: item["test_relative_l2"])["model"]
        winner_counts[winner] += 1
    summary = {
        "mesh": {"nx": nx, "ny": ny},
        "defect": {"family": "Gaussian narrowing of one gap", "sigma_m": SIGMA,
                   "center_m": DEFECT_CENTER,
                   "severity_range": list(SEVERITIES),
                   "lower_bound_reason": "Cases below 0.2 approach a branch-cancellation signal floor in the cavity correlation."},
        "analytical_modal_lengths_m": lengths.tolist(),
        "spatial_tail_held_out_winner_counts": winner_counts,
        "decay_model_results": fit_rows,
        "heat_penalties": heat_penalties,
        "separation_prediction_scores": separation_scores,
        "mesh_check": meshes,
        "finest_defect_increment_mesh_change": mesh_change,
        "acceptance": {
            "energy_imbalance_below_1e_9": max(row["energy_imbalance"] for row in meshes) < 1e-9,
            "defect_increment_mesh_change_below_1_percent": mesh_change < 0.01,
            "all_severities_and_separations_completed": (
                len(heat_penalties) == len(SEVERITIES)
                and len(separation_rows) == len(PAIR_SEVERITIES) * len(SEPARATIONS)),
        },
    }
    (output_directory / "severity_separation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path,
                        default=PROJECT / "validation")
    parser.add_argument("--nx", type=int, default=160)
    parser.add_argument("--ny", type=int, default=48)
    args = parser.parse_args()
    run(args.output_directory, args.nx, args.ny)


if __name__ == "__main__":
    main()
