"""Held-out fitting utilities for lateral perturbation decay models."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import nnls


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    scale = max(float(np.linalg.norm(observed)), np.finfo(float).tiny)
    floor = max(float(observed.max()) * 1e-12, np.finfo(float).tiny)
    return {
        "relative_l2": float(np.linalg.norm(predicted - observed) / scale),
        "log_rmse": float(np.sqrt(np.mean((np.log(np.maximum(predicted, floor))
                                            - np.log(np.maximum(observed, floor))) ** 2))),
    }


def _fit_basis(distance: np.ndarray, magnitude: np.ndarray,
               lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    basis = np.exp(-distance[:, None] / lengths[None, :])
    amplitudes, _ = nnls(basis, magnitude)
    return amplitudes, basis @ amplitudes


def fit_decay_models(
    distance: np.ndarray,
    magnitude: np.ndarray,
    analytical_lengths: np.ndarray,
    candidate_lengths: np.ndarray | None = None,
) -> list[dict[str, object]]:
    """Fit three positive exponential models and score an untouched far-field block.

    The single- and two-exponential length scales are selected only on the
    training points. Analytical modal lengths are fixed before fitting; only
    their non-negative amplitudes are calibrated.
    """
    distance = np.asarray(distance, dtype=float)
    magnitude = np.asarray(magnitude, dtype=float)
    analytical_lengths = np.asarray(analytical_lengths, dtype=float)
    if distance.shape != magnitude.shape or distance.ndim != 1 or len(distance) < 10:
        raise ValueError("distance and magnitude need at least ten paired samples")
    if np.any(distance <= 0) or np.any(magnitude <= 0) or np.any(analytical_lengths <= 0):
        raise ValueError("distances, magnitudes, and modal lengths must be positive")
    order = np.argsort(distance)
    distance, magnitude = distance[order], magnitude[order]
    # Calibrate on the near tail and test extrapolation into the farther tail.
    # A contiguous block is deliberately harder and less leakage-prone than
    # holding out alternating, spatially adjacent cells.
    cutoff = np.quantile(distance, 2.0 / 3.0)
    train = distance <= cutoff
    test = ~train
    grid = (np.geomspace(0.003, 0.100, 90) if candidate_lengths is None
            else np.asarray(candidate_lengths, dtype=float))

    def choose(candidates: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        best = None
        for lengths in candidates:
            amplitudes, prediction = _fit_basis(distance[train], magnitude[train], lengths)
            score = _metrics(magnitude[train], prediction)["relative_l2"]
            if best is None or score < best[0]:
                best = (score, lengths.copy(), amplitudes)
        assert best is not None
        return best[1], best[2]

    single_lengths, single_amplitudes = choose([np.array([length]) for length in grid])
    pair_candidates = [np.array(pair) for pair in combinations(grid, 2)
                       if pair[1] / pair[0] >= 1.15]
    double_lengths, double_amplitudes = choose(pair_candidates)
    modal_amplitudes, _ = _fit_basis(
        distance[train], magnitude[train], analytical_lengths)

    specifications = [
        ("single_exponential", single_lengths, single_amplitudes, 2),
        ("two_exponential", double_lengths, double_amplitudes, 4),
        ("analytical_modal", analytical_lengths, modal_amplitudes,
         len(analytical_lengths)),
    ]
    results = []
    for name, lengths, amplitudes, parameter_count in specifications:
        prediction = np.exp(-distance[:, None] / lengths[None, :]) @ amplitudes
        results.append({
            "model": name,
            "lengths_m": lengths.tolist(),
            "amplitudes_W_m2": amplitudes.tolist(),
            "parameter_count": parameter_count,
            "train_count": int(train.sum()),
            "test_count": int(test.sum()),
            "train_relative_l2": _metrics(magnitude[train], prediction[train])["relative_l2"],
            "test_relative_l2": _metrics(magnitude[test], prediction[test])["relative_l2"],
            "train_log_rmse": _metrics(magnitude[train], prediction[train])["log_rmse"],
            "test_log_rmse": _metrics(magnitude[test], prediction[test])["log_rmse"],
        })
    return results


def normalized_decay_shape(separation: np.ndarray, model: dict[str, object]) -> np.ndarray:
    """Return a model's unit-amplitude decay shape for separation studies."""
    lengths = np.asarray(model["lengths_m"], dtype=float)
    amplitudes = np.asarray(model["amplitudes_W_m2"], dtype=float)
    if amplitudes.sum() <= 0:
        return np.zeros_like(np.asarray(separation, dtype=float))
    return (np.exp(-np.asarray(separation)[:, None] / lengths[None, :]) @ amplitudes
            / amplitudes.sum())
