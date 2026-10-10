"""Linearized coupled-sheet decay modes for a multilayer glazing stack."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def pane_decay_lengths(
    pane_conductance: np.ndarray,
    gap_conductance: np.ndarray,
    h_indoor: float,
    h_outdoor: float,
) -> np.ndarray:
    """Return lateral far-field decay lengths [m], longest first.

    ``pane_conductance`` is ``k*t`` [W/K] for each pane and
    ``gap_conductance`` is total gas+radiation ``h`` [W/m2/K] between panes.
    The thin-sheet linearization gives ``L v = lambda K v`` and each mode decays
    as ``exp(-sqrt(lambda) x)``.
    """

    kt = np.asarray(pane_conductance, dtype=float)
    hg = np.asarray(gap_conductance, dtype=float)
    if kt.ndim != 1 or hg.shape != (len(kt) - 1,):
        raise ValueError("need n pane conductances and n-1 gap conductances")
    if np.any(kt <= 0.0) or np.any(hg <= 0.0) or h_indoor <= 0.0 or h_outdoor <= 0.0:
        raise ValueError("all conductances must be positive")
    coupling = np.zeros((len(kt), len(kt)))
    for i, h in enumerate(hg):
        coupling[i, i] += h
        coupling[i + 1, i + 1] += h
        coupling[i, i + 1] -= h
        coupling[i + 1, i] -= h
    coupling[0, 0] += h_indoor
    coupling[-1, -1] += h_outdoor
    eigenvalues = eigh(coupling, np.diag(kt), eigvals_only=True)
    if np.any(eigenvalues <= 0.0):
        raise RuntimeError("non-positive decay eigenvalue")
    return np.sort(1.0 / np.sqrt(eigenvalues))[::-1]
