import numpy as np

from thin_igu.decay_models import fit_decay_models


def test_two_exponential_beats_single_on_held_out_biexponential_data():
    distance = np.linspace(0.006, 0.080, 31)
    magnitude = 4.0 * np.exp(-distance / 0.012) + 1.5 * np.exp(-distance / 0.036)
    fits = {item["model"]: item for item in fit_decay_models(
        distance, magnitude, np.array([0.036, 0.012]))}
    assert fits["two_exponential"]["test_relative_l2"] < fits["single_exponential"]["test_relative_l2"]
    assert fits["analytical_modal"]["test_relative_l2"] < 1e-10
