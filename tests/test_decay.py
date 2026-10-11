import numpy as np

from thin_igu.decay import pane_decay_lengths


def test_decay_modes_are_positive_and_ordered():
    lengths = pane_decay_lengths(
        np.array([0.003, 0.0007, 0.003]),
        np.array([2.0, 2.0]),
        7.7,
        25.0,
    )
    assert lengths.shape == (3,)
    assert np.all(lengths > 0.0)
    assert np.all(np.diff(lengths) <= 0.0)


def test_single_sheet_reduces_to_fin_length():
    kt, h1, h2 = 0.004, 8.0, 20.0
    length = pane_decay_lengths(np.array([kt]), np.array([]), h1, h2)[0]
    assert np.isclose(length, np.sqrt(kt / (h1 + h2)))
