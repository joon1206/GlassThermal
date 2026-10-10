import numpy as np

from thin_igu.properties import radiation_conductance, radiation_flux


def test_parallel_plate_radiation_matches_direct_formula():
    sigma = 5.670374419e-8
    t1, t2, e1, e2 = 310.0, 280.0, 0.84, 0.04
    exact = sigma * (t1**4 - t2**4) / (1 / e1 + 1 / e2 - 1)
    assert np.isclose(radiation_flux(t1, t2, e1, e2), exact, rtol=1e-14)
    assert np.isclose(radiation_conductance(t1, t2, e1, e2) * (t1 - t2), exact)
