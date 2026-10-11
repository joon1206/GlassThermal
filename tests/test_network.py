import numpy as np

from thin_igu.network import Solid, solve_stack


def test_single_slab_with_surface_films_matches_series_resistance():
    ti, to, hi, ho = 300.0, 280.0, 8.0, 25.0
    slab = Solid(0.04, 0.2)
    expected = (ti - to) / (1 / hi + slab.thickness / slab.conductivity + 1 / ho)
    result = solve_stack([slab], ti, to, hi, ho)
    assert np.isclose(result.heat_flux, expected, rtol=1e-11)
    assert result.residual_inf < 1e-9


def test_multilayer_exact_series_resistance():
    layers = [Solid(0.01, 0.5), Solid(0.03, 0.1), Solid(0.005, 1.2)]
    ti, to, hi, ho = 295.0, 260.0, 7.7, 25.0
    expected = (ti - to) / (1 / hi + sum(x.thickness / x.conductivity for x in layers) + 1 / ho)
    result = solve_stack(layers, ti, to, hi, ho)
    assert np.isclose(result.heat_flux, expected, rtol=1e-11)
