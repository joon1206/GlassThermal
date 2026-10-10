import numpy as np

from thin_igu.fvm2d import EdgeSection, solve_edge_section
from thin_igu.many_query import EdgeTopology, generate_spatial_ensemble


def test_reused_topology_matches_original_fvm():
    section = EdgeSection()
    topology = EdgeTopology(section, nx=40, ny=20)
    widths_0 = 0.010 * (1.0 - 0.25 * np.exp(-0.5 * ((topology.x - 0.12) / 0.015) ** 2))
    widths_1 = np.full_like(widths_0, 0.010)

    def field(layer, x):
        return widths_0 if layer == 0 else widths_1

    legacy = solve_edge_section(section, nx=40, ny=20, gap_width=field)
    reused = topology.solve_direct(widths_0, widths_1)
    np.testing.assert_allclose(
        reused.total_heat_W_per_m, legacy.total_heat_indoor, rtol=2e-12, atol=1e-12)


def test_spatial_ensemble_is_reproducible_and_bounded():
    x = (np.arange(24) + 0.5) * 0.30 / 24
    first = generate_spatial_ensemble(8, x, seed=17)
    second = generate_spatial_ensemble(8, x, seed=17)
    np.testing.assert_array_equal(first.width_gap_0_m, second.width_gap_0_m)
    assert first.width_gap_0_m.min() >= 0.0065
    assert first.width_gap_0_m.max() <= 0.0135
