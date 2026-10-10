import numpy as np

from thin_igu.fvm2d import EdgeSection, solve_edge_section


def test_uniform_slab_matches_1d_solution_and_closes_energy():
    # Zero gap thicknesses are not allowed by the IGU field builder, so use a
    # homogeneous material map through a section whose all conductivities match.
    section = EdgeSection(
        length=0.2,
        glass_thicknesses=(0.004, 0.004, 0.004),
        gap_widths=(0.004, 0.004),
        glass_conductivity=1.0,
        spacer_width=0.2,
        spacer_conductivity=1.0,
        h_indoor=8.0,
        h_outdoor=20.0,
        indoor_temperature=300.0,
        outdoor_temperature=280.0,
    )
    result = solve_edge_section(section, nx=20, ny=20)
    expected_flux = 20.0 / (1 / 8.0 + section.thickness / 1.0 + 1 / 20.0)
    assert np.allclose(result.heat_flux_indoor, expected_flux, rtol=2e-12)
    # Sparse direct roundoff accumulates over the 400 control volumes.
    assert result.energy_imbalance < 1e-10


def test_mesh_refinement_is_stable_for_edge_section():
    section = EdgeSection(length=0.12, spacer_width=0.012)
    coarse = solve_edge_section(section, nx=40, ny=20).total_heat_indoor
    medium = solve_edge_section(section, nx=80, ny=40).total_heat_indoor
    fine = solve_edge_section(section, nx=160, ny=80).total_heat_indoor
    assert abs(fine - medium) < abs(medium - coarse)
    assert abs(fine - medium) / abs(fine) < 0.02
