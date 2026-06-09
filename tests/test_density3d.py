import numpy as np

from dgdp.density3d import (
    CylindricalGridSpec,
    accumulate_density_grid_and_faceon_image,
    build_cylindrical_density_grid,
    grid_radial_mass_profile,
    logarithmic_radial_edges,
    particle_radial_mass_profile,
    read_cylindrical_grid_spec_hdf5,
    write_cylindrical_density_hdf5,
)
from dgdp.types import ParticleSet


def test_logarithmic_radial_edges_include_zero_and_sample_inner_radius():
    edges = logarithmic_radial_edges(r_min_kpc=0.1, r_max_kpc=30.0, n_bins=8)

    assert edges[0] == 0.0
    assert edges[-1] == 30.0
    assert np.all(np.diff(edges) > 0.0)
    assert edges[2] - edges[1] < edges[-1] - edges[-2]


def test_cylindrical_density_grid_conserves_mass_inside_grid():
    spec = CylindricalGridSpec(
        r_edges_kpc=np.array([0.0, 1.0, 2.0]),
        phi_edges_rad=np.array([-np.pi, 0.0, np.pi]),
        z_edges_kpc=np.array([-1.0, 0.0, 1.0]),
    )
    particles = ParticleSet(
        positions_kpc=np.array(
            [
                [0.5, 0.0, -0.5],
                [1.5, 0.0, 0.5],
                [3.0, 0.0, 0.0],
            ]
        ),
        masses_msun=np.array([2.0, 3.0, 5.0]),
    )

    grid = build_cylindrical_density_grid(particles, spec)

    assert np.isclose(grid.input_mass_msun, 10.0)
    assert np.isclose(grid.grid_mass_msun, 5.0)
    assert np.isclose(grid.dropped_mass_msun, 5.0)
    assert np.isclose(np.sum(grid.density_msun_per_kpc3 * grid.volumes_kpc3), 5.0)


def test_grid_radial_profile_matches_particle_histogram_inside_grid():
    spec = CylindricalGridSpec(
        r_edges_kpc=np.array([0.0, 1.0, 2.0, 4.0]),
        phi_edges_rad=np.linspace(-np.pi, np.pi, 5),
        z_edges_kpc=np.array([-1.0, 1.0]),
    )
    particles = ParticleSet(
        positions_kpc=np.array(
            [
                [0.5, 0.0, 0.0],
                [0.0, 1.5, 0.0],
                [-3.0, 0.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        ),
        masses_msun=np.array([1.0, 2.0, 4.0, 8.0]),
    )

    grid = build_cylindrical_density_grid(particles, spec)

    assert np.allclose(
        grid_radial_mass_profile(grid),
        particle_radial_mass_profile(particles, spec),
    )


def test_accumulate_density_grid_and_faceon_image_conserves_image_mass():
    spec = CylindricalGridSpec(
        r_edges_kpc=np.array([0.0, 1.0, 2.0]),
        phi_edges_rad=np.linspace(-np.pi, np.pi, 5),
        z_edges_kpc=np.array([-1.0, 1.0]),
    )
    particles = ParticleSet(
        positions_kpc=np.array([[0.5, 0.0, 0.0], [0.0, -1.5, 0.0], [0.5, 0.0, 2.0]]),
        masses_msun=np.array([2.0, 3.0, 7.0]),
    )

    grid, image = accumulate_density_grid_and_faceon_image(
        [particles],
        spec,
        image_size=16,
        radius_kpc=2.0,
    )

    assert np.isclose(grid.grid_mass_msun, 5.0)
    assert np.isclose(np.sum(image), 5.0)


def test_read_cylindrical_grid_spec_hdf5_round_trips_written_edges(tmp_path):
    spec = CylindricalGridSpec(
        r_edges_kpc=np.array([0.0, 1.0, 3.0]),
        phi_edges_rad=np.linspace(-np.pi, np.pi, 5),
        z_edges_kpc=np.array([-1.0, 0.0, 1.0]),
    )
    grid = build_cylindrical_density_grid(
        ParticleSet(
            positions_kpc=np.array([[0.5, 0.0, 0.0]]),
            masses_msun=np.array([2.0]),
        ),
        spec,
    )
    path = tmp_path / "grid.hdf5"
    write_cylindrical_density_hdf5(path, grid, attrs={})

    loaded = read_cylindrical_grid_spec_hdf5(path)

    assert np.allclose(loaded.r_edges_kpc, spec.r_edges_kpc)
    assert np.allclose(loaded.phi_edges_rad, spec.phi_edges_rad)
    assert np.allclose(loaded.z_edges_kpc, spec.z_edges_kpc)
