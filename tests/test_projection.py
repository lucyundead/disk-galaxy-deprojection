import numpy as np

from dgdp.projection import project_to_mock_image
from dgdp.synthetic import make_barred_galaxy
from dgdp.types import Geometry, ParticleSet


def test_projection_conserves_mass_without_noise_or_psf():
    particles = make_barred_galaxy(seed=31, n_particles=2000, total_mass_msun=1.0e10)
    image = project_to_mock_image(
        particles,
        Geometry(inclination_deg=0.0, disk_pa_deg=0.0, bar_angle_deg=0.0),
        image_size=128,
        pixel_scale_kpc=0.5,
        psf_sigma_pixels=0.0,
        noise_sigma_fraction=0.0,
        seed=32,
    )

    assert np.isclose(image.noiseless_image.sum(), 1.0e10)
    assert np.isclose(image.image.sum(), 1.0e10)


def test_projection_is_reproducible_with_noise_seed():
    particles = make_barred_galaxy(seed=33, n_particles=2000, total_mass_msun=1.0e10)
    geom = Geometry(inclination_deg=45.0, disk_pa_deg=10.0, bar_angle_deg=20.0)
    a = project_to_mock_image(particles, geom, 64, 0.7, 1.0, 0.01, seed=34)
    b = project_to_mock_image(particles, geom, 64, 0.7, 1.0, 0.01, seed=34)

    assert np.allclose(a.image, b.image)


def test_bar_angle_rotates_faceon_image_before_projection():
    particles = ParticleSet(
        positions_kpc=np.array([[-4.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        masses_msun=np.array([1.0, 1.0]),
    )
    horizontal = project_to_mock_image(
        particles,
        Geometry(inclination_deg=0.0, disk_pa_deg=0.0, bar_angle_deg=0.0),
        image_size=32,
        pixel_scale_kpc=0.5,
        psf_sigma_pixels=0.0,
        noise_sigma_fraction=0.0,
        seed=1,
    ).image
    vertical = project_to_mock_image(
        particles,
        Geometry(inclination_deg=0.0, disk_pa_deg=0.0, bar_angle_deg=90.0),
        image_size=32,
        pixel_scale_kpc=0.5,
        psf_sigma_pixels=0.0,
        noise_sigma_fraction=0.0,
        seed=1,
    ).image

    coords = np.arange(32) - 15.5
    x_grid, y_grid = np.meshgrid(coords, coords)
    horizontal_x_var = float(np.sum(horizontal * x_grid**2))
    horizontal_y_var = float(np.sum(horizontal * y_grid**2))
    vertical_x_var = float(np.sum(vertical * x_grid**2))
    vertical_y_var = float(np.sum(vertical * y_grid**2))

    assert horizontal_x_var > horizontal_y_var
    assert vertical_y_var > vertical_x_var
