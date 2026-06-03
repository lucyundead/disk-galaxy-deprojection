import numpy as np

from dgdp.projection import project_to_mock_image
from dgdp.synthetic import make_barred_galaxy
from dgdp.types import Geometry


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
