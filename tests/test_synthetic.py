import numpy as np

from dgdp.synthetic import make_barred_galaxy


def test_synthetic_generator_is_reproducible():
    a = make_barred_galaxy(seed=11, n_particles=2000, total_mass_msun=1.0e10)
    b = make_barred_galaxy(seed=11, n_particles=2000, total_mass_msun=1.0e10)

    assert np.allclose(a.positions_kpc, b.positions_kpc)
    assert np.allclose(a.masses_msun, b.masses_msun)


def test_synthetic_generator_conserves_mass():
    particles = make_barred_galaxy(seed=12, n_particles=2000, total_mass_msun=2.5e10)

    assert np.isclose(particles.masses_msun.sum(), 2.5e10)
    assert particles.positions_kpc.shape == (2000, 3)
