import numpy as np

from dgdp.summaries import extract_summary_vector
from dgdp.synthetic import make_barred_galaxy


def test_summary_vector_has_stable_names_and_finite_values():
    particles = make_barred_galaxy(seed=21, n_particles=3000, total_mass_msun=1.0e10)
    radial_bins = (0.0, 2.0, 4.0, 8.0)
    vertical_bins = (-2.0, -0.5, 0.5, 2.0)

    summary = extract_summary_vector(particles, radial_bins, vertical_bins)

    assert summary.names[0] == "enclosed_mass_r_le_2.000_kpc"
    assert "disk_scale_height_mad_kpc" in summary.names
    assert "bar_a2_amplitude" in summary.names
    assert np.all(np.isfinite(summary.values))


def test_enclosed_mass_profile_is_monotonic():
    particles = make_barred_galaxy(seed=22, n_particles=3000, total_mass_msun=1.0e10)
    summary = extract_summary_vector(
        particles, (0.0, 2.0, 4.0, 8.0), (-2.0, -0.5, 0.5, 2.0)
    )

    enclosed = summary.values[:3]

    assert np.all(np.diff(enclosed) >= 0.0)
