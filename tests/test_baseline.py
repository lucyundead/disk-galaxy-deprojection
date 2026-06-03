import numpy as np

from dgdp.baseline import baseline_particles_from_image
from dgdp.projection import project_to_mock_image
from dgdp.summaries import extract_summary_vector
from dgdp.synthetic import make_barred_galaxy
from dgdp.types import Geometry


def test_face_on_baseline_conserves_image_mass():
    particles = make_barred_galaxy(seed=41, n_particles=4000, total_mass_msun=1.0e10)
    geom = Geometry(inclination_deg=0.0, disk_pa_deg=0.0, bar_angle_deg=0.0)
    mock = project_to_mock_image(particles, geom, 64, 0.6, 0.0, 0.0, seed=42)
    baseline = baseline_particles_from_image(mock, vertical_scale_height_kpc=0.4)

    assert np.isclose(baseline.masses_msun.sum(), mock.image.sum())


def test_baseline_summary_names_match_true_summary_names():
    particles = make_barred_galaxy(seed=43, n_particles=4000, total_mass_msun=1.0e10)
    geom = Geometry(inclination_deg=30.0, disk_pa_deg=0.0, bar_angle_deg=0.0)
    mock = project_to_mock_image(particles, geom, 64, 0.6, 0.0, 0.0, seed=44)
    baseline = baseline_particles_from_image(mock, vertical_scale_height_kpc=0.4)

    true_summary = extract_summary_vector(
        particles, (0.0, 2.0, 4.0, 8.0), (-2.0, -0.5, 0.5, 2.0)
    )
    base_summary = extract_summary_vector(
        baseline, (0.0, 2.0, 4.0, 8.0), (-2.0, -0.5, 0.5, 2.0)
    )

    assert base_summary.names == true_summary.names
