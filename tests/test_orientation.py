import numpy as np

from dgdp.coordinates import rotate_points, rotation_matrix_z
from dgdp.orientation import align_particles_to_disk_bar_frame, estimate_bar_angle_deg
from dgdp.types import ParticleSet


def test_estimate_bar_angle_recovers_faceon_major_axis():
    positions = np.array(
        [
            [-4.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )
    rotated = rotate_points(positions, rotation_matrix_z(30.0))
    particles = ParticleSet(positions_kpc=rotated, masses_msun=np.ones(4))

    assert np.isclose(estimate_bar_angle_deg(particles, radius_kpc=5.0), 30.0)


def test_align_particles_to_disk_bar_frame_places_bar_on_x_axis():
    positions = np.array(
        [
            [-4.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, -4.0, 0.0],
            [0.0, 4.0, 0.0],
        ]
    )
    rotated = rotate_points(positions, rotation_matrix_z(45.0))
    particles = ParticleSet(positions_kpc=rotated, masses_msun=np.ones(len(rotated)))

    aligned = align_particles_to_disk_bar_frame(
        particles,
        normal_radius_kpc=10.0,
        bar_radius_kpc=5.0,
    ).particles

    assert np.var(aligned.positions_kpc[:, 0]) > np.var(aligned.positions_kpc[:, 1])
