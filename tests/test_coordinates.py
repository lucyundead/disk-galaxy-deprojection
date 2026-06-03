import numpy as np

from dgdp.coordinates import bar_frame_angle, cylindrical_radius, rotate_points, rotation_matrix_z


def test_rotation_matrix_z_rotates_x_to_y():
    points = np.array([[1.0, 0.0, 0.0]])
    rotated = rotate_points(points, rotation_matrix_z(90.0))

    assert np.allclose(rotated, np.array([[0.0, 1.0, 0.0]]), atol=1e-12)


def test_cylindrical_radius_uses_xy_plane():
    points = np.array([[3.0, 4.0, 10.0], [5.0, 12.0, -2.0]])

    assert np.allclose(cylindrical_radius(points), np.array([5.0, 13.0]))


def test_bar_frame_angle_subtracts_bar_angle():
    points = np.array([[0.0, 1.0, 0.0]])

    angle = bar_frame_angle(points, bar_angle_deg=90.0)

    assert np.allclose(angle, np.array([0.0]), atol=1e-12)
