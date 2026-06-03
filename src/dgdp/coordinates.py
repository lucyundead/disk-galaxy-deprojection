from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def rotation_matrix_x(angle_deg: float) -> NDArray[np.float64]:
    theta = np.deg2rad(angle_deg)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def rotation_matrix_z(angle_deg: float) -> NDArray[np.float64]:
    theta = np.deg2rad(angle_deg)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def rotate_points(points: NDArray[np.float64], matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(points, dtype=float) @ matrix.T


def cylindrical_radius(points: NDArray[np.float64]) -> NDArray[np.float64]:
    points = np.asarray(points, dtype=float)
    return np.hypot(points[:, 0], points[:, 1])


def bar_frame_angle(points: NDArray[np.float64], bar_angle_deg: float) -> NDArray[np.float64]:
    points = np.asarray(points, dtype=float)
    phi = np.arctan2(points[:, 1], points[:, 0])
    return np.angle(np.exp(1j * (phi - np.deg2rad(bar_angle_deg))))
