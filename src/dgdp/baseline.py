from __future__ import annotations

import numpy as np

from dgdp.coordinates import rotate_points, rotation_matrix_x, rotation_matrix_z
from dgdp.types import MockImage, ParticleSet


def baseline_particles_from_image(
    mock: MockImage,
    *,
    vertical_scale_height_kpc: float,
) -> ParticleSet:
    image = np.asarray(mock.image, dtype=float)
    size = image.shape[0]
    pixel_scale = mock.pixel_scale_kpc
    coords_1d = (np.arange(size, dtype=float) - 0.5 * (size - 1)) * pixel_scale
    x_grid, y_grid = np.meshgrid(coords_1d, coords_1d)

    mass = image.ravel()
    keep = mass > 0.0
    x = x_grid.ravel()[keep]
    y_projected = y_grid.ravel()[keep]
    mass = mass[keep]

    inc = np.deg2rad(mock.geometry.inclination_deg)
    cos_inc = max(float(np.cos(inc)), 1.0e-3)
    y_disk = y_projected / cos_inc
    z = np.zeros_like(x) + vertical_scale_height_kpc
    z[::2] *= -1.0

    points = np.column_stack([x, y_disk, z])
    points = rotate_points(points, rotation_matrix_x(-mock.geometry.inclination_deg))
    points = rotate_points(points, rotation_matrix_z(-mock.geometry.disk_pa_deg))
    return ParticleSet(positions_kpc=points, masses_msun=mass)
