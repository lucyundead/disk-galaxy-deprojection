from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from dgdp.coordinates import rotate_points, rotation_matrix_x, rotation_matrix_z
from dgdp.types import Geometry, MockImage, ParticleSet


def project_to_mock_image(
    particles: ParticleSet,
    geometry: Geometry,
    image_size: int,
    pixel_scale_kpc: float,
    psf_sigma_pixels: float,
    noise_sigma_fraction: float,
    seed: int,
) -> MockImage:
    rotated = rotate_points(particles.positions_kpc, rotation_matrix_z(geometry.disk_pa_deg))
    rotated = rotate_points(rotated, rotation_matrix_x(geometry.inclination_deg))
    xy = rotated[:, :2]

    half_size_kpc = 0.5 * image_size * pixel_scale_kpc
    edges = np.linspace(-half_size_kpc, half_size_kpc, image_size + 1)
    image, _, _ = np.histogram2d(
        xy[:, 1],
        xy[:, 0],
        bins=(edges, edges),
        weights=particles.masses_msun,
    )
    if psf_sigma_pixels > 0.0:
        image = gaussian_filter(image, sigma=psf_sigma_pixels, mode="constant")

    noiseless = image.astype(float)
    if noise_sigma_fraction > 0.0:
        rng = np.random.default_rng(seed)
        sigma = noise_sigma_fraction * max(float(noiseless.max()), 1.0)
        noisy = noiseless + rng.normal(0.0, sigma, size=noiseless.shape)
        noisy = np.clip(noisy, 0.0, None)
    else:
        noisy = noiseless.copy()

    return MockImage(
        image=noisy.astype(float),
        noiseless_image=noiseless.astype(float),
        pixel_scale_kpc=float(pixel_scale_kpc),
        geometry=geometry,
    )
