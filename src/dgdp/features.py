"""Image -> feature vector for the deprojection model (ported verbatim from the training path).

Layout (per row): pooled log image (image_feature_size**2) | metadata (3) | 2 absolute mass
features | central-flux features (len(apertures) log-fractions + 1 shape ratio). Everything
except the 2 absolute mass features is scale-invariant in the image amplitude.
"""
from __future__ import annotations

import numpy as np


def average_pool_images(images: np.ndarray, *, output_size: int) -> np.ndarray:
    images = np.asarray(images, dtype=np.float32)
    if images.ndim != 3:
        raise ValueError("images must have shape (n, height, width)")
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    height, width = images.shape[1:]
    if height % output_size != 0 or width % output_size != 0:
        raise ValueError("image dimensions must be divisible by output_size")
    y_block = height // output_size
    x_block = width // output_size
    return images.reshape(images.shape[0], output_size, y_block, output_size, x_block).mean(
        axis=(2, 4)
    )


def make_central_image_features(
    images: np.ndarray,
    inclinations_deg: np.ndarray,
    *,
    pixel_scale_kpc: float,
    aperture_semi_major_kpc: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
    min_axis_ratio: float = 0.2,
) -> np.ndarray:
    """Inclination-aware central flux features (elliptical apertures on the major/minor axes)."""
    images = np.asarray(images, dtype=np.float32)
    n_rows, n_minor, n_major = images.shape
    minor_kpc = (np.arange(n_minor, dtype=np.float32) - 0.5 * (n_minor - 1)) * pixel_scale_kpc
    major_kpc = (np.arange(n_major, dtype=np.float32) - 0.5 * (n_major - 1)) * pixel_scale_kpc
    major_sq_grid = np.broadcast_to(major_kpc[None, :] ** 2, (n_minor, n_major))
    minor_sq_grid = np.broadcast_to(minor_kpc[:, None] ** 2, (n_minor, n_major))
    total_flux = np.maximum(images.sum(axis=(1, 2)), 1.0e-12)
    axis_ratios = np.maximum(
        np.cos(np.deg2rad(np.asarray(inclinations_deg, dtype=np.float32))),
        min_axis_ratio,
    )
    largest_aperture = max(aperture_semi_major_kpc)
    fractions = np.zeros((n_rows, len(aperture_semi_major_kpc)), dtype=np.float32)
    shape_ratio = np.zeros(n_rows, dtype=np.float32)
    for ratio in np.unique(axis_ratios):
        rows = np.flatnonzero(np.isclose(axis_ratios, ratio))
        elliptical_radius = np.sqrt(major_sq_grid + minor_sq_grid / ratio**2)
        for column, semi_major in enumerate(aperture_semi_major_kpc):
            mask = elliptical_radius <= semi_major
            fractions[rows, column] = images[rows][:, mask].sum(axis=1) / total_flux[rows]
        central_mask = elliptical_radius <= largest_aperture
        central_flux = images[rows][:, central_mask]
        flux_sum = np.maximum(central_flux.sum(axis=1), 1.0e-12)
        major_moment = (central_flux * major_sq_grid[central_mask][None, :]).sum(axis=1) / flux_sum
        minor_moment = (central_flux * minor_sq_grid[central_mask][None, :]).sum(axis=1) / flux_sum
        shape_ratio[rows] = np.sqrt(minor_moment / np.maximum(major_moment, 1.0e-12))
    return np.column_stack(
        (
            np.log10(np.maximum(fractions, 1.0e-4)),
            shape_ratio,
        )
    ).astype(np.float32)


def make_features(
    images: np.ndarray,
    metadata: np.ndarray,
    *,
    baseline_grid_mass_msun: np.ndarray,
    image_feature_size: int,
    central_pixel_scale_kpc: float | None = None,
) -> np.ndarray:
    image_mass = np.sum(images, axis=(1, 2))
    normalized_images = np.divide(
        images,
        np.maximum(image_mass[:, None, None], 1.0),
        out=np.zeros_like(images, dtype=np.float32),
    )
    pooled = average_pool_images(
        np.log1p(normalized_images * images.shape[1] * images.shape[2]), output_size=image_feature_size
    )
    mass_features = np.column_stack(
        (
            np.log10(np.maximum(image_mass, 1.0)),
            np.log10(np.maximum(baseline_grid_mass_msun, 1.0)),
        )
    ).astype(np.float32)
    blocks = [
        pooled.reshape(pooled.shape[0], -1),
        np.asarray(metadata, dtype=np.float32),
        mass_features,
    ]
    if central_pixel_scale_kpc is not None:
        blocks.append(
            make_central_image_features(
                images,
                np.asarray(metadata, dtype=np.float32)[:, 0],
                pixel_scale_kpc=central_pixel_scale_kpc,
            )
        )
    return np.concatenate(blocks, axis=1).astype(np.float32)
