from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BenchmarkConfig:
    seed: int
    n_galaxies: int
    projections_per_galaxy: int
    image_size: int
    pixel_scale_kpc: float
    max_inclination_deg: float
    psf_sigma_pixels: float
    noise_sigma_fraction: float
    train_fraction: float
    val_fraction: float
    test_fraction: float
    radial_bins_kpc: tuple[float, ...]
    vertical_bins_kpc: tuple[float, ...]


@dataclass(frozen=True)
class ParticleSet:
    positions_kpc: FloatArray
    masses_msun: FloatArray
    velocities_kms: FloatArray | None = None


@dataclass(frozen=True)
class Geometry:
    inclination_deg: float
    disk_pa_deg: float
    bar_angle_deg: float


@dataclass(frozen=True)
class MockImage:
    image: FloatArray
    noiseless_image: FloatArray
    pixel_scale_kpc: float
    geometry: Geometry


@dataclass(frozen=True)
class SummaryVector:
    names: tuple[str, ...]
    values: FloatArray
