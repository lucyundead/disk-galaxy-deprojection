from __future__ import annotations

import tomllib
from pathlib import Path

from dgdp.types import BenchmarkConfig


def load_config(path: Path) -> BenchmarkConfig:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return BenchmarkConfig(
        seed=int(data["seed"]),
        n_galaxies=int(data["n_galaxies"]),
        projections_per_galaxy=int(data["projections_per_galaxy"]),
        image_size=int(data["image_size"]),
        pixel_scale_kpc=float(data["pixel_scale_kpc"]),
        max_inclination_deg=float(data["max_inclination_deg"]),
        psf_sigma_pixels=float(data["psf_sigma_pixels"]),
        noise_sigma_fraction=float(data["noise_sigma_fraction"]),
        train_fraction=float(data["train_fraction"]),
        val_fraction=float(data["val_fraction"]),
        test_fraction=float(data["test_fraction"]),
        radial_bins_kpc=tuple(float(x) for x in data["radial_bins_kpc"]),
        vertical_bins_kpc=tuple(float(x) for x in data["vertical_bins_kpc"]),
    )
