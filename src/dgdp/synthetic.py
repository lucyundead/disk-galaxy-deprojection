from __future__ import annotations

import numpy as np

from dgdp.types import ParticleSet


def make_barred_galaxy(
    *,
    seed: int,
    n_particles: int = 20_000,
    total_mass_msun: float = 5.0e10,
    disk_scale_kpc: float = 4.0,
    bar_fraction: float = 0.35,
    bar_length_kpc: float = 5.0,
    disk_height_kpc: float = 0.35,
    bar_height_kpc: float = 0.75,
) -> ParticleSet:
    rng = np.random.default_rng(seed)
    n_bar = int(round(n_particles * bar_fraction))
    n_disk = n_particles - n_bar

    disk_r = rng.gamma(shape=2.0, scale=disk_scale_kpc, size=n_disk)
    disk_phi = rng.uniform(0.0, 2.0 * np.pi, size=n_disk)
    disk_x = disk_r * np.cos(disk_phi)
    disk_y = disk_r * np.sin(disk_phi)
    disk_z = rng.laplace(0.0, disk_height_kpc, size=n_disk)

    bar_x = rng.uniform(-bar_length_kpc, bar_length_kpc, size=n_bar)
    taper = np.clip(1.0 - np.abs(bar_x) / bar_length_kpc, 0.15, 1.0)
    bar_y = rng.normal(0.0, 0.45 * taper, size=n_bar)
    bar_z = rng.laplace(0.0, bar_height_kpc, size=n_bar)

    positions = np.vstack(
        [
            np.column_stack([disk_x, disk_y, disk_z]),
            np.column_stack([bar_x, bar_y, bar_z]),
        ]
    ).astype(float)
    masses = np.full(n_particles, total_mass_msun / n_particles, dtype=float)
    return ParticleSet(positions_kpc=positions, masses_msun=masses)
