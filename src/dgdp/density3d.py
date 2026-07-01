from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable
from typing import Any

import numpy as np

from dgdp.types import ParticleSet


@dataclass(frozen=True)
class CylindricalGridSpec:
    r_edges_kpc: np.ndarray
    phi_edges_rad: np.ndarray
    z_edges_kpc: np.ndarray


@dataclass(frozen=True)
class CylindricalDensityGrid:
    density_msun_per_kpc3: np.ndarray
    mass_msun: np.ndarray
    volumes_kpc3: np.ndarray
    spec: CylindricalGridSpec
    input_mass_msun: float
    grid_mass_msun: float
    dropped_mass_msun: float

    @property
    def mass_fraction_in_grid(self) -> float:
        if self.input_mass_msun <= 0.0:
            return 0.0
        return self.grid_mass_msun / self.input_mass_msun


def logarithmic_radial_edges(*, r_min_kpc: float, r_max_kpc: float, n_bins: int) -> np.ndarray:
    if r_min_kpc <= 0.0:
        raise ValueError("r_min_kpc must be positive")
    if r_max_kpc <= r_min_kpc:
        raise ValueError("r_max_kpc must be larger than r_min_kpc")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    log_edges = np.geomspace(r_min_kpc, r_max_kpc, n_bins)
    return np.concatenate(([0.0], log_edges))


def make_cylindrical_grid_spec(
    *,
    r_min_kpc: float = 0.05,
    r_max_kpc: float = 30.0,
    n_r: int = 32,
    n_phi: int = 48,
    z_max_kpc: float = 10.0,
    n_z: int = 32,
) -> CylindricalGridSpec:
    if n_phi < 1:
        raise ValueError("n_phi must be positive")
    if z_max_kpc <= 0.0:
        raise ValueError("z_max_kpc must be positive")
    if n_z < 1:
        raise ValueError("n_z must be positive")
    return CylindricalGridSpec(
        r_edges_kpc=logarithmic_radial_edges(
            r_min_kpc=r_min_kpc,
            r_max_kpc=r_max_kpc,
            n_bins=n_r,
        ),
        phi_edges_rad=np.linspace(-np.pi, np.pi, n_phi + 1),
        z_edges_kpc=np.linspace(-z_max_kpc, z_max_kpc, n_z + 1),
    )


def cylindrical_bin_volumes(spec: CylindricalGridSpec) -> np.ndarray:
    radial_area = 0.5 * (spec.r_edges_kpc[1:] ** 2 - spec.r_edges_kpc[:-1] ** 2)
    dphi = np.diff(spec.phi_edges_rad)
    dz = np.diff(spec.z_edges_kpc)
    return radial_area[:, None, None] * dphi[None, :, None] * dz[None, None, :]


def histogram_cylindrical_mass(
    particles: ParticleSet,
    spec: CylindricalGridSpec,
) -> tuple[np.ndarray, float]:
    positions = np.asarray(particles.positions_kpc, dtype=float)
    masses = np.asarray(particles.masses_msun, dtype=float)
    radius = np.hypot(positions[:, 0], positions[:, 1])
    phi = np.arctan2(positions[:, 1], positions[:, 0])
    z = positions[:, 2]
    valid = (
        np.isfinite(radius)
        & np.isfinite(phi)
        & np.isfinite(z)
        & np.isfinite(masses)
        & (masses > 0.0)
    )
    input_mass = float(masses[valid].sum())
    if not np.any(valid):
        shape = (
            len(spec.r_edges_kpc) - 1,
            len(spec.phi_edges_rad) - 1,
            len(spec.z_edges_kpc) - 1,
        )
        return np.zeros(shape, dtype=float), 0.0
    mass_grid, _ = np.histogramdd(
        np.column_stack((radius[valid], phi[valid], z[valid])),
        bins=(spec.r_edges_kpc, spec.phi_edges_rad, spec.z_edges_kpc),
        weights=masses[valid],
    )
    return mass_grid.astype(float), input_mass


def density_grid_from_mass_grid(
    mass_grid: np.ndarray,
    spec: CylindricalGridSpec,
    *,
    input_mass_msun: float,
) -> CylindricalDensityGrid:
    volumes = cylindrical_bin_volumes(spec)
    density = np.divide(
        mass_grid,
        volumes,
        out=np.zeros_like(mass_grid, dtype=float),
        where=volumes > 0.0,
    )
    grid_mass = float(np.sum(mass_grid))
    return CylindricalDensityGrid(
        density_msun_per_kpc3=density,
        mass_msun=mass_grid.astype(float),
        volumes_kpc3=volumes,
        spec=spec,
        input_mass_msun=float(input_mass_msun),
        grid_mass_msun=grid_mass,
        dropped_mass_msun=float(input_mass_msun) - grid_mass,
    )


def accumulate_cylindrical_density_grid(
    chunks: Iterable[ParticleSet],
    spec: CylindricalGridSpec,
) -> CylindricalDensityGrid:
    shape = (
        len(spec.r_edges_kpc) - 1,
        len(spec.phi_edges_rad) - 1,
        len(spec.z_edges_kpc) - 1,
    )
    mass_grid = np.zeros(shape, dtype=float)
    input_mass = 0.0
    for particles in chunks:
        chunk_grid, chunk_input_mass = histogram_cylindrical_mass(particles, spec)
        mass_grid += chunk_grid
        input_mass += chunk_input_mass
    return density_grid_from_mass_grid(mass_grid, spec, input_mass_msun=input_mass)


def particle_faceon_mass_image(
    particles: ParticleSet,
    *,
    image_size: int,
    radius_kpc: float,
    z_min_kpc: float | None = None,
    z_max_kpc: float | None = None,
) -> np.ndarray:
    positions = np.asarray(particles.positions_kpc, dtype=float)
    masses = np.asarray(particles.masses_msun, dtype=float)
    valid = (
        np.isfinite(positions[:, 0])
        & np.isfinite(positions[:, 1])
        & np.isfinite(masses)
        & (masses > 0.0)
    )
    if z_min_kpc is not None:
        valid &= positions[:, 2] >= z_min_kpc
    if z_max_kpc is not None:
        valid &= positions[:, 2] <= z_max_kpc
    edges = np.linspace(-radius_kpc, radius_kpc, image_size + 1)
    image, _, _ = np.histogram2d(
        positions[valid, 1],
        positions[valid, 0],
        bins=(edges, edges),
        weights=masses[valid],
    )
    return image.astype(float)


def accumulate_density_grid_and_faceon_image(
    chunks: Iterable[ParticleSet],
    spec: CylindricalGridSpec,
    *,
    image_size: int,
    radius_kpc: float,
) -> tuple[CylindricalDensityGrid, np.ndarray]:
    shape = (
        len(spec.r_edges_kpc) - 1,
        len(spec.phi_edges_rad) - 1,
        len(spec.z_edges_kpc) - 1,
    )
    mass_grid = np.zeros(shape, dtype=float)
    faceon_image = np.zeros((image_size, image_size), dtype=float)
    input_mass = 0.0
    for particles in chunks:
        chunk_grid, chunk_input_mass = histogram_cylindrical_mass(particles, spec)
        mass_grid += chunk_grid
        input_mass += chunk_input_mass
        faceon_image += particle_faceon_mass_image(
            particles,
            image_size=image_size,
            radius_kpc=radius_kpc,
            z_min_kpc=float(spec.z_edges_kpc[0]),
            z_max_kpc=float(spec.z_edges_kpc[-1]),
        )
    return density_grid_from_mass_grid(mass_grid, spec, input_mass_msun=input_mass), faceon_image


def build_cylindrical_density_grid(
    particles: ParticleSet,
    spec: CylindricalGridSpec,
) -> CylindricalDensityGrid:
    return accumulate_cylindrical_density_grid([particles], spec)


def read_cylindrical_grid_spec_hdf5(path: Path) -> CylindricalGridSpec:
    import h5py  # optional (only for hdf5 I/O; not needed on the deproject inference path)
    with h5py.File(path, "r") as handle:
        return CylindricalGridSpec(
            r_edges_kpc=np.asarray(handle["r_edges_kpc"], dtype=float),
            phi_edges_rad=np.asarray(handle["phi_edges_rad"], dtype=float),
            z_edges_kpc=np.asarray(handle["z_edges_kpc"], dtype=float),
        )


def grid_radial_mass_profile(grid: CylindricalDensityGrid) -> np.ndarray:
    return np.sum(grid.mass_msun, axis=(1, 2))


def particle_radial_mass_profile(particles: ParticleSet, spec: CylindricalGridSpec) -> np.ndarray:
    positions = np.asarray(particles.positions_kpc, dtype=float)
    masses = np.asarray(particles.masses_msun, dtype=float)
    radius = np.hypot(positions[:, 0], positions[:, 1])
    z = positions[:, 2]
    valid = (
        np.isfinite(radius)
        & np.isfinite(z)
        & np.isfinite(masses)
        & (masses > 0.0)
        & (z >= spec.z_edges_kpc[0])
        & (z <= spec.z_edges_kpc[-1])
    )
    profile, _ = np.histogram(radius[valid], bins=spec.r_edges_kpc, weights=masses[valid])
    return profile.astype(float)


def project_grid_to_faceon_image(
    grid: CylindricalDensityGrid,
    *,
    image_size: int,
    radius_kpc: float,
) -> np.ndarray:
    r_centers = 0.5 * (grid.spec.r_edges_kpc[:-1] + grid.spec.r_edges_kpc[1:])
    phi_centers = 0.5 * (grid.spec.phi_edges_rad[:-1] + grid.spec.phi_edges_rad[1:])
    rr, pp = np.meshgrid(r_centers, phi_centers, indexing="ij")
    xy = np.column_stack((rr.ravel() * np.cos(pp.ravel()), rr.ravel() * np.sin(pp.ravel())))
    weights = np.sum(grid.mass_msun, axis=2).ravel()
    edges = np.linspace(-radius_kpc, radius_kpc, image_size + 1)
    image, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], bins=(edges, edges), weights=weights)
    return image.astype(float)


def write_cylindrical_density_hdf5(
    path: Path,
    grid: CylindricalDensityGrid,
    *,
    attrs: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import h5py  # optional (only for hdf5 I/O; not needed on the deproject inference path)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "density_msun_per_kpc3",
            data=grid.density_msun_per_kpc3,
            compression="gzip",
        )
        handle.create_dataset("mass_msun", data=grid.mass_msun, compression="gzip")
        handle.create_dataset("r_edges_kpc", data=grid.spec.r_edges_kpc)
        handle.create_dataset("phi_edges_rad", data=grid.spec.phi_edges_rad)
        handle.create_dataset("z_edges_kpc", data=grid.spec.z_edges_kpc)
        handle.attrs["input_mass_msun"] = grid.input_mass_msun
        handle.attrs["grid_mass_msun"] = grid.grid_mass_msun
        handle.attrs["dropped_mass_msun"] = grid.dropped_mass_msun
        handle.attrs["mass_fraction_in_grid"] = grid.mass_fraction_in_grid
        for key, value in attrs.items():
            handle.attrs[key] = value
