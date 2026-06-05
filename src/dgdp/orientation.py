from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dgdp.coordinates import rotate_points, rotation_matrix_z
from dgdp.types import ParticleSet


@dataclass(frozen=True)
class AlignmentResult:
    particles: ParticleSet
    disk_normal: np.ndarray
    bar_angle_deg: float
    orientation_method: str


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("cannot normalize zero vector")
    return np.asarray(vector, dtype=float) / norm


def faceon_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_axis = _unit(normal)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(z_axis, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    x_axis = _unit(np.cross(reference, z_axis))
    y_axis = _unit(np.cross(z_axis, x_axis))
    return x_axis, y_axis, z_axis


def estimate_disk_normal(particles: ParticleSet, *, radius_kpc: float) -> tuple[np.ndarray, str]:
    positions = np.asarray(particles.positions_kpc, dtype=float)
    masses = np.asarray(particles.masses_msun, dtype=float)
    radii = np.linalg.norm(positions, axis=1)
    mask = np.isfinite(radii) & (radii <= radius_kpc) & np.isfinite(masses) & (masses > 0.0)
    if int(mask.sum()) < 10:
        mask = np.isfinite(radii) & np.isfinite(masses) & (masses > 0.0)
    pos = positions[mask]
    weight = masses[mask]
    if particles.velocities_kms is not None and len(pos) >= 10:
        velocities = np.asarray(particles.velocities_kms, dtype=float)[mask]
        mean_vel = np.average(velocities, axis=0, weights=weight)
        angular_momentum = np.sum(weight[:, None] * np.cross(pos, velocities - mean_vel), axis=0)
        if np.linalg.norm(angular_momentum) > 0.0:
            return _unit(angular_momentum), "angular_momentum"

    centered = pos - np.average(pos, axis=0, weights=weight)
    covariance = (centered * weight[:, None]).T @ centered / float(weight.sum())
    _, eigenvectors = np.linalg.eigh(covariance)
    return _unit(eigenvectors[:, 0]), "minor_axis"


def rotate_to_faceon(particles: ParticleSet, normal: np.ndarray) -> ParticleSet:
    x_axis, y_axis, z_axis = faceon_basis(normal)
    basis = np.column_stack((x_axis, y_axis, z_axis))
    velocities = particles.velocities_kms @ basis if particles.velocities_kms is not None else None
    return ParticleSet(
        positions_kpc=particles.positions_kpc @ basis,
        masses_msun=particles.masses_msun,
        velocities_kms=velocities,
    )


def estimate_bar_angle_deg(particles: ParticleSet, *, radius_kpc: float) -> float:
    positions = np.asarray(particles.positions_kpc, dtype=float)
    masses = np.asarray(particles.masses_msun, dtype=float)
    radius = np.hypot(positions[:, 0], positions[:, 1])
    mask = np.isfinite(radius) & (radius <= radius_kpc) & np.isfinite(masses) & (masses > 0.0)
    if int(mask.sum()) < 10:
        mask = np.isfinite(radius) & np.isfinite(masses) & (masses > 0.0)
    x = positions[mask, 0]
    y = positions[mask, 1]
    weight = masses[mask]
    quadrupole = np.sum(weight * (x + 1j * y) ** 2)
    return float(np.rad2deg(0.5 * np.angle(quadrupole)))


def align_particles_to_disk_bar_frame(
    particles: ParticleSet,
    *,
    normal_radius_kpc: float,
    bar_radius_kpc: float,
) -> AlignmentResult:
    normal, method = estimate_disk_normal(particles, radius_kpc=normal_radius_kpc)
    faceon = rotate_to_faceon(particles, normal)
    bar_angle = estimate_bar_angle_deg(faceon, radius_kpc=bar_radius_kpc)
    aligned_positions = rotate_points(faceon.positions_kpc, rotation_matrix_z(-bar_angle))
    aligned_velocities = (
        rotate_points(faceon.velocities_kms, rotation_matrix_z(-bar_angle))
        if faceon.velocities_kms is not None
        else None
    )
    return AlignmentResult(
        particles=ParticleSet(
            positions_kpc=aligned_positions,
            masses_msun=faceon.masses_msun,
            velocities_kms=aligned_velocities,
        ),
        disk_normal=normal,
        bar_angle_deg=bar_angle,
        orientation_method=method,
    )
