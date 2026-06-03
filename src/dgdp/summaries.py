from __future__ import annotations

import numpy as np

from dgdp.coordinates import bar_frame_angle, cylindrical_radius
from dgdp.types import ParticleSet, SummaryVector


def _weighted_mad(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    median = float(np.interp(0.5, cdf, values))
    return float(np.average(np.abs(values - median), weights=weights))


def extract_summary_vector(
    particles: ParticleSet,
    radial_bins_kpc: tuple[float, ...],
    vertical_bins_kpc: tuple[float, ...],
    *,
    bar_angle_deg: float = 0.0,
) -> SummaryVector:
    pos = particles.positions_kpc
    mass = particles.masses_msun
    radius = cylindrical_radius(pos)
    names: list[str] = []
    values: list[float] = []

    for upper in radial_bins_kpc[1:]:
        names.append(f"enclosed_mass_r_le_{upper:.3f}_kpc")
        values.append(float(mass[radius <= upper].sum()))

    for lower, upper in zip(radial_bins_kpc[:-1], radial_bins_kpc[1:]):
        area = np.pi * (upper**2 - lower**2)
        shell_mass = float(mass[(radius >= lower) & (radius < upper)].sum())
        names.append(f"surface_density_r_{lower:.3f}_{upper:.3f}_kpc")
        values.append(shell_mass / area)

    names.append("disk_scale_height_mad_kpc")
    values.append(_weighted_mad(pos[:, 2], mass))

    phi_bar = bar_frame_angle(pos, bar_angle_deg)
    inside_bar = radius <= min(5.0, radial_bins_kpc[-1])
    if np.any(inside_bar):
        weights = mass[inside_bar]
        a2 = np.average(np.exp(2j * phi_bar[inside_bar]), weights=weights)
        bar_a2 = float(np.abs(a2))
        bar_thickness = _weighted_mad(pos[inside_bar, 2], weights)
    else:
        bar_a2 = 0.0
        bar_thickness = 0.0

    names.append("bar_thickness_mad_kpc")
    values.append(bar_thickness)
    names.append("bar_a2_amplitude")
    values.append(bar_a2)

    central_radius = radial_bins_kpc[1]
    total_mass = float(mass.sum())
    central_mass = float(mass[radius <= central_radius].sum())
    names.append("central_mass_fraction")
    values.append(central_mass / total_mass)

    return SummaryVector(names=tuple(names), values=np.asarray(values, dtype=float))
