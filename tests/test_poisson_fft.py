"""Tests for the isolated FFT Poisson/force solver against the point-mass solution."""

from __future__ import annotations

import numpy as np

from dgdp.poisson_fft import (
    GRAV_KPC_KMS2_MSUN,
    circular_velocity_profile,
    isolated_potential_and_forces,
)


def _point_mass_grid(n=40, h=1.0, mass=1.0e10):
    density = np.zeros((n, n, n))
    center = n // 2
    density[center, center, center] = mass / h**3  # one cell carries the mass
    axis = (np.arange(n) - center) * h
    return density, axis, center, mass


def test_point_mass_potential_and_force_match_kepler() -> None:
    density, axis, center, mass = _point_mass_grid()
    phi, ax, ay, az = isolated_potential_and_forces(density, spacing=axis[1] - axis[0])
    # sample several cells away along x (avoid the softened self cell)
    for offset in (5, 8, 12):
        d = offset * (axis[1] - axis[0])
        phi_pred = -GRAV_KPC_KMS2_MSUN * mass / d
        a_pred = GRAV_KPC_KMS2_MSUN * mass / d**2
        i = center + offset
        assert abs(phi[i, center, center] - phi_pred) / abs(phi_pred) < 0.03
        # force points back toward the center (-x), magnitude ~ GM/d^2
        assert ax[i, center, center] < 0
        assert abs(abs(ax[i, center, center]) - a_pred) / a_pred < 0.05


def test_potential_is_symmetric() -> None:
    density, _, center, _ = _point_mass_grid(n=32)
    phi, _, _, _ = isolated_potential_and_forces(density, spacing=1.0)
    # reflection symmetry through the mass cell: phi[c+d] == phi[c-d] along each axis
    for d in (3, 7, 11):
        assert abs(phi[center + d, center, center] - phi[center - d, center, center]) < 1e-6
        assert abs(phi[center, center + d, center] - phi[center, center - d, center]) < 1e-6
        assert abs(phi[center, center, center + d] - phi[center, center, center - d]) < 1e-6


def test_circular_velocity_matches_keplerian() -> None:
    density, axis, _, mass = _point_mass_grid(n=48, h=1.0)
    _, ax, ay, _ = isolated_potential_and_forces(density, spacing=1.0)
    edges = np.array([5.5, 6.5])
    v_c = circular_velocity_profile(ax, ay, axis, axis, axis, edges, z_slab=1.5)
    v_kepler = np.sqrt(GRAV_KPC_KMS2_MSUN * mass / 6.0)
    assert abs(v_c[0] - v_kepler) / v_kepler < 0.05
