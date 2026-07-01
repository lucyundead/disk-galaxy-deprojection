"""Tests for dgdp.agama_density (skipped unless the prebuilt AGAMA is importable)."""

from __future__ import annotations

import numpy as np
import pytest

agama = pytest.importorskip("agama")

from dgdp.agama_density import (  # noqa: E402  (after importorskip by design)
    circular_velocity,
    fourier_rz_to_agama_density,
    to_agama_density,
    to_agama_potential,
)
from dgdp.fourier_rz import fit_fourier_rz  # noqa: E402

agama.setUnits(mass=1, length=1, velocity=1)


def _disk_bar(xyz: np.ndarray) -> np.ndarray:
    """Smooth exponential disk + mild m=2 bar, thin in z (Msun/kpc^3)."""
    xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
    radius = np.hypot(xyz[:, 0], xyz[:, 1])
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    bar = 1.0 + 0.3 * np.cos(2.0 * phi) * np.exp(-radius / 2.0)
    return 1e9 * np.exp(-radius / 3.0) * np.exp(-np.abs(xyz[:, 2]) / 0.3) * bar


def test_to_agama_density_preserves_total_mass_exactly():
    target = 4.2e10
    dens = to_agama_density(_disk_bar, total_mass=target, r_max=15.0, z_max=4.0)
    assert dens.totalMass() == pytest.approx(target, rel=1e-9)


def test_wrapped_density_matches_input():
    dens = to_agama_density(_disk_bar, r_max=15.0, z_max=4.0)
    rng = np.random.default_rng(0)
    radius = rng.uniform(0.5, 8.0, 500)
    phi = rng.uniform(-np.pi, np.pi, 500)
    pts = np.column_stack([radius * np.cos(phi), radius * np.sin(phi), rng.uniform(-2.0, 2.0, 500)])
    rel = np.linalg.norm(dens.density(pts) - _disk_bar(pts)) / np.linalg.norm(_disk_bar(pts))
    assert rel < 0.1


def test_to_agama_potential_and_circular_velocity():
    dens = to_agama_density(_disk_bar, total_mass=4.2e10, r_max=15.0, z_max=4.0)
    pot = to_agama_potential(dens)
    force = np.asarray(pot.force(np.array([[2.0, 0.0, 0.0], [5.0, 0.0, 0.0]])))
    assert force.shape == (2, 3)
    assert (force[:, 0] < 0).all()  # inward radial acceleration
    v_c = circular_velocity(pot, np.array([1.0, 2.0, 4.0, 6.0]))
    assert v_c.shape == (4,)
    assert np.all(v_c > 0) and np.all(v_c < 600.0)


def test_fourier_rz_to_agama_density_conserves_mass():
    model = fit_fourier_rz(_disk_bar, n_r=16, n_z_half=8)
    target = 3.0e10
    dens = fourier_rz_to_agama_density(model, total_mass=target, r_max=15.0, z_max=4.0)
    assert dens.totalMass() == pytest.approx(target, rel=1e-9)
