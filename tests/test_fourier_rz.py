"""Tests for the even-m Fourier x (R,z) density representation."""

from __future__ import annotations

import numpy as np
import pytest

from dgdp.fourier_rz import (
    EVEN_M,
    coefficient_vector,
    derive_power_allocation,
    fit_fourier_rz,
    fit_fourier_rz_from_grid,
    harmonic_power,
    model_from_vector,
    n_coefficients,
    reconstruct_fourier_rz,
)


def _bar_disk_density(points: np.ndarray, *, amp_m2: float = 0.4) -> np.ndarray:
    """Analytic m=0 + m=2 density: exponential disk, sech^2 vertical, cos(2 phi) bar."""
    radius = np.hypot(points[:, 0], points[:, 1])
    phi = np.arctan2(points[:, 1], points[:, 0])
    vertical = 1.0 / np.cosh(points[:, 2] / 0.4) ** 2
    return np.exp(-radius / 3.0) * vertical * (1.0 + amp_m2 * np.cos(2.0 * phi))


def _eval_points(rng: np.random.Generator, n: int = 4000) -> np.ndarray:
    radius = rng.uniform(0.3, 12.0, n)
    phi = rng.uniform(-np.pi, np.pi, n)
    z = rng.uniform(-3.0, 3.0, n)
    return np.column_stack([radius * np.cos(phi), radius * np.sin(phi), z])


def test_uniform_fit_reconstructs_analytic_field() -> None:
    model = fit_fourier_rz(_bar_disk_density)
    assert model["n_coeff"] == n_coefficients(model)
    assert model["n_coeff"] == 11 * 25 * 25  # (m0 real) + 5*(re+im), each 25x25 (default grid)
    pts = _eval_points(np.random.default_rng(0))
    recon = reconstruct_fourier_rz(pts, model)
    truth = _bar_disk_density(pts)
    rel_l2 = np.linalg.norm(recon - truth) / np.linalg.norm(truth)
    assert rel_l2 < 0.10


def test_reconstruction_is_bar_symmetric_and_nonnegative() -> None:
    model = fit_fourier_rz(_bar_disk_density)
    rng = np.random.default_rng(1)
    pts = _eval_points(rng)
    flipped = pts.copy()
    flipped[:, :2] *= -1.0  # phi -> phi + pi; even-m terms are invariant
    recon = reconstruct_fourier_rz(pts, model)
    assert np.all(recon >= 0.0)
    np.testing.assert_allclose(recon, reconstruct_fourier_rz(flipped, model), rtol=1e-6, atol=1e-9)


def test_power_is_concentrated_in_m0_and_m2() -> None:
    model = fit_fourier_rz(_bar_disk_density)
    powers = harmonic_power(model)
    total = sum(powers.values())
    assert powers[0] / total > 0.7
    assert powers[2] / total > 0.01
    for m in (4, 6, 8, 10):
        assert powers[m] / total < 1e-3


def test_power_allocation_compresses_and_drops_dead_harmonics() -> None:
    # pin the coarser grid so the allocation candidates (<=14x6) can capture the budget
    model = fit_fourier_rz(_bar_disk_density, n_r=14, n_z_half=6)
    alloc, info = derive_power_allocation(model, capture=0.999)
    compact = fit_fourier_rz(_bar_disk_density, alloc=alloc, n_r=14, n_z_half=6)
    assert compact["n_coeff"] < model["n_coeff"]
    assert info["captured_fraction"] >= 0.999 - 1e-9
    # harmonics carrying negligible power must be dropped entirely
    for m in EVEN_M:
        if info["power_fraction"][m] < 1e-6:
            assert m not in alloc
    # the compressed reconstruction stays faithful to the full one
    pts = _eval_points(np.random.default_rng(2))
    full = reconstruct_fourier_rz(pts, model)
    comp = reconstruct_fourier_rz(pts, compact)
    assert np.linalg.norm(comp - full) / np.linalg.norm(full) < 0.05


def test_coefficient_vector_round_trips() -> None:
    model = fit_fourier_rz(_bar_disk_density, alloc={0: (8, 4), 2: (6, 3)})
    vector = coefficient_vector(model)
    assert vector.size == model["n_coeff"]
    rebuilt = model_from_vector(vector, model)
    pts = _eval_points(np.random.default_rng(3))
    np.testing.assert_allclose(
        reconstruct_fourier_rz(pts, model), reconstruct_fourier_rz(pts, rebuilt), rtol=1e-9, atol=1e-9
    )


def test_fit_from_grid_matches_callable() -> None:
    n_r, n_phi, n_z = 40, 64, 41
    r_centers = np.geomspace(0.1, 14.0, n_r)
    z_centers = np.linspace(-3.5, 3.5, n_z)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    rr, pp, zz = np.meshgrid(r_centers, phi, z_centers, indexing="ij")
    field = _bar_disk_density(
        np.column_stack([(rr * np.cos(pp)).ravel(), (rr * np.sin(pp)).ravel(), zz.ravel()])
    ).reshape(n_r, n_phi, n_z)
    model = fit_fourier_rz_from_grid(field, r_centers, z_centers, r_max=14.0, z_max=3.4)
    rng = np.random.default_rng(4)
    radius = rng.uniform(0.5, 12.0, 3000)
    phi_q = rng.uniform(-np.pi, np.pi, 3000)
    z_q = rng.uniform(-3.0, 3.0, 3000)
    pts = np.column_stack([radius * np.cos(phi_q), radius * np.sin(phi_q), z_q])
    recon = reconstruct_fourier_rz(pts, model)
    truth = _bar_disk_density(pts)
    assert np.linalg.norm(recon - truth) / np.linalg.norm(truth) < 0.1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
