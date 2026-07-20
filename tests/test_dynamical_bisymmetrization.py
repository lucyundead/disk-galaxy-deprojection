import numpy as np
import pytest

from dgdp.deproject import DeprojectionResult
from dgdp.density3d import cylindrical_bin_volumes, make_cylindrical_grid_spec


def _clumpy_result() -> DeprojectionResult:
    spec = make_cylindrical_grid_spec(r_min_kpc=1.0, r_max_kpc=5.0, n_r=4, n_phi=16, n_z=4)
    vol = cylindrical_bin_volumes(spec)
    r = 0.5 * (spec.r_edges_kpc[:-1] + spec.r_edges_kpc[1:])
    phi = 0.5 * (spec.phi_edges_rad[:-1] + spec.phi_edges_rad[1:])
    z = 0.5 * (spec.z_edges_kpc[:-1] + spec.z_edges_kpc[1:])

    azimuth = 1.0 + 0.30 * np.cos(2.0 * phi) + 0.18 * np.cos(5.0 * phi)
    vertical = np.array([0.10, 0.40, 0.40, 0.10])
    clumpy_vertical = 1.0 + 0.08 * np.cos(3.0 * phi)[:, None] * np.sign(z)[None, :]
    mass = (2.0 + r[:, None, None]) * azimuth[None, :, None]
    mass = mass * vertical[None, None, :] * clumpy_vertical[None, :, :]
    return DeprojectionResult(
        mass / vol,
        {"r": r, "phi": phi, "z": z},
        float(mass.sum()),
        False,
        vol,
        _ctx={"placeholder": True},
    )


def _mode_amplitude(field: np.ndarray, mode: int) -> np.ndarray:
    coeff = np.fft.rfft(field, axis=1) / field.shape[1]
    return 2.0 * np.abs(coeff[:, mode]) / coeff[:, 0].real


def test_bisymmetrize_for_dynamics_is_3d_symmetric_and_mass_preserving() -> None:
    result = _clumpy_result()
    original_density = result.density_3d.copy()
    original_mass = original_density * result._vol

    dynamical = result.bisymmetrize_for_dynamics(
        inner_radius_kpc=2.0,
        outer_radius_kpc=3.0,
    )
    dynamical_mass = dynamical.density_3d * dynamical._vol

    np.testing.assert_array_equal(result.density_3d, original_density)
    assert np.all(dynamical.density_3d >= 0.0)
    np.testing.assert_allclose(dynamical_mass.sum(axis=1), original_mass.sum(axis=1), rtol=1e-13)
    np.testing.assert_allclose(dynamical_mass.sum(), original_mass.sum(), rtol=1e-13)

    half_turn = dynamical_mass.shape[1] // 2
    np.testing.assert_allclose(
        dynamical_mass,
        np.roll(dynamical_mass, half_turn, axis=1),
        rtol=0,
        atol=1e-13,
    )
    surface = dynamical_mass.sum(axis=2)
    original_surface = original_mass.sum(axis=2)
    inner = result.grid["r"] <= 2.0
    np.testing.assert_allclose(
        _mode_amplitude(surface, 2)[inner],
        _mode_amplitude(original_surface, 2)[inner],
        rtol=1e-13,
    )
    assert np.max(_mode_amplitude(surface, 5)) < 1e-13

    outer = result.grid["r"] >= 3.0
    np.testing.assert_allclose(
        dynamical_mass[outer],
        np.broadcast_to(dynamical_mass[outer].mean(axis=1, keepdims=True), dynamical_mass[outer].shape),
        rtol=0,
        atol=1e-13,
    )
    assert dynamical._ctx is None
    assert dynamical._samples is None
    assert dynamical.reproj is None


def test_bisymmetrize_for_dynamics_rejects_invalid_transition() -> None:
    result = _clumpy_result()
    with pytest.raises(ValueError, match="inner_radius_kpc"):
        result.bisymmetrize_for_dynamics(inner_radius_kpc=3.0, outer_radius_kpc=3.0)
