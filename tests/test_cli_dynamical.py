from pathlib import Path

import numpy as np

from dgdp import cli


class _FakeResult:
    def __init__(self, *, dynamical: bool = False):
        self.dynamical = dynamical
        self.total_mass = 5.25e10
        self.relative = False
        self.calls = []

    def save(self, out_dir) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "density.npz", density_3d=np.ones((2, 4, 2)))

    def bisymmetrize_for_dynamics(self, *, inner_radius_kpc, outer_radius_kpc):
        self.calls.append((inner_radius_kpc, outer_radius_kpc))
        return _FakeResult(dynamical=True)

    def potential(self, radius, z):
        assert np.all(z == 0.0)
        return -np.asarray(radius)


def test_cli_can_write_native_and_dynamical_density_and_potential(tmp_path, monkeypatch) -> None:
    native = _FakeResult()
    monkeypatch.setattr(cli, "deproject", lambda *args, **kwargs: native)
    out = tmp_path / "out"

    rc = cli.main([
        "galaxy.fits",
        "--distance-mpc", "18.83",
        "--inclination-deg", "42",
        "--pa-pix-deg", "151.1",
        "--dynamical-output", "6", "10",
        "--potential",
        "--no-figures",
        "-o", str(out),
    ])

    assert rc == 0
    assert native.calls == [(6.0, 10.0)]
    assert (out / "density.npz").exists()
    assert (out / "potential.npz").exists()
    assert (out / "dynamical" / "density.npz").exists()
    assert (out / "dynamical" / "potential.npz").exists()
    metadata = np.load(out / "dynamical" / "postprocess.npz")
    assert metadata["product_kind"] == "bisymmetric_dynamical_density"
    np.testing.assert_array_equal(metadata["retained_modes"], [0, 2, 4])
    np.testing.assert_allclose(metadata["outer_taper_kpc"], [6.0, 10.0])
