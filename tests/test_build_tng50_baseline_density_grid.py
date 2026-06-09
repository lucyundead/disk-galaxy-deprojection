import subprocess
import sys

import h5py
import numpy as np
import pandas as pd

from dgdp.density3d import (
    CylindricalGridSpec,
    density_grid_from_mass_grid,
    write_cylindrical_density_hdf5,
)
from dgdp.types import Geometry
from scripts.build_tng50_baseline_density_grid import (
    baseline_density_grid_from_image,
    sech2_vertical_bin_weights,
)


def test_sech2_vertical_bin_weights_are_symmetric_and_normalized():
    z_edges = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    weights = sech2_vertical_bin_weights(z_edges, scale_height_kpc=0.5)

    assert np.isclose(weights.sum(), 1.0)
    assert np.allclose(weights, weights[::-1])
    assert weights[1] > weights[0]


def _write_truth_grid(path, spec):
    mass_grid = np.zeros(
        (
            len(spec.r_edges_kpc) - 1,
            len(spec.phi_edges_rad) - 1,
            len(spec.z_edges_kpc) - 1,
        ),
        dtype=float,
    )
    grid = density_grid_from_mass_grid(mass_grid, spec, input_mass_msun=0.0)
    write_cylindrical_density_hdf5(path, grid, attrs={"coordinate_frame": "disk_bar_aligned"})


def test_baseline_grid_rotates_projection_bar_angle_into_truth_frame():
    spec = CylindricalGridSpec(
        r_edges_kpc=np.array([0.0, 2.0]),
        phi_edges_rad=np.array([-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi]),
        z_edges_kpc=np.array([-0.2, 0.0, 0.2]),
    )
    image = np.zeros((5, 5), dtype=float)
    image[2, 3] = 4.0

    grid = baseline_density_grid_from_image(
        image,
        geometry=Geometry(inclination_deg=0.0, disk_pa_deg=0.0, bar_angle_deg=90.0),
        pixel_scale_kpc=1.0,
        spec=spec,
        vertical_scale_height_kpc=0.1,
    )

    phi_mass = grid.mass_msun.sum(axis=(0, 2))
    assert phi_mass[:2].sum() > 0.0
    assert np.isclose(phi_mass[2:].sum(), 0.0)
    assert np.isclose(phi_mass.sum(), grid.grid_mass_msun)


def test_smooth_baseline_grid_does_not_leave_inner_phi_holes_for_smooth_image():
    spec = CylindricalGridSpec(
        r_edges_kpc=np.array([0.0, 0.4, 0.8, 1.2]),
        phi_edges_rad=np.linspace(-np.pi, np.pi, 17),
        z_edges_kpc=np.linspace(-1.0, 1.0, 9),
    )
    coords = np.linspace(-2.0, 2.0, 41)
    x_grid, y_grid = np.meshgrid(coords, coords)
    image = np.exp(-0.5 * (x_grid**2 + y_grid**2) / 0.8**2)

    grid = baseline_density_grid_from_image(
        image,
        geometry=Geometry(inclination_deg=20.0, disk_pa_deg=0.0, bar_angle_deg=0.0),
        pixel_scale_kpc=0.1,
        spec=spec,
        vertical_scale_height_kpc=0.3,
    )

    inner_panel = grid.mass_msun[:2].sum(axis=2)
    assert np.all(inner_panel > 0.0)
    assert np.isclose(grid.mass_msun.sum(), grid.grid_mass_msun)


def test_baseline_density_grid_script_writes_projection_files_with_truth_edges(tmp_path):
    manifest = tmp_path / "manifest.csv"
    residual_table = tmp_path / "residual_table.npz"
    truth_dir = tmp_path / "truth"
    output_dir = tmp_path / "baseline"
    truth_dir.mkdir()
    spec = CylindricalGridSpec(
        r_edges_kpc=np.array([0.0, 1.0, 2.0]),
        phi_edges_rad=np.linspace(-np.pi, np.pi, 5),
        z_edges_kpc=np.array([-0.5, 0.0, 0.5]),
    )
    _write_truth_grid(truth_dir / "subhalo_101_density_cylindrical.hdf5", spec)
    pd.DataFrame(
        [
            {
                "subhalo_id": 101,
                "projection_id": 0,
                "split": "train",
                "inclination_deg": 0.0,
                "disk_pa_deg": 0.0,
                "bar_angle_deg": 0.0,
            }
        ]
    ).to_csv(manifest, index=False)
    image = np.zeros((5, 5), dtype=np.float32)
    image[2, 3] = 6.0
    np.savez_compressed(
        residual_table,
        images=image[None, :, :],
        metadata=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        split=np.array(["train"]),
        galaxy_id=np.array([101], dtype=int),
        projection_id=np.array([0], dtype=int),
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_tng50_baseline_density_grid.py",
            "--config",
            "configs/milestone1.synthetic.toml",
            "--manifest",
            str(manifest),
            "--residual-table",
            str(residual_table),
            "--truth-grid-dir",
            str(truth_dir),
            "--output-dir",
            str(output_dir),
            "--vertical-scale-height-kpc",
            "0.1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "wrote 1 baseline cylindrical density grids" in result.stdout
    catalog = pd.read_csv(output_dir / "baseline_density_grid_catalog.csv")
    assert catalog["split"].tolist() == ["train"]
    grid_path = output_dir / catalog.loc[0, "baseline_density_file"]
    with h5py.File(grid_path, "r") as handle:
        assert handle["density_msun_per_kpc3"].shape == (2, 4, 2)
        assert np.allclose(handle["r_edges_kpc"][:], spec.r_edges_kpc)
        assert np.allclose(handle["phi_edges_rad"][:], spec.phi_edges_rad)
        assert np.allclose(handle["z_edges_kpc"][:], spec.z_edges_kpc)
        assert np.isclose(handle.attrs["input_image_mass_msun"], 6.0)
        assert handle.attrs["coordinate_frame"] == "disk_bar_aligned"
