import subprocess
import sys

import numpy as np
import pandas as pd

from dgdp.density3d import (
    CylindricalGridSpec,
    density_grid_from_mass_grid,
    write_cylindrical_density_hdf5,
)


def _write_grid(path, spec, mass_value):
    shape = (
        len(spec.r_edges_kpc) - 1,
        len(spec.phi_edges_rad) - 1,
        len(spec.z_edges_kpc) - 1,
    )
    grid = density_grid_from_mass_grid(
        np.full(shape, mass_value, dtype=float),
        spec,
        input_mass_msun=float(mass_value * np.prod(shape)),
    )
    write_cylindrical_density_hdf5(path, grid, attrs={"coordinate_frame": "disk_bar_aligned"})
    return grid


def _write_minimal_inputs(tmp_path, *, baseline_spec=None):
    manifest = tmp_path / "manifest.csv"
    residual_table = tmp_path / "residual_table.npz"
    truth_dir = tmp_path / "truth"
    baseline_dir = tmp_path / "baseline"
    output = tmp_path / "density_residual_table.npz"
    truth_dir.mkdir()
    baseline_dir.mkdir()
    spec = CylindricalGridSpec(
        r_edges_kpc=np.array([0.0, 1.0, 2.0]),
        phi_edges_rad=np.linspace(-np.pi, np.pi, 5),
        z_edges_kpc=np.array([-0.5, 0.0, 0.5]),
    )
    truth = _write_grid(truth_dir / "subhalo_101_density_cylindrical.hdf5", spec, 5.0)
    baseline = _write_grid(
        baseline_dir / "subhalo_101_projection_0_baseline_density_cylindrical.hdf5",
        baseline_spec or spec,
        2.0,
    )
    pd.DataFrame(
        [
            {
                "subhalo_id": 101,
                "projection_id": 0,
                "split": "train",
                "inclination_deg": 20.0,
                "disk_pa_deg": 0.0,
                "bar_angle_deg": 45.0,
            }
        ]
    ).to_csv(manifest, index=False)
    pd.DataFrame(
        [
            {
                "subhalo_id": 101,
                "projection_id": 0,
                "split": "train",
                "baseline_density_file": "subhalo_101_projection_0_baseline_density_cylindrical.hdf5",
                "true_density_file": "subhalo_101_density_cylindrical.hdf5",
            }
        ]
    ).to_csv(baseline_dir / "baseline_density_grid_catalog.csv", index=False)
    np.savez_compressed(
        residual_table,
        images=np.ones((1, 4, 4), dtype=np.float32),
        metadata=np.array([[20.0, 0.0, 45.0]], dtype=np.float32),
        split=np.array(["train"]),
        galaxy_id=np.array([101], dtype=int),
        projection_id=np.array([0], dtype=int),
    )
    return manifest, residual_table, truth_dir, baseline_dir, output, truth, baseline


def test_density_residual_table_stores_truth_baseline_and_delta_density(tmp_path):
    manifest, residual_table, truth_dir, baseline_dir, output, truth, baseline = _write_minimal_inputs(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_tng50_density_residual_table.py",
            "--manifest",
            str(manifest),
            "--residual-table",
            str(residual_table),
            "--truth-grid-dir",
            str(truth_dir),
            "--baseline-grid-dir",
            str(baseline_dir),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "wrote density residual table" in result.stdout
    table = np.load(output)
    assert table["truth_density"].shape == (1, 2, 4, 2)
    assert np.allclose(table["delta_density"][0], truth.density_msun_per_kpc3 - baseline.density_msun_per_kpc3)
    assert np.allclose(table["baseline_density"][0], baseline.density_msun_per_kpc3)
    assert table["split"].tolist() == ["train"]
    assert table["galaxy_id"].tolist() == [101]
    assert table["projection_id"].tolist() == [0]
    assert np.allclose(table["metadata"][0], np.array([20.0, 0.0, 45.0], dtype=np.float32))


def test_density_residual_table_rejects_grid_edge_mismatch(tmp_path):
    bad_spec = CylindricalGridSpec(
        r_edges_kpc=np.array([0.0, 1.0, 3.0]),
        phi_edges_rad=np.linspace(-np.pi, np.pi, 5),
        z_edges_kpc=np.array([-0.5, 0.0, 0.5]),
    )
    manifest, residual_table, truth_dir, baseline_dir, output, _, _ = _write_minimal_inputs(
        tmp_path,
        baseline_spec=bad_spec,
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_tng50_density_residual_table.py",
            "--manifest",
            str(manifest),
            "--residual-table",
            str(residual_table),
            "--truth-grid-dir",
            str(truth_dir),
            "--baseline-grid-dir",
            str(baseline_dir),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "grid edges do not match" in result.stderr
