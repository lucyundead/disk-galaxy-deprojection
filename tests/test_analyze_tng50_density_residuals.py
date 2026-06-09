import json
import subprocess
import sys

import numpy as np

from scripts.analyze_tng50_density_residuals import (
    cylindrical_bin_volumes_from_edges,
    radial_mass_profiles,
    vertical_mass_profiles,
)


def test_profile_helpers_sum_cylindrical_mass_axes():
    mass = np.arange(16, dtype=float).reshape(2, 4, 2)

    assert np.allclose(radial_mass_profiles(mass[None, ...])[0], mass.sum(axis=(1, 2)))
    assert np.allclose(vertical_mass_profiles(mass[None, ...])[0], mass.sum(axis=(0, 1)))


def test_analyze_density_residuals_writes_diagnostics_and_pca(tmp_path):
    data_path = tmp_path / "density_residual_table.npz"
    output_dir = tmp_path / "diagnostics"
    r_edges = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    phi_edges = np.linspace(-np.pi, np.pi, 5, dtype=np.float32)
    z_edges = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
    volumes = cylindrical_bin_volumes_from_edges(r_edges, phi_edges, z_edges)
    base = np.arange(16, dtype=np.float32).reshape(2, 4, 2) + 10.0
    truth_mass = np.stack([base, base * 1.1, base * 0.9, base * 1.2])
    delta_mass = np.stack(
        [
            np.ones_like(base) * 0.1,
            np.ones_like(base) * -0.2,
            np.arange(16, dtype=np.float32).reshape(2, 4, 2) * 0.01,
            np.flip(base, axis=0) * 0.02,
        ]
    )
    baseline_mass = truth_mass - delta_mass
    np.savez_compressed(
        data_path,
        images=np.zeros((4, 8, 8), dtype=np.float32),
        truth_density=(truth_mass / volumes[None, ...]).astype(np.float32),
        baseline_density=(baseline_mass / volumes[None, ...]).astype(np.float32),
        delta_density=(delta_mass / volumes[None, ...]).astype(np.float32),
        metadata=np.array(
            [
                [20.0, 0.0, 0.0],
                [20.0, 0.0, 45.0],
                [40.0, 0.0, 0.0],
                [60.0, 0.0, 90.0],
            ],
            dtype=np.float32,
        ),
        split=np.array(["train", "train", "val", "test"]),
        galaxy_id=np.array([1, 2, 3, 4], dtype=int),
        projection_id=np.array([0, 0, 0, 0], dtype=int),
        r_edges_kpc=r_edges,
        phi_edges_rad=phi_edges,
        z_edges_kpc=z_edges,
        truth_grid_mass_msun=truth_mass.sum(axis=(1, 2, 3)).astype(np.float32),
        baseline_grid_mass_msun=baseline_mass.sum(axis=(1, 2, 3)).astype(np.float32),
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_tng50_density_residuals.py",
            "--data",
            str(data_path),
            "--output-dir",
            str(output_dir),
            "--n-components",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "wrote density residual diagnostics" in result.stdout
    diagnostics = json.loads((output_dir / "density_residual_diagnostics.json").read_text())
    assert diagnostics["n_rows"] == 4
    assert diagnostics["density_shape"] == [2, 4, 2]
    assert diagnostics["split_counts"] == {"test": 1, "train": 2, "val": 1}
    assert diagnostics["baseline_total_mass_mae_msun"] > 0.0
    assert diagnostics["training_readiness"]["mass_scale_ok"] is True
    assert "inclination_deg" in diagnostics["grouped_baseline_delta_mae_msun"]
    assert "bar_angle_deg" in diagnostics["grouped_baseline_delta_mae_msun"]

    pca = np.load(output_dir / "density_residual_pca.npz")
    assert pca["coefficients"].shape == (4, 2)
    assert pca["components"].shape == (2, 16)
    assert pca["mean"].shape == (16,)
    assert pca["explained_variance_ratio"].shape == (2,)
    assert pca["fit_split"].item() == "train"
    assert int(pca["n_train"].item()) == 2
