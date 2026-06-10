import json
import subprocess
import sys

import numpy as np

from scripts.analyze_tng50_density_residuals import cylindrical_bin_volumes_from_edges


def _write_tiny_probabilistic_inputs(tmp_path):
    pca_path = tmp_path / "density_residual_pca.npz"
    density_path = tmp_path / "density_residual_table.npz"
    rng = np.random.default_rng(21)
    n_rows = 8
    images = rng.random((n_rows, 8, 8), dtype=np.float32)
    metadata = np.array(
        [
            [20.0, 0.0, 0.0],
            [20.0, 0.0, 45.0],
            [40.0, 0.0, 0.0],
            [40.0, 0.0, 45.0],
            [60.0, 0.0, 0.0],
            [60.0, 0.0, 45.0],
            [20.0, 0.0, 90.0],
            [60.0, 0.0, 90.0],
        ],
        dtype=np.float32,
    )
    coefficients = np.column_stack(
        (
            images.reshape(n_rows, -1).mean(axis=1),
            metadata[:, 0] / 60.0,
            metadata[:, 2] / 90.0,
        )
    ).astype(np.float32)
    components = rng.normal(size=(3, 16)).astype(np.float32) * 0.01
    mean = np.zeros(16, dtype=np.float32)
    split = np.array(["train", "train", "train", "train", "val", "val", "test", "test"])
    r_edges = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    phi_edges = np.linspace(-np.pi, np.pi, 5, dtype=np.float32)
    z_edges = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
    volumes = cylindrical_bin_volumes_from_edges(r_edges, phi_edges, z_edges)
    truth_mass = np.ones((n_rows, 2, 4, 2), dtype=np.float32) * 10.0
    delta_mass = (coefficients @ components).reshape(n_rows, 2, 4, 2) * 100.0
    baseline_mass = truth_mass - delta_mass
    np.savez_compressed(
        pca_path,
        coefficients=coefficients,
        components=components,
        mean=mean,
        explained_variance_ratio=np.array([0.5, 0.3, 0.1], dtype=np.float32),
        split=split,
        galaxy_id=np.arange(n_rows),
        projection_id=np.zeros(n_rows, dtype=int),
        metadata=metadata,
        r_edges_kpc=r_edges,
        phi_edges_rad=phi_edges,
        z_edges_kpc=z_edges,
        fit_split=np.array("train"),
        n_train=np.array(4, dtype=np.int32),
    )
    np.savez_compressed(
        density_path,
        images=images,
        truth_density=(truth_mass / volumes[None, ...]).astype(np.float32),
        baseline_density=(baseline_mass / volumes[None, ...]).astype(np.float32),
        delta_density=(delta_mass / volumes[None, ...]).astype(np.float32),
        metadata=metadata,
        split=split,
        galaxy_id=np.arange(n_rows),
        projection_id=np.zeros(n_rows, dtype=int),
        r_edges_kpc=r_edges,
        phi_edges_rad=phi_edges,
        z_edges_kpc=z_edges,
        truth_grid_mass_msun=truth_mass.sum(axis=(1, 2, 3)).astype(np.float32),
        baseline_grid_mass_msun=baseline_mass.sum(axis=(1, 2, 3)).astype(np.float32),
    )
    return pca_path, density_path


def test_train_density_residual_pca_mdn_writes_posterior_outputs(tmp_path):
    pca_path, density_path = _write_tiny_probabilistic_inputs(tmp_path)
    output_dir = tmp_path / "pca_mdn"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_density_residual_pca_mdn.py",
            "--pca",
            str(pca_path),
            "--density-table",
            str(density_path),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "3",
            "--hidden-dim",
            "8",
            "--image-feature-size",
            "4",
            "--batch-size",
            "2",
            "--n-components",
            "2",
            "--n-samples",
            "5",
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "wrote probabilistic density residual PCA model" in result.stdout
    assert (output_dir / "density_residual_pca_mdn.pt").exists()
    assert (output_dir / "density_residual_pca_mdn_predictions.npz").exists()
    metrics = json.loads((output_dir / "density_residual_pca_mdn_metrics.json").read_text())
    assert metrics["n_train"] == 4
    assert metrics["n_val"] == 2
    assert metrics["n_test"] == 2
    assert "best_val_nll" in metrics
    assert "test_coeff_coverage_68" in metrics
    assert "test_posterior_mean_cell_mass_mae_msun" in metrics
    predictions = np.load(output_dir / "density_residual_pca_mdn_predictions.npz")
    assert predictions["sampled_coefficients"].shape == (8, 5, 3)
    assert predictions["posterior_mean_delta_mass"].shape == (8, 2, 4, 2)


def test_train_density_residual_pca_mdn_accepts_central_features(tmp_path):
    pca_path, density_path = _write_tiny_probabilistic_inputs(tmp_path)
    output_dir = tmp_path / "pca_mdn_central"

    subprocess.run(
        [
            sys.executable,
            "scripts/train_density_residual_pca_mdn.py",
            "--pca",
            str(pca_path),
            "--density-table",
            str(density_path),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "2",
            "--hidden-dim",
            "8",
            "--image-feature-size",
            "4",
            "--central-pixel-scale-kpc",
            "0.5",
            "--batch-size",
            "2",
            "--n-components",
            "1",
            "--n-samples",
            "3",
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads((output_dir / "density_residual_pca_mdn_metrics.json").read_text())
    assert metrics["central_pixel_scale_kpc"] == 0.5
