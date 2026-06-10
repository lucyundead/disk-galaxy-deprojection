import json
import subprocess
import sys

import numpy as np

from scripts.analyze_tng50_density_residuals import cylindrical_bin_volumes_from_edges
from scripts.train_density_residual_pca import (
    average_pool_images,
    make_central_image_features,
    make_density_residual_features,
)


def _write_tiny_training_inputs(tmp_path):
    pca_path = tmp_path / "density_residual_pca.npz"
    density_path = tmp_path / "density_residual_table.npz"
    rng = np.random.default_rng(10)
    images = rng.random((6, 8, 8), dtype=np.float32)
    metadata = np.array(
        [
            [20.0, 0.0, 0.0],
            [20.0, 0.0, 45.0],
            [40.0, 0.0, 0.0],
            [40.0, 0.0, 45.0],
            [60.0, 0.0, 0.0],
            [60.0, 0.0, 45.0],
        ],
        dtype=np.float32,
    )
    coefficients = np.column_stack(
        (
            images.reshape(6, -1).mean(axis=1),
            metadata[:, 0] / 60.0,
            metadata[:, 2] / 45.0,
        )
    ).astype(np.float32)
    components = rng.normal(size=(3, 16)).astype(np.float32) * 0.01
    mean = np.zeros(16, dtype=np.float32)
    split = np.array(["train", "train", "train", "val", "test", "test"])
    r_edges = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    phi_edges = np.linspace(-np.pi, np.pi, 5, dtype=np.float32)
    z_edges = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
    volumes = cylindrical_bin_volumes_from_edges(r_edges, phi_edges, z_edges)
    truth_mass = np.ones((6, 2, 4, 2), dtype=np.float32) * 10.0
    baseline_mass = truth_mass - (coefficients @ components).reshape(6, 2, 4, 2) * 100.0
    np.savez_compressed(
        pca_path,
        coefficients=coefficients,
        components=components,
        mean=mean,
        explained_variance_ratio=np.array([0.5, 0.3, 0.1], dtype=np.float32),
        split=split,
        galaxy_id=np.arange(6),
        projection_id=np.zeros(6, dtype=int),
        metadata=metadata,
        r_edges_kpc=r_edges,
        phi_edges_rad=phi_edges,
        z_edges_kpc=z_edges,
        fit_split=np.array("train"),
        n_train=np.array(3, dtype=np.int32),
    )
    np.savez_compressed(
        density_path,
        images=images,
        truth_density=(truth_mass / volumes[None, ...]).astype(np.float32),
        baseline_density=(baseline_mass / volumes[None, ...]).astype(np.float32),
        delta_density=((truth_mass - baseline_mass) / volumes[None, ...]).astype(np.float32),
        metadata=metadata,
        split=split,
        galaxy_id=np.arange(6),
        projection_id=np.zeros(6, dtype=int),
        r_edges_kpc=r_edges,
        phi_edges_rad=phi_edges,
        z_edges_kpc=z_edges,
        truth_grid_mass_msun=truth_mass.sum(axis=(1, 2, 3)).astype(np.float32),
        baseline_grid_mass_msun=baseline_mass.sum(axis=(1, 2, 3)).astype(np.float32),
    )
    return pca_path, density_path


def test_feature_builder_pools_images_and_adds_mass_features():
    images = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    metadata = np.ones((2, 3), dtype=np.float32)
    baseline_mass = np.array([10.0, 20.0], dtype=np.float32)

    pooled = average_pool_images(images, output_size=2)
    features = make_density_residual_features(
        images,
        metadata,
        baseline_grid_mass_msun=baseline_mass,
        image_feature_size=2,
    )

    assert pooled.shape == (2, 2, 2)
    assert features.shape == (2, 2 * 2 + 3 + 2)
    assert np.all(np.isfinite(features))


def test_central_image_features_track_concentration_and_inclination():
    size = 32
    pixel_scale = 0.5
    coords = (np.arange(size, dtype=np.float32) - 0.5 * (size - 1)) * pixel_scale
    x_grid = coords[None, :]
    y_grid = coords[:, None]
    sigma = 1.0
    face_on = np.exp(-(x_grid**2 + y_grid**2) / (2.0 * sigma**2)).astype(np.float32)
    inclined = np.exp(-(x_grid**2 + (y_grid / 0.5) ** 2) / (2.0 * sigma**2)).astype(np.float32)
    images = np.stack((face_on, inclined))
    inclinations = np.array([0.0, 60.0], dtype=np.float32)

    features = make_central_image_features(
        images,
        inclinations,
        pixel_scale_kpc=pixel_scale,
    )

    assert features.shape == (2, 5)
    assert np.all(np.isfinite(features))
    log_fractions = features[:, :4]
    assert np.all(np.diff(log_fractions, axis=1) >= 0.0)
    shape_ratio = features[:, 4]
    assert abs(shape_ratio[0] - 1.0) < 0.1
    assert abs(shape_ratio[1] - 0.5) < 0.1
    fractions = 10.0 ** log_fractions
    assert abs(fractions[1, 1] - fractions[0, 1]) < 0.05


def test_feature_builder_appends_central_features():
    images = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    metadata = np.ones((2, 3), dtype=np.float32)
    baseline_mass = np.array([10.0, 20.0], dtype=np.float32)

    base = make_density_residual_features(
        images,
        metadata,
        baseline_grid_mass_msun=baseline_mass,
        image_feature_size=2,
    )
    with_central = make_density_residual_features(
        images,
        metadata,
        baseline_grid_mass_msun=baseline_mass,
        image_feature_size=2,
        central_pixel_scale_kpc=0.5,
    )

    assert with_central.shape == (2, base.shape[1] + 5)
    assert np.all(np.isfinite(with_central))


def test_train_density_residual_pca_script_writes_model_predictions_and_metrics(tmp_path):
    pca_path, density_path = _write_tiny_training_inputs(tmp_path)
    output_dir = tmp_path / "model"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_density_residual_pca.py",
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
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "wrote density residual PCA model" in result.stdout
    assert (output_dir / "density_residual_pca_mlp.pt").exists()
    assert (output_dir / "density_residual_pca_predictions.npz").exists()
    metrics = json.loads((output_dir / "density_residual_pca_metrics.json").read_text())
    assert metrics["n_train"] == 3
    assert metrics["n_val"] == 1
    assert metrics["n_test"] == 2
    assert "best_epoch" in metrics
    assert "best_val_mse" in metrics
    assert metrics["preserve_baseline_total_mass"] is True
    assert "test_model_coeff_rmse" in metrics
    assert "test_baseline_cell_mass_mae_msun" in metrics
    assert "test_corrected_cell_mass_mae_msun" in metrics
