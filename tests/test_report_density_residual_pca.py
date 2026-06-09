import json
import subprocess
import sys

import numpy as np

from scripts.analyze_tng50_density_residuals import cylindrical_bin_volumes_from_edges


def _write_tiny_report_inputs(tmp_path):
    table_path = tmp_path / "density_residual_table.npz"
    pca_path = tmp_path / "density_residual_pca.npz"
    predictions_path = tmp_path / "density_residual_pca_predictions.npz"
    r_edges = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    phi_edges = np.linspace(-np.pi, np.pi, 5, dtype=np.float32)
    z_edges = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    volumes = cylindrical_bin_volumes_from_edges(r_edges, phi_edges, z_edges)
    base = np.arange(16, dtype=np.float32).reshape(2, 4, 2) + 20.0
    truth_mass = np.stack(
        [
            base,
            base * 1.1,
            base * 0.9,
            base * 1.2,
            base * 0.8,
            base * 1.3,
        ]
    )
    residual = np.stack(
        [
            np.ones_like(base) * 1.0,
            np.ones_like(base) * 2.0,
            np.ones_like(base) * 0.5,
            np.ones_like(base) * 0.25,
            np.ones_like(base) * 1.5,
            np.ones_like(base) * 0.75,
        ]
    )
    baseline_mass = truth_mass - residual
    predicted_delta = residual * np.array([0.8, 1.0, 0.7, 0.5, 0.9, 0.6])[:, None, None, None]
    split = np.array(["train", "train", "train", "val", "test", "test"])
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
    galaxy_id = np.arange(6, dtype=int)
    projection_id = np.arange(6, dtype=int)
    np.savez_compressed(
        table_path,
        images=np.zeros((6, 8, 8), dtype=np.float32),
        truth_density=(truth_mass / volumes[None, ...]).astype(np.float32),
        baseline_density=(baseline_mass / volumes[None, ...]).astype(np.float32),
        delta_density=(residual / volumes[None, ...]).astype(np.float32),
        metadata=metadata,
        split=split,
        galaxy_id=galaxy_id,
        projection_id=projection_id,
        r_edges_kpc=r_edges,
        phi_edges_rad=phi_edges,
        z_edges_kpc=z_edges,
        truth_grid_mass_msun=truth_mass.sum(axis=(1, 2, 3)).astype(np.float32),
        baseline_grid_mass_msun=baseline_mass.sum(axis=(1, 2, 3)).astype(np.float32),
    )
    np.savez_compressed(
        pca_path,
        explained_variance_ratio=np.array([0.6, 0.2], dtype=np.float32),
        split=split,
        galaxy_id=galaxy_id,
        projection_id=projection_id,
        metadata=metadata,
    )
    np.savez_compressed(
        predictions_path,
        predicted_delta_mass=predicted_delta.astype(np.float32),
        predicted_coefficients=np.zeros((6, 2), dtype=np.float32),
        true_coefficients=np.zeros((6, 2), dtype=np.float32),
        split=split,
        galaxy_id=galaxy_id,
        projection_id=projection_id,
        metadata=metadata,
    )
    return table_path, pca_path, predictions_path


def test_report_density_residual_pca_writes_metrics_markdown_and_png(tmp_path):
    table_path, pca_path, predictions_path = _write_tiny_report_inputs(tmp_path)
    output_dir = tmp_path / "report"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/report_density_residual_pca.py",
            "--density-table",
            str(table_path),
            "--pca",
            str(pca_path),
            "--predictions",
            str(predictions_path),
            "--output-dir",
            str(output_dir),
            "--max-examples",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "wrote density residual PCA report" in result.stdout
    metrics = json.loads((output_dir / "density_residual_pca_report_metrics.json").read_text())
    assert metrics["n_test"] == 2
    assert metrics["test_model_cell_mass_mae_msun"] < metrics["test_baseline_cell_mass_mae_msun"]
    assert "test_mean_train_residual_cell_mass_mae_msun" in metrics
    assert "test_radial_profile" in metrics["profile_metrics"]
    assert "test_vertical_profile" in metrics["profile_metrics"]
    assert "test_m2_profile" in metrics["profile_metrics"]
    assert "inclination_deg" in metrics["stratified_metrics"]
    assert "bar_angle_deg" in metrics["stratified_metrics"]
    markdown = (output_dir / "density_residual_pca_report.md").read_text()
    assert "How To Read The Figures" in markdown
    assert "truth, baseline, PCA corrected, and mean train residual" in markdown
    assert list((output_dir / "figures").glob("test_example_*.png"))
    payload = np.load(output_dir / "density_residual_pca_figure_payload.npz")
    assert payload["truth_mass"].shape == (1, 2, 4, 2)
    assert payload["model_mass"].shape == (1, 2, 4, 2)
