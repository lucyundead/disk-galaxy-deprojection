import json
import sys

import numpy as np

from scripts import report_milestone2b_physical_summaries
from scripts.analyze_tng50_density_residuals import cylindrical_bin_volumes_from_edges


def _write_tiny_physical_inputs(tmp_path):
    density_path = tmp_path / "density_table.npz"
    pca_path = tmp_path / "pca.npz"
    predictions_path = tmp_path / "predictions.npz"
    final_metrics_path = tmp_path / "final_metrics.json"

    n_rows = 5
    r_edges = np.array([0.0, 1.0, 3.0], dtype=np.float32)
    phi_edges = np.linspace(-np.pi, np.pi, 5, dtype=np.float32)
    z_edges = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    volumes = cylindrical_bin_volumes_from_edges(r_edges, phi_edges, z_edges)
    truth_mass = np.ones((n_rows, 2, 4, 2), dtype=np.float32) * 10.0
    baseline_mass = truth_mass.copy()
    baseline_mass[:, 0, :, :] *= 0.8
    baseline_mass[:, 1, :, :] *= 1.2
    split = np.array(["train", "val", "test", "test", "test"])
    metadata = np.array(
        [
            [20.0, 0.0, 0.0],
            [20.0, 0.0, 45.0],
            [20.0, 0.0, 0.0],
            [40.0, 0.0, 45.0],
            [60.0, 0.0, 90.0],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        density_path,
        truth_density=(truth_mass / volumes[None, ...]).astype(np.float32),
        baseline_density=(baseline_mass / volumes[None, ...]).astype(np.float32),
        split=split,
        metadata=metadata,
        galaxy_id=np.arange(n_rows),
        projection_id=np.arange(n_rows),
        r_edges_kpc=r_edges,
        phi_edges_rad=phi_edges,
        z_edges_kpc=z_edges,
        baseline_grid_mass_msun=baseline_mass.sum(axis=(1, 2, 3)).astype(np.float32),
    )
    components = np.zeros((2, 16), dtype=np.float32)
    components[0, :8] = 0.1
    components[1, 8:] = -0.1
    mean = np.zeros(16, dtype=np.float32)
    np.savez_compressed(
        pca_path,
        components=components,
        mean=mean,
    )
    true_coefficients = np.zeros((n_rows, 2), dtype=np.float32)
    sampled_coefficients = np.zeros((n_rows, 4, 2), dtype=np.float32)
    sampled_coefficients[:, :, 0] = np.array([-0.4, -0.1, 0.1, 0.4], dtype=np.float32)
    sampled_coefficients[:, :, 1] = np.array([-0.3, -0.1, 0.1, 0.3], dtype=np.float32)
    posterior_mean_mass = baseline_mass + (truth_mass - baseline_mass) * 0.5
    np.savez_compressed(
        predictions_path,
        sampled_coefficients=sampled_coefficients,
        posterior_mean_mass=posterior_mean_mass.astype(np.float32),
        true_coefficients=true_coefficients,
        split=split,
        metadata=metadata,
    )
    final_metrics_path.write_text(
        json.dumps(
            {
                "selected_model": {
                    "run_id": "components_1_seed_test",
                    "inclination_temperature_by_inclination": {
                        "20": {"temperature_scale": 0.8},
                        "40": {"temperature_scale": 0.9},
                        "60": {"temperature_scale": 1.0},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return density_path, pca_path, predictions_path, final_metrics_path


def test_report_milestone2b_physical_summaries_writes_metrics_and_markdown(
    tmp_path,
    monkeypatch,
):
    density_path, pca_path, predictions_path, final_metrics_path = _write_tiny_physical_inputs(
        tmp_path,
    )
    output_dir = tmp_path / "physical"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_milestone2b_physical_summaries.py",
            "--density-table",
            str(density_path),
            "--pca",
            str(pca_path),
            "--predictions",
            str(predictions_path),
            "--final-evaluation-metrics",
            str(final_metrics_path),
            "--output-dir",
            str(output_dir),
            "--sample-batch-size",
            "2",
        ],
    )

    report_milestone2b_physical_summaries.main()

    metrics = json.loads((output_dir / "milestone2b_physical_summary_metrics.json").read_text())
    assert metrics["selected_run_id"] == "components_1_seed_test"
    assert metrics["n_test"] == 3
    for key in (
        "radial_profile",
        "vertical_profile",
        "vertical_rms_height_kpc",
        "central_mass_fraction_r_lt_2kpc",
        "bar_axis_mass_fraction",
    ):
        assert key in metrics["summary_metrics"]
        assert "mdn_coverage_68" in metrics["summary_metrics"][key]
    markdown = (output_dir / "milestone2b_physical_summary_evaluation.md").read_text()
    assert "Physical Summary Posterior Evaluation" in markdown
    assert "bar-axis mass fraction" in markdown
