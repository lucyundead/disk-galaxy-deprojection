import json
import subprocess
import sys

import numpy as np

from tests.test_train_density_residual_pca_mdn import _write_tiny_probabilistic_inputs


def test_train_total_mass_correction_writes_predictions_and_metrics(tmp_path):
    _, density_path = _write_tiny_probabilistic_inputs(tmp_path)
    output_dir = tmp_path / "total_mass_correction"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_total_mass_correction.py",
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
            "--n-samples",
            "5",
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "wrote total mass correction model" in result.stdout
    assert (output_dir / "total_mass_correction.pt").exists()
    metrics = json.loads((output_dir / "total_mass_correction_metrics.json").read_text())
    assert metrics["train"]["n_rows"] == 4
    assert metrics["val"]["n_rows"] == 2
    assert metrics["test"]["n_rows"] == 2
    assert "log_ratio_mae" in metrics["test"]
    assert "baseline_total_mass_fractional_mae" in metrics["test"]
    assert "corrected_total_mass_fractional_mae" in metrics["test"]
    predictions = np.load(output_dir / "total_mass_correction_predictions.npz")
    assert predictions["sampled_log_ratio"].shape == (8, 5)
    assert predictions["mean_log_ratio"].shape == (8,)
    assert predictions["true_log_ratio"].shape == (8,)
