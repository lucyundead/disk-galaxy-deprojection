import json
import subprocess
import sys

import numpy as np

from tests.test_train_density_residual_pca_mdn import _write_tiny_probabilistic_inputs


def test_train_central_fraction_correction_writes_predictions_and_metrics(tmp_path):
    _, density_path = _write_tiny_probabilistic_inputs(tmp_path)
    predictions_path = tmp_path / "mdn_predictions.npz"
    rng = np.random.default_rng(11)
    np.savez_compressed(
        predictions_path,
        posterior_mean_mass=(9.0 + rng.random((8, 2, 4, 2))).astype(np.float32),
    )
    output_dir = tmp_path / "central_fraction_correction"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_central_fraction_correction.py",
            "--density-table",
            str(density_path),
            "--predictions",
            str(predictions_path),
            "--output-dir",
            str(output_dir),
            "--central-radius-kpc",
            "1.5",
            "--epochs",
            "3",
            "--hidden-dim",
            "8",
            "--image-feature-size",
            "4",
            "--central-pixel-scale-kpc",
            "0.5",
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

    assert "wrote central fraction correction model" in result.stdout
    assert (output_dir / "central_fraction_correction.pt").exists()
    metrics = json.loads((output_dir / "central_fraction_correction_metrics.json").read_text())
    assert metrics["central_radius_kpc"] == 1.5
    assert metrics["train"]["n_rows"] == 4
    assert "logit_delta_mae" in metrics["test"]
    assert "uncorrected_fraction_bias" in metrics["test"]
    assert "corrected_fraction_bias" in metrics["test"]
    predictions = np.load(output_dir / "central_fraction_correction_predictions.npz")
    assert predictions["sampled_logit_delta"].shape == (8, 5)
    assert predictions["mean_logit_delta"].shape == (8,)
    assert float(predictions["central_radius_kpc"]) == 1.5
