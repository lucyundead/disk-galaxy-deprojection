import csv
import json
import subprocess
import sys

import numpy as np

from scripts import sweep_density_residual_pca_mdn


def test_sweep_density_residual_pca_mdn_aggregates_runs_and_temperature(tmp_path, monkeypatch):
    pca_path = tmp_path / "density_residual_pca.npz"
    density_path = tmp_path / "density_residual_table.npz"
    pca_path.write_bytes(b"placeholder")
    density_path.write_bytes(b"placeholder")
    output_dir = tmp_path / "sweep"
    commands = []

    def fake_run(command, check):
        commands.append(command)
        run_dir = output_dir / command[command.index("--output-dir") + 1]
        seed = int(command[command.index("--seed") + 1])
        n_components = int(command[command.index("--n-components") + 1])
        run_dir.mkdir(parents=True)
        (run_dir / "density_residual_pca_mdn_metrics.json").write_text(
            json.dumps(
                {
                    "best_epoch": seed,
                    "best_val_nll": float(n_components),
                    "n_components": n_components,
                    "n_samples": 6,
                    "test_posterior_mean_cell_mass_mae_msun": 1000.0 + seed + n_components,
                    "test_posterior_mean_coeff_rmse": 0.1 * n_components,
                    "val_coeff_coverage_68": 0.5,
                    "test_coeff_coverage_68": 1.0,
                    "n_test": 2,
                }
            ),
            encoding="utf-8",
        )
        true = np.array(
            [
                [0.0, 0.0],
                [0.1, -0.1],
                [0.2, -0.2],
                [0.3, -0.3],
                [0.4, -0.4],
                [0.5, -0.5],
            ],
            dtype=np.float32,
        )
        offsets = np.linspace(-0.5, 0.5, 6, dtype=np.float32)
        sampled = true[:, None, :] + offsets[None, :, None] * 0.5
        split = np.array(["train", "val", "val", "val", "test", "test"])
        metadata = np.array(
            [
                [20.0, 0.0, 0.0],
                [20.0, 0.0, 45.0],
                [40.0, 0.0, 45.0],
                [60.0, 0.0, 90.0],
                [40.0, 0.0, 90.0],
                [60.0, 0.0, 90.0],
            ],
            dtype=np.float32,
        )
        np.savez_compressed(
            run_dir / "density_residual_pca_mdn_predictions.npz",
            sampled_coefficients=sampled,
            true_coefficients=true,
            split=split,
            metadata=metadata,
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sweep_density_residual_pca_mdn.py",
            "--pca",
            str(pca_path),
            "--density-table",
            str(density_path),
            "--output-dir",
            str(output_dir),
            "--seeds",
            "11",
            "12",
            "--n-components",
            "1",
            "3",
            "--epochs",
            "2",
            "--device",
            "cpu",
        ],
    )

    sweep_density_residual_pca_mdn.main()

    rows = list(csv.DictReader((output_dir / "density_residual_pca_mdn_sweep.csv").open()))
    assert len(rows) == 4
    assert len(commands) == 4
    assert rows[0]["run_id"] == "components_1_seed_11"
    assert rows[0]["test_posterior_mean_cell_mass_mae_msun"] == "1012.0"
    assert float(rows[0]["temperature_scale_from_val"]) > 0.0
    assert "temperature_test_coeff_coverage_68" in rows[0]
    assert "inclination_temperature_test_coeff_coverage_68" in rows[0]
    summary = json.loads((output_dir / "density_residual_pca_mdn_sweep.json").read_text())
    assert summary["n_runs"] == 4
    assert summary["best_by_test_mae"]["run_id"] == "components_1_seed_11"
    assert "by_inclination" in summary["runs"][0]
    assert "inclination_temperature_by_inclination" in summary["runs"][0]
    assert sorted(summary["runs"][0]["inclination_temperature_by_inclination"]) == [
        "20",
        "40",
        "60",
    ]
