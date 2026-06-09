import json
import sys

from scripts import report_milestone2b_final_evaluation


def test_report_milestone2b_final_evaluation_writes_selected_default(tmp_path, monkeypatch):
    pca_metrics_path = tmp_path / "density_residual_pca_report_metrics.json"
    sweep_path = tmp_path / "density_residual_pca_mdn_sweep.json"
    output_dir = tmp_path / "final"

    pca_metrics_path.write_text(
        json.dumps(
            {
                "n_test": 12,
                "n_test_galaxies": 4,
                "pca_n_components": 8,
                "pca_cumulative_explained_variance": 0.9,
                "test_baseline_cell_mass_mae_msun": 100.0,
                "test_baseline_total_mass_fractional_mae": 0.05,
                "test_mean_train_residual_cell_mass_mae_msun": 70.0,
                "test_mean_train_cell_mass_mae_improvement_fraction": 0.3,
                "test_mean_train_residual_total_mass_fractional_mae": 0.05,
                "test_model_cell_mass_mae_msun": 40.0,
                "test_model_cell_mass_mae_improvement_fraction": 0.6,
                "test_model_total_mass_fractional_mae": 0.05,
            }
        ),
        encoding="utf-8",
    )
    sweep_path.write_text(
        json.dumps(
            {
                "n_runs": 3,
                "target_coverage": 0.68,
                "test_coeff_coverage_68_mean": 0.75,
                "temperature_test_coeff_coverage_68_mean": 0.67,
                "inclination_temperature_test_coeff_coverage_68_mean": 0.68,
                "runs": [
                    {
                        "run_id": "components_1_seed_1",
                        "seed": 1,
                        "n_components": 1,
                        "test_posterior_mean_cell_mass_mae_msun": 38.0,
                        "test_posterior_mean_total_mass_fractional_mae": 0.05,
                        "test_coeff_coverage_68": 0.72,
                        "temperature_test_coeff_coverage_68": 0.66,
                        "inclination_temperature_test_coeff_coverage_68": 0.69,
                        "inclination_temperature_by_inclination": {
                            "20": {"test_coeff_coverage_68": 0.68, "temperature_scale": 0.8},
                            "40": {"test_coeff_coverage_68": 0.69, "temperature_scale": 0.8},
                            "60": {"test_coeff_coverage_68": 0.70, "temperature_scale": 0.9},
                        },
                    },
                    {
                        "run_id": "components_1_seed_2",
                        "seed": 2,
                        "n_components": 1,
                        "test_posterior_mean_cell_mass_mae_msun": 42.0,
                        "test_posterior_mean_total_mass_fractional_mae": 0.05,
                        "test_coeff_coverage_68": 0.74,
                        "temperature_test_coeff_coverage_68": 0.67,
                        "inclination_temperature_test_coeff_coverage_68": 0.68,
                        "inclination_temperature_by_inclination": {},
                    },
                    {
                        "run_id": "components_3_seed_1",
                        "seed": 1,
                        "n_components": 3,
                        "test_posterior_mean_cell_mass_mae_msun": 37.0,
                        "test_posterior_mean_total_mass_fractional_mae": 0.05,
                        "test_coeff_coverage_68": 0.76,
                        "temperature_test_coeff_coverage_68": 0.68,
                        "inclination_temperature_test_coeff_coverage_68": 0.68,
                        "inclination_temperature_by_inclination": {},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_milestone2b_final_evaluation.py",
            "--pca-report-metrics",
            str(pca_metrics_path),
            "--mdn-sweep",
            str(sweep_path),
            "--output-dir",
            str(output_dir),
            "--selected-n-components",
            "1",
        ],
    )

    report_milestone2b_final_evaluation.main()

    metrics = json.loads((output_dir / "milestone2b_final_evaluation_metrics.json").read_text())
    assert metrics["selected_model"]["run_id"] == "components_1_seed_1"
    assert metrics["selected_model"]["n_components"] == 1
    assert metrics["selected_model"]["improvement_over_baseline_fraction"] == 0.62
    assert len(metrics["comparison_rows"]) == 4

    markdown = (output_dir / "milestone2b_final_evaluation.md").read_text()
    assert "1-component MDN + inclination-aware temperature diagnostics" in markdown
    assert "| MDN posterior mean | 3.800e1 | 62.00% |" in markdown
    assert "--n-components 1" in markdown
