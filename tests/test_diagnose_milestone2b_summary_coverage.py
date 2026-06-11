import json
import sys

from scripts import diagnose_milestone2b_summary_coverage
from tests.test_report_milestone2b_physical_summaries import (
    _write_tiny_central_fraction_predictions,
    _write_tiny_m2_predictions,
    _write_tiny_physical_inputs,
    _write_tiny_total_mass_predictions,
)


def test_diagnose_milestone2b_summary_coverage_writes_metrics_and_markdown(
    tmp_path,
    monkeypatch,
):
    density_path, pca_path, predictions_path, final_metrics_path = _write_tiny_physical_inputs(
        tmp_path,
    )
    output_dir = tmp_path / "coverage_diagnostics"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diagnose_milestone2b_summary_coverage.py",
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
            "--bootstrap-draws",
            "50",
            "--seed",
            "123",
        ],
    )

    diagnose_milestone2b_summary_coverage.main()

    metrics = json.loads(
        (output_dir / "milestone2b_summary_coverage_diagnostics_metrics.json").read_text()
    )
    assert metrics["selected_run_id"] == "components_1_seed_test"
    assert metrics["target_coverage"] == 0.68
    assert metrics["bootstrap_draws"] == 50
    assert metrics["n_test_galaxies"] == 3
    for summary_name in (
        "radial_profile",
        "vertical_profile",
        "central_mass_fraction_r_lt_2kpc",
    ):
        summary = metrics["summary_coverage"][summary_name]
        for mode_name in (
            "coefficient_temperature",
            "physical_global_temperature",
            "physical_summary_temperature",
            "physical_summary_inclination_temperature",
        ):
            mode = summary["modes"][mode_name]
            ci_low, ci_high = mode["test_coverage_68_ci95"]
            assert 0.0 <= ci_low <= ci_high <= 1.0
            assert 0.0 <= mode["test_coverage_68"] <= 1.0
        bias = summary["bias_spread"]
        assert "mean_z" in bias
        assert len(bias["mean_z_ci95"]) == 2
        assert bias["std_z"] >= 0.0
        assert 0.0 <= bias["mean_pit"] <= 1.0
        assert summary["decision"] in (
            diagnose_milestone2b_summary_coverage.DECISION_CONSISTENT,
            diagnose_milestone2b_summary_coverage.DECISION_SPREAD,
            diagnose_milestone2b_summary_coverage.DECISION_BIAS,
        )
        assert "inclination_deg" in summary["stratified"]
        assert "bar_angle_deg" in summary["stratified"]
    total_mass = metrics["total_mass_constraint"]
    assert "test" in total_mass["by_split"]
    assert "radial_profile_bias_explained_fraction" in total_mass
    assert len(metrics["decisions"]) == len(metrics["summary_coverage"])
    markdown = (
        output_dir / "milestone2b_summary_coverage_diagnostics.md"
    ).read_text()
    assert "Summary Coverage Diagnostics" in markdown
    assert "Galaxy-Bootstrap 95% CI" in markdown
    assert "Total-Mass Constraint Contribution" in markdown
    assert "## Decision" in markdown


def test_diagnose_milestone2b_summary_coverage_applies_total_mass_correction(
    tmp_path,
    monkeypatch,
):
    density_path, pca_path, predictions_path, final_metrics_path = _write_tiny_physical_inputs(
        tmp_path,
    )
    total_mass_path = _write_tiny_total_mass_predictions(tmp_path)
    central_fraction_path = _write_tiny_central_fraction_predictions(tmp_path)
    m2_path = _write_tiny_m2_predictions(tmp_path)
    output_dir = tmp_path / "coverage_diagnostics_corrected"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diagnose_milestone2b_summary_coverage.py",
            "--density-table",
            str(density_path),
            "--pca",
            str(pca_path),
            "--predictions",
            str(predictions_path),
            "--final-evaluation-metrics",
            str(final_metrics_path),
            "--total-mass-predictions",
            str(total_mass_path),
            "--central-fraction-predictions",
            str(central_fraction_path),
            "--m2-predictions",
            str(m2_path),
            "--output-dir",
            str(output_dir),
            "--sample-batch-size",
            "2",
            "--bootstrap-draws",
            "50",
            "--seed",
            "123",
        ],
    )

    diagnose_milestone2b_summary_coverage.main()

    metrics = json.loads(
        (output_dir / "milestone2b_summary_coverage_diagnostics_metrics.json").read_text()
    )
    assert metrics["total_mass_correction_applied"] is True
    assert metrics["central_fraction_correction_applied"] is True
    assert metrics["m2_correction_applied"] is True
    assert "radial_profile" in metrics["summary_coverage"]
    markdown = (
        output_dir / "milestone2b_summary_coverage_diagnostics.md"
    ).read_text()
    assert "total-mass correction applied: `yes`" in markdown
    assert "central-fraction correction applied: `yes`" in markdown
    assert "m=2 correction applied: `yes`" in markdown
