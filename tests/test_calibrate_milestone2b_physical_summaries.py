import json
import sys

from scripts import calibrate_milestone2b_physical_summaries
from tests.test_report_milestone2b_physical_summaries import _write_tiny_physical_inputs


def test_calibrate_milestone2b_physical_summaries_compares_temperature_modes(
    tmp_path,
    monkeypatch,
):
    density_path, pca_path, predictions_path, final_metrics_path = _write_tiny_physical_inputs(
        tmp_path,
    )
    output_dir = tmp_path / "physical_calibration"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate_milestone2b_physical_summaries.py",
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

    calibrate_milestone2b_physical_summaries.main()

    metrics = json.loads(
        (output_dir / "milestone2b_physical_summary_calibration_metrics.json").read_text()
    )
    assert metrics["selected_run_id"] == "components_1_seed_test"
    assert metrics["target_coverage"] == 0.68
    for summary_name in (
        "radial_profile",
        "vertical_profile",
        "central_mass_fraction_r_lt_2kpc",
    ):
        summary = metrics["summary_calibration"][summary_name]
        assert "coefficient_temperature" in summary
        assert "physical_global_temperature" in summary
        assert "physical_summary_temperature" in summary
        assert "physical_summary_inclination_temperature" in summary
        assert summary["physical_summary_temperature"]["temperature_scale"] > 0.0
    markdown = (
        output_dir / "milestone2b_physical_summary_calibration.md"
    ).read_text()
    assert "Physical Summary Calibration" in markdown
    assert "per-summary + inclination" in markdown
