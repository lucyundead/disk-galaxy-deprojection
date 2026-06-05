import numpy as np

from scripts.diagnose_summary_residual import (
    build_grouped_table,
    build_per_summary_table,
)


def _predictions():
    truth = np.array(
        [
            [8.0, 2.0],
            [25.0, 5.0],
            [16.0, 4.0],
            [32.0, 8.0],
        ]
    )
    baseline = np.array(
        [
            [10.0, 1.0],
            [20.0, 7.0],
            [20.0, 3.0],
            [24.0, 10.0],
        ]
    )
    corrected = np.array(
        [
            [9.0, 2.5],
            [23.0, 4.5],
            [15.0, 4.5],
            [30.0, 7.0],
        ]
    )
    return {
        "summary_names": np.array(["enclosed_mass_5_kpc", "height_mad"]),
        "truth": truth,
        "baseline": baseline,
        "corrected_mean": corrected,
        "lower": corrected - 1.5,
        "upper": corrected + 1.5,
        "metadata": np.array(
            [
                [20.0, 0.0, 0.0],
                [20.0, 0.0, 45.0],
                [40.0, 0.0, 0.0],
                [40.0, 0.0, 45.0],
            ]
        ),
    }


def test_per_summary_table_reports_improvements_and_coverage():
    table = build_per_summary_table(_predictions())

    assert table["summary_name"].tolist() == ["enclosed_mass_5_kpc", "height_mad"]
    assert set(
        [
            "baseline_mae",
            "corrected_mae",
            "baseline_mean_relative_abs_error",
            "corrected_mean_relative_abs_error",
            "coverage_68",
        ]
    ).issubset(table.columns)
    assert np.all(table["corrected_mae"] < table["baseline_mae"])
    assert np.all((0.0 <= table["coverage_68"]) & (table["coverage_68"] <= 1.0))


def test_grouped_table_reports_inclination_and_bar_angle_slices():
    grouped = build_grouped_table(_predictions())

    assert set(grouped["group_type"]) == {
        "inclination",
        "bar_angle",
        "inclination_bar_angle",
    }
    assert grouped[grouped["group_type"] == "inclination"]["inclination_deg"].tolist() == [
        20.0,
        40.0,
    ]
    assert grouped[grouped["group_type"] == "bar_angle"]["bar_angle_deg"].tolist() == [
        0.0,
        45.0,
    ]
    assert np.all(grouped["corrected_mae"] < grouped["baseline_mae"])
