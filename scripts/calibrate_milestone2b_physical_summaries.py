from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from report_milestone2b_physical_summaries import (
        DEFAULT_DENSITY_TABLE,
        DEFAULT_FINAL_METRICS,
        DEFAULT_PCA,
        DEFAULT_PREDICTIONS,
        _apply_central_fraction_logit_shift,
        _calibrated_sample_batches,
        _central_radial_mask,
        _fmt,
        _mass_grids,
        _radial_band_ids,
        _rescale_rows_to_total_mass,
        _row_temperature_scales,
        _scale_m2_harmonic_bands,
        _summary_functions,
        _temperature_scales_by_inclination,
    )
except ModuleNotFoundError:
    from scripts.report_milestone2b_physical_summaries import (
        DEFAULT_DENSITY_TABLE,
        DEFAULT_FINAL_METRICS,
        DEFAULT_PCA,
        DEFAULT_PREDICTIONS,
        _apply_central_fraction_logit_shift,
        _calibrated_sample_batches,
        _central_radial_mask,
        _fmt,
        _mass_grids,
        _radial_band_ids,
        _rescale_rows_to_total_mass,
        _row_temperature_scales,
        _scale_m2_harmonic_bands,
        _summary_functions,
        _temperature_scales_by_inclination,
    )


DEFAULT_OUTPUT_DIR = Path("outputs/tng50_milestone2b/milestone2b_physical_summary_calibration")
TARGET_COVERAGE = 0.68


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--density-table", type=Path, default=DEFAULT_DENSITY_TABLE)
    parser.add_argument("--pca", type=Path, default=DEFAULT_PCA)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--final-evaluation-metrics", type=Path, default=DEFAULT_FINAL_METRICS)
    parser.add_argument("--total-mass-predictions", type=Path, default=None)
    parser.add_argument("--central-fraction-predictions", type=Path, default=None)
    parser.add_argument("--m2-predictions", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--central-radius-kpc", type=float, default=2.0)
    parser.add_argument("--bar-half-angle-deg", type=float, default=30.0)
    parser.add_argument("--sample-batch-size", type=int, default=8)
    return parser.parse_args()


def _scale_candidates() -> np.ndarray:
    return np.concatenate(
        (
            np.linspace(0.25, 2.0, 36),
            np.linspace(2.1, 8.0, 60),
        )
    )


def _scaled_interval(samples: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(samples, axis=1, keepdims=True)
    scaled = center + scale * (samples - center)
    return np.quantile(scaled, 0.16, axis=1), np.quantile(scaled, 0.84, axis=1)


def _coverage_counts(samples: np.ndarray, truth: np.ndarray, *, scale: float) -> tuple[int, int]:
    lower, upper = _scaled_interval(samples, scale)
    inside = (truth >= lower) & (truth <= upper)
    return int(np.sum(inside)), int(inside.size)


def _coverage(samples: np.ndarray, truth: np.ndarray, *, scale: float) -> float:
    covered, total = _coverage_counts(samples, truth, scale=scale)
    return covered / total if total else float("nan")


def _mean_width(samples: np.ndarray, *, scale: float) -> float:
    lower, upper = _scaled_interval(samples, scale)
    return float(np.mean(upper - lower))


def _coverage_with_row_scales(
    samples: np.ndarray,
    truth: np.ndarray,
    row_scales: np.ndarray,
) -> float:
    center = np.mean(samples, axis=1, keepdims=True)
    shape = (row_scales.shape[0], 1) + (1,) * (samples.ndim - 2)
    scaled = center + row_scales.reshape(shape) * (samples - center)
    lower = np.quantile(scaled, 0.16, axis=1)
    upper = np.quantile(scaled, 0.84, axis=1)
    return float(np.mean((truth >= lower) & (truth <= upper)))


def _mean_width_with_row_scales(samples: np.ndarray, row_scales: np.ndarray) -> float:
    center = np.mean(samples, axis=1, keepdims=True)
    shape = (row_scales.shape[0], 1) + (1,) * (samples.ndim - 2)
    scaled = center + row_scales.reshape(shape) * (samples - center)
    lower = np.quantile(scaled, 0.16, axis=1)
    upper = np.quantile(scaled, 0.84, axis=1)
    return float(np.mean(upper - lower))


def _select_scale(samples: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    if not np.any(mask):
        return 1.0, float("nan")
    scored = [
        (float(scale), _coverage(samples[mask], truth[mask], scale=float(scale)))
        for scale in _scale_candidates()
    ]
    return min(scored, key=lambda item: (abs(item[1] - TARGET_COVERAGE), item[0]))


def _select_global_scale(
    sample_summaries: dict[str, np.ndarray],
    truth_summaries: dict[str, np.ndarray],
    val_mask: np.ndarray,
) -> tuple[float, float]:
    if not np.any(val_mask):
        return 1.0, float("nan")
    scored = []
    for scale in _scale_candidates():
        covered = 0
        total = 0
        for name, samples in sample_summaries.items():
            truth = truth_summaries[name]
            item_covered, item_total = _coverage_counts(
                samples[val_mask],
                truth[val_mask],
                scale=float(scale),
            )
            covered += item_covered
            total += item_total
        scored.append((float(scale), covered / total if total else float("nan")))
    return min(scored, key=lambda item: (abs(item[1] - TARGET_COVERAGE), item[0]))


def _mae(candidate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(candidate - truth)))


def _fit_inclination_scales(
    samples: np.ndarray,
    truth: np.ndarray,
    split: np.ndarray,
    inclinations: np.ndarray,
    *,
    fallback_scale: float,
) -> dict[str, dict[str, float | int]]:
    output = {}
    values = sorted(float(x) for x in np.unique(inclinations[(split == "val") | (split == "test")]))
    for value in values:
        key = f"{value:g}"
        val_mask = (split == "val") & np.isclose(inclinations, value)
        test_mask = (split == "test") & np.isclose(inclinations, value)
        if np.any(val_mask):
            scale, val_coverage = _select_scale(samples, truth, val_mask)
        else:
            scale = fallback_scale
            val_coverage = float("nan")
        output[key] = {
            "temperature_scale": scale,
            "val_coverage_68": val_coverage,
            "test_coverage_68": (
                _coverage(samples[test_mask], truth[test_mask], scale=scale)
                if np.any(test_mask)
                else float("nan")
            ),
            "n_val": int(np.sum(val_mask)),
            "n_test": int(np.sum(test_mask)),
        }
    return output


def _row_scales_from_inclination(
    inclinations: np.ndarray,
    scales_by_inclination: dict[str, dict[str, float | int]],
    *,
    fallback_scale: float,
) -> np.ndarray:
    scales = []
    for inclination in inclinations:
        row = scales_by_inclination.get(f"{float(inclination):g}")
        scales.append(float(row["temperature_scale"]) if row is not None else fallback_scale)
    return np.asarray(scales, dtype=np.float32)


def _summary_labels() -> dict[str, str]:
    return {
        "radial_profile": "radial mass profile",
        "vertical_profile": "vertical mass profile",
        "vertical_rms_height_kpc": "vertical RMS height",
        "central_mass_fraction_r_lt_2kpc": "central mass fraction",
        "bar_axis_mass_fraction": "bar-axis mass fraction",
        "bar_frame_m2_profile": "bar-frame m=2 profile",
    }


def _total_mass_targets(
    table: np.lib.npyio.NpzFile,
    sampled_coefficients: np.ndarray,
    total_mass_predictions: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, np.ndarray]:
    baseline_total = table["baseline_grid_mass_msun"].astype(np.float32)
    sampled_log_ratio = total_mass_predictions["sampled_log_ratio"].astype(np.float32)
    mean_log_ratio = total_mass_predictions["mean_log_ratio"].astype(np.float32)
    if sampled_log_ratio.shape[0] != baseline_total.shape[0]:
        raise ValueError(
            "total-mass predictions row count does not match the density table: "
            f"{sampled_log_ratio.shape[0]} vs {baseline_total.shape[0]}"
        )
    if sampled_log_ratio.shape[1] != sampled_coefficients.shape[1]:
        raise ValueError(
            "total-mass predictions sample count does not match the coefficient samples: "
            f"{sampled_log_ratio.shape[1]} vs {sampled_coefficients.shape[1]}"
        )
    sampled_targets = baseline_total[:, None] * np.exp(sampled_log_ratio)
    mean_targets = baseline_total * np.exp(mean_log_ratio)
    return sampled_targets.astype(np.float32), mean_targets.astype(np.float32)


def _central_fraction_shifts(
    table: np.lib.npyio.NpzFile,
    sampled_coefficients: np.ndarray,
    central_fraction_predictions: np.lib.npyio.NpzFile,
    central_radius_kpc: float,
) -> tuple[np.ndarray, np.ndarray]:
    sampled_shift = central_fraction_predictions["sampled_logit_delta"].astype(np.float32)
    mean_shift = central_fraction_predictions["mean_logit_delta"].astype(np.float32)
    head_radius = float(central_fraction_predictions["central_radius_kpc"])
    if not np.isclose(head_radius, central_radius_kpc):
        raise ValueError(
            "central-fraction predictions were trained for radius "
            f"{head_radius} kpc but the evaluation uses {central_radius_kpc} kpc"
        )
    n_rows = int(table["split"].shape[0])
    if sampled_shift.shape[0] != n_rows:
        raise ValueError(
            "central-fraction predictions row count does not match the density table: "
            f"{sampled_shift.shape[0]} vs {n_rows}"
        )
    if sampled_shift.shape[1] != sampled_coefficients.shape[1]:
        raise ValueError(
            "central-fraction predictions sample count does not match the coefficient "
            f"samples: {sampled_shift.shape[1]} vs {sampled_coefficients.shape[1]}"
        )
    return sampled_shift, mean_shift


def _m2_scales(
    table: np.lib.npyio.NpzFile,
    sampled_coefficients: np.ndarray,
    m2_predictions: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sampled_scales = m2_predictions["sampled_log_m2_delta"].astype(np.float32)
    mean_scales = m2_predictions["mean_log_m2_delta"].astype(np.float32)
    n_radial_bins = int(table["r_edges_kpc"].shape[0]) - 1
    head_radial_bins = int(m2_predictions["n_radial_bins"])
    if head_radial_bins != n_radial_bins:
        raise ValueError(
            "m2 predictions were trained for a grid with "
            f"{head_radial_bins} radial bins but the density table has {n_radial_bins}"
        )
    n_rows = int(table["split"].shape[0])
    if sampled_scales.shape[0] != n_rows:
        raise ValueError(
            "m2 predictions row count does not match the density table: "
            f"{sampled_scales.shape[0]} vs {n_rows}"
        )
    if sampled_scales.shape[1] != sampled_coefficients.shape[1]:
        raise ValueError(
            "m2 predictions sample count does not match the coefficient samples: "
            f"{sampled_scales.shape[1]} vs {sampled_coefficients.shape[1]}"
        )
    band_ids = _radial_band_ids(n_radial_bins, sampled_scales.shape[2])
    return sampled_scales, mean_scales, band_ids


def _build_summary_arrays(
    *,
    table: np.lib.npyio.NpzFile,
    pca: np.lib.npyio.NpzFile,
    predictions: np.lib.npyio.NpzFile,
    final_metrics: dict[str, Any],
    central_radius_kpc: float,
    bar_half_angle_deg: float,
    sample_batch_size: int,
    total_mass_predictions: np.lib.npyio.NpzFile | None = None,
    central_fraction_predictions: np.lib.npyio.NpzFile | None = None,
    m2_predictions: np.lib.npyio.NpzFile | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    truth_mass, baseline_mass = _mass_grids(table)
    posterior_mean_mass = predictions["posterior_mean_mass"].astype(np.float32)
    metadata = table["metadata"].astype(np.float32)
    sampled_coefficients = predictions["sampled_coefficients"].astype(np.float32)
    sampled_total_targets = None
    if total_mass_predictions is not None:
        sampled_total_targets, mean_total_targets = _total_mass_targets(
            table,
            sampled_coefficients,
            total_mass_predictions,
        )
        posterior_mean_mass = _rescale_rows_to_total_mass(
            posterior_mean_mass,
            mean_total_targets,
        )
    sampled_central_shift = None
    central_mask = None
    if central_fraction_predictions is not None:
        sampled_central_shift, mean_central_shift = _central_fraction_shifts(
            table,
            sampled_coefficients,
            central_fraction_predictions,
            central_radius_kpc,
        )
        central_mask = _central_radial_mask(table["r_edges_kpc"], central_radius_kpc)
        posterior_mean_mass = _apply_central_fraction_logit_shift(
            posterior_mean_mass,
            mean_central_shift,
            central_mask,
        )
    sampled_m2_scales = None
    m2_band_ids = None
    if m2_predictions is not None:
        sampled_m2_scales, mean_m2_scales, m2_band_ids = _m2_scales(
            table,
            sampled_coefficients,
            m2_predictions,
        )
        posterior_mean_mass = _scale_m2_harmonic_bands(
            posterior_mean_mass,
            mean_m2_scales,
            m2_band_ids,
        )
    row_scales = _row_temperature_scales(
        metadata,
        _temperature_scales_by_inclination(final_metrics),
    )
    summary_functions = _summary_functions(
        r_edges_kpc=table["r_edges_kpc"],
        phi_edges_rad=table["phi_edges_rad"],
        z_edges_kpc=table["z_edges_kpc"],
        central_radius_kpc=central_radius_kpc,
        bar_half_angle_deg=bar_half_angle_deg,
    )
    truth_summaries = {name: fn(truth_mass) for name, fn in summary_functions.items()}
    baseline_summaries = {name: fn(baseline_mass) for name, fn in summary_functions.items()}
    mean_summaries = {name: fn(posterior_mean_mass) for name, fn in summary_functions.items()}
    sample_summaries: dict[str, list[np.ndarray]] = {name: [] for name in summary_functions}
    for sample_mass in _calibrated_sample_batches(
        sampled_coefficients=sampled_coefficients,
        pca=pca,
        baseline_mass=baseline_mass,
        baseline_grid_mass_msun=table["baseline_grid_mass_msun"].astype(np.float32),
        row_scales=row_scales,
        batch_size=sample_batch_size,
        target_total_mass_msun=sampled_total_targets,
        central_logit_shift=sampled_central_shift,
        central_radial_mask=central_mask,
        m2_log_scales=sampled_m2_scales,
        m2_band_ids=m2_band_ids,
    ):
        for name, fn in summary_functions.items():
            sample_summaries[name].append(fn(sample_mass))
    return (
        truth_summaries,
        baseline_summaries,
        mean_summaries,
        {name: np.concatenate(chunks, axis=0) for name, chunks in sample_summaries.items()},
    )


def build_metrics(
    *,
    table: np.lib.npyio.NpzFile,
    pca: np.lib.npyio.NpzFile,
    predictions: np.lib.npyio.NpzFile,
    final_metrics: dict[str, Any],
    central_radius_kpc: float,
    bar_half_angle_deg: float,
    sample_batch_size: int,
    total_mass_predictions: np.lib.npyio.NpzFile | None = None,
    central_fraction_predictions: np.lib.npyio.NpzFile | None = None,
    m2_predictions: np.lib.npyio.NpzFile | None = None,
) -> dict[str, Any]:
    split = table["split"].astype(str)
    metadata = table["metadata"].astype(np.float32)
    val_mask = split == "val"
    test_mask = split == "test"
    if not np.any(val_mask):
        raise ValueError("physical summary calibration requires at least one validation row")
    if not np.any(test_mask):
        raise ValueError("physical summary calibration requires at least one test row")
    truth_summaries, baseline_summaries, mean_summaries, sample_summaries = _build_summary_arrays(
        table=table,
        pca=pca,
        predictions=predictions,
        final_metrics=final_metrics,
        central_radius_kpc=central_radius_kpc,
        bar_half_angle_deg=bar_half_angle_deg,
        sample_batch_size=sample_batch_size,
        total_mass_predictions=total_mass_predictions,
        central_fraction_predictions=central_fraction_predictions,
        m2_predictions=m2_predictions,
    )
    global_scale, global_val_coverage = _select_global_scale(
        sample_summaries,
        truth_summaries,
        val_mask,
    )
    summary_calibration = {}
    for name, samples in sample_summaries.items():
        truth = truth_summaries[name]
        summary_scale, summary_val_coverage = _select_scale(samples, truth, val_mask)
        inclination_scales = _fit_inclination_scales(
            samples,
            truth,
            split,
            metadata[:, 0],
            fallback_scale=summary_scale,
        )
        inclination_row_scales = _row_scales_from_inclination(
            metadata[:, 0],
            inclination_scales,
            fallback_scale=summary_scale,
        )
        baseline_mae = _mae(baseline_summaries[name][test_mask], truth[test_mask])
        mean_mae = _mae(mean_summaries[name][test_mask], truth[test_mask])
        summary_calibration[name] = {
            "baseline_mae": baseline_mae,
            "mdn_posterior_mean_mae": mean_mae,
            "mdn_improvement_over_baseline_fraction": (
                (baseline_mae - mean_mae) / baseline_mae if baseline_mae > 0.0 else 0.0
            ),
            "coefficient_temperature": {
                "temperature_scale": 1.0,
                "test_coverage_68": _coverage(samples[test_mask], truth[test_mask], scale=1.0),
                "test_mean_68_interval_width": _mean_width(samples[test_mask], scale=1.0),
            },
            "physical_global_temperature": {
                "temperature_scale": global_scale,
                "val_coverage_68": global_val_coverage,
                "test_coverage_68": _coverage(
                    samples[test_mask],
                    truth[test_mask],
                    scale=global_scale,
                ),
                "test_mean_68_interval_width": _mean_width(
                    samples[test_mask],
                    scale=global_scale,
                ),
            },
            "physical_summary_temperature": {
                "temperature_scale": summary_scale,
                "val_coverage_68": summary_val_coverage,
                "test_coverage_68": _coverage(
                    samples[test_mask],
                    truth[test_mask],
                    scale=summary_scale,
                ),
                "test_mean_68_interval_width": _mean_width(
                    samples[test_mask],
                    scale=summary_scale,
                ),
            },
            "physical_summary_inclination_temperature": {
                "temperature_scales_by_inclination": inclination_scales,
                "test_coverage_68": _coverage_with_row_scales(
                    samples[test_mask],
                    truth[test_mask],
                    inclination_row_scales[test_mask],
                ),
                "test_mean_68_interval_width": _mean_width_with_row_scales(
                    samples[test_mask],
                    inclination_row_scales[test_mask],
                ),
            },
        }
    return {
        "selected_run_id": final_metrics["selected_model"]["run_id"],
        "target_coverage": TARGET_COVERAGE,
        "total_mass_correction_applied": total_mass_predictions is not None,
        "central_fraction_correction_applied": central_fraction_predictions is not None,
        "m2_correction_applied": m2_predictions is not None,
        "n_val": int(np.sum(val_mask)),
        "n_test": int(np.sum(test_mask)),
        "global_physical_temperature_scale": global_scale,
        "global_physical_val_coverage_68": global_val_coverage,
        "summary_calibration": summary_calibration,
    }


def _write_markdown(path: Path, metrics: dict[str, Any]) -> None:
    labels = _summary_labels()
    lines = [
        "# Milestone 2b Physical Summary Calibration",
        "",
        "Date: 2026-06-09",
        "",
        "## Scope",
        "",
        "This report calibrates the selected MDN posterior in physical-summary "
        "space. It reuses the selected MDN posterior samples, keeps the posterior "
        "mean fixed, and fits temperature scales on validation summaries before "
        "evaluating held-out test coverage. The tested modes are coefficient "
        "temperature, physical global temperature, per-summary temperature, and "
        "per-summary + inclination temperature.",
        "",
        f"- selected run: `{metrics['selected_run_id']}`",
        f"- total-mass correction applied: "
        f"`{'yes' if metrics.get('total_mass_correction_applied') else 'no'}`",
        f"- central-fraction correction applied: "
        f"`{'yes' if metrics.get('central_fraction_correction_applied') else 'no'}`",
        f"- m=2 correction applied: "
        f"`{'yes' if metrics.get('m2_correction_applied') else 'no'}`",
        f"- validation rows: `{metrics['n_val']}`",
        f"- test rows: `{metrics['n_test']}`",
        f"- target coverage: `{metrics['target_coverage']:.2f}`",
        f"- global physical temperature scale: `{metrics['global_physical_temperature_scale']:.3g}`",
        "",
        "## Test Coverage By Calibration Mode",
        "",
        "| Summary | Coeff. temp | Physical global | Per-summary | Per-summary + inclination |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, label in labels.items():
        item = metrics["summary_calibration"][key]
        lines.append(
            "| "
            f"{label} | "
            f"{item['coefficient_temperature']['test_coverage_68']:.3f} | "
            f"{item['physical_global_temperature']['test_coverage_68']:.3f} | "
            f"{item['physical_summary_temperature']['test_coverage_68']:.3f} | "
            f"{item['physical_summary_inclination_temperature']['test_coverage_68']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Per-Summary Temperature Scales",
            "",
            "| Summary | Baseline MAE | MDN mean MAE | Improvement | Summary scale | Summary+inclination width |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, label in labels.items():
        item = metrics["summary_calibration"][key]
        lines.append(
            "| "
            f"{label} | "
            f"{_fmt(item['baseline_mae'])} | "
            f"{_fmt(item['mdn_posterior_mean_mae'])} | "
            f"{100.0 * item['mdn_improvement_over_baseline_fraction']:.2f}% | "
            f"{item['physical_summary_temperature']['temperature_scale']:.3g} | "
            f"{_fmt(item['physical_summary_inclination_temperature']['test_mean_68_interval_width'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Physical-summary calibration tests whether simple post-processing can "
            "repair the undercoverage seen after PCA-coefficient calibration. A "
            "useful calibration should bring test coverage near 0.68 without "
            "requiring very large intervals.",
            "",
            "The result is mixed. A single physical global temperature improves "
            "the undercovered radial, central, and m=2 summaries, but it "
            "overcovers vertical structure and bar-axis mass fraction. Per-summary "
            "temperature calibration brings the bar-frame m=2 profile close to "
            "nominal coverage, while per-summary + inclination calibration brings "
            "central mass fraction and m=2 close to nominal.",
            "",
            "Calibration alone is not enough for a fully calibrated physical-summary "
            "posterior. Radial profile coverage remains low even after physical "
            "temperature scaling, while vertical structure becomes overcovered. "
            "The next modeling step should train or calibrate a lightweight "
            "summary-specific uncertainty model rather than moving to a larger "
            "full-3D generator.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        table = stack.enter_context(np.load(args.density_table))
        pca = stack.enter_context(np.load(args.pca))
        predictions = stack.enter_context(np.load(args.predictions))
        total_mass_predictions = (
            stack.enter_context(np.load(args.total_mass_predictions))
            if args.total_mass_predictions is not None
            else None
        )
        central_fraction_predictions = (
            stack.enter_context(np.load(args.central_fraction_predictions))
            if args.central_fraction_predictions is not None
            else None
        )
        m2_predictions = (
            stack.enter_context(np.load(args.m2_predictions))
            if args.m2_predictions is not None
            else None
        )
        metrics = build_metrics(
            table=table,
            pca=pca,
            predictions=predictions,
            final_metrics=json.loads(args.final_evaluation_metrics.read_text(encoding="utf-8")),
            central_radius_kpc=args.central_radius_kpc,
            bar_half_angle_deg=args.bar_half_angle_deg,
            sample_batch_size=args.sample_batch_size,
            total_mass_predictions=total_mass_predictions,
            central_fraction_predictions=central_fraction_predictions,
            m2_predictions=m2_predictions,
        )
    (args.output_dir / "milestone2b_physical_summary_calibration_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(args.output_dir / "milestone2b_physical_summary_calibration.md", metrics)
    print(f"wrote Milestone 2b physical summary calibration to {args.output_dir}")


if __name__ == "__main__":
    main()
