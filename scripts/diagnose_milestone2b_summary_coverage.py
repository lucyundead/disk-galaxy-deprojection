from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from calibrate_milestone2b_physical_summaries import (
        TARGET_COVERAGE,
        _build_summary_arrays,
        _coverage,
        _coverage_with_row_scales,
        _fit_inclination_scales,
        _row_scales_from_inclination,
        _scaled_interval,
        _select_global_scale,
        _select_scale,
        _summary_labels,
    )
    from report_milestone2b_physical_summaries import (
        DEFAULT_DENSITY_TABLE,
        DEFAULT_FINAL_METRICS,
        DEFAULT_PCA,
        DEFAULT_PREDICTIONS,
        _fmt,
        _mass_grids,
    )
except ModuleNotFoundError:
    from scripts.calibrate_milestone2b_physical_summaries import (
        TARGET_COVERAGE,
        _build_summary_arrays,
        _coverage,
        _coverage_with_row_scales,
        _fit_inclination_scales,
        _row_scales_from_inclination,
        _scaled_interval,
        _select_global_scale,
        _select_scale,
        _summary_labels,
    )
    from scripts.report_milestone2b_physical_summaries import (
        DEFAULT_DENSITY_TABLE,
        DEFAULT_FINAL_METRICS,
        DEFAULT_PCA,
        DEFAULT_PREDICTIONS,
        _fmt,
        _mass_grids,
    )


DEFAULT_OUTPUT_DIR = Path("outputs/tng50_milestone2b/milestone2b_summary_coverage_diagnostics")

DECISION_CONSISTENT = "consistent with calibrated"
DECISION_SPREAD = "spread-miscalibrated (scale fix sufficient)"
DECISION_BIAS = "bias-dominated (structural fix needed)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--density-table", type=Path, default=DEFAULT_DENSITY_TABLE)
    parser.add_argument("--pca", type=Path, default=DEFAULT_PCA)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--final-evaluation-metrics", type=Path, default=DEFAULT_FINAL_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--central-radius-kpc", type=float, default=2.0)
    parser.add_argument("--bar-half-angle-deg", type=float, default=30.0)
    parser.add_argument("--sample-batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260610)
    return parser.parse_args()


def _inside_with_scale(samples: np.ndarray, truth: np.ndarray, scale: float) -> np.ndarray:
    lower, upper = _scaled_interval(samples, scale)
    return (truth >= lower) & (truth <= upper)


def _inside_with_row_scales(
    samples: np.ndarray,
    truth: np.ndarray,
    row_scales: np.ndarray,
) -> np.ndarray:
    center = np.mean(samples, axis=1, keepdims=True)
    shape = (row_scales.shape[0], 1) + (1,) * (samples.ndim - 2)
    scaled = center + row_scales.reshape(shape) * (samples - center)
    lower = np.quantile(scaled, 0.16, axis=1)
    upper = np.quantile(scaled, 0.84, axis=1)
    return (truth >= lower) & (truth <= upper)


def _galaxy_bootstrap_mean(
    values: np.ndarray,
    galaxy_ids: np.ndarray,
    *,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    unique = np.unique(galaxy_ids)
    sums = np.array(
        [np.sum(values[galaxy_ids == galaxy], dtype=np.float64) for galaxy in unique]
    )
    counts = np.array([float(values[galaxy_ids == galaxy].size) for galaxy in unique])
    point = float(np.sum(sums) / np.sum(counts))
    indices = rng.integers(0, unique.size, size=(draws, unique.size))
    resampled = np.sum(sums[indices], axis=1) / np.sum(counts[indices], axis=1)
    return point, float(np.quantile(resampled, 0.025)), float(np.quantile(resampled, 0.975))


def _z_scores(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    center = np.mean(samples, axis=1)
    spread = np.maximum(np.std(samples, axis=1), 1.0e-12)
    return (truth - center) / spread


def _pit_values(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.mean(samples <= np.expand_dims(truth, axis=1), axis=1)


def _decision(ci_low: float, ci_high: float, mean_z: float, std_z: float) -> str:
    if ci_low <= TARGET_COVERAGE <= ci_high:
        return DECISION_CONSISTENT
    if abs(mean_z) >= 0.5 * max(std_z, 1.0e-9):
        return DECISION_BIAS
    return DECISION_SPREAD


def _stratified_stats(
    inside: np.ndarray,
    z_values: np.ndarray,
    labels: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    output = {}
    for value in sorted(float(x) for x in np.unique(labels)):
        mask = np.isclose(labels, value)
        output[f"{value:g}"] = {
            "n_rows": int(np.sum(mask)),
            "coverage_68": float(np.mean(inside[mask])),
            "mean_z": float(np.mean(z_values[mask])),
            "std_z": float(np.std(z_values[mask])),
        }
    return output


def _total_mass_diagnostics(
    *,
    truth_mass: np.ndarray,
    baseline_mass: np.ndarray,
    split: np.ndarray,
    truth_radial: np.ndarray,
    sample_mean_radial: np.ndarray,
    test_mask: np.ndarray,
) -> dict[str, Any]:
    truth_total = np.sum(truth_mass, axis=(1, 2, 3), dtype=np.float64)
    baseline_total = np.sum(baseline_mass, axis=(1, 2, 3), dtype=np.float64)
    fractional_offset = np.divide(
        truth_total - baseline_total,
        truth_total,
        out=np.zeros_like(truth_total),
        where=truth_total > 0.0,
    )
    by_split = {}
    for name in ("train", "val", "test"):
        mask = split == name
        if not np.any(mask):
            continue
        by_split[name] = {
            "n_rows": int(np.sum(mask)),
            "mean_fractional_offset": float(np.mean(fractional_offset[mask])),
            "std_fractional_offset": float(np.std(fractional_offset[mask])),
            "mean_abs_fractional_offset": float(np.mean(np.abs(fractional_offset[mask]))),
        }
    implied_bias = truth_radial[test_mask] * fractional_offset[test_mask, None]
    observed_bias = truth_radial[test_mask] - sample_mean_radial[test_mask]
    observed_total = float(np.sum(np.abs(observed_bias)))
    explained_fraction = (
        float(np.sum(np.abs(implied_bias)) / observed_total) if observed_total > 0.0 else float("nan")
    )
    if np.std(implied_bias) > 0.0 and np.std(observed_bias) > 0.0:
        correlation = float(np.corrcoef(implied_bias.ravel(), observed_bias.ravel())[0, 1])
    else:
        correlation = float("nan")
    return {
        "note": (
            "Posterior samples share the baseline total grid mass by construction, so the "
            "fractional offset between truth and baseline total mass is an uncorrectable "
            "error floor for absolute-mass summaries."
        ),
        "by_split": by_split,
        "radial_profile_bias_explained_fraction": explained_fraction,
        "radial_profile_bias_correlation": correlation,
    }


def build_metrics(
    *,
    table: np.lib.npyio.NpzFile,
    pca: np.lib.npyio.NpzFile,
    predictions: np.lib.npyio.NpzFile,
    final_metrics: dict[str, Any],
    central_radius_kpc: float,
    bar_half_angle_deg: float,
    sample_batch_size: int,
    bootstrap_draws: int,
    seed: int,
) -> dict[str, Any]:
    split = table["split"].astype(str)
    metadata = table["metadata"].astype(np.float32)
    galaxy_ids = table["galaxy_id"].astype(int)
    val_mask = split == "val"
    test_mask = split == "test"
    if not np.any(val_mask):
        raise ValueError("summary coverage diagnostics require at least one validation row")
    if not np.any(test_mask):
        raise ValueError("summary coverage diagnostics require at least one test row")
    truth_summaries, _, _, sample_summaries = _build_summary_arrays(
        table=table,
        pca=pca,
        predictions=predictions,
        final_metrics=final_metrics,
        central_radius_kpc=central_radius_kpc,
        bar_half_angle_deg=bar_half_angle_deg,
        sample_batch_size=sample_batch_size,
    )
    rng = np.random.default_rng(seed)
    test_galaxies = galaxy_ids[test_mask]
    global_scale, _ = _select_global_scale(sample_summaries, truth_summaries, val_mask)
    summary_diagnostics = {}
    decisions = {}
    for name, samples in sample_summaries.items():
        truth = truth_summaries[name]
        summary_scale, _ = _select_scale(samples, truth, val_mask)
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
        mode_inside = {
            "coefficient_temperature": (
                1.0,
                _inside_with_scale(samples[test_mask], truth[test_mask], 1.0),
                _coverage(samples[val_mask], truth[val_mask], scale=1.0),
            ),
            "physical_global_temperature": (
                global_scale,
                _inside_with_scale(samples[test_mask], truth[test_mask], global_scale),
                _coverage(samples[val_mask], truth[val_mask], scale=global_scale),
            ),
            "physical_summary_temperature": (
                summary_scale,
                _inside_with_scale(samples[test_mask], truth[test_mask], summary_scale),
                _coverage(samples[val_mask], truth[val_mask], scale=summary_scale),
            ),
            "physical_summary_inclination_temperature": (
                float("nan"),
                _inside_with_row_scales(
                    samples[test_mask],
                    truth[test_mask],
                    inclination_row_scales[test_mask],
                ),
                _coverage_with_row_scales(
                    samples[val_mask],
                    truth[val_mask],
                    inclination_row_scales[val_mask],
                ),
            ),
        }
        modes = {}
        for mode_name, (scale, inside, val_coverage) in mode_inside.items():
            point, ci_low, ci_high = _galaxy_bootstrap_mean(
                inside.astype(np.float64),
                test_galaxies,
                draws=bootstrap_draws,
                rng=rng,
            )
            modes[mode_name] = {
                "temperature_scale": scale,
                "val_coverage_68": val_coverage,
                "test_coverage_68": point,
                "test_coverage_68_ci95": [ci_low, ci_high],
                "ci_contains_target": bool(ci_low <= TARGET_COVERAGE <= ci_high),
            }
        z_values = _z_scores(samples[test_mask], truth[test_mask])
        pit = _pit_values(samples[test_mask], truth[test_mask])
        mean_z, mean_z_ci_low, mean_z_ci_high = _galaxy_bootstrap_mean(
            z_values,
            test_galaxies,
            draws=bootstrap_draws,
            rng=rng,
        )
        std_z = float(np.std(z_values))
        per_summary_mode = modes["physical_summary_temperature"]
        decisions[name] = _decision(
            per_summary_mode["test_coverage_68_ci95"][0],
            per_summary_mode["test_coverage_68_ci95"][1],
            mean_z,
            std_z,
        )
        summary_inside = mode_inside["physical_summary_temperature"][1]
        summary_diagnostics[name] = {
            "modes": modes,
            "bias_spread": {
                "mean_z": mean_z,
                "mean_z_ci95": [mean_z_ci_low, mean_z_ci_high],
                "std_z": std_z,
                "mean_pit": float(np.mean(pit)),
                "bias_ratio": float(abs(mean_z) / max(std_z, 1.0e-9)),
            },
            "stratified": {
                "inclination_deg": _stratified_stats(
                    summary_inside,
                    z_values,
                    metadata[test_mask, 0],
                ),
                "bar_angle_deg": _stratified_stats(
                    summary_inside,
                    z_values,
                    metadata[test_mask, 2],
                ),
            },
            "decision": decisions[name],
        }
    truth_mass, baseline_mass = _mass_grids(table)
    total_mass = _total_mass_diagnostics(
        truth_mass=truth_mass,
        baseline_mass=baseline_mass,
        split=split,
        truth_radial=truth_summaries["radial_profile"],
        sample_mean_radial=np.mean(sample_summaries["radial_profile"], axis=1),
        test_mask=test_mask,
    )
    return {
        "selected_run_id": final_metrics["selected_model"]["run_id"],
        "target_coverage": TARGET_COVERAGE,
        "bootstrap_draws": int(bootstrap_draws),
        "seed": int(seed),
        "n_val": int(np.sum(val_mask)),
        "n_test": int(np.sum(test_mask)),
        "n_val_galaxies": int(np.unique(galaxy_ids[val_mask]).size),
        "n_test_galaxies": int(np.unique(test_galaxies).size),
        "global_physical_temperature_scale": global_scale,
        "summary_coverage": summary_diagnostics,
        "total_mass_constraint": total_mass,
        "decisions": decisions,
    }


def _write_markdown(path: Path, metrics: dict[str, Any]) -> None:
    labels = _summary_labels()
    lines = [
        "# Milestone 2b Summary Coverage Diagnostics",
        "",
        "Date: 2026-06-10",
        "",
        "## Scope",
        "",
        "This report quantifies how much of the physical-summary miscoverage seen after "
        "temperature calibration is statistically meaningful, given that test coverage is "
        f"estimated from only `{metrics['n_test_galaxies']}` held-out galaxies "
        f"(`{metrics['n_test']}` correlated projection rows). Coverage confidence intervals "
        "use a galaxy-level cluster bootstrap. Bias/spread decomposition uses per-row "
        "z-scores and PIT values of the uncalibrated posterior samples.",
        "",
        f"- selected run: `{metrics['selected_run_id']}`",
        f"- target coverage: `{metrics['target_coverage']:.2f}`",
        f"- bootstrap draws: `{metrics['bootstrap_draws']}` (seed `{metrics['seed']}`)",
        f"- validation galaxies/rows: `{metrics['n_val_galaxies']}` / `{metrics['n_val']}`",
        f"- test galaxies/rows: `{metrics['n_test_galaxies']}` / `{metrics['n_test']}`",
        "",
        "## Test Coverage With Galaxy-Bootstrap 95% CI (Per-Summary Temperature)",
        "",
        "| Summary | Test coverage | 95% CI | CI contains 0.68 |",
        "| --- | ---: | ---: | --- |",
    ]
    for key, label in labels.items():
        mode = metrics["summary_coverage"][key]["modes"]["physical_summary_temperature"]
        ci_low, ci_high = mode["test_coverage_68_ci95"]
        lines.append(
            f"| {label} | {mode['test_coverage_68']:.3f} | "
            f"[{ci_low:.3f}, {ci_high:.3f}] | "
            f"{'yes' if mode['ci_contains_target'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Test Coverage By Calibration Mode (Point Estimates)",
            "",
            "| Summary | Coeff. temp | Physical global | Per-summary | Per-summary + inclination |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, label in labels.items():
        modes = metrics["summary_coverage"][key]["modes"]
        lines.append(
            "| "
            f"{label} | "
            f"{modes['coefficient_temperature']['test_coverage_68']:.3f} | "
            f"{modes['physical_global_temperature']['test_coverage_68']:.3f} | "
            f"{modes['physical_summary_temperature']['test_coverage_68']:.3f} | "
            f"{modes['physical_summary_inclination_temperature']['test_coverage_68']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Bias/Spread Decomposition (Uncalibrated Samples, Test Rows)",
            "",
            "A summary whose miscoverage is driven by interval width has `mean z ~ 0` and "
            "`std z != 1`; temperature scaling can repair it. A summary with `|mean z|` "
            "comparable to `std z` is bias-dominated and cannot be repaired by any "
            "spread-only calibration.",
            "",
            "| Summary | mean z | mean z 95% CI | std z | mean PIT | bias ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, label in labels.items():
        bias = metrics["summary_coverage"][key]["bias_spread"]
        ci_low, ci_high = bias["mean_z_ci95"]
        lines.append(
            "| "
            f"{label} | "
            f"{bias['mean_z']:.3f} | "
            f"[{ci_low:.3f}, {ci_high:.3f}] | "
            f"{bias['std_z']:.3f} | "
            f"{bias['mean_pit']:.3f} | "
            f"{bias['bias_ratio']:.3f} |"
        )
    total_mass = metrics["total_mass_constraint"]
    lines.extend(
        [
            "",
            "## Total-Mass Constraint Contribution",
            "",
            total_mass["note"],
            "",
            "| Split | Rows | Mean offset | Std offset | Mean abs offset |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for split_name, item in total_mass["by_split"].items():
        lines.append(
            "| "
            f"{split_name} | "
            f"{item['n_rows']} | "
            f"{item['mean_fractional_offset']:.4f} | "
            f"{item['std_fractional_offset']:.4f} | "
            f"{item['mean_abs_fractional_offset']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"- fraction of observed radial-profile bias magnitude explained by the "
            f"total-mass constraint: "
            f"`{_fmt(total_mass['radial_profile_bias_explained_fraction'])}`",
            f"- correlation between constraint-implied and observed radial bias: "
            f"`{_fmt(total_mass['radial_profile_bias_correlation'])}`",
            "",
            "## Decision",
            "",
            "Decisions use the per-summary temperature mode: a summary whose bootstrap CI "
            "contains the target coverage is treated as consistent with calibrated; "
            "otherwise the bias ratio separates spread-miscalibrated from bias-dominated.",
            "",
            "| Summary | Decision |",
            "| --- | --- |",
        ]
    )
    for key, label in labels.items():
        lines.append(f"| {label} | {metrics['decisions'][key]} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (
        np.load(args.density_table) as table,
        np.load(args.pca) as pca,
        np.load(args.predictions) as predictions,
    ):
        metrics = build_metrics(
            table=table,
            pca=pca,
            predictions=predictions,
            final_metrics=json.loads(args.final_evaluation_metrics.read_text(encoding="utf-8")),
            central_radius_kpc=args.central_radius_kpc,
            bar_half_angle_deg=args.bar_half_angle_deg,
            sample_batch_size=args.sample_batch_size,
            bootstrap_draws=args.bootstrap_draws,
            seed=args.seed,
        )
    (args.output_dir / "milestone2b_summary_coverage_diagnostics_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(
        args.output_dir / "milestone2b_summary_coverage_diagnostics.md",
        metrics,
    )
    print(f"wrote Milestone 2b summary coverage diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
