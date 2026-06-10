from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DENSITY_TABLE = Path("outputs/tng50_milestone2b/density_residual_table.npz")
DEFAULT_PCA = Path("outputs/tng50_milestone2b/density_residual_diagnostics/density_residual_pca.npz")
DEFAULT_PREDICTIONS = Path(
    "outputs/tng50_milestone2b/density_residual_pca_mdn_sweep/"
    "components_1_seed_20260610/density_residual_pca_mdn_predictions.npz"
)
DEFAULT_FINAL_METRICS = Path(
    "outputs/tng50_milestone2b/milestone2b_final_evaluation/"
    "milestone2b_final_evaluation_metrics.json"
)
DEFAULT_OUTPUT_DIR = Path("outputs/tng50_milestone2b/milestone2b_physical_summary_evaluation")


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
    return parser.parse_args()


def cylindrical_bin_volumes_from_edges(
    r_edges_kpc: np.ndarray,
    phi_edges_rad: np.ndarray,
    z_edges_kpc: np.ndarray,
) -> np.ndarray:
    radial_area = 0.5 * (np.asarray(r_edges_kpc)[1:] ** 2 - np.asarray(r_edges_kpc)[:-1] ** 2)
    dphi = np.diff(phi_edges_rad)
    dz = np.diff(z_edges_kpc)
    return radial_area[:, None, None] * dphi[None, :, None] * dz[None, None, :]


def _mass_grids(table: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    volumes = cylindrical_bin_volumes_from_edges(
        table["r_edges_kpc"],
        table["phi_edges_rad"],
        table["z_edges_kpc"],
    ).astype(np.float32)
    truth_mass = table["truth_density"].astype(np.float32) * volumes[None, ...]
    baseline_mass = table["baseline_density"].astype(np.float32) * volumes[None, ...]
    return truth_mass.astype(np.float32), baseline_mass.astype(np.float32)


def _reconstruct_delta_mass_from_coefficients(
    coefficients: np.ndarray,
    *,
    components: np.ndarray,
    mean: np.ndarray,
    scale_msun: np.ndarray,
    grid_shape: tuple[int, int, int],
) -> np.ndarray:
    target = mean[None, :] + coefficients @ components
    return (target * scale_msun[:, None]).reshape((coefficients.shape[0], *grid_shape)).astype(
        np.float32
    )


def _preserve_total_mass(reference_mass: np.ndarray, corrected_mass: np.ndarray) -> np.ndarray:
    reference_total = np.sum(reference_mass, axis=(1, 2, 3))
    corrected_total = np.sum(corrected_mass, axis=(1, 2, 3))
    scale = np.divide(
        reference_total,
        corrected_total,
        out=np.ones_like(reference_total, dtype=np.float32),
        where=corrected_total > 0.0,
    )
    return (corrected_mass * scale[:, None, None, None]).astype(np.float32)


def _corrected_mass_from_delta(baseline_mass: np.ndarray, delta_mass: np.ndarray) -> np.ndarray:
    corrected = np.clip(baseline_mass + delta_mass, 0.0, None)
    return _preserve_total_mass(baseline_mass, corrected)


def _rescale_rows_to_total_mass(mass: np.ndarray, target_total_msun: np.ndarray) -> np.ndarray:
    current_total = np.sum(mass, axis=tuple(range(1, mass.ndim)))
    scale = np.divide(
        target_total_msun.astype(np.float32),
        current_total.astype(np.float32),
        out=np.ones_like(current_total, dtype=np.float32),
        where=current_total > 0.0,
    )
    return (mass * scale.reshape((-1,) + (1,) * (mass.ndim - 1))).astype(np.float32)


def radial_mass_profiles(mass: np.ndarray) -> np.ndarray:
    return np.sum(mass, axis=(-2, -1))


def vertical_mass_profiles(mass: np.ndarray) -> np.ndarray:
    if mass.ndim == 4:
        return np.sum(mass, axis=(1, 2))
    return np.sum(mass, axis=(2, 3))


def azimuthal_m2_profiles(mass: np.ndarray, phi_edges_rad: np.ndarray) -> np.ndarray:
    phi_centers = 0.5 * (phi_edges_rad[:-1] + phi_edges_rad[1:])
    radial_phi_mass = np.sum(mass, axis=-1)
    complex_m2 = np.sum(radial_phi_mass * np.exp(2j * phi_centers), axis=-1)
    total = np.sum(radial_phi_mass, axis=-1)
    return np.divide(np.abs(complex_m2), total, out=np.zeros_like(total, dtype=float), where=total > 0.0)


def _total_mass(mass: np.ndarray) -> np.ndarray:
    return np.sum(mass, axis=tuple(range(mass.ndim - 3, mass.ndim)))


def vertical_rms_height_kpc(mass: np.ndarray, z_edges_kpc: np.ndarray) -> np.ndarray:
    z_centers = 0.5 * (z_edges_kpc[:-1] + z_edges_kpc[1:])
    weights = z_centers**2
    weighted = np.sum(mass * weights.reshape((1,) * (mass.ndim - 1) + (-1,)), axis=(-3, -2, -1))
    total = _total_mass(mass)
    return np.sqrt(np.divide(weighted, total, out=np.zeros_like(weighted, dtype=float), where=total > 0.0))


def central_mass_fraction(mass: np.ndarray, r_edges_kpc: np.ndarray, *, radius_kpc: float) -> np.ndarray:
    r_centers = 0.5 * (r_edges_kpc[:-1] + r_edges_kpc[1:])
    radial_mask = r_centers < radius_kpc
    if mass.ndim == 4:
        central = np.sum(mass[:, radial_mask, :, :], axis=(1, 2, 3))
    else:
        central = np.sum(mass[:, :, radial_mask, :, :], axis=(2, 3, 4))
    total = _total_mass(mass)
    return np.divide(central, total, out=np.zeros_like(central, dtype=float), where=total > 0.0)


def bar_axis_mass_fraction(
    mass: np.ndarray,
    phi_edges_rad: np.ndarray,
    *,
    half_angle_deg: float,
) -> np.ndarray:
    phi_centers = 0.5 * (phi_edges_rad[:-1] + phi_edges_rad[1:])
    distance_to_bar_axis = np.abs(((phi_centers + 0.5 * np.pi) % np.pi) - 0.5 * np.pi)
    phi_mask = distance_to_bar_axis <= np.deg2rad(half_angle_deg)
    if mass.ndim == 4:
        bar_mass = np.sum(mass[:, :, phi_mask, :], axis=(1, 2, 3))
    else:
        bar_mass = np.sum(mass[:, :, :, phi_mask, :], axis=(2, 3, 4))
    total = _total_mass(mass)
    return np.divide(bar_mass, total, out=np.zeros_like(bar_mass, dtype=float), where=total > 0.0)


def _mae(candidate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(candidate - truth)))


def _coverage(samples: np.ndarray, truth: np.ndarray) -> float:
    lower = np.quantile(samples, 0.16, axis=1)
    upper = np.quantile(samples, 0.84, axis=1)
    inside = (truth >= lower) & (truth <= upper)
    return float(np.mean(inside))


def _mean_interval_width(samples: np.ndarray) -> float:
    lower = np.quantile(samples, 0.16, axis=1)
    upper = np.quantile(samples, 0.84, axis=1)
    return float(np.mean(upper - lower))


def _temperature_scales_by_inclination(final_metrics: dict[str, Any]) -> dict[str, float]:
    by_inclination = final_metrics["selected_model"]["inclination_temperature_by_inclination"]
    return {key: float(value["temperature_scale"]) for key, value in by_inclination.items()}


def _row_temperature_scales(metadata: np.ndarray, scales_by_inclination: dict[str, float]) -> np.ndarray:
    scales = []
    for inclination in metadata[:, 0]:
        scales.append(scales_by_inclination.get(f"{float(inclination):g}", 1.0))
    return np.asarray(scales, dtype=np.float32)


def _calibrated_sample_batches(
    *,
    sampled_coefficients: np.ndarray,
    pca: np.lib.npyio.NpzFile,
    baseline_mass: np.ndarray,
    baseline_grid_mass_msun: np.ndarray,
    row_scales: np.ndarray,
    batch_size: int,
    target_total_mass_msun: np.ndarray | None = None,
):
    n_rows, n_samples, n_coefficients = sampled_coefficients.shape
    grid_shape = tuple(int(x) for x in baseline_mass.shape[1:])
    for start in range(0, n_rows, batch_size):
        stop = min(start + batch_size, n_rows)
        coefficient_batch = sampled_coefficients[start:stop]
        center = coefficient_batch.mean(axis=1, keepdims=True)
        scaled_coefficients = center + row_scales[start:stop, None, None] * (
            coefficient_batch - center
        )
        flat_coefficients = scaled_coefficients.reshape(-1, n_coefficients)
        repeated_scale = np.repeat(baseline_grid_mass_msun[start:stop], n_samples)
        delta = _reconstruct_delta_mass_from_coefficients(
            flat_coefficients,
            components=pca["components"].astype(np.float32),
            mean=pca["mean"].astype(np.float32),
            scale_msun=repeated_scale.astype(np.float32),
            grid_shape=grid_shape,
        )
        repeated_baseline = np.repeat(baseline_mass[start:stop, None, ...], n_samples, axis=1)
        corrected = _corrected_mass_from_delta(
            repeated_baseline.reshape((-1, *grid_shape)).astype(np.float32),
            delta,
        )
        if target_total_mass_msun is not None:
            corrected = _rescale_rows_to_total_mass(
                corrected,
                target_total_mass_msun[start:stop].reshape(-1),
            )
        yield corrected.reshape((stop - start, n_samples, *grid_shape))


def _summary_functions(
    *,
    r_edges_kpc: np.ndarray,
    phi_edges_rad: np.ndarray,
    z_edges_kpc: np.ndarray,
    central_radius_kpc: float,
    bar_half_angle_deg: float,
) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    return {
        "radial_profile": radial_mass_profiles,
        "vertical_profile": vertical_mass_profiles,
        "vertical_rms_height_kpc": lambda mass: vertical_rms_height_kpc(mass, z_edges_kpc),
        "central_mass_fraction_r_lt_2kpc": lambda mass: central_mass_fraction(
            mass,
            r_edges_kpc,
            radius_kpc=central_radius_kpc,
        ),
        "bar_axis_mass_fraction": lambda mass: bar_axis_mass_fraction(
            mass,
            phi_edges_rad,
            half_angle_deg=bar_half_angle_deg,
        ),
        "bar_frame_m2_profile": lambda mass: azimuthal_m2_profiles(mass, phi_edges_rad),
    }


def _summary_metrics(
    *,
    truth_summary: np.ndarray,
    baseline_summary: np.ndarray,
    mdn_mean_summary: np.ndarray,
    sample_summary: np.ndarray,
) -> dict[str, float]:
    baseline_mae = _mae(baseline_summary, truth_summary)
    mdn_mae = _mae(mdn_mean_summary, truth_summary)
    return {
        "baseline_mae": baseline_mae,
        "mdn_posterior_mean_mae": mdn_mae,
        "mdn_improvement_over_baseline_fraction": (
            (baseline_mae - mdn_mae) / baseline_mae if baseline_mae > 0.0 else 0.0
        ),
        "mdn_coverage_68": _coverage(sample_summary, truth_summary),
        "mdn_mean_68_interval_width": _mean_interval_width(sample_summary),
    }


def _stratified_summary_metrics(
    *,
    truth_summary: np.ndarray,
    baseline_summary: np.ndarray,
    mdn_mean_summary: np.ndarray,
    sample_summary: np.ndarray,
    labels: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    output = {}
    for value in sorted(float(x) for x in np.unique(labels)):
        mask = np.isclose(labels, value)
        output[f"{value:g}"] = {
            "n": int(np.sum(mask)),
            **_summary_metrics(
                truth_summary=truth_summary[mask],
                baseline_summary=baseline_summary[mask],
                mdn_mean_summary=mdn_mean_summary[mask],
                sample_summary=sample_summary[mask],
            ),
        }
    return output


def build_metrics(
    *,
    table: np.lib.npyio.NpzFile,
    pca: np.lib.npyio.NpzFile,
    predictions: np.lib.npyio.NpzFile,
    final_metrics: dict[str, Any],
    central_radius_kpc: float,
    bar_half_angle_deg: float,
    sample_batch_size: int,
) -> dict[str, Any]:
    truth_mass, baseline_mass = _mass_grids(table)
    posterior_mean_mass = predictions["posterior_mean_mass"].astype(np.float32)
    split = table["split"].astype(str)
    metadata = table["metadata"].astype(np.float32)
    test_mask = split == "test"
    if not np.any(test_mask):
        raise ValueError("physical summary evaluation requires at least one test row")
    test_indices = np.flatnonzero(test_mask)
    truth_test = truth_mass[test_mask]
    baseline_test = baseline_mass[test_mask]
    mean_test = posterior_mean_mass[test_mask]
    metadata_test = metadata[test_mask]
    sampled_coefficients = predictions["sampled_coefficients"].astype(np.float32)[test_mask]
    row_scales = _row_temperature_scales(
        metadata_test,
        _temperature_scales_by_inclination(final_metrics),
    )
    baseline_grid_mass_msun = table["baseline_grid_mass_msun"].astype(np.float32)[test_mask]

    summary_functions = _summary_functions(
        r_edges_kpc=table["r_edges_kpc"],
        phi_edges_rad=table["phi_edges_rad"],
        z_edges_kpc=table["z_edges_kpc"],
        central_radius_kpc=central_radius_kpc,
        bar_half_angle_deg=bar_half_angle_deg,
    )
    truth_summaries = {name: fn(truth_test) for name, fn in summary_functions.items()}
    baseline_summaries = {name: fn(baseline_test) for name, fn in summary_functions.items()}
    mean_summaries = {name: fn(mean_test) for name, fn in summary_functions.items()}
    sample_summaries: dict[str, list[np.ndarray]] = {name: [] for name in summary_functions}
    for sample_mass in _calibrated_sample_batches(
        sampled_coefficients=sampled_coefficients,
        pca=pca,
        baseline_mass=baseline_test,
        baseline_grid_mass_msun=baseline_grid_mass_msun,
        row_scales=row_scales,
        batch_size=sample_batch_size,
    ):
        for name, fn in summary_functions.items():
            sample_summaries[name].append(fn(sample_mass))
    stacked_samples = {
        name: np.concatenate(chunks, axis=0) for name, chunks in sample_summaries.items()
    }
    summary_metrics = {
        name: _summary_metrics(
            truth_summary=truth_summaries[name],
            baseline_summary=baseline_summaries[name],
            mdn_mean_summary=mean_summaries[name],
            sample_summary=stacked_samples[name],
        )
        for name in summary_functions
    }
    stratified_metrics = {
        "inclination_deg": {
            name: _stratified_summary_metrics(
                truth_summary=truth_summaries[name],
                baseline_summary=baseline_summaries[name],
                mdn_mean_summary=mean_summaries[name],
                sample_summary=stacked_samples[name],
                labels=metadata_test[:, 0],
            )
            for name in summary_functions
        },
        "bar_angle_deg": {
            name: _stratified_summary_metrics(
                truth_summary=truth_summaries[name],
                baseline_summary=baseline_summaries[name],
                mdn_mean_summary=mean_summaries[name],
                sample_summary=stacked_samples[name],
                labels=metadata_test[:, 2],
            )
            for name in summary_functions
        },
    }
    return {
        "selected_run_id": final_metrics["selected_model"]["run_id"],
        "n_test": int(np.sum(test_mask)),
        "n_test_galaxies": int(len(set(table["galaxy_id"][test_mask].astype(int).tolist()))),
        "n_posterior_samples": int(sampled_coefficients.shape[1]),
        "central_radius_kpc": float(central_radius_kpc),
        "bar_half_angle_deg": float(bar_half_angle_deg),
        "test_indices": test_indices.astype(int).tolist(),
        "summary_metrics": summary_metrics,
        "stratified_metrics": stratified_metrics,
    }


def _fmt(value: float) -> str:
    if abs(value) >= 1.0e4:
        return f"{value:.3e}".replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")
    return f"{value:.4g}"


def _write_markdown(path: Path, metrics: dict[str, Any]) -> None:
    names = {
        "radial_profile": "radial mass profile",
        "vertical_profile": "vertical mass profile",
        "vertical_rms_height_kpc": "vertical RMS height",
        "central_mass_fraction_r_lt_2kpc": "central mass fraction",
        "bar_axis_mass_fraction": "bar-axis mass fraction",
        "bar_frame_m2_profile": "bar-frame m=2 profile",
    }
    lines = [
        "# Milestone 2b Physical Summary Posterior Evaluation",
        "",
        "Date: 2026-06-09",
        "",
        "## Scope",
        "",
        "This report evaluates the selected calibrated MDN posterior on physical "
        "summaries derived from the coarse cylindrical 3D mass grid. It uses the "
        "frozen Milestone 2b default run and applies the inclination-aware "
        "temperature scales before computing 68 percent posterior intervals.",
        "",
        f"- selected run: `{metrics['selected_run_id']}`",
        f"- held-out test projections: `{metrics['n_test']}` from "
        f"`{metrics['n_test_galaxies']}` galaxies",
        f"- posterior samples per row: `{metrics['n_posterior_samples']}`",
        f"- central aperture: `R < {metrics['central_radius_kpc']:g} kpc`",
        f"- bar-axis aperture: within `{metrics['bar_half_angle_deg']:g} deg` of the "
        "bar major axis",
        "",
        "## Held-Out Physical Summaries",
        "",
        "| Summary | Baseline MAE | MDN posterior-mean MAE | Improvement | 68% coverage | Mean 68% width |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, label in names.items():
        item = metrics["summary_metrics"][key]
        lines.append(
            "| "
            f"{label} | "
            f"{_fmt(item['baseline_mae'])} | "
            f"{_fmt(item['mdn_posterior_mean_mae'])} | "
            f"{100.0 * item['mdn_improvement_over_baseline_fraction']:.2f}% | "
            f"{item['mdn_coverage_68']:.3f} | "
            f"{_fmt(item['mdn_mean_68_interval_width'])} |"
        )
    lines.extend(
        [
            "",
            "## Inclination-Stratified Coverage",
            "",
            "| Summary | 20 deg | 40 deg | 60 deg |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for key, label in names.items():
        bins = metrics["stratified_metrics"]["inclination_deg"][key]
        lines.append(
            "| "
            f"{label} | "
            f"{bins['20']['mdn_coverage_68']:.3f} | "
            f"{bins['40']['mdn_coverage_68']:.3f} | "
            f"{bins['60']['mdn_coverage_68']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Bar-Angle-Stratified Coverage",
            "",
            "| Summary | 0 deg | 45 deg | 90 deg |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for key, label in names.items():
        bins = metrics["stratified_metrics"]["bar_angle_deg"][key]
        lines.append(
            "| "
            f"{label} | "
            f"{bins['0']['mdn_coverage_68']:.3f} | "
            f"{bins['45']['mdn_coverage_68']:.3f} | "
            f"{bins['90']['mdn_coverage_68']:.3f} |"
        )
    vertical_rms = metrics["summary_metrics"]["vertical_rms_height_kpc"]
    bar_axis = metrics["summary_metrics"]["bar_axis_mass_fraction"]
    radial = metrics["summary_metrics"]["radial_profile"]
    central = metrics["summary_metrics"]["central_mass_fraction_r_lt_2kpc"]
    m2 = metrics["summary_metrics"]["bar_frame_m2_profile"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These checks ask whether the calibrated PCA-residual posterior is useful "
            "for physical galaxy summaries, not only PCA coefficients. Strong "
            "results should combine lower posterior-mean MAE than the geometric "
            "baseline with coverage near the nominal 0.68 target. Summaries with "
            "low coverage identify where the current MDN uncertainty is still too "
            "narrow after inclination-aware coefficient calibration.",
            "",
            "The posterior mean improves every physical summary in this report. The "
            "strongest point-prediction gains are vertical structure: vertical "
            f"profile MAE improves by {100.0 * metrics['summary_metrics']['vertical_profile']['mdn_improvement_over_baseline_fraction']:.2f}% "
            "and vertical RMS height improves by "
            f"{100.0 * vertical_rms['mdn_improvement_over_baseline_fraction']:.2f}%.",
            "",
            "Coverage is mixed. Vertical RMS height and bar-axis mass fraction are "
            f"close to nominal at {vertical_rms['mdn_coverage_68']:.3f} and "
            f"{bar_axis['mdn_coverage_68']:.3f}. Radial profile, central mass "
            "fraction, and bar-frame m=2 profile remain undercovered at "
            f"{radial['mdn_coverage_68']:.3f}, {central['mdn_coverage_68']:.3f}, "
            f"and {m2['mdn_coverage_68']:.3f}.",
            "",
            "Recommendation: keep the selected MDN as the Milestone 2b point and "
            "coefficient-uncertainty baseline, but do not yet claim fully "
            "calibrated physical-summary posteriors. The next modeling step should "
            "calibrate or train against physical summaries directly before moving "
            "to a larger full-3D posterior generator.",
        ]
    )
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
        )
    (args.output_dir / "milestone2b_physical_summary_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(args.output_dir / "milestone2b_physical_summary_evaluation.md", metrics)
    print(f"wrote Milestone 2b physical summary evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
