"""Scan sensitivity of the adopted MDN posterior to inclination errors.

The realistic deployment scenario is that the observer's inclination estimate
carries an error. Inclination enters the pipeline twice: through the geometric
baseline (deprojection stretch and sech^2 vertical spread) and through the
model conditioning (metadata and central elliptical features). This scan
evaluates the trained model end-to-end with perturbed inclination everywhere
an observer would use it, while the truth grids and images stay fixed at the
true viewing geometry.

For each scenario (deterministic offsets and one Gaussian-error draw) the scan
rebuilds the baseline from the image at the perturbed inclination, rebuilds the
features, re-samples the MDN posterior with common sampling noise (paired
across scenarios), and reports physical-summary bias, MAE, and 68 percent
coverage against truth, stratified by true inclination.

The correction heads are deliberately excluded: the reference scenario is
computed through the identical pipeline, so scenario-minus-reference
differences isolate the inclination sensitivity of the core model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from build_tng50_baseline_density_grid import baseline_density_grid_from_image
from dgdp.density3d import CylindricalGridSpec, cylindrical_bin_volumes
from dgdp.models.mdn import SummaryResidualMDN
from dgdp.types import Geometry
from report_milestone2b_physical_summaries import (
    azimuthal_m2_profiles,
    bar_axis_mass_fraction,
    central_mass_fraction,
    radial_mass_profiles,
    vertical_mass_profiles,
    vertical_rms_height_kpc,
)
from train_density_residual_pca import (
    make_density_residual_features,
    reconstruct_delta_mass_from_coefficients,
)
from train_density_residual_pca_mdn import corrected_mass_from_delta

DEFAULT_TABLE = Path("outputs/tng50_milestone2c_clean3d/density_residual_table.npz")
DEFAULT_PCA = Path(
    "outputs/tng50_milestone2c_clean3d/density_residual_diagnostics/density_residual_pca.npz"
)
DEFAULT_RUN_DIR = Path(
    "outputs/tng50_milestone2c_clean3d/density_residual_pca_mdn_sweep_central/"
    "components_1_seed_20260608"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/tng50_milestone2c_clean3d/milestone2c_inclination_sensitivity"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--density-table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--pca", type=Path, default=DEFAULT_PCA)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--offsets-deg",
        type=float,
        nargs="+",
        default=[-5.0, -3.0, 3.0, 5.0],
    )
    parser.add_argument("--gaussian-sigma-deg", type=float, default=3.0)
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--sample-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--pixel-scale-kpc", type=float, default=0.35)
    parser.add_argument("--vertical-scale-height-kpc", type=float, default=0.4)
    parser.add_argument("--central-radius-kpc", type=float, default=2.0)
    parser.add_argument("--bar-half-angle-deg", type=float, default=30.0)
    return parser.parse_args()


def clip_inclination_deg(inclination_deg: np.ndarray) -> np.ndarray:
    return np.clip(inclination_deg, 1.0, 89.0)


def rebuild_baseline_masses(
    images: np.ndarray,
    inclination_deg: np.ndarray,
    disk_pa_deg: np.ndarray,
    bar_angle_deg: np.ndarray,
    *,
    spec: CylindricalGridSpec,
    pixel_scale_kpc: float,
    vertical_scale_height_kpc: float,
) -> np.ndarray:
    masses = np.empty(
        (
            images.shape[0],
            len(spec.r_edges_kpc) - 1,
            len(spec.phi_edges_rad) - 1,
            len(spec.z_edges_kpc) - 1,
        ),
        dtype=np.float32,
    )
    for index in range(images.shape[0]):
        geometry = Geometry(
            inclination_deg=float(inclination_deg[index]),
            disk_pa_deg=float(disk_pa_deg[index]),
            bar_angle_deg=float(bar_angle_deg[index]),
        )
        grid = baseline_density_grid_from_image(
            np.asarray(images[index], dtype=float),
            geometry=geometry,
            pixel_scale_kpc=pixel_scale_kpc,
            spec=spec,
            vertical_scale_height_kpc=vertical_scale_height_kpc,
        )
        masses[index] = grid.mass_msun.astype(np.float32)
    return masses


def summary_values(
    mass: np.ndarray,
    *,
    r_edges_kpc: np.ndarray,
    phi_edges_rad: np.ndarray,
    z_edges_kpc: np.ndarray,
    central_radius_kpc: float,
    bar_half_angle_deg: float,
) -> dict[str, np.ndarray]:
    return {
        "radial_mass_profile": radial_mass_profiles(mass),
        "vertical_mass_profile": vertical_mass_profiles(mass),
        "vertical_rms_height": vertical_rms_height_kpc(mass, z_edges_kpc),
        "central_mass_fraction": central_mass_fraction(
            mass,
            r_edges_kpc,
            radius_kpc=central_radius_kpc,
        ),
        "bar_axis_mass_fraction": bar_axis_mass_fraction(
            mass,
            phi_edges_rad,
            half_angle_deg=bar_half_angle_deg,
        ),
        "m2_profile": azimuthal_m2_profiles(mass, phi_edges_rad),
    }


def _coverage_68(samples: np.ndarray, truth: np.ndarray) -> float:
    lower = np.quantile(samples, 0.16, axis=1)
    upper = np.quantile(samples, 0.84, axis=1)
    inside = (truth >= lower) & (truth <= upper)
    return float(np.mean(inside))


def _summary_metrics(
    posterior_mean: dict[str, np.ndarray],
    sampled: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    *,
    row_mask: np.ndarray,
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for name, truth_values in truth.items():
        mean_values = posterior_mean[name][row_mask]
        sample_values = sampled[name][row_mask]
        truth_masked = truth_values[row_mask]
        metrics[name] = {
            "bias": float(np.mean(mean_values - truth_masked)),
            "mae": float(np.mean(np.abs(mean_values - truth_masked))),
            "coverage_68": _coverage_68(sample_values, truth_masked),
        }
    return metrics


def run_scenario(
    *,
    label: str,
    delta_inclination_deg: np.ndarray,
    images: np.ndarray,
    metadata: np.ndarray,
    truth_mass: np.ndarray,
    spec: CylindricalGridSpec,
    model: SummaryResidualMDN,
    normalization: dict[str, np.ndarray],
    components: np.ndarray,
    pca_mean: np.ndarray,
    args: argparse.Namespace,
    image_feature_size: int,
    central_pixel_scale_kpc: float | None,
    truth_summaries: dict[str, np.ndarray],
    table_baseline_mass: np.ndarray | None,
) -> dict[str, object]:
    n_rows = images.shape[0]
    inclination_obs = clip_inclination_deg(metadata[:, 0] + delta_inclination_deg)
    baseline_mass = rebuild_baseline_masses(
        images,
        inclination_obs,
        metadata[:, 1],
        metadata[:, 2],
        spec=spec,
        pixel_scale_kpc=args.pixel_scale_kpc,
        vertical_scale_height_kpc=args.vertical_scale_height_kpc,
    )
    baseline_grid_mass = np.sum(baseline_mass, axis=(1, 2, 3)).astype(np.float32)
    metadata_obs = metadata.copy()
    metadata_obs[:, 0] = inclination_obs.astype(np.float32)
    features = make_density_residual_features(
        images,
        metadata_obs,
        baseline_grid_mass_msun=baseline_grid_mass,
        image_feature_size=image_feature_size,
        central_pixel_scale_kpc=central_pixel_scale_kpc,
    )
    x = (features - normalization["x_mean"][None, :]) / normalization["x_scale"][None, :]
    torch.manual_seed(args.seed)
    with torch.no_grad():
        sampled_z = model.sample(
            torch.tensor(x, dtype=torch.float32),
            n_samples=args.n_samples,
        ).numpy()
    sampled_coefficients = (
        sampled_z * normalization["y_scale"][None, None, :]
        + normalization["y_mean"][None, None, :]
    ).astype(np.float32)
    posterior_mean_coefficients = sampled_coefficients.mean(axis=1).astype(np.float32)

    grid_shape = baseline_mass.shape[1:]
    posterior_mean_delta = reconstruct_delta_mass_from_coefficients(
        posterior_mean_coefficients,
        components=components,
        mean=pca_mean,
        scale_msun=baseline_grid_mass,
        grid_shape=grid_shape,
    )
    posterior_mean_mass = corrected_mass_from_delta(
        baseline_mass,
        posterior_mean_delta,
        preserve_baseline_total_mass=True,
    )
    summary_kwargs = {
        "r_edges_kpc": spec.r_edges_kpc,
        "phi_edges_rad": spec.phi_edges_rad,
        "z_edges_kpc": spec.z_edges_kpc,
        "central_radius_kpc": args.central_radius_kpc,
        "bar_half_angle_deg": args.bar_half_angle_deg,
    }
    posterior_mean_summaries = summary_values(posterior_mean_mass, **summary_kwargs)

    sampled_summaries: dict[str, list[np.ndarray]] = {
        name: [] for name in truth_summaries
    }
    for start in range(0, args.n_samples, args.sample_batch_size):
        stop = min(start + args.sample_batch_size, args.n_samples)
        batch = stop - start
        flat_coefficients = sampled_coefficients[:, start:stop, :].reshape(
            n_rows * batch, -1
        )
        flat_delta = reconstruct_delta_mass_from_coefficients(
            flat_coefficients,
            components=components,
            mean=pca_mean,
            scale_msun=np.repeat(baseline_grid_mass, batch),
            grid_shape=grid_shape,
        )
        flat_mass = corrected_mass_from_delta(
            np.repeat(baseline_mass, batch, axis=0),
            flat_delta,
            preserve_baseline_total_mass=True,
        )
        batch_summaries = summary_values(flat_mass, **summary_kwargs)
        for name, values in batch_summaries.items():
            sampled_summaries[name].append(
                values.reshape((n_rows, batch) + values.shape[1:])
            )
    sampled_stacked = {
        name: np.concatenate(chunks, axis=1)
        for name, chunks in sampled_summaries.items()
    }

    truth_total = np.sum(truth_mass, axis=(1, 2, 3))
    scenario: dict[str, object] = {
        "label": label,
        "delta_inclination_mean_deg": float(np.mean(delta_inclination_deg)),
        "delta_inclination_std_deg": float(np.std(delta_inclination_deg)),
        "baseline_total_mass_fractional_mae": float(
            np.mean(np.abs(baseline_grid_mass - truth_total) / np.maximum(truth_total, 1.0))
        ),
        "summaries": _summary_metrics(
            posterior_mean_summaries,
            sampled_stacked,
            truth_summaries,
            row_mask=np.ones(n_rows, dtype=bool),
        ),
        "by_true_inclination": {
            str(int(value)): _summary_metrics(
                posterior_mean_summaries,
                sampled_stacked,
                truth_summaries,
                row_mask=np.isclose(metadata[:, 0], value),
            )
            for value in np.unique(metadata[:, 0])
        },
    }
    if table_baseline_mass is not None:
        rel_diff = np.abs(baseline_mass - table_baseline_mass) / np.maximum(
            np.max(np.abs(table_baseline_mass), axis=(1, 2, 3), keepdims=True),
            1.0,
        )
        scenario["baseline_rebuild_max_rel_diff"] = float(np.max(rel_diff))
    return scenario


def _write_markdown(path: Path, metrics: dict[str, object]) -> None:
    scenarios: list[dict[str, object]] = metrics["scenarios"]  # type: ignore[assignment]
    reference = next(s for s in scenarios if s["label"] == "reference")
    lines = [
        "# Milestone 2c Inclination Sensitivity Scan",
        "",
        f"- adopted run: `{metrics['run_dir']}`",
        f"- split: `{metrics['split']}` ({metrics['n_rows']} rows)",
        "- correction heads: excluded (paired comparison against the reference)",
        f"- baseline rebuild check (reference vs table): max rel diff "
        f"`{reference.get('baseline_rebuild_max_rel_diff', float('nan')):.2e}`",
        "",
        "## Posterior-Mean MAE By Scenario",
        "",
    ]
    names = list(reference["summaries"].keys())  # type: ignore[index]
    header = "| Scenario | baseline mass err | " + " | ".join(names) + " |"
    lines.append(header)
    lines.append("| --- | ---: | " + " | ".join("---:" for _ in names) + " |")
    for scenario in scenarios:
        summaries = scenario["summaries"]  # type: ignore[index]
        cells = [f"{summaries[name]['mae']:.4g}" for name in names]
        lines.append(
            f"| {scenario['label']} | "
            f"{scenario['baseline_total_mass_fractional_mae']:.4f} | "
            + " | ".join(cells)
            + " |"
        )
    lines += ["", "## Posterior-Mean Bias By Scenario", ""]
    lines.append(header)
    lines.append("| --- | ---: | " + " | ".join("---:" for _ in names) + " |")
    for scenario in scenarios:
        summaries = scenario["summaries"]  # type: ignore[index]
        cells = [f"{summaries[name]['bias']:+.4g}" for name in names]
        lines.append(
            f"| {scenario['label']} | "
            f"{scenario['baseline_total_mass_fractional_mae']:.4f} | "
            + " | ".join(cells)
            + " |"
        )
    lines += ["", "## Coverage 68 By Scenario", ""]
    lines.append(header)
    lines.append("| --- | ---: | " + " | ".join("---:" for _ in names) + " |")
    for scenario in scenarios:
        summaries = scenario["summaries"]  # type: ignore[index]
        cells = [f"{summaries[name]['coverage_68']:.3f}" for name in names]
        lines.append(
            f"| {scenario['label']} | "
            f"{scenario['baseline_total_mass_fractional_mae']:.4f} | "
            + " | ".join(cells)
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.density_table) as table:
        split = table["split"].astype(str)
        mask = split == args.split
        spec = CylindricalGridSpec(
            r_edges_kpc=np.asarray(table["r_edges_kpc"], dtype=float),
            phi_edges_rad=np.asarray(table["phi_edges_rad"], dtype=float),
            z_edges_kpc=np.asarray(table["z_edges_kpc"], dtype=float),
        )
        volumes = cylindrical_bin_volumes(spec).astype(np.float32)
        images = np.asarray(table["images"][mask], dtype=np.float32)
        metadata = np.asarray(table["metadata"][mask], dtype=np.float32)
        truth_mass = (
            np.asarray(table["truth_density"][mask], dtype=np.float32) * volumes[None]
        )
        table_baseline_mass = (
            np.asarray(table["baseline_density"][mask], dtype=np.float32)
            * volumes[None]
        )

    with np.load(args.pca) as pca:
        components = pca["components"].astype(np.float32)
        pca_mean = pca["mean"].astype(np.float32)

    checkpoint = torch.load(
        args.run_dir / "density_residual_pca_mdn.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = SummaryResidualMDN(
        input_dim=int(checkpoint["input_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        n_components=int(checkpoint["n_components"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with np.load(args.run_dir / "density_residual_pca_mdn_normalization.npz") as norm:
        normalization = {key: norm[key].astype(np.float32) for key in norm.files}

    truth_summaries = summary_values(
        truth_mass,
        r_edges_kpc=spec.r_edges_kpc,
        phi_edges_rad=spec.phi_edges_rad,
        z_edges_kpc=spec.z_edges_kpc,
        central_radius_kpc=args.central_radius_kpc,
        bar_half_angle_deg=args.bar_half_angle_deg,
    )

    rng = np.random.default_rng(args.seed)
    n_rows = images.shape[0]
    scenario_deltas: list[tuple[str, np.ndarray]] = [
        ("reference", np.zeros(n_rows)),
    ]
    for offset in args.offsets_deg:
        scenario_deltas.append((f"offset_{offset:+g}_deg", np.full(n_rows, offset)))
    if args.gaussian_sigma_deg > 0.0:
        scenario_deltas.append(
            (
                f"gaussian_sigma_{args.gaussian_sigma_deg:g}_deg",
                rng.normal(0.0, args.gaussian_sigma_deg, size=n_rows),
            )
        )

    scenarios = []
    for label, delta in scenario_deltas:
        scenario = run_scenario(
            label=label,
            delta_inclination_deg=delta,
            images=images,
            metadata=metadata,
            truth_mass=truth_mass,
            spec=spec,
            model=model,
            normalization=normalization,
            components=components,
            pca_mean=pca_mean,
            args=args,
            image_feature_size=int(checkpoint["image_feature_size"]),
            central_pixel_scale_kpc=(
                float(checkpoint["central_pixel_scale_kpc"])
                if checkpoint["central_pixel_scale_kpc"] is not None
                else None
            ),
            truth_summaries=truth_summaries,
            table_baseline_mass=(table_baseline_mass if label == "reference" else None),
        )
        scenarios.append(scenario)
        print(f"scenario {label}: done")

    metrics: dict[str, object] = {
        "run_dir": str(args.run_dir),
        "split": args.split,
        "n_rows": int(n_rows),
        "n_samples": int(args.n_samples),
        "seed": int(args.seed),
        "offsets_deg": [float(v) for v in args.offsets_deg],
        "gaussian_sigma_deg": float(args.gaussian_sigma_deg),
        "vertical_scale_height_kpc": float(args.vertical_scale_height_kpc),
        "scenarios": scenarios,
    }
    (args.output_dir / "inclination_sensitivity_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    _write_markdown(args.output_dir / "inclination_sensitivity.md", metrics)
    print(f"wrote inclination sensitivity scan to {args.output_dir}")


if __name__ == "__main__":
    main()
