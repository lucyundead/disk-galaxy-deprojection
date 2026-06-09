from __future__ import annotations

import argparse
import json
import struct
import zlib
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--density-table", type=Path, required=True)
    parser.add_argument("--pca", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=6)
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


def radial_mass_profiles(mass_grids: np.ndarray) -> np.ndarray:
    return np.sum(mass_grids, axis=(2, 3))


def vertical_mass_profiles(mass_grids: np.ndarray) -> np.ndarray:
    return np.sum(mass_grids, axis=(1, 2))


def azimuthal_m2_amplitude_profiles(mass_grids: np.ndarray, phi_edges_rad: np.ndarray) -> np.ndarray:
    phi_centers = 0.5 * (phi_edges_rad[:-1] + phi_edges_rad[1:])
    radial_phi_mass = np.sum(mass_grids, axis=3)
    complex_m2 = np.sum(radial_phi_mass * np.exp(2j * phi_centers)[None, None, :], axis=2)
    total = np.sum(radial_phi_mass, axis=2)
    return np.divide(np.abs(complex_m2), total, out=np.zeros_like(total, dtype=float), where=total > 0.0)


def _make_mass_grids(table: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    volumes = cylindrical_bin_volumes_from_edges(
        table["r_edges_kpc"],
        table["phi_edges_rad"],
        table["z_edges_kpc"],
    ).astype(np.float32)
    truth_mass = table["truth_density"].astype(np.float32) * volumes[None, ...]
    baseline_mass = table["baseline_density"].astype(np.float32) * volumes[None, ...]
    delta_mass = table["delta_density"].astype(np.float32) * volumes[None, ...]
    return truth_mass, baseline_mass, delta_mass


def preserve_total_mass(reference_mass: np.ndarray, corrected_mass: np.ndarray) -> np.ndarray:
    reference_total = np.sum(reference_mass, axis=(1, 2, 3))
    corrected_total = np.sum(corrected_mass, axis=(1, 2, 3))
    scale = np.divide(
        reference_total,
        corrected_total,
        out=np.ones_like(reference_total, dtype=np.float32),
        where=corrected_total > 0.0,
    )
    return (corrected_mass * scale[:, None, None, None]).astype(np.float32)


def corrected_mass_from_delta(baseline_mass: np.ndarray, delta_mass: np.ndarray) -> np.ndarray:
    corrected = np.clip(baseline_mass + delta_mass, 0.0, None)
    return preserve_total_mass(baseline_mass, corrected)


def _mae(values: np.ndarray) -> float:
    return float(np.mean(np.abs(values)))


def _fractional_mae(candidate: np.ndarray, truth: np.ndarray) -> float:
    truth_total = np.sum(truth, axis=(1, 2, 3))
    candidate_total = np.sum(candidate, axis=(1, 2, 3))
    return float(np.mean(np.abs(candidate_total - truth_total) / np.maximum(truth_total, 1.0)))


def _profile_mae(candidate_profile: np.ndarray, truth_profile: np.ndarray) -> float:
    return _mae(candidate_profile - truth_profile)


def _candidate_metrics(
    *,
    label: str,
    candidate: np.ndarray,
    truth: np.ndarray,
    metrics: dict[str, object],
) -> None:
    metrics[f"{label}_cell_mass_mae_msun"] = _mae(candidate - truth)
    metrics[f"{label}_total_mass_fractional_mae"] = _fractional_mae(candidate, truth)


def _profile_metrics(
    *,
    truth_mass: np.ndarray,
    candidates: dict[str, np.ndarray],
    phi_edges_rad: np.ndarray,
) -> dict[str, dict[str, float]]:
    truth_radial = radial_mass_profiles(truth_mass)
    truth_vertical = vertical_mass_profiles(truth_mass)
    truth_m2 = azimuthal_m2_amplitude_profiles(truth_mass, phi_edges_rad)
    result: dict[str, dict[str, float]] = {
        "test_radial_profile": {},
        "test_vertical_profile": {},
        "test_m2_profile": {},
    }
    for name, mass in candidates.items():
        result["test_radial_profile"][f"{name}_mae_msun"] = _profile_mae(
            radial_mass_profiles(mass),
            truth_radial,
        )
        result["test_vertical_profile"][f"{name}_mae_msun"] = _profile_mae(
            vertical_mass_profiles(mass),
            truth_vertical,
        )
        result["test_m2_profile"][f"{name}_mae"] = _profile_mae(
            azimuthal_m2_amplitude_profiles(mass, phi_edges_rad),
            truth_m2,
        )
    return result


def _stratified_metrics(
    *,
    truth_mass: np.ndarray,
    baseline_mass: np.ndarray,
    model_mass: np.ndarray,
    mean_train_mass: np.ndarray,
    labels: np.ndarray,
) -> dict[str, dict[str, dict[str, float | int]]]:
    result: dict[str, dict[str, dict[str, float | int]]] = {}
    for raw_label in np.unique(labels):
        mask = labels == raw_label
        key = f"{float(raw_label):g}"
        truth = truth_mass[mask]
        result[key] = {
            "n": int(np.sum(mask)),
            "baseline_cell_mass_mae_msun": _mae(baseline_mass[mask] - truth),
            "model_cell_mass_mae_msun": _mae(model_mass[mask] - truth),
            "mean_train_residual_cell_mass_mae_msun": _mae(mean_train_mass[mask] - truth),
        }
    return result


def compute_report(
    table: np.lib.npyio.NpzFile,
    pca: np.lib.npyio.NpzFile,
    predictions: np.lib.npyio.NpzFile,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    truth_mass, baseline_mass, delta_mass = _make_mass_grids(table)
    split = table["split"].astype(str)
    metadata = table["metadata"].astype(np.float32)
    predicted_delta_mass = predictions["predicted_delta_mass"].astype(np.float32)
    if predicted_delta_mass.shape != truth_mass.shape:
        raise ValueError("predicted_delta_mass shape must match the density table mass grid shape")
    train_mask = split == "train"
    test_mask = split == "test"
    if not np.any(train_mask):
        raise ValueError("mean train residual baseline requires at least one train row")
    if not np.any(test_mask):
        raise ValueError("report requires at least one test row")

    model_mass = corrected_mass_from_delta(baseline_mass, predicted_delta_mass)
    mean_train_delta_mass = np.mean(delta_mass[train_mask], axis=0)
    mean_train_mass = corrected_mass_from_delta(
        baseline_mass,
        np.broadcast_to(mean_train_delta_mass, baseline_mass.shape),
    )

    truth_test = truth_mass[test_mask]
    baseline_test = baseline_mass[test_mask]
    model_test = model_mass[test_mask]
    mean_train_test = mean_train_mass[test_mask]
    metrics: dict[str, object] = {
        "n_rows": int(len(split)),
        "n_test": int(np.sum(test_mask)),
        "n_test_galaxies": int(len(set(table["galaxy_id"][test_mask].astype(int).tolist()))),
        "pca_n_components": int(pca.get("explained_variance_ratio", np.array([])).shape[0]),
        "pca_cumulative_explained_variance": float(
            np.sum(pca.get("explained_variance_ratio", np.array([], dtype=np.float32)))
        ),
    }
    _candidate_metrics(label="test_baseline", candidate=baseline_test, truth=truth_test, metrics=metrics)
    _candidate_metrics(label="test_model", candidate=model_test, truth=truth_test, metrics=metrics)
    _candidate_metrics(
        label="test_mean_train_residual",
        candidate=mean_train_test,
        truth=truth_test,
        metrics=metrics,
    )
    baseline_mae = float(metrics["test_baseline_cell_mass_mae_msun"])
    metrics["test_model_cell_mass_mae_improvement_fraction"] = (
        (baseline_mae - float(metrics["test_model_cell_mass_mae_msun"])) / baseline_mae
        if baseline_mae > 0.0
        else 0.0
    )
    metrics["test_mean_train_cell_mass_mae_improvement_fraction"] = (
        (baseline_mae - float(metrics["test_mean_train_residual_cell_mass_mae_msun"]))
        / baseline_mae
        if baseline_mae > 0.0
        else 0.0
    )
    metrics["profile_metrics"] = _profile_metrics(
        truth_mass=truth_test,
        candidates={
            "baseline": baseline_test,
            "model": model_test,
            "mean_train_residual": mean_train_test,
        },
        phi_edges_rad=table["phi_edges_rad"],
    )
    metrics["stratified_metrics"] = {
        "inclination_deg": _stratified_metrics(
            truth_mass=truth_test,
            baseline_mass=baseline_test,
            model_mass=model_test,
            mean_train_mass=mean_train_test,
            labels=metadata[test_mask, 0],
        ),
        "bar_angle_deg": _stratified_metrics(
            truth_mass=truth_test,
            baseline_mass=baseline_test,
            model_mass=model_test,
            mean_train_mass=mean_train_test,
            labels=metadata[test_mask, 2],
        ),
    }
    arrays = {
        "truth_mass": truth_mass,
        "baseline_mass": baseline_mass,
        "model_mass": model_mass,
        "mean_train_mass": mean_train_mass,
        "test_indices": np.flatnonzero(test_mask),
        "galaxy_id": table["galaxy_id"],
        "projection_id": table["projection_id"],
        "metadata": metadata,
    }
    return metrics, arrays


def _recommendation(metrics: dict[str, object]) -> str:
    model_improvement = float(metrics["test_model_cell_mass_mae_improvement_fraction"])
    mean_improvement = float(metrics["test_mean_train_cell_mass_mae_improvement_fraction"])
    model_mae = float(metrics["test_model_cell_mass_mae_msun"])
    mean_mae = float(metrics["test_mean_train_residual_cell_mass_mae_msun"])
    total_mae = float(metrics["test_model_total_mass_fractional_mae"])
    if model_improvement > 0.1 and model_mae < mean_mae:
        return (
            "Proceed to a probabilistic residual model over PCA coefficients before scaling up "
            "the network. The deterministic model improves spatial mass placement beyond the "
            "mean train residual, while total mass remains unchanged by construction."
        )
    if mean_improvement > 0.1 and mean_mae <= model_mae:
        return (
            "Revise the feature or target scaling before a larger model. The mean train residual "
            "matches or beats the image-conditioned model, so the current features are not yet "
            "doing enough projection-specific work."
        )
    if total_mae > 0.05:
        return (
            "Keep the spatial residual experiment, but add a separate total-mass correction only "
            "if total stellar mass becomes a target. The current correction intentionally preserves "
            "baseline total mass."
        )
    return (
        "Stay with the compact deterministic setup for another diagnostic pass before using the "
        "cluster for a larger model."
    )


def write_markdown_report(path: Path, metrics: dict[str, object]) -> None:
    profile_metrics = metrics["profile_metrics"]
    recommendation = _recommendation(metrics)
    lines = [
        "# Milestone 2b Density Residual PCA Report",
        "",
        "## Held-Out Test Metrics",
        "",
        f"- Test projections: {metrics['n_test']}",
        f"- Test galaxies: {metrics['n_test_galaxies']}",
        f"- Baseline cell-mass MAE: {metrics['test_baseline_cell_mass_mae_msun']:.6g} Msun",
        f"- Corrected cell-mass MAE: {metrics['test_model_cell_mass_mae_msun']:.6g} Msun",
        "- Mean train residual cell-mass MAE: "
        f"{metrics['test_mean_train_residual_cell_mass_mae_msun']:.6g} Msun",
        "- Corrected improvement over baseline: "
        f"{100.0 * metrics['test_model_cell_mass_mae_improvement_fraction']:.2f} percent",
        "- Mean train residual improvement over baseline: "
        f"{100.0 * metrics['test_mean_train_cell_mass_mae_improvement_fraction']:.2f} percent",
        f"- Corrected total-mass fractional MAE: {metrics['test_model_total_mass_fractional_mae']:.6g}",
        "",
        "## Profile Diagnostics",
        "",
        "- Radial profile model MAE: "
        f"{profile_metrics['test_radial_profile']['model_mae_msun']:.6g} Msun",
        "- Vertical profile model MAE: "
        f"{profile_metrics['test_vertical_profile']['model_mae_msun']:.6g} Msun",
        f"- Bar-frame m=2 model MAE: {profile_metrics['test_m2_profile']['model_mae']:.6g}",
        "",
        "## How To Read The Figures",
        "",
        "Each example figure compares the same held-out projection in four mass maps: "
        "truth, baseline, PCA corrected, and mean train residual. The maps show "
        "log stellar mass in the bar-aligned cylindrical `R-phi` grid after summing "
        "over vertical bins; brighter cells contain more mass.",
        "",
        "The residual panels show candidate minus truth. A good correction should "
        "make the PCA corrected residual less structured and lower amplitude than "
        "the baseline residual. The mean train residual panel asks whether the "
        "model is doing more than applying an average correction learned from the "
        "training galaxies.",
        "",
        "## Recommendation",
        "",
        recommendation,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack("!I", len(data))
        + tag
        + data
        + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def _write_grayscale_png(path: Path, image: np.ndarray) -> None:
    clipped = np.asarray(image, dtype=np.uint8)
    height, width = clipped.shape
    raw_rows = b"".join(b"\x00" + clipped[row].tobytes() for row in range(height))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw_rows))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def _scale_to_uint8(image: np.ndarray, *, vmin: float, vmax: float) -> np.ndarray:
    if vmax <= vmin:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (np.asarray(image) - vmin) / (vmax - vmin)
    return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)


def _plot_example(path: Path, arrays: dict[str, np.ndarray], row_index: int) -> None:
    try:
        _plot_example_matplotlib(path, arrays, row_index)
        return
    except ImportError:
        pass
    _plot_example_fallback(path, arrays, row_index)


def _plot_example_matplotlib(path: Path, arrays: dict[str, np.ndarray], row_index: int) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    mass_maps = {
        "Truth": arrays["truth_mass"][row_index],
        "Baseline": arrays["baseline_mass"][row_index],
        "PCA corrected": arrays["model_mass"][row_index],
        "Mean train residual": arrays["mean_train_mass"][row_index],
    }
    truth_panel = np.sum(mass_maps["Truth"], axis=2)
    image_panels = {
        name: np.log10(np.maximum(np.sum(mass, axis=2), 1.0)) for name, mass in mass_maps.items()
    }
    residual_panels = {
        name: np.sum(mass, axis=2) - truth_panel
        for name, mass in mass_maps.items()
        if name != "Truth"
    }
    vmin = min(float(np.min(image)) for image in image_panels.values())
    vmax = max(float(np.max(image)) for image in image_panels.values())
    residual_limit = max(float(np.max(np.abs(panel))) for panel in residual_panels.values())
    residual_limit = max(residual_limit, 1.0)
    fig, axes = plt.subplots(2, 4, figsize=(10.5, 5.3), constrained_layout=True)
    for axis, (name, image) in zip(axes[0], image_panels.items(), strict=True):
        im = axis.imshow(image.T, origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
        axis.set_title(name)
        axis.set_xlabel("R bin")
        axis.set_ylabel("phi bin")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.86, label="log10 mass per R-phi cell")

    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.0,
        0.8,
        "Residual panels\n(candidate - truth)\n\nLess color structure\nmeans closer to truth.",
        va="top",
    )
    for axis, (name, residual) in zip(axes[1, 1:], residual_panels.items(), strict=True):
        im_resid = axis.imshow(
            residual.T,
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            vmin=-residual_limit,
            vmax=residual_limit,
        )
        axis.set_title(f"{name} residual")
        axis.set_xlabel("R bin")
        axis.set_ylabel("phi bin")
    fig.colorbar(im_resid, ax=axes[1, 1:].tolist(), shrink=0.86, label="Msun per R-phi cell")
    galaxy = int(arrays["galaxy_id"][row_index])
    projection = int(arrays["projection_id"][row_index])
    inc, _, bar_angle = arrays["metadata"][row_index]
    fig.suptitle(
        f"Held-out subhalo {galaxy}, projection {projection}: "
        f"inclination={inc:g} deg, bar angle={bar_angle:g} deg",
        y=1.02,
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_example_fallback(path: Path, arrays: dict[str, np.ndarray], row_index: int) -> None:
    names = [
        ("truth", arrays["truth_mass"][row_index]),
        ("baseline", arrays["baseline_mass"][row_index]),
        ("corrected", arrays["model_mass"][row_index]),
        ("mean train", arrays["mean_train_mass"][row_index]),
    ]
    images = [(name, np.log10(np.maximum(np.sum(mass, axis=2), 1.0))) for name, mass in names]
    vmin = min(float(np.min(image)) for _, image in images)
    vmax = max(float(np.max(image)) for _, image in images)
    panels = [_scale_to_uint8(image.T, vmin=vmin, vmax=vmax) for _, image in images]
    separator = np.full((panels[0].shape[0], 2), 255, dtype=np.uint8)
    tiled = np.concatenate(
        [item for panel in panels for item in (panel, separator)][:-1],
        axis=1,
    )
    _write_grayscale_png(path, tiled)


def write_figure_payload(output_dir: Path, arrays: dict[str, np.ndarray], selected: list[int]) -> Path:
    path = output_dir / "density_residual_pca_figure_payload.npz"
    np.savez_compressed(
        path,
        truth_mass=arrays["truth_mass"][selected].astype(np.float32),
        baseline_mass=arrays["baseline_mass"][selected].astype(np.float32),
        model_mass=arrays["model_mass"][selected].astype(np.float32),
        mean_train_mass=arrays["mean_train_mass"][selected].astype(np.float32),
        galaxy_id=arrays["galaxy_id"][selected],
        projection_id=arrays["projection_id"][selected],
        metadata=arrays["metadata"][selected].astype(np.float32),
    )
    return path


def write_figures(output_dir: Path, arrays: dict[str, np.ndarray], *, max_examples: int) -> list[str]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for existing in list(figure_dir.glob("test_example_*.png")) + list(
        figure_dir.glob("test_example_*.pdf")
    ):
        existing.unlink()
    written = []
    selected = []
    seen_galaxies = set()
    for row_index in arrays["test_indices"]:
        galaxy = int(arrays["galaxy_id"][row_index])
        if galaxy in seen_galaxies:
            continue
        selected.append(int(row_index))
        seen_galaxies.add(galaxy)
        if len(selected) >= max(0, max_examples):
            break
    write_figure_payload(output_dir, arrays, selected)
    for row_index in selected:
        galaxy = int(arrays["galaxy_id"][row_index])
        projection = int(arrays["projection_id"][row_index])
        path = figure_dir / f"test_example_subhalo_{galaxy}_projection_{projection}.png"
        _plot_example(path, arrays, int(row_index))
        written.append(str(path))
    return written


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (
        np.load(args.density_table) as table,
        np.load(args.pca) as pca,
        np.load(args.predictions) as predictions,
    ):
        metrics, arrays = compute_report(table, pca, predictions)
        metrics["figures"] = write_figures(args.output_dir, arrays, max_examples=args.max_examples)
    metrics_path = args.output_dir / "density_residual_pca_report_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_report(args.output_dir / "density_residual_pca_report.md", metrics)
    print(f"wrote density residual PCA report to {args.output_dir}")


if __name__ == "__main__":
    main()
