from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-components", type=int, default=32)
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


def _mae(values: np.ndarray) -> float:
    return float(np.mean(np.abs(values)))


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _split_counts(split: np.ndarray) -> dict[str, int]:
    labels, counts = np.unique(split.astype(str), return_counts=True)
    return {str(label): int(count) for label, count in zip(labels, counts, strict=True)}


def _group_mae(values: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    result = {}
    for label in np.unique(labels):
        mask = labels == label
        key = f"{float(label):g}"
        result[key] = _mae(values[mask])
    return result


def _make_mass_grids(table: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    volumes = cylindrical_bin_volumes_from_edges(
        table["r_edges_kpc"],
        table["phi_edges_rad"],
        table["z_edges_kpc"],
    ).astype(np.float32)
    truth_mass = table["truth_density"].astype(np.float32) * volumes[None, ...]
    baseline_mass = table["baseline_density"].astype(np.float32) * volumes[None, ...]
    delta_mass = table["delta_density"].astype(np.float32) * volumes[None, ...]
    return truth_mass, baseline_mass, delta_mass, volumes


def compute_diagnostics(table: np.lib.npyio.NpzFile) -> dict[str, object]:
    truth_mass, baseline_mass, delta_mass, _ = _make_mass_grids(table)
    metadata = table["metadata"]
    split = table["split"].astype(str)
    truth_total = np.sum(truth_mass, axis=(1, 2, 3))
    baseline_total = np.sum(baseline_mass, axis=(1, 2, 3))
    delta_total = np.sum(delta_mass, axis=(1, 2, 3))
    radial_delta = radial_mass_profiles(delta_mass)
    vertical_delta = vertical_mass_profiles(delta_mass)
    truth_m2 = azimuthal_m2_amplitude_profiles(truth_mass, table["phi_edges_rad"])
    baseline_m2 = azimuthal_m2_amplitude_profiles(baseline_mass, table["phi_edges_rad"])
    per_row_delta_mae = np.mean(np.abs(delta_mass), axis=(1, 2, 3))
    fractional_mass_mae = float(
        np.mean(np.abs(baseline_total - truth_total) / np.maximum(truth_total, 1.0))
    )
    mass_scale_ok = fractional_mass_mae < 0.25
    return {
        "n_rows": int(len(table["galaxy_id"])),
        "n_galaxies": int(len(set(table["galaxy_id"].astype(int).tolist()))),
        "density_shape": [int(x) for x in truth_mass.shape[1:]],
        "image_shape": [int(x) for x in table["images"].shape[1:]],
        "split_counts": _split_counts(split),
        "baseline_total_mass_mae_msun": _mae(baseline_total - truth_total),
        "baseline_total_mass_bias_msun": float(np.mean(baseline_total - truth_total)),
        "baseline_total_mass_fractional_mae": fractional_mass_mae,
        "delta_grid_mass_mae_msun": _mae(delta_total),
        "delta_cell_mass_mae_msun": _mae(delta_mass),
        "radial_delta_profile_mae_msun": _mae(radial_delta),
        "vertical_delta_profile_mae_msun": _mae(vertical_delta),
        "m2_amplitude_mae": _mae(baseline_m2 - truth_m2),
        "grouped_baseline_delta_mae_msun": {
            "inclination_deg": _group_mae(per_row_delta_mae, metadata[:, 0]),
            "bar_angle_deg": _group_mae(per_row_delta_mae, metadata[:, 2]),
        },
        "training_readiness": {
            "mass_scale_ok": bool(mass_scale_ok),
            "recommendation": (
                "ok_for_compressed_residual_experiment"
                if mass_scale_ok
                else "fix_image_truth_mass_contract_before_training"
            ),
        },
    }


def fit_delta_mass_fraction_pca(
    table: np.lib.npyio.NpzFile,
    *,
    n_components: int,
) -> dict[str, np.ndarray]:
    _, _, delta_mass, _ = _make_mass_grids(table)
    truth_mass = np.asarray(table["truth_grid_mass_msun"], dtype=np.float32)
    target = delta_mass.reshape(delta_mass.shape[0], -1) / np.maximum(truth_mass[:, None], 1.0)
    split = table["split"].astype(str)
    train_mask = split == "train"
    if int(np.sum(train_mask)) < 2:
        raise ValueError("PCA compression requires at least two train rows")
    train_target = target[train_mask]
    mean = np.mean(train_target, axis=0)
    centered_train = train_target - mean[None, :]
    _, singular_values, vt = np.linalg.svd(centered_train, full_matrices=False)
    max_components = min(int(n_components), vt.shape[0])
    components = vt[:max_components].astype(np.float32)
    centered_all = target - mean[None, :]
    coefficients = centered_all @ components.T
    train_variance = np.sum(np.square(centered_train))
    explained = np.square(singular_values[:max_components])
    explained_ratio = np.divide(
        explained,
        train_variance,
        out=np.zeros_like(explained, dtype=float),
        where=train_variance > 0.0,
    )
    reconstruction = mean[None, :] + coefficients @ components
    residual = target - reconstruction
    return {
        "coefficients": coefficients.astype(np.float32),
        "components": components,
        "mean": mean.astype(np.float32),
        "explained_variance_ratio": explained_ratio.astype(np.float32),
        "target": target.astype(np.float32),
        "reconstruction_rel_rmse": np.array(
            _rmse(residual) / max(_rmse(target), 1.0e-12),
            dtype=np.float32,
        ),
        "fit_split": np.array("train"),
        "n_train": np.array(int(np.sum(train_mask)), dtype=np.int32),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.data) as table:
        diagnostics = compute_diagnostics(table)
        pca = fit_delta_mass_fraction_pca(table, n_components=args.n_components)
        extra = {
            "pca_n_components": int(pca["components"].shape[0]),
            "pca_cumulative_explained_variance": float(np.sum(pca["explained_variance_ratio"])),
            "pca_reconstruction_rel_rmse": float(pca["reconstruction_rel_rmse"]),
        }
        diagnostics = {**diagnostics, **extra}
        np.savez_compressed(
            args.output_dir / "density_residual_pca.npz",
            **pca,
            r_edges_kpc=table["r_edges_kpc"],
            phi_edges_rad=table["phi_edges_rad"],
            z_edges_kpc=table["z_edges_kpc"],
            split=table["split"],
            galaxy_id=table["galaxy_id"],
            projection_id=table["projection_id"],
            metadata=table["metadata"],
        )
    (args.output_dir / "density_residual_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote density residual diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
