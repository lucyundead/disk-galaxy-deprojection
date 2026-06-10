from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca", type=Path, required=True)
    parser.add_argument("--density-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--image-feature-size", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--allow-total-mass-change", action="store_true")
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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


def average_pool_images(images: np.ndarray, *, output_size: int) -> np.ndarray:
    images = np.asarray(images, dtype=np.float32)
    if images.ndim != 3:
        raise ValueError("images must have shape (n, height, width)")
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    height, width = images.shape[1:]
    if height % output_size != 0 or width % output_size != 0:
        raise ValueError("image dimensions must be divisible by output_size")
    y_block = height // output_size
    x_block = width // output_size
    return images.reshape(images.shape[0], output_size, y_block, output_size, x_block).mean(
        axis=(2, 4)
    )


def make_central_image_features(
    images: np.ndarray,
    inclinations_deg: np.ndarray,
    *,
    pixel_scale_kpc: float,
    aperture_semi_major_kpc: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
    min_axis_ratio: float = 0.2,
) -> np.ndarray:
    """Inclination-aware central flux features.

    Image axis 0 is the apparent minor axis (inclination compresses sky y by
    cos i in the projection convention of dgdp.projection), axis 1 the major
    axis. Elliptical apertures use semi-major axis a along the major axis and
    semi-minor a*cos(i) along the minor axis.
    """
    images = np.asarray(images, dtype=np.float32)
    n_rows, n_minor, n_major = images.shape
    minor_kpc = (np.arange(n_minor, dtype=np.float32) - 0.5 * (n_minor - 1)) * pixel_scale_kpc
    major_kpc = (np.arange(n_major, dtype=np.float32) - 0.5 * (n_major - 1)) * pixel_scale_kpc
    major_sq_grid = np.broadcast_to(major_kpc[None, :] ** 2, (n_minor, n_major))
    minor_sq_grid = np.broadcast_to(minor_kpc[:, None] ** 2, (n_minor, n_major))
    total_flux = np.maximum(images.sum(axis=(1, 2)), 1.0e-12)
    axis_ratios = np.maximum(
        np.cos(np.deg2rad(np.asarray(inclinations_deg, dtype=np.float32))),
        min_axis_ratio,
    )
    largest_aperture = max(aperture_semi_major_kpc)
    fractions = np.zeros((n_rows, len(aperture_semi_major_kpc)), dtype=np.float32)
    shape_ratio = np.zeros(n_rows, dtype=np.float32)
    for ratio in np.unique(axis_ratios):
        rows = np.flatnonzero(np.isclose(axis_ratios, ratio))
        elliptical_radius = np.sqrt(major_sq_grid + minor_sq_grid / ratio**2)
        for column, semi_major in enumerate(aperture_semi_major_kpc):
            mask = elliptical_radius <= semi_major
            fractions[rows, column] = images[rows][:, mask].sum(axis=1) / total_flux[rows]
        central_mask = elliptical_radius <= largest_aperture
        central_flux = images[rows][:, central_mask]
        flux_sum = np.maximum(central_flux.sum(axis=1), 1.0e-12)
        major_moment = (central_flux * major_sq_grid[central_mask][None, :]).sum(axis=1) / flux_sum
        minor_moment = (central_flux * minor_sq_grid[central_mask][None, :]).sum(axis=1) / flux_sum
        shape_ratio[rows] = np.sqrt(minor_moment / np.maximum(major_moment, 1.0e-12))
    return np.column_stack(
        (
            np.log10(np.maximum(fractions, 1.0e-4)),
            shape_ratio,
        )
    ).astype(np.float32)


def make_density_residual_features(
    images: np.ndarray,
    metadata: np.ndarray,
    *,
    baseline_grid_mass_msun: np.ndarray,
    image_feature_size: int,
    central_pixel_scale_kpc: float | None = None,
) -> np.ndarray:
    image_mass = np.sum(images, axis=(1, 2))
    normalized_images = np.divide(
        images,
        np.maximum(image_mass[:, None, None], 1.0),
        out=np.zeros_like(images, dtype=np.float32),
    )
    pooled = average_pool_images(np.log1p(normalized_images * images.shape[1] * images.shape[2]), output_size=image_feature_size)
    mass_features = np.column_stack(
        (
            np.log10(np.maximum(image_mass, 1.0)),
            np.log10(np.maximum(baseline_grid_mass_msun, 1.0)),
        )
    ).astype(np.float32)
    blocks = [
        pooled.reshape(pooled.shape[0], -1),
        np.asarray(metadata, dtype=np.float32),
        mass_features,
    ]
    if central_pixel_scale_kpc is not None:
        blocks.append(
            make_central_image_features(
                images,
                np.asarray(metadata, dtype=np.float32)[:, 0],
                pixel_scale_kpc=central_pixel_scale_kpc,
            )
        )
    return np.concatenate(blocks, axis=1).astype(np.float32)


def standardize_with_train(
    values: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values[train_mask].mean(axis=0)
    scale = values[train_mask].std(axis=0)
    scale = np.where(scale > 1.0e-6, scale, 1.0)
    return ((values - mean) / scale).astype(np.float32), mean.astype(np.float32), scale.astype(np.float32)


class DensityResidualMLP(nn.Module):
    def __init__(self, *, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _make_batches(indices: np.ndarray, *, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    shuffled = np.array(indices, copy=True)
    rng.shuffle(shuffled)
    return [shuffled[start : start + batch_size] for start in range(0, len(shuffled), batch_size)]


def _mass_grids(table: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    volumes = cylindrical_bin_volumes_from_edges(
        table["r_edges_kpc"],
        table["phi_edges_rad"],
        table["z_edges_kpc"],
    ).astype(np.float32)
    truth_mass = table["truth_density"].astype(np.float32) * volumes[None, ...]
    baseline_mass = table["baseline_density"].astype(np.float32) * volumes[None, ...]
    return truth_mass, baseline_mass, volumes


def reconstruct_delta_mass_from_coefficients(
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


def _mae(values: np.ndarray) -> float:
    return float(np.mean(np.abs(values)))


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def compute_mass_metrics(
    *,
    truth_mass: np.ndarray,
    baseline_mass: np.ndarray,
    predicted_delta_mass: np.ndarray,
    split: np.ndarray,
    preserve_baseline_total_mass: bool,
) -> dict[str, float | int]:
    corrected_mass = np.clip(baseline_mass + predicted_delta_mass, 0.0, None)
    if preserve_baseline_total_mass:
        baseline_total_all = np.sum(baseline_mass, axis=(1, 2, 3))
        corrected_total_all = np.sum(corrected_mass, axis=(1, 2, 3))
        scale = np.divide(
            baseline_total_all,
            corrected_total_all,
            out=np.ones_like(baseline_total_all, dtype=np.float32),
            where=corrected_total_all > 0.0,
        )
        corrected_mass = corrected_mass * scale[:, None, None, None]
    metrics: dict[str, float | int] = {}
    for label in ("train", "val", "test"):
        mask = split == label
        metrics[f"n_{label}"] = int(np.sum(mask))
        if not np.any(mask):
            continue
        baseline_error = baseline_mass[mask] - truth_mass[mask]
        corrected_error = corrected_mass[mask] - truth_mass[mask]
        truth_total = np.sum(truth_mass[mask], axis=(1, 2, 3))
        baseline_total = np.sum(baseline_mass[mask], axis=(1, 2, 3))
        corrected_total = np.sum(corrected_mass[mask], axis=(1, 2, 3))
        metrics[f"{label}_baseline_cell_mass_mae_msun"] = _mae(baseline_error)
        metrics[f"{label}_corrected_cell_mass_mae_msun"] = _mae(corrected_error)
        metrics[f"{label}_baseline_total_mass_fractional_mae"] = float(
            np.mean(np.abs(baseline_total - truth_total) / np.maximum(truth_total, 1.0))
        )
        metrics[f"{label}_corrected_total_mass_fractional_mae"] = float(
            np.mean(np.abs(corrected_total - truth_total) / np.maximum(truth_total, 1.0))
        )
    return metrics


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = _choose_device(args.device)

    with np.load(args.pca) as pca, np.load(args.density_table) as table:
        split = pca["split"].astype(str)
        train_mask = split == "train"
        if int(np.sum(train_mask)) < 1:
            raise ValueError("training requires at least one train row")
        features = make_density_residual_features(
            table["images"].astype(np.float32),
            pca["metadata"].astype(np.float32),
            baseline_grid_mass_msun=table["baseline_grid_mass_msun"].astype(np.float32),
            image_feature_size=args.image_feature_size,
        )
        coefficients = pca["coefficients"].astype(np.float32)
        x_all, x_mean, x_scale = standardize_with_train(features, train_mask)
        y_all, y_mean, y_scale = standardize_with_train(coefficients, train_mask)

        model = DensityResidualMLP(
            input_dim=x_all.shape[1],
            output_dim=coefficients.shape[1],
            hidden_dim=args.hidden_dim,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        loss_fn = nn.MSELoss()
        train_indices = np.flatnonzero(train_mask)
        x_tensor = torch.tensor(x_all, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y_all, dtype=torch.float32, device=device)

        val_mask = split == "val"
        val_indices = np.flatnonzero(val_mask)
        best_epoch = 0
        best_val = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
        for epoch in range(args.epochs):
            model.train()
            losses = []
            for batch in _make_batches(train_indices, batch_size=args.batch_size, rng=rng):
                optimizer.zero_grad()
                prediction = model(x_tensor[batch])
                loss = loss_fn(prediction, y_tensor[batch])
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            model.eval()
            with torch.no_grad():
                eval_indices = val_indices if len(val_indices) else train_indices
                val_loss = float(loss_fn(model(x_tensor[eval_indices]), y_tensor[eval_indices]).item())
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epoch == 0 or epoch == args.epochs - 1:
                print(
                    f"epoch={epoch + 1} "
                    f"train_mse={float(np.mean(losses)):.6f} "
                    f"val_mse={val_loss:.6f}"
                )
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                break

        model.eval()
        model.load_state_dict(best_state)
        with torch.no_grad():
            pred_z = model(x_tensor).cpu().numpy()
        pred_coefficients = pred_z * y_scale[None, :] + y_mean[None, :]
        coeff_mean_baseline = np.zeros_like(coefficients)

        truth_mass, baseline_mass, _ = _mass_grids(table)
        grid_shape = tuple(int(x) for x in truth_mass.shape[1:])
        predicted_delta_mass = reconstruct_delta_mass_from_coefficients(
            pred_coefficients,
            components=pca["components"].astype(np.float32),
            mean=pca["mean"].astype(np.float32),
            scale_msun=table["baseline_grid_mass_msun"].astype(np.float32),
            grid_shape=grid_shape,
        )
        metrics = compute_mass_metrics(
            truth_mass=truth_mass,
            baseline_mass=baseline_mass,
            predicted_delta_mass=predicted_delta_mass,
            split=split,
            preserve_baseline_total_mass=not args.allow_total_mass_change,
        )
        for label in ("train", "val", "test"):
            mask = split == label
            if np.any(mask):
                metrics[f"{label}_mean_coeff_rmse"] = _rmse(coeff_mean_baseline[mask] - coefficients[mask])
                metrics[f"{label}_model_coeff_rmse"] = _rmse(pred_coefficients[mask] - coefficients[mask])
        metrics["device"] = str(device)
        metrics["image_feature_size"] = int(args.image_feature_size)
        metrics["hidden_dim"] = int(args.hidden_dim)
        metrics["epochs"] = int(args.epochs)
        metrics["best_epoch"] = int(best_epoch)
        metrics["best_val_mse"] = float(best_val)
        metrics["weight_decay"] = float(args.weight_decay)
        metrics["patience"] = int(args.patience)
        metrics["preserve_baseline_total_mass"] = bool(not args.allow_total_mass_change)

        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": x_all.shape[1],
                "output_dim": coefficients.shape[1],
                "hidden_dim": args.hidden_dim,
                "image_feature_size": args.image_feature_size,
            },
            args.output_dir / "density_residual_pca_mlp.pt",
        )
        np.savez_compressed(
            args.output_dir / "density_residual_pca_normalization.npz",
            x_mean=x_mean,
            x_scale=x_scale,
            y_mean=y_mean,
            y_scale=y_scale,
        )
        np.savez_compressed(
            args.output_dir / "density_residual_pca_predictions.npz",
            predicted_coefficients=pred_coefficients.astype(np.float32),
            true_coefficients=coefficients.astype(np.float32),
            predicted_delta_mass=predicted_delta_mass.astype(np.float32),
            split=split,
            galaxy_id=pca["galaxy_id"],
            projection_id=pca["projection_id"],
            metadata=pca["metadata"],
        )

    (args.output_dir / "density_residual_pca_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote density residual PCA model to {args.output_dir / 'density_residual_pca_mlp.pt'}")


if __name__ == "__main__":
    main()
