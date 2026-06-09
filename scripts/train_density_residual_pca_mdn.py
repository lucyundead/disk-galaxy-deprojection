from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from dgdp.models.mdn import SummaryResidualMDN
from train_density_residual_pca import (
    _choose_device,
    _make_batches,
    _mass_grids,
    _rmse,
    make_density_residual_features,
    reconstruct_delta_mass_from_coefficients,
    standardize_with_train,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca", type=Path, required=True)
    parser.add_argument("--density-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--image-feature-size", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--allow-total-mass-change", action="store_true")
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def _preserve_baseline_total_mass(
    baseline_mass: np.ndarray,
    corrected_mass: np.ndarray,
) -> np.ndarray:
    baseline_total = np.sum(baseline_mass, axis=(1, 2, 3))
    corrected_total = np.sum(corrected_mass, axis=(1, 2, 3))
    scale = np.divide(
        baseline_total,
        corrected_total,
        out=np.ones_like(baseline_total, dtype=np.float32),
        where=corrected_total > 0.0,
    )
    return (corrected_mass * scale[:, None, None, None]).astype(np.float32)


def corrected_mass_from_delta(
    baseline_mass: np.ndarray,
    delta_mass: np.ndarray,
    *,
    preserve_baseline_total_mass: bool,
) -> np.ndarray:
    corrected = np.clip(baseline_mass + delta_mass, 0.0, None)
    if preserve_baseline_total_mass:
        corrected = _preserve_baseline_total_mass(baseline_mass, corrected)
    return corrected.astype(np.float32)


def _mae(values: np.ndarray) -> float:
    return float(np.mean(np.abs(values)))


def _fractional_total_mass_mae(candidate: np.ndarray, truth: np.ndarray) -> float:
    truth_total = np.sum(truth, axis=(1, 2, 3))
    candidate_total = np.sum(candidate, axis=(1, 2, 3))
    return float(np.mean(np.abs(candidate_total - truth_total) / np.maximum(truth_total, 1.0)))


def _coefficient_coverage(
    samples: np.ndarray,
    truth: np.ndarray,
    *,
    lower_q: float = 0.16,
    upper_q: float = 0.84,
) -> float:
    lower = np.quantile(samples, lower_q, axis=1)
    upper = np.quantile(samples, upper_q, axis=1)
    inside = (truth >= lower) & (truth <= upper)
    return float(np.mean(inside))


def _add_split_metrics(
    metrics: dict[str, float | int | str | bool],
    *,
    label: str,
    mask: np.ndarray,
    truth_coefficients: np.ndarray,
    sampled_coefficients: np.ndarray,
    posterior_mean_coefficients: np.ndarray,
    truth_mass: np.ndarray,
    baseline_mass: np.ndarray,
    posterior_mean_mass: np.ndarray,
) -> None:
    metrics[f"n_{label}"] = int(np.sum(mask))
    if not np.any(mask):
        return
    metrics[f"{label}_posterior_mean_coeff_rmse"] = _rmse(
        posterior_mean_coefficients[mask] - truth_coefficients[mask]
    )
    metrics[f"{label}_coeff_coverage_68"] = _coefficient_coverage(
        sampled_coefficients[mask],
        truth_coefficients[mask],
    )
    metrics[f"{label}_baseline_cell_mass_mae_msun"] = _mae(baseline_mass[mask] - truth_mass[mask])
    metrics[f"{label}_posterior_mean_cell_mass_mae_msun"] = _mae(
        posterior_mean_mass[mask] - truth_mass[mask]
    )
    metrics[f"{label}_baseline_total_mass_fractional_mae"] = _fractional_total_mass_mae(
        baseline_mass[mask],
        truth_mass[mask],
    )
    metrics[f"{label}_posterior_mean_total_mass_fractional_mae"] = _fractional_total_mass_mae(
        posterior_mean_mass[mask],
        truth_mass[mask],
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = _choose_device(args.device)

    with np.load(args.pca) as pca, np.load(args.density_table) as table:
        split = pca["split"].astype(str)
        train_mask = split == "train"
        val_mask = split == "val"
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

        model = SummaryResidualMDN(
            input_dim=x_all.shape[1],
            output_dim=coefficients.shape[1],
            hidden_dim=args.hidden_dim,
            n_components=args.n_components,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        x_tensor = torch.tensor(x_all, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y_all, dtype=torch.float32, device=device)
        train_indices = np.flatnonzero(train_mask)
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
                loss = model.negative_log_likelihood(x_tensor[batch], y_tensor[batch])
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            model.eval()
            with torch.no_grad():
                eval_indices = val_indices if len(val_indices) else train_indices
                val_nll = float(
                    model.negative_log_likelihood(
                        x_tensor[eval_indices],
                        y_tensor[eval_indices],
                    ).item()
                )
            if val_nll < best_val:
                best_val = val_nll
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epoch == 0 or epoch == args.epochs - 1:
                print(
                    f"epoch={epoch + 1} "
                    f"train_nll={float(np.mean(losses)):.6f} "
                    f"val_nll={val_nll:.6f}"
                )
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            sampled_z = model.sample(x_tensor, n_samples=args.n_samples).cpu().numpy()
        sampled_coefficients = sampled_z * y_scale[None, None, :] + y_mean[None, None, :]
        posterior_mean_coefficients = sampled_coefficients.mean(axis=1).astype(np.float32)

        truth_mass, baseline_mass, _ = _mass_grids(table)
        grid_shape = tuple(int(x) for x in truth_mass.shape[1:])
        posterior_mean_delta_mass = reconstruct_delta_mass_from_coefficients(
            posterior_mean_coefficients,
            components=pca["components"].astype(np.float32),
            mean=pca["mean"].astype(np.float32),
            scale_msun=table["baseline_grid_mass_msun"].astype(np.float32),
            grid_shape=grid_shape,
        )
        posterior_mean_mass = corrected_mass_from_delta(
            baseline_mass,
            posterior_mean_delta_mass,
            preserve_baseline_total_mass=not args.allow_total_mass_change,
        )
        metrics: dict[str, float | int | str | bool] = {
            "device": str(device),
            "image_feature_size": int(args.image_feature_size),
            "hidden_dim": int(args.hidden_dim),
            "epochs": int(args.epochs),
            "best_epoch": int(best_epoch),
            "best_val_nll": float(best_val),
            "n_components": int(args.n_components),
            "n_samples": int(args.n_samples),
            "weight_decay": float(args.weight_decay),
            "patience": int(args.patience),
            "preserve_baseline_total_mass": bool(not args.allow_total_mass_change),
        }
        for label in ("train", "val", "test"):
            _add_split_metrics(
                metrics,
                label=label,
                mask=split == label,
                truth_coefficients=coefficients,
                sampled_coefficients=sampled_coefficients,
                posterior_mean_coefficients=posterior_mean_coefficients,
                truth_mass=truth_mass,
                baseline_mass=baseline_mass,
                posterior_mean_mass=posterior_mean_mass,
            )

        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": x_all.shape[1],
                "output_dim": coefficients.shape[1],
                "hidden_dim": args.hidden_dim,
                "n_components": args.n_components,
                "image_feature_size": args.image_feature_size,
            },
            args.output_dir / "density_residual_pca_mdn.pt",
        )
        np.savez_compressed(
            args.output_dir / "density_residual_pca_mdn_normalization.npz",
            x_mean=x_mean,
            x_scale=x_scale,
            y_mean=y_mean,
            y_scale=y_scale,
        )
        np.savez_compressed(
            args.output_dir / "density_residual_pca_mdn_predictions.npz",
            sampled_coefficients=sampled_coefficients.astype(np.float32),
            posterior_mean_coefficients=posterior_mean_coefficients.astype(np.float32),
            true_coefficients=coefficients.astype(np.float32),
            posterior_mean_delta_mass=posterior_mean_delta_mass.astype(np.float32),
            posterior_mean_mass=posterior_mean_mass.astype(np.float32),
            split=split,
            galaxy_id=pca["galaxy_id"],
            projection_id=pca["projection_id"],
            metadata=pca["metadata"],
        )

    (args.output_dir / "density_residual_pca_mdn_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"best_epoch={best_epoch} best_val_nll={best_val:.6f}")
    print(
        "wrote probabilistic density residual PCA model to "
        f"{args.output_dir / 'density_residual_pca_mdn.pt'}"
    )


if __name__ == "__main__":
    main()
