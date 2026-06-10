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
    make_density_residual_features,
    standardize_with_train,
)


DEFAULT_DENSITY_TABLE = Path("outputs/tng50_milestone2b/density_residual_table.npz")
DEFAULT_OUTPUT_DIR = Path("outputs/tng50_milestone2b/total_mass_correction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--density-table", type=Path, default=DEFAULT_DENSITY_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--image-feature-size", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-components", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def _coverage_68(samples: np.ndarray, truth: np.ndarray) -> float:
    lower = np.quantile(samples, 0.16, axis=1)
    upper = np.quantile(samples, 0.84, axis=1)
    return float(np.mean((truth >= lower) & (truth <= upper)))


def _split_metrics(
    *,
    mask: np.ndarray,
    true_log_ratio: np.ndarray,
    mean_log_ratio: np.ndarray,
    sampled_log_ratio: np.ndarray,
    truth_total_msun: np.ndarray,
    baseline_total_msun: np.ndarray,
) -> dict[str, float | int]:
    corrected_total = baseline_total_msun[mask] * np.exp(mean_log_ratio[mask])
    truth_total = truth_total_msun[mask]
    return {
        "n_rows": int(np.sum(mask)),
        "log_ratio_mae": float(np.mean(np.abs(mean_log_ratio[mask] - true_log_ratio[mask]))),
        "log_ratio_coverage_68": _coverage_68(sampled_log_ratio[mask], true_log_ratio[mask]),
        "baseline_total_mass_fractional_mae": float(
            np.mean(np.abs(baseline_total_msun[mask] - truth_total) / truth_total)
        ),
        "corrected_total_mass_fractional_mae": float(
            np.mean(np.abs(corrected_total - truth_total) / truth_total)
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = _choose_device(args.device)

    with np.load(args.density_table) as table:
        split = table["split"].astype(str)
        train_mask = split == "train"
        val_mask = split == "val"
        if int(np.sum(train_mask)) < 1:
            raise ValueError("training requires at least one train row")
        truth_total_msun = table["truth_grid_mass_msun"].astype(np.float64)
        baseline_total_msun = table["baseline_grid_mass_msun"].astype(np.float64)
        true_log_ratio = np.log(truth_total_msun / baseline_total_msun).astype(np.float32)
        features = make_density_residual_features(
            table["images"].astype(np.float32),
            table["metadata"].astype(np.float32),
            baseline_grid_mass_msun=table["baseline_grid_mass_msun"].astype(np.float32),
            image_feature_size=args.image_feature_size,
        )
        x_all, x_mean, x_scale = standardize_with_train(features, train_mask)
        y_all, y_mean, y_scale = standardize_with_train(
            true_log_ratio[:, None],
            train_mask,
        )

        model = SummaryResidualMDN(
            input_dim=x_all.shape[1],
            output_dim=1,
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
        sampled_log_ratio = (sampled_z * y_scale[None, None, :] + y_mean[None, None, :])[:, :, 0]
        mean_log_ratio = sampled_log_ratio.mean(axis=1).astype(np.float32)

        metrics: dict[str, float | int | str | dict[str, float | int]] = {
            "device": str(device),
            "image_feature_size": int(args.image_feature_size),
            "hidden_dim": int(args.hidden_dim),
            "epochs": int(args.epochs),
            "best_epoch": int(best_epoch),
            "best_val_nll": float(best_val),
            "n_components": int(args.n_components),
            "n_samples": int(args.n_samples),
            "seed": int(args.seed),
        }
        for label in ("train", "val", "test"):
            mask = split == label
            if not np.any(mask):
                metrics[label] = {"n_rows": 0}
                continue
            metrics[label] = _split_metrics(
                mask=mask,
                true_log_ratio=true_log_ratio,
                mean_log_ratio=mean_log_ratio,
                sampled_log_ratio=sampled_log_ratio,
                truth_total_msun=truth_total_msun,
                baseline_total_msun=baseline_total_msun,
            )

        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": x_all.shape[1],
                "output_dim": 1,
                "hidden_dim": args.hidden_dim,
                "n_components": args.n_components,
                "image_feature_size": args.image_feature_size,
            },
            args.output_dir / "total_mass_correction.pt",
        )
        np.savez_compressed(
            args.output_dir / "total_mass_correction_normalization.npz",
            x_mean=x_mean,
            x_scale=x_scale,
            y_mean=y_mean,
            y_scale=y_scale,
        )
        np.savez_compressed(
            args.output_dir / "total_mass_correction_predictions.npz",
            sampled_log_ratio=sampled_log_ratio.astype(np.float32),
            mean_log_ratio=mean_log_ratio,
            true_log_ratio=true_log_ratio,
            split=split,
            galaxy_id=table["galaxy_id"],
            projection_id=table["projection_id"],
            metadata=table["metadata"],
        )

    (args.output_dir / "total_mass_correction_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"best_epoch={best_epoch} best_val_nll={best_val:.6f}")
    print(f"wrote total mass correction model to {args.output_dir / 'total_mass_correction.pt'}")


if __name__ == "__main__":
    main()
