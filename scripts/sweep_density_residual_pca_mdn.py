from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


TARGET_COVERAGE = 0.68


def _value_key(value: float) -> str:
    return f"{value:g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca", type=Path, required=True)
    parser.add_argument("--density-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260608, 20260609, 20260610])
    parser.add_argument("--n-components", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--image-feature-size", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _coverage_counts(
    samples: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    *,
    scale: float = 1.0,
) -> tuple[int, int]:
    if not np.any(mask):
        return 0, 0
    masked_samples = samples[mask]
    center = masked_samples.mean(axis=1, keepdims=True)
    scaled_samples = center + scale * (masked_samples - center)
    lower = np.quantile(scaled_samples, 0.16, axis=1)
    upper = np.quantile(scaled_samples, 0.84, axis=1)
    inside = (truth[mask] >= lower) & (truth[mask] <= upper)
    return int(np.sum(inside)), int(inside.size)


def _coverage(samples: np.ndarray, truth: np.ndarray, mask: np.ndarray, *, scale: float = 1.0) -> float:
    covered, total = _coverage_counts(samples, truth, mask, scale=scale)
    if total == 0:
        return float("nan")
    return covered / total


def _select_temperature_scale(samples: np.ndarray, truth: np.ndarray, val_mask: np.ndarray) -> tuple[float, float]:
    if not np.any(val_mask):
        return 1.0, _coverage(samples, truth, val_mask, scale=1.0)
    candidates = np.linspace(0.25, 2.0, 36)
    scored = [
        (float(scale), _coverage(samples, truth, val_mask, scale=float(scale)))
        for scale in candidates
    ]
    return min(scored, key=lambda item: (abs(item[1] - TARGET_COVERAGE), item[0]))


def _coverage_by_value(
    samples: np.ndarray,
    truth: np.ndarray,
    split: np.ndarray,
    values: np.ndarray,
    *,
    scale: float,
) -> dict[str, float]:
    output = {}
    for value in sorted(float(x) for x in np.unique(values[split == "test"])):
        mask = (split == "test") & np.isclose(values, value)
        output[_value_key(value)] = _coverage(samples, truth, mask, scale=scale)
    return output


def _coverage_with_value_scales(
    samples: np.ndarray,
    truth: np.ndarray,
    split: np.ndarray,
    values: np.ndarray,
    *,
    split_label: str,
    scale_by_value: dict[str, float],
    fallback_scale: float,
) -> float:
    covered = 0
    total = 0
    for value in sorted(float(x) for x in np.unique(values[split == split_label])):
        mask = (split == split_label) & np.isclose(values, value)
        scale = scale_by_value.get(_value_key(value), fallback_scale)
        value_covered, value_total = _coverage_counts(samples, truth, mask, scale=scale)
        covered += value_covered
        total += value_total
    if total == 0:
        return float("nan")
    return covered / total


def _temperature_by_value(
    samples: np.ndarray,
    truth: np.ndarray,
    split: np.ndarray,
    values: np.ndarray,
    *,
    fallback_scale: float,
) -> dict[str, dict[str, float | int]]:
    output = {}
    all_values = sorted(float(x) for x in np.unique(values[(split == "val") | (split == "test")]))
    for value in all_values:
        key = _value_key(value)
        val_mask = (split == "val") & np.isclose(values, value)
        test_mask = (split == "test") & np.isclose(values, value)
        if np.any(val_mask):
            scale, val_coverage = _select_temperature_scale(samples, truth, val_mask)
        else:
            scale = fallback_scale
            val_coverage = float("nan")
        output[key] = {
            "temperature_scale": scale,
            "val_coeff_coverage_68": val_coverage,
            "test_coeff_coverage_68": _coverage(samples, truth, test_mask, scale=scale),
            "n_val": int(np.sum(val_mask)),
            "n_test": int(np.sum(test_mask)),
        }
    return output


def _calibration_metrics(predictions_path: Path) -> dict[str, Any]:
    with np.load(predictions_path) as predictions:
        samples = predictions["sampled_coefficients"].astype(np.float32)
        truth = predictions["true_coefficients"].astype(np.float32)
        split = predictions["split"].astype(str)
        metadata = predictions["metadata"].astype(np.float32)

    val_mask = split == "val"
    test_mask = split == "test"
    scale, val_coverage = _select_temperature_scale(samples, truth, val_mask)
    test_coverage = _coverage(samples, truth, test_mask, scale=scale)
    raw_test_coverage = _coverage(samples, truth, test_mask, scale=1.0)
    inclination_temperature = _temperature_by_value(
        samples,
        truth,
        split,
        metadata[:, 0],
        fallback_scale=scale,
    )
    inclination_scales = {
        key: float(value["temperature_scale"])
        for key, value in inclination_temperature.items()
    }
    return {
        "temperature_scale_from_val": scale,
        "temperature_val_coeff_coverage_68": val_coverage,
        "temperature_test_coeff_coverage_68": test_coverage,
        "raw_test_coeff_coverage_68_from_predictions": raw_test_coverage,
        "inclination_temperature_val_coeff_coverage_68": _coverage_with_value_scales(
            samples,
            truth,
            split,
            metadata[:, 0],
            split_label="val",
            scale_by_value=inclination_scales,
            fallback_scale=scale,
        ),
        "inclination_temperature_test_coeff_coverage_68": _coverage_with_value_scales(
            samples,
            truth,
            split,
            metadata[:, 0],
            split_label="test",
            scale_by_value=inclination_scales,
            fallback_scale=scale,
        ),
        "inclination_temperature_by_inclination": inclination_temperature,
        "by_inclination": _coverage_by_value(samples, truth, split, metadata[:, 0], scale=scale),
        "by_bar_angle": _coverage_by_value(samples, truth, split, metadata[:, 2], scale=scale),
    }


def _run_training(args: argparse.Namespace, *, seed: int, n_components: int, run_dir: Path) -> None:
    if args.skip_existing and (run_dir / "density_residual_pca_mdn_metrics.json").exists():
        return
    command = [
        sys.executable,
        "scripts/train_density_residual_pca_mdn.py",
        "--pca",
        str(args.pca),
        "--density-table",
        str(args.density_table),
        "--output-dir",
        str(run_dir),
        "--epochs",
        str(args.epochs),
        "--hidden-dim",
        str(args.hidden_dim),
        "--image-feature-size",
        str(args.image_feature_size),
        "--batch-size",
        str(args.batch_size),
        "--n-components",
        str(n_components),
        "--n-samples",
        str(args.n_samples),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--patience",
        str(args.patience),
        "--seed",
        str(seed),
        "--device",
        args.device,
    ]
    subprocess.run(command, check=True)


def _run_row(args: argparse.Namespace, *, seed: int, n_components: int) -> dict[str, Any]:
    run_id = f"components_{n_components}_seed_{seed}"
    run_dir = args.output_dir / run_id
    _run_training(args, seed=seed, n_components=n_components, run_dir=run_dir)
    metrics_path = run_dir / "density_residual_pca_mdn_metrics.json"
    predictions_path = run_dir / "density_residual_pca_mdn_predictions.npz"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    calibration = _calibration_metrics(predictions_path)
    return {
        "run_id": run_id,
        "seed": seed,
        "n_components": n_components,
        "run_dir": str(run_dir),
        **metrics,
        **calibration,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    scalar_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if not isinstance(value, dict)
        }
    )
    preferred = ["run_id", "seed", "n_components", "run_dir"]
    fieldnames = preferred + [key for key in scalar_keys if key not in preferred]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best = min(rows, key=lambda row: row["test_posterior_mean_cell_mass_mae_msun"])
    coverage = np.array([row["test_coeff_coverage_68"] for row in rows], dtype=float)
    calibrated = np.array([row["temperature_test_coeff_coverage_68"] for row in rows], dtype=float)
    inclination_calibrated = np.array(
        [row["inclination_temperature_test_coeff_coverage_68"] for row in rows],
        dtype=float,
    )
    return {
        "n_runs": len(rows),
        "target_coverage": TARGET_COVERAGE,
        "best_by_test_mae": best,
        "test_coeff_coverage_68_mean": float(np.mean(coverage)),
        "test_coeff_coverage_68_std": float(np.std(coverage)),
        "temperature_test_coeff_coverage_68_mean": float(np.mean(calibrated)),
        "temperature_test_coeff_coverage_68_std": float(np.std(calibrated)),
        "inclination_temperature_test_coeff_coverage_68_mean": float(
            np.mean(inclination_calibrated)
        ),
        "inclination_temperature_test_coeff_coverage_68_std": float(
            np.std(inclination_calibrated)
        ),
        "runs": rows,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _run_row(args, seed=seed, n_components=n_components)
        for n_components in args.n_components
        for seed in args.seeds
    ]
    _write_csv(args.output_dir / "density_residual_pca_mdn_sweep.csv", rows)
    (args.output_dir / "density_residual_pca_mdn_sweep.json").write_text(
        json.dumps(_json_ready(_summary(rows)), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote density residual PCA MDN sweep to {args.output_dir}")


if __name__ == "__main__":
    main()
