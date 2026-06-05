from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dgdp.metrics import interval_coverage, mean_absolute_error
from dgdp.models.mdn import SummaryResidualMDN
from dgdp.train import make_feature_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def load_corrected_predictions(
    run_dir: Path,
    *,
    split_name: str,
    n_samples: int,
) -> dict[str, np.ndarray]:
    table = np.load(run_dir / "residual_table.npz", allow_pickle=True)
    norm = np.load(run_dir / "normalization.npz")
    checkpoint = torch.load(run_dir / "summary_residual_mdn.pt", map_location="cpu")

    model = SummaryResidualMDN(
        input_dim=int(checkpoint["input_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        n_components=int(checkpoint["n_components"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    x_all = make_feature_matrix(table["images"], table["baseline"], table["metadata"])
    x_all = (x_all - norm["x_mean"]) / norm["x_scale"]
    split = table["split"].astype(str)
    mask = split == split_name
    x_eval = torch.tensor(x_all[mask], dtype=torch.float32)

    samples_z = model.sample(x_eval, n_samples=n_samples).numpy()
    samples = samples_z * norm["y_scale"][None, None, :] + norm["y_mean"][None, None, :]
    baseline = table["baseline"][mask]
    truth = table["truth"][mask]
    corrected_samples = baseline[:, None, :] + samples
    corrected_mean = corrected_samples.mean(axis=1)
    lower = np.quantile(corrected_samples, 0.16, axis=1)
    upper = np.quantile(corrected_samples, 0.84, axis=1)

    return {
        "baseline": baseline,
        "truth": truth,
        "corrected_mean": corrected_mean,
        "lower": lower,
        "upper": upper,
        "metadata": table["metadata"][mask],
        "galaxy_id": table["galaxy_id"][mask],
        "projection_id": table["projection_id"][mask],
        "summary_names": table["summary_names"].astype(str),
    }


def _relative_abs_error(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    truth_abs = np.abs(truth)
    scale = max(float(np.median(truth_abs)), 1.0)
    denominator = np.maximum(truth_abs, 1.0e-6 * scale)
    return np.abs(prediction - truth) / denominator


def build_per_summary_table(predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    baseline = predictions["baseline"]
    corrected = predictions["corrected_mean"]
    truth = predictions["truth"]
    lower = predictions["lower"]
    upper = predictions["upper"]
    rows = []
    for index, name in enumerate(predictions["summary_names"]):
        baseline_mae = mean_absolute_error(baseline[:, index], truth[:, index])
        corrected_mae = mean_absolute_error(corrected[:, index], truth[:, index])
        baseline_rel = float(np.mean(_relative_abs_error(baseline[:, index], truth[:, index])))
        corrected_rel = float(np.mean(_relative_abs_error(corrected[:, index], truth[:, index])))
        rows.append(
            {
                "summary_name": name,
                "n": int(len(truth)),
                "truth_median": float(np.median(truth[:, index])),
                "truth_median_abs": float(np.median(np.abs(truth[:, index]))),
                "baseline_mae": baseline_mae,
                "corrected_mae": corrected_mae,
                "mae_improvement_fraction": (
                    (baseline_mae - corrected_mae) / baseline_mae if baseline_mae > 0.0 else 0.0
                ),
                "baseline_mean_relative_abs_error": baseline_rel,
                "corrected_mean_relative_abs_error": corrected_rel,
                "relative_error_improvement_fraction": (
                    (baseline_rel - corrected_rel) / baseline_rel if baseline_rel > 0.0 else 0.0
                ),
                "coverage_68": interval_coverage(
                    lower[:, index],
                    upper[:, index],
                    truth[:, index],
                ),
            }
        )
    return pd.DataFrame(rows)


def build_grouped_table(predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    metadata = predictions["metadata"]
    frame = pd.DataFrame(
        {
            "inclination_deg": metadata[:, 0],
            "bar_angle_deg": metadata[:, 2],
        }
    )
    baseline = predictions["baseline"]
    corrected = predictions["corrected_mean"]
    truth = predictions["truth"]
    lower = predictions["lower"]
    upper = predictions["upper"]
    rows = []

    group_specs = [
        ("inclination", ["inclination_deg"]),
        ("bar_angle", ["bar_angle_deg"]),
        ("inclination_bar_angle", ["inclination_deg", "bar_angle_deg"]),
    ]
    for group_type, columns in group_specs:
        for keys, group in frame.groupby(columns, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            mask = group.index.to_numpy()
            baseline_mae = mean_absolute_error(baseline[mask], truth[mask])
            corrected_mae = mean_absolute_error(corrected[mask], truth[mask])
            row = {
                "group_type": group_type,
                "n": int(len(mask)),
                "baseline_mae": baseline_mae,
                "corrected_mae": corrected_mae,
                "mae_improvement_fraction": (
                    (baseline_mae - corrected_mae) / baseline_mae if baseline_mae > 0.0 else 0.0
                ),
                "coverage_68": interval_coverage(lower[mask], upper[mask], truth[mask]),
                "inclination_deg": np.nan,
                "bar_angle_deg": np.nan,
            }
            for column, key in zip(columns, keys):
                row[column] = float(key)
            rows.append(row)
    return pd.DataFrame(rows)


def _short_name(name: str) -> str:
    return (
        name.replace("enclosed_mass_", "M(")
        .replace("surface_density_", "Sigma_")
        .replace("_kpc", "")
        .replace("_mad", "")
        .replace("_amplitude", "")
    )


def _try_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None
    return plt


def write_plots(
    output_dir: Path,
    predictions: dict[str, np.ndarray],
    per_summary: pd.DataFrame,
    grouped: pd.DataFrame,
) -> None:
    plt = _try_import_matplotlib()
    if plt is None:
        print("matplotlib not available; skipped diagnostic plots")
        return

    names = per_summary["summary_name"].tolist()
    x = np.arange(len(names))
    labels = [_short_name(name) for name in names]

    fig, ax = plt.subplots(figsize=(13.5, 5.5), dpi=150)
    ax.bar(x - 0.2, per_summary["baseline_mae"], width=0.4, label="baseline")
    ax.bar(x + 0.2, per_summary["corrected_mae"], width=0.4, label="corrected")
    ax.set_yscale("log")
    ax.set_ylabel("MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "per_summary_mae.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13.5, 5.5), dpi=150)
    ax.bar(x - 0.2, per_summary["baseline_mean_relative_abs_error"], width=0.4, label="baseline")
    ax.bar(x + 0.2, per_summary["corrected_mean_relative_abs_error"], width=0.4, label="corrected")
    ax.set_yscale("log")
    ax.set_ylabel("mean relative absolute error")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "per_summary_relative_error.png")
    plt.close(fig)

    n_cols = 5
    n_rows = int(np.ceil(len(names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows), dpi=150)
    axes_flat = np.atleast_1d(axes).ravel()
    truth = predictions["truth"]
    baseline = predictions["baseline"]
    corrected = predictions["corrected_mean"]
    for index, ax in enumerate(axes_flat):
        if index >= len(names):
            ax.axis("off")
            continue
        y_true = truth[:, index]
        ax.scatter(y_true, baseline[:, index], s=8, alpha=0.45, label="baseline")
        ax.scatter(y_true, corrected[:, index], s=8, alpha=0.45, label="corrected")
        finite = np.isfinite(y_true)
        low = float(np.nanmin([y_true[finite].min(), baseline[:, index].min(), corrected[:, index].min()]))
        high = float(np.nanmax([y_true[finite].max(), baseline[:, index].max(), corrected[:, index].max()]))
        ax.plot([low, high], [low, high], color="black", linewidth=0.8)
        ax.set_title(_short_name(names[index]), fontsize=8)
        ax.tick_params(labelsize=7)
    handles, legend_labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_dir / "truth_vs_prediction_all_summaries.png")
    plt.close(fig)

    for group_type, filename in [
        ("inclination", "mae_by_inclination.png"),
        ("bar_angle", "mae_by_bar_angle.png"),
    ]:
        subset = grouped[grouped["group_type"] == group_type].sort_values(
            "inclination_deg" if group_type == "inclination" else "bar_angle_deg"
        )
        x_col = "inclination_deg" if group_type == "inclination" else "bar_angle_deg"
        fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=150)
        ax.plot(subset[x_col], subset["baseline_mae"], marker="o", label="baseline")
        ax.plot(subset[x_col], subset["corrected_mae"], marker="o", label="corrected")
        ax.set_xlabel(x_col)
        ax.set_ylabel("aggregate MAE")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.run_dir / "diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_corrected_predictions(
        args.run_dir,
        split_name=args.split,
        n_samples=args.n_samples,
    )
    per_summary = build_per_summary_table(predictions)
    grouped = build_grouped_table(predictions)

    per_summary.to_csv(output_dir / "per_summary_metrics.csv", index=False)
    grouped.to_csv(output_dir / "grouped_metrics.csv", index=False)
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        baseline=predictions["baseline"],
        corrected_mean=predictions["corrected_mean"],
        truth=predictions["truth"],
        lower=predictions["lower"],
        upper=predictions["upper"],
        metadata=predictions["metadata"],
        galaxy_id=predictions["galaxy_id"],
        projection_id=predictions["projection_id"],
        summary_names=predictions["summary_names"],
    )
    aggregate = {
        "split": args.split,
        "n_examples": int(len(predictions["truth"])),
        "baseline_mae": mean_absolute_error(predictions["baseline"], predictions["truth"]),
        "corrected_mae": mean_absolute_error(
            predictions["corrected_mean"],
            predictions["truth"],
        ),
        "coverage_68": interval_coverage(
            predictions["lower"],
            predictions["upper"],
            predictions["truth"],
        ),
    }
    (output_dir / "diagnostic_summary.json").write_text(
        json.dumps(aggregate, indent=2),
        encoding="utf-8",
    )
    if not args.no_plots:
        write_plots(output_dir, predictions, per_summary, grouped)
    print(f"wrote diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
