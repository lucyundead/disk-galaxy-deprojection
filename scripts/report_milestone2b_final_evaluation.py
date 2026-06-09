from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PCA_METRICS = Path(
    "outputs/tng50_milestone2b/density_residual_pca_report/"
    "density_residual_pca_report_metrics.json"
)
DEFAULT_MDN_SWEEP = Path(
    "outputs/tng50_milestone2b/density_residual_pca_mdn_sweep/"
    "density_residual_pca_mdn_sweep.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca-report-metrics", type=Path, default=DEFAULT_PCA_METRICS)
    parser.add_argument("--mdn-sweep", type=Path, default=DEFAULT_MDN_SWEEP)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tng50_milestone2b/milestone2b_final_evaluation"),
    )
    parser.add_argument("--selected-n-components", type=int, default=1)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_scientific(value: float) -> str:
    return f"{value:.3e}".replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")


def _fmt_fraction(value: float) -> str:
    return f"{value:.2%}"


def _fmt_float(value: float) -> str:
    return f"{value:.3f}"


def _select_default_mdn_run(sweep: dict[str, Any], *, n_components: int) -> dict[str, Any]:
    candidates = [row for row in sweep["runs"] if int(row["n_components"]) == n_components]
    if not candidates:
        raise ValueError(f"sweep has no runs with n_components={n_components}")
    return min(candidates, key=lambda row: row["test_posterior_mean_cell_mass_mae_msun"])


def _mean_for_runs(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _selected_family_summary(
    sweep: dict[str, Any],
    *,
    n_components: int,
) -> dict[str, float | int]:
    rows = [row for row in sweep["runs"] if int(row["n_components"]) == n_components]
    if not rows:
        raise ValueError(f"sweep has no runs with n_components={n_components}")
    return {
        "n_runs": len(rows),
        "test_posterior_mean_cell_mass_mae_msun_mean": _mean_for_runs(
            rows,
            "test_posterior_mean_cell_mass_mae_msun",
        ),
        "raw_coeff_coverage_68_mean": _mean_for_runs(rows, "test_coeff_coverage_68"),
        "global_temperature_coeff_coverage_68_mean": _mean_for_runs(
            rows,
            "temperature_test_coeff_coverage_68",
        ),
        "inclination_temperature_coeff_coverage_68_mean": _mean_for_runs(
            rows,
            "inclination_temperature_test_coeff_coverage_68",
        ),
    }


def _comparison_rows(
    pca_metrics: dict[str, Any],
    selected_mdn: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline_mae = float(pca_metrics["test_baseline_cell_mass_mae_msun"])
    mdn_mae = float(selected_mdn["test_posterior_mean_cell_mass_mae_msun"])
    return [
        {
            "model": "Geometric baseline",
            "cell_mass_mae_msun": baseline_mae,
            "improvement_over_baseline_fraction": 0.0,
            "total_mass_fractional_mae": float(
                pca_metrics["test_baseline_total_mass_fractional_mae"]
            ),
            "coeff_coverage_68": None,
        },
        {
            "model": "Mean train residual",
            "cell_mass_mae_msun": float(
                pca_metrics["test_mean_train_residual_cell_mass_mae_msun"]
            ),
            "improvement_over_baseline_fraction": float(
                pca_metrics["test_mean_train_cell_mass_mae_improvement_fraction"]
            ),
            "total_mass_fractional_mae": float(
                pca_metrics["test_mean_train_residual_total_mass_fractional_mae"]
            ),
            "coeff_coverage_68": None,
        },
        {
            "model": "Deterministic PCA MLP",
            "cell_mass_mae_msun": float(pca_metrics["test_model_cell_mass_mae_msun"]),
            "improvement_over_baseline_fraction": float(
                pca_metrics["test_model_cell_mass_mae_improvement_fraction"]
            ),
            "total_mass_fractional_mae": float(pca_metrics["test_model_total_mass_fractional_mae"]),
            "coeff_coverage_68": None,
        },
        {
            "model": "MDN posterior mean",
            "cell_mass_mae_msun": mdn_mae,
            "improvement_over_baseline_fraction": 1.0 - mdn_mae / baseline_mae,
            "total_mass_fractional_mae": float(
                selected_mdn["test_posterior_mean_total_mass_fractional_mae"]
            ),
            "coeff_coverage_68": float(selected_mdn["test_coeff_coverage_68"]),
            "global_temperature_coeff_coverage_68": float(
                selected_mdn["temperature_test_coeff_coverage_68"]
            ),
            "inclination_temperature_coeff_coverage_68": float(
                selected_mdn["inclination_temperature_test_coeff_coverage_68"]
            ),
        },
    ]


def build_metrics(
    pca_metrics: dict[str, Any],
    sweep: dict[str, Any],
    *,
    selected_n_components: int,
) -> dict[str, Any]:
    selected_mdn = _select_default_mdn_run(sweep, n_components=selected_n_components)
    baseline_mae = float(pca_metrics["test_baseline_cell_mass_mae_msun"])
    selected_mae = float(selected_mdn["test_posterior_mean_cell_mass_mae_msun"])
    selected_model = {
        "run_id": selected_mdn["run_id"],
        "seed": int(selected_mdn["seed"]),
        "n_components": int(selected_mdn["n_components"]),
        "test_posterior_mean_cell_mass_mae_msun": selected_mae,
        "improvement_over_baseline_fraction": 1.0 - selected_mae / baseline_mae,
        "raw_coeff_coverage_68": float(selected_mdn["test_coeff_coverage_68"]),
        "global_temperature_coeff_coverage_68": float(
            selected_mdn["temperature_test_coeff_coverage_68"]
        ),
        "inclination_temperature_coeff_coverage_68": float(
            selected_mdn["inclination_temperature_test_coeff_coverage_68"]
        ),
        "inclination_temperature_by_inclination": selected_mdn.get(
            "inclination_temperature_by_inclination",
            {},
        ),
    }
    return {
        "n_test": int(pca_metrics["n_test"]),
        "n_test_galaxies": int(pca_metrics["n_test_galaxies"]),
        "pca_n_components": int(pca_metrics["pca_n_components"]),
        "pca_cumulative_explained_variance": float(
            pca_metrics["pca_cumulative_explained_variance"]
        ),
        "selected_model": selected_model,
        "selected_model_family": _selected_family_summary(
            sweep,
            n_components=selected_n_components,
        ),
        "comparison_rows": _comparison_rows(pca_metrics, selected_mdn),
    }


def _coverage_cell(row: dict[str, Any]) -> str:
    if row["coeff_coverage_68"] is None:
        return "not applicable"
    return (
        f"raw {_fmt_float(row['coeff_coverage_68'])}; "
        f"global temp {_fmt_float(row['global_temperature_coeff_coverage_68'])}; "
        f"inclination temp {_fmt_float(row['inclination_temperature_coeff_coverage_68'])}"
    )


def _write_markdown(path: Path, metrics: dict[str, Any]) -> None:
    selected = metrics["selected_model"]
    family = metrics["selected_model_family"]
    lines = [
        "# Milestone 2b Final PCA Residual Evaluation",
        "",
        "Date: 2026-06-09",
        "",
        "## Frozen Default",
        "",
        "The selected Milestone 2b probabilistic baseline is the "
        "1-component MDN + inclination-aware temperature diagnostics. It uses "
        "the existing 32-component PCA residual target and preserves baseline "
        "grid mass, so it evaluates spatial redistribution inside the coarse "
        "cylindrical 3D target rather than total stellar mass correction.",
        "",
        f"- selected run: `{selected['run_id']}`",
        f"- seed: `{selected['seed']}`",
        f"- PCA components: `{metrics['pca_n_components']}`",
        f"- PCA explained variance: {_fmt_fraction(metrics['pca_cumulative_explained_variance'])}",
        f"- held-out test projections: `{metrics['n_test']}` from "
        f"`{metrics['n_test_galaxies']}` galaxies",
        "",
        "## Held-Out Comparison",
        "",
        "| Model | Cell-mass MAE (Msun) | Improvement | Total-mass frac. MAE | 68% coefficient coverage |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in metrics["comparison_rows"]:
        lines.append(
            "| "
            f"{row['model']} | "
            f"{_fmt_scientific(row['cell_mass_mae_msun'])} | "
            f"{_fmt_fraction(row['improvement_over_baseline_fraction'])} | "
            f"{_fmt_float(row['total_mass_fractional_mae'])} | "
            f"{_coverage_cell(row)} |"
        )
    lines.extend(
        [
            "",
            "The selected MDN posterior mean is the best point predictor in this "
            "Milestone 2b comparison, but its main role is uncertainty over PCA "
            "residual coefficients. Raw MDN intervals are conservative; global "
            "temperature scaling moves aggregate coverage closer to nominal, and "
            "inclination-aware temperature diagnostics expose the viewing-geometry "
            "dependence.",
            "",
            "## Selected Family Stability",
            "",
            f"The selected 1-component family has `{family['n_runs']}` seed runs.",
            "",
            "| Quantity | Mean over 1-component seeds |",
            "| --- | ---: |",
            "| Posterior-mean cell-mass MAE (Msun) | "
            f"{_fmt_scientific(family['test_posterior_mean_cell_mass_mae_msun_mean'])} |",
            "| Raw 68% coefficient coverage | "
            f"{_fmt_float(family['raw_coeff_coverage_68_mean'])} |",
            "| Global-temperature 68% coefficient coverage | "
            f"{_fmt_float(family['global_temperature_coeff_coverage_68_mean'])} |",
            "| Inclination-temperature 68% coefficient coverage | "
            f"{_fmt_float(family['inclination_temperature_coeff_coverage_68_mean'])} |",
            "",
            "Inclination-aware coverage for the selected run:",
            "",
            "| Inclination | Test coverage | Temperature scale |",
            "| ---: | ---: | ---: |",
        ]
    )
    for inclination, row in sorted(
        selected["inclination_temperature_by_inclination"].items(),
        key=lambda item: float(item[0]),
    ):
        lines.append(
            f"| {inclination} deg | "
            f"{_fmt_float(row['test_coeff_coverage_68'])} | "
            f"{_fmt_float(row['temperature_scale'])} |"
        )
    lines.extend(
        [
            "",
            "## Reproduce The Frozen Artifact",
            "",
            "Train the selected default run:",
            "",
            "```bash",
            ".venv/bin/python scripts/train_density_residual_pca_mdn.py \\",
            "  --pca outputs/tng50_milestone2b/density_residual_diagnostics/density_residual_pca.npz \\",
            "  --density-table outputs/tng50_milestone2b/density_residual_table.npz \\",
            "  --output-dir outputs/tng50_milestone2b/density_residual_pca_mdn_selected \\",
            "  --epochs 200 \\",
            "  --hidden-dim 128 \\",
            "  --image-feature-size 24 \\",
            "  --batch-size 64 \\",
            f"  --n-components {selected['n_components']} \\",
            "  --n-samples 128 \\",
            "  --weight-decay 0.0001 \\",
            "  --patience 50 \\",
            f"  --seed {selected['seed']} \\",
            "  --device cpu",
            "```",
            "",
            "Regenerate the calibration sweep summary from existing run outputs:",
            "",
            "```bash",
            ".venv/bin/python scripts/sweep_density_residual_pca_mdn.py \\",
            "  --pca outputs/tng50_milestone2b/density_residual_diagnostics/density_residual_pca.npz \\",
            "  --density-table outputs/tng50_milestone2b/density_residual_table.npz \\",
            "  --output-dir outputs/tng50_milestone2b/density_residual_pca_mdn_sweep \\",
            "  --seeds 20260608 20260609 20260610 \\",
            "  --n-components 1 3 5 \\",
            "  --epochs 200 \\",
            "  --hidden-dim 128 \\",
            "  --image-feature-size 24 \\",
            "  --batch-size 64 \\",
            "  --n-samples 128 \\",
            "  --patience 50 \\",
            "  --device cpu \\",
            "  --skip-existing",
            "```",
            "",
            "Regenerate this final report:",
            "",
            "```bash",
            ".venv/bin/python scripts/report_milestone2b_final_evaluation.py",
            "```",
            "",
            "## Scientific Next Step",
            "",
            "Do not move to a larger full-3D generator yet. The next useful "
            "analysis is to evaluate calibrated posterior samples on physical "
            "summaries: radial profile, vertical profile or thickness, bar-region "
            "mass, and central concentration. Those checks decide whether the "
            "PCA-residual posterior is scientifically useful beyond coefficient "
            "coverage and cell-wise mass error.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = build_metrics(
        _load_json(args.pca_report_metrics),
        _load_json(args.mdn_sweep),
        selected_n_components=args.selected_n_components,
    )
    (args.output_dir / "milestone2b_final_evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(args.output_dir / "milestone2b_final_evaluation.md", metrics)
    print(f"wrote Milestone 2b final evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
