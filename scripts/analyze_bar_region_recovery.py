"""Quantify how well the adopted MDN posterior recovers the bar region.

The science driver is bar-region gas dynamics: torques depend on the bar's
quadrupole amplitude and phase and on its vertical thickness, all within a few
kpc, while the grid and global metrics span the full 30 kpc disk. This script
evaluates recovery in bar-scaled regions (R < L_bar per galaxy, from the
manifest), separately from the transition zone and outer disk:

- region mass budget (fraction of grid mass) for truth/baseline/posterior;
- cell-level relative MAE per region;
- m=2 Fourier amplitude AND phase per region (phase errors map to spurious
  torques in a downstream potential);
- mass-weighted RMS thickness in the bar region;
- a peanut proxy: RMS z along the bar versus perpendicular to it within
  R in [0.3, 0.8] L_bar, truth versus posterior;
- 68 percent coverage of bar-region mass fraction and bar A2 from the
  sampled posterior coefficients.

Rows aggregate to galaxies (mean over projections) before population stats,
matching the galaxy-clustered conventions of the diagnostics scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from train_density_residual_pca import (
    _mass_grids,
    reconstruct_delta_mass_from_coefficients,
)

PEANUT_INNER_FRAC = 0.3
PEANUT_OUTER_FRAC = 0.8
SECTOR_HALF_WIDTH_RAD = np.pi / 6.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--density-table", type=Path, required=True)
    parser.add_argument("--pca", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--no-preserve-total-mass",
        action="store_true",
        help="match a run trained with --allow-total-mass-change",
    )
    return parser.parse_args()


def _bin_centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def _bar_lengths_by_galaxy(manifest_path: Path) -> dict[int, float]:
    lengths: dict[int, float] = {}
    with open(manifest_path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            lengths[int(row["subhalo_id"])] = float(row["bar_length"])
    return lengths


def _corrected_from_delta(
    baseline: np.ndarray, delta: np.ndarray, preserve_total: bool
) -> np.ndarray:
    corrected = np.clip(baseline + delta, 0.0, None)
    if preserve_total:
        axes = tuple(range(1, corrected.ndim))
        baseline_total = np.sum(baseline, axis=axes)
        corrected_total = np.sum(corrected, axis=axes)
        ratio = np.divide(
            baseline_total,
            corrected_total,
            out=np.ones_like(baseline_total),
            where=corrected_total > 0,
        )
        corrected = corrected * ratio.reshape((-1,) + (1,) * (corrected.ndim - 1))
    return corrected


def _m2_amp_phase(mass_rphi: np.ndarray, phi_centers: np.ndarray) -> tuple[float, float]:
    """m=2 Fourier amplitude (normalized) and phase from a (R, phi) mass map."""
    total = float(mass_rphi.sum())
    if total <= 0:
        return 0.0, 0.0
    weights = mass_rphi.sum(axis=0)
    complex_sum = np.sum(weights * np.exp(-2j * phi_centers))
    amp = float(np.abs(complex_sum) / total)
    phase = float(0.5 * np.angle(complex_sum))
    return amp, phase


def _wrap_m2_phase_deg(delta_rad: float) -> float:
    """Wrap an m=2 phase difference into [-90, 90) degrees."""
    period = np.pi
    wrapped = (delta_rad + period / 2.0) % period - period / 2.0
    return float(np.degrees(wrapped))


def _rms_z(mass: np.ndarray, z_centers: np.ndarray, cell_mask: np.ndarray) -> float:
    selected = np.where(cell_mask, mass, 0.0)
    total = float(selected.sum())
    if total <= 0:
        return 0.0
    z_sq = np.broadcast_to(z_centers[None, None, :] ** 2, mass.shape)
    return float(np.sqrt(np.sum(selected * z_sq) / total))


def _region_metrics(
    truth: np.ndarray,
    candidate: np.ndarray,
    region_mask: np.ndarray,
) -> dict[str, float]:
    truth_region = truth[region_mask]
    cand_region = candidate[region_mask]
    truth_total = float(truth.sum())
    cand_total = float(candidate.sum())
    # Fractions are self-normalized per grid so the error measures spatial
    # redistribution only; the candidate total is pinned to the baseline total
    # by the preserve-total step and would otherwise leak into this metric.
    metrics = {
        "mass_fraction_true": float(truth_region.sum()) / truth_total,
        "mass_fraction_err": float(cand_region.sum()) / cand_total
        - float(truth_region.sum()) / truth_total,
        "cell_rel_mae": float(np.mean(np.abs(cand_region - truth_region)))
        / max(float(np.mean(truth_region)), 1e-30),
    }
    return metrics


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table = np.load(args.density_table)
    split_mask = table["split"] == args.split
    r_edges = table["r_edges_kpc"]
    phi_centers = _bin_centers(table["phi_edges_rad"])
    z_centers = _bin_centers(table["z_edges_kpc"])
    r_centers = _bin_centers(r_edges)

    truth_mass_all, baseline_mass_all, _ = _mass_grids(table)
    truth_mass = truth_mass_all[split_mask]
    baseline_mass = baseline_mass_all[split_mask]
    del truth_mass_all, baseline_mass_all
    grid_shape = truth_mass.shape[1:]
    galaxy_ids = table["galaxy_id"][split_mask]
    metadata = table["metadata"][split_mask]
    scale_msun = table["baseline_grid_mass_msun"][split_mask].astype(np.float32)

    preds = np.load(args.predictions)
    pred_mask = preds["split"] == args.split
    projection_ids = table["projection_id"][split_mask]
    if not (
        np.array_equal(preds["galaxy_id"][pred_mask], galaxy_ids)
        and np.array_equal(preds["projection_id"][pred_mask], projection_ids)
    ):
        raise SystemExit("predictions row order does not match the density table")
    posterior_mean_mass = preds["posterior_mean_mass"][pred_mask]
    sampled_coefficients = preds["sampled_coefficients"][pred_mask]
    mean_coefficients = preds["posterior_mean_coefficients"][pred_mask]

    pca = np.load(args.pca)
    components = pca["components"].astype(np.float32)
    pca_mean = pca["mean"].astype(np.float32)
    preserve_total = not args.no_preserve_total_mass

    # Self-check: rebuilding the posterior-mean grid from the stored mean
    # coefficients must reproduce the stored grid, proving we use the same
    # transform for the per-sample reconstructions below.
    rebuilt_delta = reconstruct_delta_mass_from_coefficients(
        mean_coefficients,
        components=components,
        mean=pca_mean,
        scale_msun=scale_msun,
        grid_shape=grid_shape,
    )
    rebuilt = _corrected_from_delta(baseline_mass, rebuilt_delta, preserve_total)
    rebuild_rel = float(
        np.max(np.abs(rebuilt - posterior_mean_mass))
        / max(float(np.mean(posterior_mean_mass)), 1e-30)
    )
    del rebuilt, rebuilt_delta
    if rebuild_rel > 1e-3:
        raise SystemExit(f"posterior-mean rebuild mismatch: {rebuild_rel:.3e}")

    bar_lengths = _bar_lengths_by_galaxy(args.manifest)

    along_bar = (np.abs(phi_centers) < SECTOR_HALF_WIDTH_RAD) | (
        np.abs(np.abs(phi_centers) - np.pi) < SECTOR_HALF_WIDTH_RAD
    )
    perp_bar = np.abs(np.abs(phi_centers) - np.pi / 2.0) < SECTOR_HALF_WIDTH_RAD

    row_records: list[dict] = []
    n_rows = truth_mass.shape[0]
    for idx in range(n_rows):
        gid = int(galaxy_ids[idx])
        length = bar_lengths[gid]
        truth = truth_mass[idx]
        baseline = baseline_mass[idx]
        posterior = posterior_mean_mass[idx]

        regions = {
            "bar": r_centers < length,
            "transition": (r_centers >= length) & (r_centers < 1.5 * length),
            "outer": r_centers >= 1.5 * length,
        }
        record: dict = {
            "galaxy_id": gid,
            "bar_length_kpc": length,
            "inclination_deg": float(metadata[idx, 0]),
        }
        for name, radial_mask in regions.items():
            cell_mask = np.zeros(grid_shape, dtype=bool)
            cell_mask[radial_mask] = True
            record[name] = {
                "posterior": _region_metrics(truth, posterior, cell_mask),
                "baseline": _region_metrics(truth, baseline, cell_mask),
            }
            for label, cand in (("posterior", posterior), ("baseline", baseline)):
                amp_true, phase_true = _m2_amp_phase(
                    truth[radial_mask].sum(axis=2), phi_centers
                )
                amp_cand, phase_cand = _m2_amp_phase(
                    cand[radial_mask].sum(axis=2), phi_centers
                )
                record[name][label]["m2_amp_true"] = amp_true
                record[name][label]["m2_amp_err"] = amp_cand - amp_true
                record[name][label]["m2_phase_abs_err_deg"] = abs(
                    _wrap_m2_phase_deg(phase_cand - phase_true)
                )

        bar_cells = np.zeros(grid_shape, dtype=bool)
        bar_cells[regions["bar"]] = True
        rms_true = _rms_z(truth, z_centers, bar_cells)
        record["bar_rms_z_true_kpc"] = rms_true
        record["bar_rms_z_posterior_ratio"] = (
            _rms_z(posterior, z_centers, bar_cells) / rms_true if rms_true > 0 else 0.0
        )
        record["bar_rms_z_baseline_ratio"] = (
            _rms_z(baseline, z_centers, bar_cells) / rms_true if rms_true > 0 else 0.0
        )

        peanut_radial = (r_centers >= PEANUT_INNER_FRAC * length) & (
            r_centers < PEANUT_OUTER_FRAC * length
        )
        major_mask = np.zeros(grid_shape, dtype=bool)
        major_mask[np.ix_(peanut_radial, along_bar)] = True
        minor_mask = np.zeros(grid_shape, dtype=bool)
        minor_mask[np.ix_(peanut_radial, perp_bar)] = True
        for label, grid in (("true", truth), ("posterior", posterior)):
            major = _rms_z(grid, z_centers, major_mask)
            minor = _rms_z(grid, z_centers, minor_mask)
            record[f"peanut_ratio_{label}"] = major / minor if minor > 0 else 0.0

        # Coverage of bar-region summaries from the sampled posterior.
        sample_delta = reconstruct_delta_mass_from_coefficients(
            sampled_coefficients[idx],
            components=components,
            mean=pca_mean,
            scale_msun=np.full(
                sampled_coefficients.shape[1], scale_msun[idx], dtype=np.float32
            ),
            grid_shape=grid_shape,
        )
        samples = _corrected_from_delta(
            np.broadcast_to(baseline, sample_delta.shape), sample_delta, preserve_total
        )
        sample_totals = samples.sum(axis=(1, 2, 3))
        bar_frac_samples = (
            samples[:, regions["bar"]].sum(axis=(1, 2, 3)) / sample_totals
        )
        amp_samples = np.empty(samples.shape[0], dtype=np.float64)
        for s in range(samples.shape[0]):
            amp_samples[s], _ = _m2_amp_phase(
                samples[s][regions["bar"]].sum(axis=2), phi_centers
            )
        del sample_delta, samples
        truth_total = float(truth.sum())
        bar_frac_true = float(truth[bar_cells].sum()) / truth_total
        amp_true_bar, _ = _m2_amp_phase(truth[regions["bar"]].sum(axis=2), phi_centers)
        for key, sample_values, true_value in (
            ("bar_mass_fraction", bar_frac_samples, bar_frac_true),
            ("bar_m2_amp", amp_samples, amp_true_bar),
        ):
            lo, hi = np.quantile(sample_values, [0.16, 0.84])
            record[f"coverage_{key}"] = bool(lo <= true_value <= hi)

        row_records.append(record)

    # Aggregate rows -> galaxies -> population statistics.
    by_galaxy: dict[int, list[dict]] = {}
    for record in row_records:
        by_galaxy.setdefault(record["galaxy_id"], []).append(record)

    def galaxy_means(extract) -> np.ndarray:
        return np.array(
            [np.mean([extract(r) for r in recs]) for recs in by_galaxy.values()]
        )

    def population(values: np.ndarray) -> dict[str, float]:
        return {
            "median": float(np.median(values)),
            "p16": float(np.percentile(values, 16)),
            "p84": float(np.percentile(values, 84)),
        }

    summary: dict = {
        "predictions": str(args.predictions),
        "split": args.split,
        "n_rows": n_rows,
        "n_galaxies": len(by_galaxy),
        "rebuild_max_rel_diff": rebuild_rel,
        "regions": {},
    }
    for name in ("bar", "transition", "outer"):
        region_summary: dict = {}
        for label in ("posterior", "baseline"):
            region_summary[label] = {
                "cell_rel_mae": population(
                    galaxy_means(lambda r: r[name][label]["cell_rel_mae"])
                ),
                "mass_fraction_err": population(
                    galaxy_means(lambda r: r[name][label]["mass_fraction_err"])
                ),
                "m2_amp_err": population(
                    galaxy_means(lambda r: r[name][label]["m2_amp_err"])
                ),
                "m2_phase_abs_err_deg": population(
                    galaxy_means(lambda r: r[name][label]["m2_phase_abs_err_deg"])
                ),
            }
        region_summary["mass_fraction_true"] = population(
            galaxy_means(lambda r: r[name]["posterior"]["mass_fraction_true"])
        )
        region_summary["m2_amp_true"] = population(
            galaxy_means(lambda r: r[name]["posterior"]["m2_amp_true"])
        )
        summary["regions"][name] = region_summary

    summary["bar_rms_z_true_kpc"] = population(
        galaxy_means(lambda r: r["bar_rms_z_true_kpc"])
    )
    summary["bar_rms_z_posterior_ratio"] = population(
        galaxy_means(lambda r: r["bar_rms_z_posterior_ratio"])
    )
    summary["bar_rms_z_baseline_ratio"] = population(
        galaxy_means(lambda r: r["bar_rms_z_baseline_ratio"])
    )

    peanut_true = galaxy_means(lambda r: r["peanut_ratio_true"])
    peanut_post = galaxy_means(lambda r: r["peanut_ratio_posterior"])
    summary["peanut_ratio_true"] = population(peanut_true)
    summary["peanut_ratio_posterior"] = population(peanut_post)
    if np.std(peanut_true) > 0 and np.std(peanut_post) > 0:
        summary["peanut_ratio_pearson_r"] = float(
            np.corrcoef(peanut_true, peanut_post)[0, 1]
        )
    summary["peanut_sign_agreement"] = float(
        np.mean(np.sign(peanut_true - 1.0) == np.sign(peanut_post - 1.0))
    )

    for key in ("bar_mass_fraction", "bar_m2_amp"):
        per_galaxy = galaxy_means(lambda r: float(r[f"coverage_{key}"]))
        summary[f"coverage68_{key}"] = float(np.mean(per_galaxy))

    # Persist per-galaxy values so population claims are artifact-backed.
    summary["per_galaxy"] = {
        str(gid): {
            "bar_length_kpc": recs[0]["bar_length_kpc"],
            "bar_cell_rel_mae": float(
                np.mean([r["bar"]["posterior"]["cell_rel_mae"] for r in recs])
            ),
            "bar_m2_phase_abs_err_deg": float(
                np.mean([r["bar"]["posterior"]["m2_phase_abs_err_deg"] for r in recs])
            ),
            "bar_rms_z_posterior_ratio": float(
                np.mean([r["bar_rms_z_posterior_ratio"] for r in recs])
            ),
            "peanut_ratio_true": float(
                np.mean([r["peanut_ratio_true"] for r in recs])
            ),
            "peanut_ratio_posterior": float(
                np.mean([r["peanut_ratio_posterior"] for r in recs])
            ),
        }
        for gid, recs in by_galaxy.items()
    }

    json_path = args.output_dir / "bar_region_recovery.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Bar-Region Recovery (adopted 2c run)",
        "",
        f"- predictions: `{args.predictions}`",
        f"- split: `{args.split}` ({n_rows} rows, {len(by_galaxy)} galaxies)",
        f"- rebuild check (stored vs recomputed posterior mean): `{rebuild_rel:.2e}`",
        "",
        "## Recovery By Region (median [p16, p84] over galaxies)",
        "",
        "| Region | metric | posterior | baseline |",
        "| --- | --- | --- | --- |",
    ]

    def fmt(stats: dict[str, float], scale: float = 1.0, digits: int = 3) -> str:
        return (
            f"{stats['median'] * scale:.{digits}f} "
            f"[{stats['p16'] * scale:.{digits}f}, {stats['p84'] * scale:.{digits}f}]"
        )

    for name in ("bar", "transition", "outer"):
        region = summary["regions"][name]
        lines.append(
            f"| {name} (true mass frac {region['mass_fraction_true']['median']:.3f}) "
            f"| cell rel MAE | {fmt(region['posterior']['cell_rel_mae'])} "
            f"| {fmt(region['baseline']['cell_rel_mae'])} |"
        )
        lines.append(
            f"| | mass fraction err | {fmt(region['posterior']['mass_fraction_err'])} "
            f"| {fmt(region['baseline']['mass_fraction_err'])} |"
        )
        lines.append(
            f"| | m2 amp err (true {region['m2_amp_true']['median']:.3f}) "
            f"| {fmt(region['posterior']['m2_amp_err'])} "
            f"| {fmt(region['baseline']['m2_amp_err'])} |"
        )
        lines.append(
            f"| | m2 phase abs err [deg] "
            f"| {fmt(region['posterior']['m2_phase_abs_err_deg'], digits=2)} "
            f"| {fmt(region['baseline']['m2_phase_abs_err_deg'], digits=2)} |"
        )
    lines += [
        "",
        "## Vertical Structure In The Bar Region",
        "",
        f"- true RMS z: {fmt(summary['bar_rms_z_true_kpc'], digits=3)} kpc",
        f"- posterior / true RMS z: {fmt(summary['bar_rms_z_posterior_ratio'])}",
        f"- baseline / true RMS z: {fmt(summary['bar_rms_z_baseline_ratio'])}",
        f"- peanut ratio (along/perp), truth: {fmt(summary['peanut_ratio_true'])}",
        f"- peanut ratio, posterior: {fmt(summary['peanut_ratio_posterior'])}",
        f"- peanut Pearson r (galaxies): "
        f"{summary.get('peanut_ratio_pearson_r', float('nan')):.3f}",
        f"- peanut sign agreement: {summary['peanut_sign_agreement']:.3f}",
        "",
        "## Bar-Region Coverage (raw posterior, target 0.68)",
        "",
        f"- bar mass fraction: {summary['coverage68_bar_mass_fraction']:.3f}",
        f"- bar m2 amplitude: {summary['coverage68_bar_m2_amp']:.3f}",
        "",
    ]
    (args.output_dir / "bar_region_recovery.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
