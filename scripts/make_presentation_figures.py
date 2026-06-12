"""Presentation-quality figures for the Milestone 2c results.

Produces, in --output-dir:
- example_reconstruction_<subhalo>.png : truth / baseline / posterior face-on
  surface density in PHYSICAL x-y coordinates (the report figures plot R-phi
  index maps, which read poorly on slides), plus a posterior residual panel;
- pca_component_gallery.png : face-on maps of the leading PCA components;
- peanut_ratio_scatter.png : per-galaxy vertical anisotropy, truth vs
  posterior, from the bar-region recovery analysis;
- inclination_sensitivity.png : bar-frame summary MAE versus inclination
  offset, from the sensitivity scan;
- bar_region_mae.png : cell-level relative MAE by bar-scaled region.

All inputs are existing artifacts; this script does no inference.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from train_density_residual_pca import _mass_grids

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.frameon": False,
    }
)

DEFAULT_EXAMPLES = [307485, 375073, 390932, 396628, 411449, 63864]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--density-table", type=Path, required=True)
    parser.add_argument("--pca", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bar-region-json", type=Path, required=True)
    parser.add_argument("--sensitivity-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--example-subhalos", type=int, nargs="*", default=DEFAULT_EXAMPLES)
    parser.add_argument("--example-projection", type=int, default=6)
    parser.add_argument("--display-radius-kpc", type=float, default=10.0)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def _faceon_vertices(
    r_edges: np.ndarray, phi_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rr, pp = np.meshgrid(r_edges, phi_edges, indexing="ij")
    return rr * np.cos(pp), rr * np.sin(pp)


def _surface_density(mass_grid: np.ndarray, r_edges: np.ndarray, phi_edges: np.ndarray) -> np.ndarray:
    """z-summed mass per face-on cell area, Msun / kpc^2; mass_grid is (R, phi, z)."""
    area = 0.5 * (r_edges[1:, None] ** 2 - r_edges[:-1, None] ** 2) * np.diff(phi_edges)[None, :]
    return mass_grid.sum(axis=2) / area


def plot_examples(args, table, preds) -> list[str]:
    truth_mass, baseline_mass, _ = _mass_grids(table)
    posterior_mass = preds["posterior_mean_mass"]
    r_edges = table["r_edges_kpc"].astype(np.float64)
    phi_edges = table["phi_edges_rad"].astype(np.float64)
    keep = r_edges <= args.display_radius_kpc + 1e-6
    n_r = int(keep[1:].sum())
    x_v, y_v = _faceon_vertices(r_edges[: n_r + 1], phi_edges)

    bar_lengths = {
        int(row["subhalo_id"]): float(row["bar_length"])
        for row in csv.DictReader(open(args.manifest, encoding="utf-8"))
    }

    written = []
    for subhalo in args.example_subhalos:
        rows = np.flatnonzero(
            (table["galaxy_id"] == subhalo)
            & (table["projection_id"] == args.example_projection)
        )
        if rows.size == 0:
            print(f"skip subhalo {subhalo}: not in table")
            continue
        idx = int(rows[0])
        panels = {
            "Truth": _surface_density(truth_mass[idx], r_edges, phi_edges)[:n_r],
            "Geometric baseline": _surface_density(baseline_mass[idx], r_edges, phi_edges)[:n_r],
            "MDN posterior mean": _surface_density(posterior_mass[idx], r_edges, phi_edges)[:n_r],
        }
        vmax = max(float(p.max()) for p in panels.values())
        norm = LogNorm(vmin=vmax * 3e-3, vmax=vmax)
        # Bar-contrast view: surface density relative to its own azimuthal
        # average at each radius. Bars appear as lobes along the x axis; this
        # is what a log colormap over the full dynamic range cannot show.
        contrasts = {
            name: panel / np.maximum(panel.mean(axis=1, keepdims=True), 1e-30) - 1.0
            for name, panel in panels.items()
        }

        fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.8), constrained_layout=True)
        for ax, (name, panel) in zip(axes[0], panels.items(), strict=True):
            mesh = ax.pcolormesh(x_v, y_v, panel, norm=norm, cmap="magma", rasterized=True)
            ax.set_title(name)
        fig.colorbar(
            mesh, ax=axes[0].tolist(), shrink=0.9, label=r"$\Sigma_\star$ [M$_\odot$ kpc$^{-2}$]"
        )
        for ax, (name, contrast) in zip(axes[1], contrasts.items(), strict=True):
            mesh_c = ax.pcolormesh(
                x_v, y_v, contrast, cmap="RdBu_r", vmin=-0.8, vmax=0.8, rasterized=True
            )
            ax.set_title(f"{name}: bar contrast")
        fig.colorbar(
            mesh_c,
            ax=axes[1].tolist(),
            shrink=0.9,
            label=r"$\Sigma / \langle\Sigma\rangle_\phi - 1$",
        )

        length = bar_lengths.get(int(subhalo))
        theta = np.linspace(0, 2 * np.pi, 200)
        for ax, edge_color in [(a, "w") for a in axes[0]] + [(a, "k") for a in axes[1]]:
            if length:
                ax.plot(
                    length * np.cos(theta),
                    length * np.sin(theta),
                    linestyle="--",
                    color=edge_color,
                    lw=1.1,
                    alpha=0.8,
                )
            ax.set_aspect("equal")
            ax.set_xlim(-args.display_radius_kpc, args.display_radius_kpc)
            ax.set_ylim(-args.display_radius_kpc, args.display_radius_kpc)
        for ax in axes[1]:
            ax.set_xlabel("x [kpc] (bar axis)")
        axes[0, 0].set_ylabel("y [kpc]")
        axes[1, 0].set_ylabel("y [kpc]")
        inc = float(table["metadata"][idx, 0])
        fig.suptitle(
            f"Held-out subhalo {subhalo} - viewed at i={inc:g} deg, deprojected face-on "
            f"(dashed: bar radius {length:.1f} kpc)",
            fontsize=15,
        )
        out = args.output_dir / f"example_reconstruction_{subhalo}.png"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(out.name)
    return written


def plot_pca_gallery(args, table, pca) -> str:
    r_edges = table["r_edges_kpc"].astype(np.float64)
    phi_edges = table["phi_edges_rad"].astype(np.float64)
    grid_shape = table["truth_density"].shape[1:]
    keep_r = r_edges <= 8.0 + 1e-6
    n_r = int(keep_r[1:].sum())
    x_v, y_v = _faceon_vertices(r_edges[: n_r + 1], phi_edges)
    components = pca["components"].reshape(-1, *grid_shape)
    evr = pca["explained_variance_ratio"]

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.6), constrained_layout=True)
    for k, ax in enumerate(axes.ravel()):
        comp_map = components[k].sum(axis=2)[:n_r]
        lim = float(np.percentile(np.abs(comp_map), 99.5))
        ax.pcolormesh(
            x_v, y_v, comp_map, cmap="RdBu_r", vmin=-lim, vmax=lim, rasterized=True
        )
        ax.set_aspect("equal")
        ax.set_title(f"PC{k + 1}  ({100 * evr[k]:.0f}% of variance)")
        ax.set_xlabel("x [kpc] (bar axis)")
        if k % 3 == 0:
            ax.set_ylabel("y [kpc]")
    fig.suptitle(
        "Leading residual shape patterns (face-on, red = baseline lacks mass)",
        fontsize=15,
    )
    out = args.output_dir / "pca_component_gallery.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return out.name


def plot_peanut_scatter(args) -> str:
    summary = json.loads(args.bar_region_json.read_text(encoding="utf-8"))
    per_galaxy = summary["per_galaxy"]
    true_ratio = np.array([g["peanut_ratio_true"] for g in per_galaxy.values()])
    post_ratio = np.array([g["peanut_ratio_posterior"] for g in per_galaxy.values()])
    bar_length = np.array([g["bar_length_kpc"] for g in per_galaxy.values()])
    r = float(np.corrcoef(true_ratio, post_ratio)[0, 1])

    fig, ax = plt.subplots(figsize=(6.4, 6.0), constrained_layout=True)
    sc = ax.scatter(true_ratio, post_ratio, c=bar_length, cmap="viridis", s=60, zorder=3)
    lims = [
        min(true_ratio.min(), post_ratio.min()) - 0.02,
        max(true_ratio.max(), post_ratio.max()) + 0.02,
    ]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.6)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("truth: RMS$_z$(along bar) / RMS$_z$(perpendicular)")
    ax.set_ylabel("posterior: same ratio")
    ax.set_title(f"Bar vertical anisotropy, per held-out galaxy (r = {r:.2f})")
    fig.colorbar(sc, ax=ax, shrink=0.85, label="bar radius [kpc]")
    out = args.output_dir / "peanut_ratio_scatter.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return out.name


SENSITIVITY_SUMMARIES = {
    "central_mass_fraction": "central mass fraction",
    "bar_axis_mass_fraction": "bar-axis mass fraction",
    "m2_profile": "m=2 profile",
    "radial_mass_profile": "radial mass profile",
}


def plot_sensitivity(args) -> str:
    metrics = json.loads(args.sensitivity_json.read_text(encoding="utf-8"))
    scenarios = {s["label"]: s for s in metrics["scenarios"]}
    reference = scenarios["reference"]["summaries"]
    offsets = [-5.0, -3.0, 0.0, 3.0, 5.0]
    labels = ["offset_-5_deg", "offset_-3_deg", "reference", "offset_+3_deg", "offset_+5_deg"]

    fig, ax = plt.subplots(figsize=(8.4, 5.8), constrained_layout=True)
    for key, name in SENSITIVITY_SUMMARIES.items():
        ratios = [
            scenarios[label]["summaries"][key]["mae"] / reference[key]["mae"]
            for label in labels
        ]
        ax.plot(offsets, ratios, "o-", lw=2, ms=7, label=name)
        gauss = scenarios["gaussian_sigma_3_deg"]["summaries"][key]["mae"] / reference[key]["mae"]
        ax.plot(0.0, gauss, "*", ms=13, color=ax.lines[-1].get_color(), zorder=4)
    ax.axhline(1.0, color="k", lw=0.8, alpha=0.5)
    ax.plot([], [], "k*", ms=12, label=r"random scatter $\sigma_i = 3$ deg")
    ax.set_xlabel(r"systematic inclination error $\Delta i$ [deg]")
    ax.set_ylabel("test MAE relative to true inclination")
    ax.set_title("Overestimating inclination is the damaging direction")
    ax.annotate(
        "thin-disk axis-ratio\ninversion errs this way",
        xy=(3.4, 1.45),
        fontsize=11,
        ha="left",
        color="0.3",
    )
    ax.legend(loc="upper left", fontsize=11)
    out = args.output_dir / "inclination_sensitivity.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return out.name


def plot_bar_region_mae(args) -> str:
    summary = json.loads(args.bar_region_json.read_text(encoding="utf-8"))
    regions = ["bar", "transition", "outer"]
    display = ["bar\n(R < L$_{bar}$)", "transition\n(1-1.5 L$_{bar}$)", "outer disk\n(> 1.5 L$_{bar}$)"]
    fig, ax = plt.subplots(figsize=(7.6, 5.4), constrained_layout=True)
    width = 0.36
    xs = np.arange(len(regions))
    for shift, label, color in ((-width / 2, "baseline", "0.65"), (width / 2, "posterior", "#b3403c")):
        med = [summary["regions"][r][label]["cell_rel_mae"]["median"] for r in regions]
        lo = [med[i] - summary["regions"][r][label]["cell_rel_mae"]["p16"] for i, r in enumerate(regions)]
        hi = [summary["regions"][r][label]["cell_rel_mae"]["p84"] - med[i] for i, r in enumerate(regions)]
        ax.bar(
            xs + shift,
            med,
            width,
            yerr=[lo, hi],
            capsize=4,
            color=color,
            label=f"{'geometric baseline' if label == 'baseline' else 'MDN posterior'}",
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(display)
    ax.set_ylabel("cell-level relative MAE")
    ax.set_title("The bar region is the best-recovered part of the galaxy")
    ax.legend()
    out = args.output_dir / "bar_region_mae.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return out.name


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = np.load(args.density_table)
    preds = np.load(args.predictions)
    pca = np.load(args.pca)
    written = plot_examples(args, table, preds)
    written.append(plot_pca_gallery(args, table, pca))
    written.append(plot_peanut_scatter(args))
    written.append(plot_sensitivity(args))
    written.append(plot_bar_region_mae(args))
    for name in written:
        print(f"wrote {args.output_dir / name}")


if __name__ == "__main__":
    main()
