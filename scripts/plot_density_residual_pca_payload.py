from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _setup_matplotlib():
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
    return plt


def plot_example(payload: np.lib.npyio.NpzFile, index: int, output_dir: Path) -> list[Path]:
    plt = _setup_matplotlib()
    mass_maps = {
        "Truth": payload["truth_mass"][index],
        "Baseline": payload["baseline_mass"][index],
        "PCA corrected": payload["model_mass"][index],
        "Mean train residual": payload["mean_train_mass"][index],
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
        image_artist = axis.imshow(
            image.T,
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(name)
        axis.set_xlabel("R bin")
        axis.set_ylabel("phi bin")
    fig.colorbar(
        image_artist,
        ax=axes[0].tolist(),
        shrink=0.86,
        label="log10 mass per R-phi cell",
    )

    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.0,
        0.84,
        "Residual panels\n(candidate - truth)\n\nLess color structure\nmeans closer to truth.",
        va="top",
    )
    for axis, (name, residual) in zip(axes[1, 1:], residual_panels.items(), strict=True):
        residual_artist = axis.imshow(
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
    fig.colorbar(
        residual_artist,
        ax=axes[1, 1:].tolist(),
        shrink=0.86,
        label="Msun per R-phi cell",
    )
    galaxy = int(payload["galaxy_id"][index])
    projection = int(payload["projection_id"][index])
    inc, _, bar_angle = payload["metadata"][index]
    fig.suptitle(
        f"Held-out subhalo {galaxy}, projection {projection}: "
        f"inclination={inc:g} deg, bar angle={bar_angle:g} deg",
        y=1.02,
    )
    stem = output_dir / f"test_example_subhalo_{galaxy}_projection_{projection}"
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.payload) as payload:
        for index in range(payload["truth_mass"].shape[0]):
            plot_example(payload, index, args.output_dir)
    print(f"wrote matplotlib density residual figures to {args.output_dir}")


if __name__ == "__main__":
    main()
