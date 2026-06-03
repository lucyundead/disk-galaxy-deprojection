from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dgdp.baseline import baseline_particles_from_image
from dgdp.config import load_config
from dgdp.dataset import build_residual_row
from dgdp.manifest import make_synthetic_manifest, validate_split_by_galaxy
from dgdp.projection import project_to_mock_image
from dgdp.summaries import extract_summary_vector
from dgdp.synthetic import make_barred_galaxy
from dgdp.types import Geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/milestone1_synthetic"))
    parser.add_argument("--n-galaxies", type=int, default=None)
    parser.add_argument("--projections-per-galaxy", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    n_galaxies = args.n_galaxies or cfg.n_galaxies
    projections_per_galaxy = args.projections_per_galaxy or cfg.projections_per_galaxy
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = make_synthetic_manifest(
        seed=cfg.seed,
        n_galaxies=n_galaxies,
        projections_per_galaxy=projections_per_galaxy,
        max_inclination_deg=cfg.max_inclination_deg,
    )
    if not validate_split_by_galaxy(manifest):
        raise RuntimeError("manifest leaks galaxy IDs across splits")

    rows = []
    for row in manifest.itertuples(index=False):
        particles = make_barred_galaxy(
            seed=int(row.galaxy_id),
            n_particles=6000,
            total_mass_msun=5.0e10,
        )
        geometry = Geometry(
            inclination_deg=float(row.inclination_deg),
            disk_pa_deg=float(row.disk_pa_deg),
            bar_angle_deg=float(row.bar_angle_deg),
        )
        mock = project_to_mock_image(
            particles,
            geometry,
            cfg.image_size,
            cfg.pixel_scale_kpc,
            cfg.psf_sigma_pixels,
            cfg.noise_sigma_fraction,
            seed=int(row.seed),
        )
        baseline_particles = baseline_particles_from_image(mock, vertical_scale_height_kpc=0.4)
        true_summary = extract_summary_vector(
            particles,
            cfg.radial_bins_kpc,
            cfg.vertical_bins_kpc,
            bar_angle_deg=geometry.bar_angle_deg,
        )
        baseline_summary = extract_summary_vector(
            baseline_particles,
            cfg.radial_bins_kpc,
            cfg.vertical_bins_kpc,
            bar_angle_deg=geometry.bar_angle_deg,
        )
        rows.append(
            build_residual_row(
                galaxy_id=int(row.galaxy_id),
                projection_id=int(row.projection_id),
                split=str(row.split),
                image=mock.image,
                true_summary=true_summary,
                baseline_summary=baseline_summary,
                geometry=geometry,
            )
        )

    np.savez_compressed(
        output_dir / "residual_table.npz",
        images=np.stack([r["image"] for r in rows]),
        baseline=np.stack([r["baseline"] for r in rows]),
        truth=np.stack([r["truth"] for r in rows]),
        delta=np.stack([r["delta"] for r in rows]),
        metadata=np.stack([r["metadata"] for r in rows]),
        split=np.array([r["split"] for r in rows]),
        galaxy_id=np.array([r["galaxy_id"] for r in rows], dtype=int),
        projection_id=np.array([r["projection_id"] for r in rows], dtype=int),
        summary_names=np.array(rows[0]["summary_names"], dtype=str),
    )
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    print(f"wrote residual table to {output_dir / 'residual_table.npz'}")


if __name__ == "__main__":
    main()
