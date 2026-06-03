from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dgdp.baseline import baseline_particles_from_image
from dgdp.config import load_config
from dgdp.dataset import build_residual_row
from dgdp.projection import project_to_mock_image
from dgdp.summaries import extract_summary_vector
from dgdp.tng50 import load_particle_set_hdf5
from dgdp.types import Geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--particle-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    manifest = pd.read_csv(args.manifest)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    if 'inclination_deg' not in manifest.columns:
        manifest['inclination_deg'] = rng.uniform(0, cfg.max_inclination_deg, size=len(manifest))
    if 'disk_pa_deg' not in manifest.columns:
        manifest['disk_pa_deg'] = rng.uniform(0, 180, size=len(manifest))
    if 'bar_angle_deg' not in manifest.columns:
        manifest['bar_angle_deg'] = rng.uniform(0, 180, size=len(manifest))
    if 'projection_id' not in manifest.columns:
        manifest['projection_id'] = 0

    rows = []
    for row in manifest.itertuples(index=False):
        subhalo_id = int(row.subhalo_id)
        projection_id = int(getattr(row, 'projection_id', 0))
        geometry = Geometry(
            inclination_deg=float(getattr(row, 'inclination_deg', 0.0)),
            disk_pa_deg=float(getattr(row, 'disk_pa_deg', 0.0)),
            bar_angle_deg=float(getattr(row, 'bar_angle_deg', 0.0)),
        )
        particles = load_particle_set_hdf5(
            args.particle_dir / f'subhalo_{subhalo_id}.hdf5',
            length_unit_kpc=1.0,
            mass_unit_msun=1.0,
        )
        mock = project_to_mock_image(
            particles,
            geometry,
            cfg.image_size,
            cfg.pixel_scale_kpc,
            cfg.psf_sigma_pixels,
            cfg.noise_sigma_fraction,
            seed=cfg.seed + subhalo_id * 100 + projection_id,
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
                galaxy_id=subhalo_id,
                projection_id=projection_id,
                split=str(row.split),
                image=mock.image,
                true_summary=true_summary,
                baseline_summary=baseline_summary,
                geometry=geometry,
            )
        )

    np.savez_compressed(
        output_dir / 'residual_table.npz',
        images=np.stack([r['image'] for r in rows]),
        baseline=np.stack([r['baseline'] for r in rows]),
        truth=np.stack([r['truth'] for r in rows]),
        delta=np.stack([r['delta'] for r in rows]),
        metadata=np.stack([r['metadata'] for r in rows]),
        split=np.array([r['split'] for r in rows]),
        galaxy_id=np.array([r['galaxy_id'] for r in rows], dtype=int),
        projection_id=np.array([r['projection_id'] for r in rows], dtype=int),
        summary_names=np.array(rows[0]['summary_names'], dtype=str),
    )
    manifest.to_csv(output_dir / 'manifest.csv', index=False)
    print(f'wrote TNG50 residual table to {output_dir / chr(34)}residual_table.npz{chr(34)}')


if __name__ == '__main__':
    main()
