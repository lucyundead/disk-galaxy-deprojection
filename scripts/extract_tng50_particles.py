from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from dgdp.tng50 import load_subhalo_stars_from_chunks, write_particle_set_hdf5
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--tng-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--snapshot', type=int, default=99)
    parser.add_argument('--max-particles-per-galaxy', type=int, default=80000)
    parser.add_argument('--hubble-param', type=float, default=0.6774)
    return parser.parse_args()
def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    wrote = 0
    for row in manifest.itertuples(index=False):
        subhalo_id = int(row.subhalo_id)
        center = np.array(
            [
                float(row.subhalo_pos_x_ckpc_h),
                float(row.subhalo_pos_y_ckpc_h),
                float(row.subhalo_pos_z_ckpc_h),
            ]
        )
        particles = load_subhalo_stars_from_chunks(
            snap_dir=args.tng_root / f'snapdir_{args.snapshot:03d}',
            offsets_path=args.tng_root / 'postprocessing' / 'offsets' / f'offsets_{args.snapshot:03d}.hdf5',
            subhalo_id=subhalo_id,
            star_particle_count=int(row.star_particles),
            subhalo_center_ckpc_h=center,
            snapshot=args.snapshot,
            hubble_param=args.hubble_param,
            max_particles=args.max_particles_per_galaxy,
        )
        write_particle_set_hdf5(output_dir / f'subhalo_{subhalo_id}.hdf5', particles, subhalo_id=subhalo_id)
        wrote += 1

    print(f'wrote {wrote} compact particle files to {output_dir}')
if __name__ == '__main__':
    main()