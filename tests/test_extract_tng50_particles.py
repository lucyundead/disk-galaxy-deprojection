import subprocess
import sys

import h5py
import numpy as np
import pandas as pd


def test_extract_tng50_particles_writes_compact_files(tmp_path):
    tng_root = tmp_path / 'TNG50-1'
    snap_dir = tng_root / 'snapdir_099'
    offsets_dir = tng_root / 'postprocessing' / 'offsets'
    snap_dir.mkdir(parents=True)
    offsets_dir.mkdir(parents=True)
    with h5py.File(offsets_dir / 'offsets_099.hdf5', 'w') as handle:
        subhalo = handle.create_group('Subhalo')
        offsets = np.zeros((1, 6), dtype=np.int64)
        subhalo['SnapByType'] = offsets
    with h5py.File(snap_dir / 'snap_099.0.hdf5', 'w') as handle:
        header = handle.create_group('Header')
        stars = handle.create_group('PartType4')
        header.attrs['NumPart_ThisFile'] = np.array([0, 0, 0, 0, 2, 0])
        stars['Coordinates'] = np.ones((2, 3))
        stars['Masses'] = np.ones(2)
        stars['Velocities'] = np.zeros((2, 3))
        stars['GFM_StellarFormationTime'] = np.ones(2)
    manifest = tmp_path / 'manifest.csv'
    pd.DataFrame(
        {
            'subhalo_id': [0],
            'subhalo_pos_x_ckpc_h': [0.0],
            'subhalo_pos_y_ckpc_h': [0.0],
            'subhalo_pos_z_ckpc_h': [0.0],
            'star_particles': [2],
            'split': ['train'],
        }
    ).to_csv(manifest, index=False)
    output_dir = tmp_path / 'particles'

    result = subprocess.run(
        [
            sys.executable,
            'scripts/extract_tng50_particles.py',
            '--tng-root',
            str(tng_root),
            '--manifest',
            str(manifest),
            '--output-dir',
            str(output_dir),
            '--max-particles-per-galaxy',
            '10',
            '--hubble-param',
            '1.0',
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert 'wrote 1 compact particle files' in result.stdout
    assert (output_dir / 'subhalo_0.hdf5').exists()
