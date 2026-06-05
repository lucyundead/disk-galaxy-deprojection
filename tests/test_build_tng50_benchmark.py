import subprocess
import sys

import numpy as np
import pandas as pd

from dgdp.tng50 import write_particle_set_hdf5
from dgdp.types import ParticleSet


def test_build_tng50_benchmark_writes_residual_table(tmp_path):
    particle_dir = tmp_path / 'particles'
    particle_dir.mkdir()
    manifest = tmp_path / 'manifest.csv'
    pd.DataFrame(
        {
            'subhalo_id': [0],
            'projection_id': [0],
            'split': ['train'],
            'inclination_deg': [0.0],
            'disk_pa_deg': [0.0],
            'bar_angle_deg': [0.0],
        }
    ).to_csv(manifest, index=False)
    particles = ParticleSet(
        positions_kpc=np.random.default_rng(1).normal(size=(2000, 3)),
        masses_msun=np.ones(2000) * 1.0e6,
    )
    write_particle_set_hdf5(particle_dir / 'subhalo_0.hdf5', particles, subhalo_id=0)
    output_dir = tmp_path / 'run'

    result = subprocess.run(
        [
            sys.executable,
            'scripts/build_tng50_benchmark.py',
            '--config',
            'configs/milestone1.synthetic.toml',
            '--manifest',
            str(manifest),
            '--particle-dir',
            str(particle_dir),
            '--output-dir',
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert 'wrote TNG50 residual table' in result.stdout
    assert (output_dir / 'residual_table.npz').exists()


def test_build_tng50_benchmark_expands_default_projection_grid(tmp_path):
    particle_dir = tmp_path / 'particles'
    particle_dir.mkdir()
    manifest = tmp_path / 'manifest.csv'
    pd.DataFrame(
        {
            'subhalo_id': [7],
            'split': ['train'],
        }
    ).to_csv(manifest, index=False)
    rng = np.random.default_rng(2)
    positions = rng.normal(size=(2000, 3))
    positions[:, 2] *= 0.1
    particles = ParticleSet(
        positions_kpc=positions,
        masses_msun=np.ones(2000) * 1.0e6,
        velocities_kms=np.column_stack((-positions[:, 1], positions[:, 0], np.zeros(2000))),
    )
    write_particle_set_hdf5(particle_dir / 'subhalo_7.hdf5', particles, subhalo_id=7)
    output_dir = tmp_path / 'run_grid'

    subprocess.run(
        [
            sys.executable,
            'scripts/build_tng50_benchmark.py',
            '--config',
            'configs/milestone1.synthetic.toml',
            '--manifest',
            str(manifest),
            '--particle-dir',
            str(particle_dir),
            '--output-dir',
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    table = np.load(output_dir / 'residual_table.npz')
    expanded = pd.read_csv(output_dir / 'manifest.csv')
    assert table['images'].shape[0] == 9
    assert expanded['inclination_deg'].tolist() == [20.0, 20.0, 20.0, 40.0, 40.0, 40.0, 60.0, 60.0, 60.0]
    assert expanded['bar_angle_deg'].tolist() == [0.0, 45.0, 90.0, 0.0, 45.0, 90.0, 0.0, 45.0, 90.0]
