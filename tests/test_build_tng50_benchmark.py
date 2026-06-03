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
