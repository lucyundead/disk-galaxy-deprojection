import subprocess
import sys

import numpy as np
import pandas as pd

from dgdp.tng50 import write_particle_set_hdf5
from dgdp.types import ParticleSet


def test_plot_tng50_faceon_density_writes_pngs_and_gallery(tmp_path):
    particle_dir = tmp_path / "particles"
    particle_dir.mkdir()
    theta = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    radius = np.linspace(1.0, 10.0, 128)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    positions = np.column_stack((x, y, np.zeros_like(x)))
    velocities = np.column_stack((-y, x, np.zeros_like(x)))
    particles = ParticleSet(
        positions_kpc=positions,
        masses_msun=np.ones(len(theta)) * 1.0e6,
        velocities_kms=velocities,
    )
    write_particle_set_hdf5(particle_dir / "subhalo_42.hdf5", particles, subhalo_id=42)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "subhalo_id": [42],
            "split": ["train"],
            "bar_a2_catalog": [0.3],
            "bar_length_catalog": [2.5],
        }
    ).to_csv(manifest, index=False)
    output_dir = tmp_path / "plots"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/plot_tng50_faceon_density.py",
            "--manifest",
            str(manifest),
            "--particle-dir",
            str(particle_dir),
            "--output-dir",
            str(output_dir),
            "--image-size",
            "64",
            "--radius-kpc",
            "12",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    png = output_dir / "subhalo_42_faceon_density.png"
    assert "wrote 1 face-on density plots" in result.stdout
    assert png.read_bytes().startswith(b"\x89PNG")
    assert (output_dir / "montage.png").read_bytes().startswith(b"\x89PNG")
    assert (output_dir / "gallery.html").exists()
    assert (output_dir / "faceon_density_metadata.csv").exists()
