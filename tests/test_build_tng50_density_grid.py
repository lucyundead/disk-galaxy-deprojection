import subprocess
import sys

import h5py
import numpy as np
import pandas as pd

from dgdp.tng50 import write_particle_set_hdf5
from dgdp.types import ParticleSet


def test_density_grid_script_writes_grid_files_from_particle_dir(tmp_path):
    manifest = tmp_path / "manifest.csv"
    particle_dir = tmp_path / "particles"
    output_dir = tmp_path / "density"
    pd.DataFrame(
        [
            {
                "subhalo_id": 101,
                "snapshot": 99,
                "star_particles": 4,
                "subhalo_pos_x_ckpc_h": 0.0,
                "subhalo_pos_y_ckpc_h": 0.0,
                "subhalo_pos_z_ckpc_h": 0.0,
                "split": "train",
            }
        ]
    ).to_csv(manifest, index=False)
    particles = ParticleSet(
        positions_kpc=np.array(
            [
                [0.5, 0.0, 0.0],
                [0.0, 1.0, 0.2],
                [-1.5, 0.0, -0.2],
                [0.0, -2.0, 0.0],
            ]
        ),
        masses_msun=np.array([1.0, 2.0, 3.0, 4.0]),
        velocities_kms=np.array(
            [
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
    )
    write_particle_set_hdf5(particle_dir / "subhalo_101.hdf5", particles, subhalo_id=101)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_tng50_density_grid.py",
            "--manifest",
            str(manifest),
            "--particle-dir",
            str(particle_dir),
            "--output-dir",
            str(output_dir),
            "--r-min-kpc",
            "0.1",
            "--r-max-kpc",
            "5.0",
            "--n-r",
            "8",
            "--n-phi",
            "12",
            "--z-max-kpc",
            "2.0",
            "--n-z",
            "8",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "wrote 1 cylindrical density grids" in result.stdout
    assert (output_dir / "density_grid_catalog.csv").exists()
    with h5py.File(output_dir / "subhalo_101_density_cylindrical.hdf5", "r") as handle:
        assert handle["density_msun_per_kpc3"].shape == (8, 12, 8)
        assert handle["mass_msun"].shape == (8, 12, 8)
        assert np.isclose(handle.attrs["grid_mass_msun"], 10.0)
        assert np.isfinite(handle.attrs["faceon_image_l1_fraction"])
        assert handle.attrs["coordinate_frame"] == "disk_bar_aligned"
