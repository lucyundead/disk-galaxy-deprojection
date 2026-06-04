import subprocess
import sys

import h5py
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


def test_plot_tng50_faceon_density_streams_from_tng_chunks(tmp_path):
    tng_root = tmp_path / "TNG50-1"
    snap_dir = tng_root / "snapdir_099"
    offsets_dir = tng_root / "postprocessing" / "offsets"
    snap_dir.mkdir(parents=True)
    offsets_dir.mkdir(parents=True)
    with h5py.File(offsets_dir / "offsets_099.hdf5", "w") as handle:
        subhalo = handle.create_group("Subhalo")
        subhalo["SnapByType"] = np.zeros((1, 6), dtype=np.int64)

    theta = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    radius = np.linspace(1.0, 10.0, 128)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    coords = np.column_stack((x, y, np.zeros_like(x)))
    velocities = np.column_stack((-y, x, np.zeros_like(x)))
    with h5py.File(snap_dir / "snap_099.0.hdf5", "w") as handle:
        header = handle.create_group("Header")
        stars = handle.create_group("PartType4")
        header.attrs["NumPart_ThisFile"] = np.array([0, 0, 0, 0, len(coords), 0])
        stars["Coordinates"] = coords
        stars["Masses"] = np.ones(len(coords))
        stars["Velocities"] = velocities
        stars["GFM_StellarFormationTime"] = np.ones(len(coords))

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "subhalo_id": [0],
            "split": ["train"],
            "star_particles": [len(coords)],
            "subhalo_pos_x_ckpc_h": [0.0],
            "subhalo_pos_y_ckpc_h": [0.0],
            "subhalo_pos_z_ckpc_h": [0.0],
            "bar_a2_catalog": [0.3],
            "bar_length_catalog": [2.5],
        }
    ).to_csv(manifest, index=False)
    output_dir = tmp_path / "streamed_plots"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/plot_tng50_faceon_density.py",
            "--manifest",
            str(manifest),
            "--tng-root",
            str(tng_root),
            "--output-dir",
            str(output_dir),
            "--image-size",
            "64",
            "--radius-kpc",
            "12",
            "--hubble-param",
            "1.0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "wrote 1 face-on density plots" in result.stdout
    assert (output_dir / "subhalo_0_faceon_density.png").read_bytes().startswith(b"\x89PNG")
