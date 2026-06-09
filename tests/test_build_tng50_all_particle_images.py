import subprocess
import sys

import numpy as np
import pandas as pd

from dgdp.orientation import align_particles_to_disk_bar_frame
from dgdp.projection import project_to_mock_image
from dgdp.tng50 import write_particle_set_hdf5
from dgdp.types import Geometry, ParticleSet


def test_all_particle_image_script_writes_clean_projection_from_fallback_particles(tmp_path):
    particle_dir = tmp_path / "particles"
    particle_dir.mkdir()
    manifest = tmp_path / "manifest.csv"
    output_dir = tmp_path / "run"
    theta = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    positions = np.column_stack(
        (
            3.0 * np.cos(theta),
            1.5 * np.sin(theta),
            np.zeros_like(theta),
        )
    )
    particles = ParticleSet(
        positions_kpc=positions,
        masses_msun=np.ones(len(theta)) * 2.0,
        velocities_kms=np.column_stack((-positions[:, 1], positions[:, 0], np.zeros(len(theta)))),
    )
    write_particle_set_hdf5(particle_dir / "subhalo_101.hdf5", particles, subhalo_id=101)
    pd.DataFrame(
        [
            {
                "subhalo_id": 101,
                "snapshot": 99,
                "star_particles": len(theta),
                "subhalo_pos_x_ckpc_h": 0.0,
                "subhalo_pos_y_ckpc_h": 0.0,
                "subhalo_pos_z_ckpc_h": 0.0,
                "projection_id": 0,
                "split": "train",
                "inclination_deg": 0.0,
                "disk_pa_deg": 0.0,
                "bar_angle_deg": 0.0,
            }
        ]
    ).to_csv(manifest, index=False)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_tng50_all_particle_images.py",
            "--config",
            "configs/milestone2b.clean3d.toml",
            "--manifest",
            str(manifest),
            "--particle-dir",
            str(particle_dir),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "wrote all-particle TNG50 image table" in result.stdout
    table = np.load(output_dir / "residual_table.npz")
    assert table["images"].shape == (1, 192, 192)
    assert table["summary_names"].shape == (0,)
    aligned = align_particles_to_disk_bar_frame(particles, normal_radius_kpc=30.0, bar_radius_kpc=5.0).particles
    expected = project_to_mock_image(
        aligned,
        Geometry(inclination_deg=0.0, disk_pa_deg=0.0, bar_angle_deg=0.0),
        image_size=192,
        pixel_scale_kpc=0.35,
        psf_sigma_pixels=0.0,
        noise_sigma_fraction=0.0,
        seed=0,
    ).image
    assert np.allclose(table["images"][0], expected)
    diagnostics = pd.read_csv(output_dir / "all_particle_image_catalog.csv")
    assert diagnostics["source_particles"].tolist() == ["compact_particle_file"]
    assert np.isclose(diagnostics["image_mass_msun"].iloc[0], expected.sum())
