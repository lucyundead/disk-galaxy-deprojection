import json
import subprocess
import sys

import numpy as np
import torch

from dgdp.density3d import CylindricalGridSpec, cylindrical_bin_volumes
from dgdp.models.mdn import SummaryResidualMDN
from dgdp.types import Geometry
from scripts.build_tng50_baseline_density_grid import baseline_density_grid_from_image


def _write_tiny_scan_inputs(tmp_path, *, scale_height_kpc):
    rng = np.random.default_rng(7)
    n_rows = 4
    image_size = 8
    spec = CylindricalGridSpec(
        r_edges_kpc=np.array([0.0, 0.5, 1.0, 1.5]),
        phi_edges_rad=np.linspace(-np.pi, np.pi, 5),
        z_edges_kpc=np.array([-0.5, 0.0, 0.5]),
    )
    volumes = cylindrical_bin_volumes(spec)
    metadata = np.array(
        [
            [20.0, 0.0, 0.0],
            [20.0, 0.0, 45.0],
            [40.0, 0.0, 0.0],
            [40.0, 0.0, 45.0],
        ],
        dtype=np.float32,
    )
    coords = np.linspace(-0.7, 0.7, image_size)
    x_grid, y_grid = np.meshgrid(coords, coords)
    images = []
    baseline_density = []
    truth_density = []
    for index in range(n_rows):
        image = np.exp(-0.5 * (x_grid**2 + y_grid**2) / 0.3**2).astype(np.float32)
        image = image * (1.0 + 0.1 * index) * 1.0e8
        images.append(image)
        grid = baseline_density_grid_from_image(
            np.asarray(image, dtype=float),
            geometry=Geometry(
                inclination_deg=float(metadata[index, 0]),
                disk_pa_deg=float(metadata[index, 1]),
                bar_angle_deg=float(metadata[index, 2]),
            ),
            pixel_scale_kpc=0.2,
            spec=spec,
            vertical_scale_height_kpc=scale_height_kpc,
        )
        baseline_density.append(grid.density_msun_per_kpc3.astype(np.float32))
        truth_mass = grid.mass_msun * (1.0 + 0.05 * rng.standard_normal(grid.mass_msun.shape))
        truth_density.append((np.clip(truth_mass, 0.0, None) / volumes).astype(np.float32))

    table_path = tmp_path / "density_residual_table.npz"
    np.savez_compressed(
        table_path,
        images=np.stack(images),
        metadata=metadata,
        truth_density=np.stack(truth_density),
        baseline_density=np.stack(baseline_density),
        split=np.array(["test"] * n_rows),
        galaxy_id=np.arange(n_rows, dtype=int),
        projection_id=np.zeros(n_rows, dtype=int),
        r_edges_kpc=spec.r_edges_kpc.astype(np.float32),
        phi_edges_rad=spec.phi_edges_rad.astype(np.float32),
        z_edges_kpc=spec.z_edges_kpc.astype(np.float32),
    )

    n_cells = volumes.size
    n_coefficients = 3
    pca_path = tmp_path / "density_residual_pca.npz"
    np.savez_compressed(
        pca_path,
        components=rng.standard_normal((n_coefficients, n_cells)).astype(np.float32) * 1.0e-3,
        mean=np.zeros(n_cells, dtype=np.float32),
    )

    image_feature_size = 2
    input_dim = image_feature_size**2 + metadata.shape[1] + 2
    model = SummaryResidualMDN(
        input_dim=input_dim,
        output_dim=n_coefficients,
        hidden_dim=8,
        n_components=1,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "output_dim": n_coefficients,
            "hidden_dim": 8,
            "n_components": 1,
            "image_feature_size": image_feature_size,
            "central_pixel_scale_kpc": None,
        },
        run_dir / "density_residual_pca_mdn.pt",
    )
    np.savez_compressed(
        run_dir / "density_residual_pca_mdn_normalization.npz",
        x_mean=np.zeros(input_dim, dtype=np.float32),
        x_scale=np.ones(input_dim, dtype=np.float32),
        y_mean=np.zeros(n_coefficients, dtype=np.float32),
        y_scale=np.ones(n_coefficients, dtype=np.float32),
    )
    return table_path, pca_path, run_dir


def test_scan_inclination_sensitivity_script(tmp_path):
    scale_height_kpc = 0.2
    table_path, pca_path, run_dir = _write_tiny_scan_inputs(
        tmp_path,
        scale_height_kpc=scale_height_kpc,
    )
    output_dir = tmp_path / "scan"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/scan_inclination_sensitivity.py",
            "--density-table",
            str(table_path),
            "--pca",
            str(pca_path),
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(output_dir),
            "--offsets-deg",
            "3",
            "--gaussian-sigma-deg",
            "0",
            "--n-samples",
            "8",
            "--sample-batch-size",
            "4",
            "--pixel-scale-kpc",
            "0.2",
            "--vertical-scale-height-kpc",
            str(scale_height_kpc),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "wrote inclination sensitivity scan" in result.stdout
    metrics = json.loads(
        (output_dir / "inclination_sensitivity_metrics.json").read_text(encoding="utf-8")
    )
    labels = [scenario["label"] for scenario in metrics["scenarios"]]
    assert labels == ["reference", "offset_+3_deg"]

    reference, offset = metrics["scenarios"]
    assert reference["baseline_rebuild_max_rel_diff"] < 1.0e-5
    assert reference["delta_inclination_mean_deg"] == 0.0
    assert offset["delta_inclination_mean_deg"] == 3.0

    for scenario in metrics["scenarios"]:
        summaries = scenario["summaries"]
        assert set(summaries) == {
            "radial_mass_profile",
            "vertical_mass_profile",
            "vertical_rms_height",
            "central_mass_fraction",
            "bar_axis_mass_fraction",
            "m2_profile",
        }
        for values in summaries.values():
            assert np.isfinite(values["bias"])
            assert np.isfinite(values["mae"])
            assert 0.0 <= values["coverage_68"] <= 1.0
        assert set(scenario["by_true_inclination"]) == {"20", "40"}

    assert (output_dir / "inclination_sensitivity.md").exists()
