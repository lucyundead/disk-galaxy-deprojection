import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from dgdp.density3d import CylindricalGridSpec, cylindrical_bin_volumes

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_inputs(tmp_path):
    rng = np.random.default_rng(11)
    spec = CylindricalGridSpec(
        r_edges_kpc=np.array([0.0, 1.0, 2.0, 3.0]),
        phi_edges_rad=np.linspace(-np.pi, np.pi, 9),
        z_edges_kpc=np.linspace(-1.0, 1.0, 5),
    )
    volumes = cylindrical_bin_volumes(spec).astype(np.float32)
    grid_shape = volumes.shape
    n_rows = 4
    truth_density = rng.uniform(0.5, 2.0, size=(n_rows, *grid_shape)).astype(np.float32)
    truth_mass = truth_density * volumes[None, ...]
    grid_mass = truth_mass.sum(axis=(1, 2, 3)).astype(np.float32)

    table_path = tmp_path / "table.npz"
    np.savez(
        table_path,
        truth_density=truth_density,
        baseline_density=truth_density.copy(),
        delta_density=np.zeros_like(truth_density),
        metadata=np.array(
            [[20.0, 0.0, 0.0], [40.0, 0.0, 0.0], [20.0, 0.0, 0.0], [40.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        split=np.array(["test"] * n_rows),
        galaxy_id=np.array([7, 7, 9, 9], dtype=np.int64),
        projection_id=np.array([0, 1, 0, 1], dtype=np.int64),
        r_edges_kpc=spec.r_edges_kpc.astype(np.float32),
        phi_edges_rad=spec.phi_edges_rad.astype(np.float32),
        z_edges_kpc=spec.z_edges_kpc.astype(np.float32),
        truth_grid_mass_msun=grid_mass,
        baseline_grid_mass_msun=grid_mass,
    )

    n_cells = int(np.prod(grid_shape))
    n_components = 2
    pca_path = tmp_path / "pca.npz"
    np.savez(
        pca_path,
        components=rng.normal(0.0, 0.01, size=(n_components, n_cells)).astype(
            np.float32
        ),
        mean=np.zeros(n_cells, dtype=np.float32),
    )

    # Zero mean coefficients reproduce the baseline exactly, and the baseline
    # equals the truth, so every error metric must vanish. The samples get a
    # small symmetric spread so the 68 percent interval has finite width and
    # brackets the truth (identical samples would make the interval degenerate
    # and coverage sensitive to float32 summation order).
    sampled = np.zeros((n_rows, 3, n_components), dtype=np.float32)
    sampled[:, 0, 0] = -0.01
    sampled[:, 2, 0] = 0.01
    predictions_path = tmp_path / "predictions.npz"
    np.savez(
        predictions_path,
        sampled_coefficients=sampled,
        posterior_mean_coefficients=np.zeros((n_rows, n_components), dtype=np.float32),
        posterior_mean_mass=truth_mass.astype(np.float32),
        split=np.array(["test"] * n_rows),
        galaxy_id=np.array([7, 7, 9, 9], dtype=np.int64),
        projection_id=np.array([0, 1, 0, 1], dtype=np.int64),
        metadata=np.zeros((n_rows, 3), dtype=np.float32),
    )

    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "subhalo_id,bar_length\n7,1.2\n9,1.4\n", encoding="utf-8"
    )
    return table_path, pca_path, predictions_path, manifest_path


def test_bar_region_recovery_exact_posterior(tmp_path):
    table_path, pca_path, predictions_path, manifest_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "analyze_bar_region_recovery.py"),
            "--predictions",
            str(predictions_path),
            "--density-table",
            str(table_path),
            "--pca",
            str(pca_path),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    summary = json.loads((output_dir / "bar_region_recovery.json").read_text())
    assert summary["n_rows"] == 4
    assert summary["n_galaxies"] == 2
    assert summary["rebuild_max_rel_diff"] < 1e-5

    for region in ("bar", "transition", "outer"):
        for label in ("posterior", "baseline"):
            stats = summary["regions"][region][label]
            assert abs(stats["cell_rel_mae"]["median"]) < 1e-6
            assert abs(stats["mass_fraction_err"]["median"]) < 1e-6
            assert abs(stats["m2_amp_err"]["median"]) < 1e-6
            assert abs(stats["m2_phase_abs_err_deg"]["median"]) < 1e-3

    assert abs(summary["bar_rms_z_posterior_ratio"]["median"] - 1.0) < 1e-6
    assert abs(summary["bar_rms_z_baseline_ratio"]["median"] - 1.0) < 1e-6
    assert (
        abs(
            summary["peanut_ratio_posterior"]["median"]
            - summary["peanut_ratio_true"]["median"]
        )
        < 1e-6
    )
    assert summary["coverage68_bar_mass_fraction"] == 1.0
    assert summary["coverage68_bar_m2_amp"] == 1.0
    assert (output_dir / "bar_region_recovery.md").exists()
