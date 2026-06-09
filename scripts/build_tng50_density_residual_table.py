from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from dgdp.density3d import CylindricalGridSpec, read_cylindrical_grid_spec_hdf5
from dgdp.manifest import validate_split_by_galaxy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--residual-table", type=Path, required=True)
    parser.add_argument("--truth-grid-dir", type=Path, required=True)
    parser.add_argument("--baseline-grid-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _truth_grid_path(truth_grid_dir: Path, subhalo_id: int) -> Path:
    return truth_grid_dir / f"subhalo_{subhalo_id}_density_cylindrical.hdf5"


def _load_density(path: Path) -> tuple[np.ndarray, float]:
    with h5py.File(path, "r") as handle:
        density = np.asarray(handle["density_msun_per_kpc3"], dtype=np.float32)
        grid_mass = float(handle.attrs["grid_mass_msun"])
    if not np.all(np.isfinite(density)):
        raise ValueError(f"density contains non-finite values: {path}")
    return density, grid_mass


def _specs_match(a: CylindricalGridSpec, b: CylindricalGridSpec) -> bool:
    return bool(
        np.allclose(a.r_edges_kpc, b.r_edges_kpc)
        and np.allclose(a.phi_edges_rad, b.phi_edges_rad)
        and np.allclose(a.z_edges_kpc, b.z_edges_kpc)
    )


def _require_matching_specs(truth_path: Path, baseline_path: Path) -> CylindricalGridSpec:
    truth_spec = read_cylindrical_grid_spec_hdf5(truth_path)
    baseline_spec = read_cylindrical_grid_spec_hdf5(baseline_path)
    if not _specs_match(truth_spec, baseline_spec):
        raise ValueError(f"grid edges do not match: {truth_path} vs {baseline_path}")
    return truth_spec


def _manifest_for_table(manifest: pd.DataFrame, table: np.lib.npyio.NpzFile) -> pd.DataFrame:
    if len(manifest) != len(table["galaxy_id"]):
        raise ValueError("manifest row count must match residual table rows")
    manifest = manifest.reset_index(drop=True).copy()
    if not np.array_equal(manifest["subhalo_id"].to_numpy(dtype=int), table["galaxy_id"].astype(int)):
        raise ValueError("manifest subhalo_id order must match residual table galaxy_id")
    if not np.array_equal(manifest["projection_id"].to_numpy(dtype=int), table["projection_id"].astype(int)):
        raise ValueError("manifest projection_id order must match residual table projection_id")
    return manifest


def _baseline_catalog_index(catalog: pd.DataFrame) -> dict[tuple[int, int], object]:
    return {
        (int(row.subhalo_id), int(row.projection_id)): row
        for row in catalog.itertuples(index=False)
    }


def build_density_residual_table(
    *,
    manifest: pd.DataFrame,
    table: np.lib.npyio.NpzFile,
    truth_grid_dir: Path,
    baseline_grid_dir: Path,
) -> dict[str, np.ndarray]:
    manifest = _manifest_for_table(manifest, table)
    baseline_catalog = pd.read_csv(baseline_grid_dir / "baseline_density_grid_catalog.csv")
    baseline_by_key = _baseline_catalog_index(baseline_catalog)

    truth_density = []
    baseline_density = []
    truth_grid_mass = []
    baseline_grid_mass = []
    true_files = []
    baseline_files = []
    spec: CylindricalGridSpec | None = None
    for row in manifest.itertuples(index=False):
        subhalo_id = int(row.subhalo_id)
        projection_id = int(row.projection_id)
        baseline_row = baseline_by_key[(subhalo_id, projection_id)]
        truth_path = truth_grid_dir / str(getattr(baseline_row, "true_density_file", ""))
        if not truth_path.exists():
            truth_path = _truth_grid_path(truth_grid_dir, subhalo_id)
        baseline_rel = Path(str(baseline_row.baseline_density_file))
        baseline_path = baseline_grid_dir / baseline_rel
        row_spec = _require_matching_specs(truth_path, baseline_path)
        if spec is None:
            spec = row_spec
        elif not _specs_match(spec, row_spec):
            raise ValueError("grid edges do not match across residual table rows")

        truth_grid, truth_mass = _load_density(truth_path)
        baseline_grid, baseline_mass = _load_density(baseline_path)
        if np.any(baseline_grid < 0.0):
            raise ValueError(f"baseline density contains negative values: {baseline_path}")
        truth_density.append(truth_grid)
        baseline_density.append(baseline_grid)
        truth_grid_mass.append(truth_mass)
        baseline_grid_mass.append(baseline_mass)
        true_files.append(str(truth_path.name))
        baseline_files.append(str(baseline_rel))

    if spec is None:
        raise ValueError("residual table has no rows")

    truth_array = np.stack(truth_density).astype(np.float32)
    baseline_array = np.stack(baseline_density).astype(np.float32)
    return {
        "images": np.asarray(table["images"], dtype=np.float32),
        "truth_density": truth_array,
        "baseline_density": baseline_array,
        "delta_density": (truth_array - baseline_array).astype(np.float32),
        "metadata": np.asarray(table["metadata"], dtype=np.float32),
        "split": np.asarray(table["split"]),
        "galaxy_id": np.asarray(table["galaxy_id"], dtype=int),
        "projection_id": np.asarray(table["projection_id"], dtype=int),
        "r_edges_kpc": spec.r_edges_kpc.astype(np.float32),
        "phi_edges_rad": spec.phi_edges_rad.astype(np.float32),
        "z_edges_kpc": spec.z_edges_kpc.astype(np.float32),
        "truth_grid_mass_msun": np.asarray(truth_grid_mass, dtype=np.float32),
        "baseline_grid_mass_msun": np.asarray(baseline_grid_mass, dtype=np.float32),
        "true_density_file": np.asarray(true_files, dtype=str),
        "baseline_density_file": np.asarray(baseline_files, dtype=str),
    }


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    if not validate_split_by_galaxy(manifest.rename(columns={"subhalo_id": "galaxy_id"})):
        raise RuntimeError("manifest leaks subhalo IDs across splits")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(args.residual_table) as table:
        arrays = build_density_residual_table(
            manifest=manifest,
            table=table,
            truth_grid_dir=args.truth_grid_dir,
            baseline_grid_dir=args.baseline_grid_dir,
        )
    np.savez_compressed(args.output, **arrays)
    diagnostics = {
        "n_rows": int(len(arrays["galaxy_id"])),
        "density_shape": list(arrays["truth_density"].shape[1:]),
        "median_abs_delta_density": float(np.median(np.abs(arrays["delta_density"]))),
    }
    diagnostics_path = args.output.parent / "density_residual_table_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(f"wrote density residual table to {args.output}")


if __name__ == "__main__":
    main()
