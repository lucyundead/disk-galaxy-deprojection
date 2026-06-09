from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dgdp.config import load_config
from dgdp.coordinates import rotate_points, rotation_matrix_x, rotation_matrix_z
from dgdp.density3d import (
    CylindricalDensityGrid,
    CylindricalGridSpec,
    density_grid_from_mass_grid,
    read_cylindrical_grid_spec_hdf5,
    write_cylindrical_density_hdf5,
)
from dgdp.manifest import validate_split_by_galaxy
from dgdp.projection import project_to_mock_image
from dgdp.types import Geometry, ParticleSet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--residual-table", type=Path, required=True)
    parser.add_argument("--truth-grid-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vertical-scale-height-kpc", type=float, default=0.4)
    return parser.parse_args()


def truth_grid_path(truth_grid_dir: Path, subhalo_id: int) -> Path:
    return truth_grid_dir / f"subhalo_{subhalo_id}_density_cylindrical.hdf5"


def baseline_grid_path(subhalo_id: int, projection_id: int) -> Path:
    return Path(f"subhalo_{subhalo_id}_projection_{projection_id}_baseline_density_cylindrical.hdf5")


def sech2_vertical_bin_weights(z_edges_kpc: np.ndarray, *, scale_height_kpc: float) -> np.ndarray:
    if scale_height_kpc <= 0.0:
        raise ValueError("scale_height_kpc must be positive")
    z_edges = np.asarray(z_edges_kpc, dtype=float)
    antiderivative = np.tanh(z_edges / scale_height_kpc)
    weights = scale_height_kpc * np.diff(antiderivative)
    norm = float(np.sum(weights))
    if norm <= 0.0:
        raise ValueError("vertical grid must span a nonzero sech^2 integral")
    return weights / norm


def _bilinear_image_mass_per_sky_area(
    image: np.ndarray,
    *,
    x_sky_kpc: np.ndarray,
    y_sky_kpc: np.ndarray,
    pixel_scale_kpc: float,
) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError("image must be a square 2D array")
    size = image.shape[0]
    col = x_sky_kpc / pixel_scale_kpc + 0.5 * (size - 1)
    row = y_sky_kpc / pixel_scale_kpc + 0.5 * (size - 1)
    valid = (row >= 0.0) & (row <= size - 1) & (col >= 0.0) & (col <= size - 1)
    row0 = np.floor(np.clip(row, 0.0, size - 1)).astype(int)
    col0 = np.floor(np.clip(col, 0.0, size - 1)).astype(int)
    row1 = np.minimum(row0 + 1, size - 1)
    col1 = np.minimum(col0 + 1, size - 1)
    dy = np.clip(row - row0, 0.0, 1.0)
    dx = np.clip(col - col0, 0.0, 1.0)
    values = (
        image[row0, col0] * (1.0 - dx) * (1.0 - dy)
        + image[row0, col1] * dx * (1.0 - dy)
        + image[row1, col0] * (1.0 - dx) * dy
        + image[row1, col1] * dx * dy
    )
    values = np.where(valid, values, 0.0)
    return values / pixel_scale_kpc**2


def deprojected_surface_mass_grid_from_image(
    image: np.ndarray,
    *,
    geometry: Geometry,
    pixel_scale_kpc: float,
    spec: CylindricalGridSpec,
) -> np.ndarray:
    r_edges = np.asarray(spec.r_edges_kpc, dtype=float)
    phi_edges = np.asarray(spec.phi_edges_rad, dtype=float)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    rr, pp = np.meshgrid(r_centers, phi_centers, indexing="ij")
    points = np.column_stack(
        (
            (rr * np.cos(pp)).ravel(),
            (rr * np.sin(pp)).ravel(),
            np.zeros(rr.size, dtype=float),
        )
    )
    projected = rotate_points(points, rotation_matrix_z(geometry.bar_angle_deg))
    projected = rotate_points(projected, rotation_matrix_z(geometry.disk_pa_deg))
    projected = rotate_points(projected, rotation_matrix_x(geometry.inclination_deg))
    cos_inc = max(float(np.cos(np.deg2rad(geometry.inclination_deg))), 1.0e-3)
    sky_mass_per_area = _bilinear_image_mass_per_sky_area(
        image,
        x_sky_kpc=projected[:, 0],
        y_sky_kpc=projected[:, 1],
        pixel_scale_kpc=pixel_scale_kpc,
    ).reshape(rr.shape)
    surface_density = sky_mass_per_area * cos_inc
    radial_area = 0.5 * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
    dphi = np.diff(phi_edges)
    return surface_density * radial_area[:, None] * dphi[None, :]


def particles_from_cylindrical_mass_grid(grid: CylindricalDensityGrid) -> ParticleSet:
    r_centers = 0.5 * (grid.spec.r_edges_kpc[:-1] + grid.spec.r_edges_kpc[1:])
    phi_centers = 0.5 * (grid.spec.phi_edges_rad[:-1] + grid.spec.phi_edges_rad[1:])
    z_centers = 0.5 * (grid.spec.z_edges_kpc[:-1] + grid.spec.z_edges_kpc[1:])
    rr, pp, zz = np.meshgrid(r_centers, phi_centers, z_centers, indexing="ij")
    mass = grid.mass_msun.ravel()
    keep = mass > 0.0
    positions = np.column_stack(
        (
            (rr.ravel() * np.cos(pp.ravel()))[keep],
            (rr.ravel() * np.sin(pp.ravel()))[keep],
            zz.ravel()[keep],
        )
    )
    return ParticleSet(positions_kpc=positions, masses_msun=mass[keep])


def baseline_density_grid_from_image(
    image: np.ndarray,
    *,
    geometry: Geometry,
    pixel_scale_kpc: float,
    spec: CylindricalGridSpec,
    vertical_scale_height_kpc: float,
) -> CylindricalDensityGrid:
    surface_mass = deprojected_surface_mass_grid_from_image(
        image,
        geometry=geometry,
        pixel_scale_kpc=pixel_scale_kpc,
        spec=spec,
    )
    z_weights = sech2_vertical_bin_weights(
        spec.z_edges_kpc,
        scale_height_kpc=vertical_scale_height_kpc,
    )
    mass_grid = surface_mass[:, :, None] * z_weights[None, None, :]
    return density_grid_from_mass_grid(
        mass_grid,
        spec,
        input_mass_msun=float(np.sum(image)),
    )


def _manifest_for_table(manifest: pd.DataFrame, table: np.lib.npyio.NpzFile) -> pd.DataFrame:
    if len(manifest) != len(table["galaxy_id"]):
        raise ValueError("manifest row count must match residual table rows")
    manifest = manifest.reset_index(drop=True).copy()
    table_subhalo = np.asarray(table["galaxy_id"], dtype=int)
    table_projection = np.asarray(table["projection_id"], dtype=int)
    if not np.array_equal(manifest["subhalo_id"].to_numpy(dtype=int), table_subhalo):
        raise ValueError("manifest subhalo_id order must match residual table galaxy_id")
    if not np.array_equal(manifest["projection_id"].to_numpy(dtype=int), table_projection):
        raise ValueError("manifest projection_id order must match residual table projection_id")
    return manifest


def build_one_baseline_density_grid(
    *,
    row: object,
    image: np.ndarray,
    truth_grid_dir: Path,
    pixel_scale_kpc: float,
    vertical_scale_height_kpc: float,
) -> tuple[Path, dict[str, float | int | str], CylindricalDensityGrid]:
    subhalo_id = int(getattr(row, "subhalo_id"))
    projection_id = int(getattr(row, "projection_id"))
    geometry = Geometry(
        inclination_deg=float(getattr(row, "inclination_deg")),
        disk_pa_deg=float(getattr(row, "disk_pa_deg")),
        bar_angle_deg=float(getattr(row, "bar_angle_deg")),
    )
    true_path = truth_grid_path(truth_grid_dir, subhalo_id)
    spec = read_cylindrical_grid_spec_hdf5(true_path)
    grid = baseline_density_grid_from_image(
        image,
        geometry=geometry,
        pixel_scale_kpc=pixel_scale_kpc,
        spec=spec,
        vertical_scale_height_kpc=vertical_scale_height_kpc,
    )
    particles = particles_from_cylindrical_mass_grid(grid)
    reprojected = project_to_mock_image(
        particles,
        geometry,
        image_size=int(np.asarray(image).shape[0]),
        pixel_scale_kpc=pixel_scale_kpc,
        psf_sigma_pixels=0.0,
        noise_sigma_fraction=0.0,
        seed=0,
    ).image
    image_mass = float(np.sum(image))
    reprojection_l1 = float(np.sum(np.abs(reprojected - image)))
    rel_path = baseline_grid_path(subhalo_id, projection_id)
    row_data: dict[str, float | int | str] = {
        "subhalo_id": subhalo_id,
        "projection_id": projection_id,
        "split": str(getattr(row, "split")),
        "baseline_density_file": str(rel_path),
        "true_density_file": str(true_path.name),
        "input_image_mass_msun": image_mass,
        "input_mass_msun": grid.input_mass_msun,
        "grid_mass_msun": grid.grid_mass_msun,
        "dropped_mass_msun": grid.dropped_mass_msun,
        "mass_fraction_in_grid": grid.mass_fraction_in_grid,
        "reprojection_l1_msun": reprojection_l1,
        "reprojection_l1_fraction": reprojection_l1 / max(image_mass, 1.0),
        "inclination_deg": geometry.inclination_deg,
        "disk_pa_deg": geometry.disk_pa_deg,
        "bar_angle_deg": geometry.bar_angle_deg,
        "vertical_scale_height_kpc": float(vertical_scale_height_kpc),
        "r_min_log_kpc": float(spec.r_edges_kpc[1]),
        "r_max_kpc": float(spec.r_edges_kpc[-1]),
        "n_r": len(spec.r_edges_kpc) - 1,
        "n_phi": len(spec.phi_edges_rad) - 1,
        "n_z": len(spec.z_edges_kpc) - 1,
    }
    return rel_path, row_data, grid


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    manifest = pd.read_csv(args.manifest)
    if not validate_split_by_galaxy(manifest.rename(columns={"subhalo_id": "galaxy_id"})):
        raise RuntimeError("manifest leaks subhalo IDs across splits")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with np.load(args.residual_table) as table:
        manifest = _manifest_for_table(manifest, table)
        images = np.asarray(table["images"], dtype=float)
        for index, row in enumerate(manifest.itertuples(index=False)):
            rel_path, row_data, grid = build_one_baseline_density_grid(
                row=row,
                image=images[index],
                truth_grid_dir=args.truth_grid_dir,
                pixel_scale_kpc=cfg.pixel_scale_kpc,
                vertical_scale_height_kpc=args.vertical_scale_height_kpc,
            )
            attrs = {
                **row_data,
                "coordinate_frame": "disk_bar_aligned",
                "source": "surface_density_baseline_from_mock_image",
                "vertical_profile": "sech2",
                "density_unit": "Msun/kpc^3",
                "mass_unit": "Msun",
                "length_unit": "kpc",
            }
            write_cylindrical_density_hdf5(output_dir / rel_path, grid, attrs=attrs)
            rows.append(row_data)

    catalog = pd.DataFrame(rows)
    catalog.to_csv(output_dir / "baseline_density_grid_catalog.csv", index=False)
    diagnostics = {
        "n_projections": int(len(catalog)),
        "median_mass_fraction_in_grid": float(catalog["mass_fraction_in_grid"].median()),
        "min_mass_fraction_in_grid": float(catalog["mass_fraction_in_grid"].min()),
        "max_mass_fraction_in_grid": float(catalog["mass_fraction_in_grid"].max()),
        "median_reprojection_l1_fraction": float(catalog["reprojection_l1_fraction"].median()),
    }
    (output_dir / "baseline_density_grid_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {len(catalog)} baseline cylindrical density grids to {output_dir}")


if __name__ == "__main__":
    main()
