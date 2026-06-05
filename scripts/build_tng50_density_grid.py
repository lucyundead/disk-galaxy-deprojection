from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from dgdp.coordinates import rotate_points, rotation_matrix_z
from dgdp.density3d import (
    CylindricalDensityGrid,
    CylindricalGridSpec,
    accumulate_density_grid_and_faceon_image,
    make_cylindrical_grid_spec,
    project_grid_to_faceon_image,
    write_cylindrical_density_hdf5,
)
from dgdp.orientation import align_particles_to_disk_bar_frame, rotate_to_faceon
from dgdp.tng50 import iter_subhalo_stars_from_chunks, load_particle_set_hdf5
from dgdp.types import ParticleSet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--particle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tng-root", type=Path, default=None)
    parser.add_argument("--snapshot", type=int, default=99)
    parser.add_argument("--hubble-param", type=float, default=0.6774)
    parser.add_argument("--normal-radius-kpc", type=float, default=30.0)
    parser.add_argument("--bar-radius-kpc", type=float, default=5.0)
    parser.add_argument("--r-min-kpc", type=float, default=0.05)
    parser.add_argument("--r-max-kpc", type=float, default=30.0)
    parser.add_argument("--n-r", type=int, default=32)
    parser.add_argument("--n-phi", type=int, default=48)
    parser.add_argument("--z-max-kpc", type=float, default=10.0)
    parser.add_argument("--n-z", type=int, default=32)
    parser.add_argument("--check-image-size", type=int, default=128)
    return parser.parse_args()


def unique_manifest_galaxies(manifest: pd.DataFrame) -> pd.DataFrame:
    return manifest.drop_duplicates(subset=["subhalo_id"], keep="first").reset_index(drop=True)


def _center_from_row(row: object) -> np.ndarray:
    return np.array(
        [
            float(getattr(row, "subhalo_pos_x_ckpc_h")),
            float(getattr(row, "subhalo_pos_y_ckpc_h")),
            float(getattr(row, "subhalo_pos_z_ckpc_h")),
        ]
    )


def _align_to_existing_frame(
    particles: ParticleSet,
    *,
    disk_normal: np.ndarray,
    bar_angle_deg: float,
) -> ParticleSet:
    faceon = rotate_to_faceon(particles, disk_normal)
    positions = rotate_points(faceon.positions_kpc, rotation_matrix_z(-bar_angle_deg))
    velocities = (
        rotate_points(faceon.velocities_kms, rotation_matrix_z(-bar_angle_deg))
        if faceon.velocities_kms is not None
        else None
    )
    return ParticleSet(
        positions_kpc=positions,
        masses_msun=faceon.masses_msun,
        velocities_kms=velocities,
    )


def _source_chunks(
    *,
    row: object,
    tng_root: Path | None,
    snapshot: int,
    hubble_param: float,
    fallback_particles: ParticleSet,
) -> Iterable[ParticleSet]:
    if tng_root is None:
        return [fallback_particles]
    center = _center_from_row(row)
    return iter_subhalo_stars_from_chunks(
        snap_dir=tng_root / f"snapdir_{snapshot:03d}",
        offsets_path=tng_root / "postprocessing" / "offsets" / f"offsets_{snapshot:03d}.hdf5",
        subhalo_id=int(getattr(row, "subhalo_id")),
        star_particle_count=int(getattr(row, "star_particles")),
        subhalo_center_ckpc_h=center,
        snapshot=snapshot,
        hubble_param=hubble_param,
    )


def _aligned_chunks(
    chunks: Iterable[ParticleSet],
    *,
    disk_normal: np.ndarray,
    bar_angle_deg: float,
) -> Iterator[ParticleSet]:
    for chunk in chunks:
        yield _align_to_existing_frame(
            chunk,
            disk_normal=disk_normal,
            bar_angle_deg=bar_angle_deg,
        )


def build_one_density_grid(
    *,
    row: object,
    particle_dir: Path,
    tng_root: Path | None,
    spec: CylindricalGridSpec,
    snapshot: int,
    hubble_param: float,
    normal_radius_kpc: float,
    bar_radius_kpc: float,
    check_image_size: int,
) -> tuple[Path, dict[str, float | int | str], CylindricalDensityGrid]:
    subhalo_id = int(getattr(row, "subhalo_id"))
    orientation_particles = load_particle_set_hdf5(
        particle_dir / f"subhalo_{subhalo_id}.hdf5",
        length_unit_kpc=1.0,
        mass_unit_msun=1.0,
    )
    alignment = align_particles_to_disk_bar_frame(
        orientation_particles,
        normal_radius_kpc=normal_radius_kpc,
        bar_radius_kpc=bar_radius_kpc,
    )
    raw_chunks = _source_chunks(
        row=row,
        tng_root=tng_root,
        snapshot=snapshot,
        hubble_param=hubble_param,
        fallback_particles=orientation_particles,
    )
    grid, particle_image = accumulate_density_grid_and_faceon_image(
        _aligned_chunks(
            raw_chunks,
            disk_normal=alignment.disk_normal,
            bar_angle_deg=alignment.bar_angle_deg,
        ),
        spec,
        image_size=check_image_size,
        radius_kpc=float(spec.r_edges_kpc[-1]),
    )
    grid_image = project_grid_to_faceon_image(
        grid,
        image_size=check_image_size,
        radius_kpc=float(spec.r_edges_kpc[-1]),
    )
    particle_image_mass = float(np.sum(particle_image))
    grid_image_mass = float(np.sum(grid_image))
    image_l1 = float(np.sum(np.abs(grid_image - particle_image)))
    image_l1_fraction = image_l1 / max(particle_image_mass, 1.0)
    rel_path = Path(f"subhalo_{subhalo_id}_density_cylindrical.hdf5")
    row_data: dict[str, float | int | str] = {
        "subhalo_id": subhalo_id,
        "snapshot": snapshot,
        "density_file": str(rel_path),
        "source_particles": "tng_chunks_all_formed_stars" if tng_root is not None else "compact_particle_file",
        "input_mass_msun": grid.input_mass_msun,
        "grid_mass_msun": grid.grid_mass_msun,
        "dropped_mass_msun": grid.dropped_mass_msun,
        "mass_fraction_in_grid": grid.mass_fraction_in_grid,
        "faceon_particle_image_mass_msun": particle_image_mass,
        "faceon_grid_image_mass_msun": grid_image_mass,
        "faceon_image_l1_msun": image_l1,
        "faceon_image_l1_fraction": image_l1_fraction,
        "orientation_method": alignment.orientation_method,
        "disk_normal_x": float(alignment.disk_normal[0]),
        "disk_normal_y": float(alignment.disk_normal[1]),
        "disk_normal_z": float(alignment.disk_normal[2]),
        "bar_angle_deg": float(alignment.bar_angle_deg),
        "r_min_log_kpc": float(spec.r_edges_kpc[1]),
        "r_max_kpc": float(spec.r_edges_kpc[-1]),
        "n_r": len(spec.r_edges_kpc) - 1,
        "n_phi": len(spec.phi_edges_rad) - 1,
        "n_z": len(spec.z_edges_kpc) - 1,
    }
    return rel_path, row_data, grid


def main() -> None:
    args = parse_args()
    manifest = unique_manifest_galaxies(pd.read_csv(args.manifest))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = make_cylindrical_grid_spec(
        r_min_kpc=args.r_min_kpc,
        r_max_kpc=args.r_max_kpc,
        n_r=args.n_r,
        n_phi=args.n_phi,
        z_max_kpc=args.z_max_kpc,
        n_z=args.n_z,
    )

    rows = []
    for row in manifest.itertuples(index=False):
        rel_path, row_data, grid = build_one_density_grid(
            row=row,
            particle_dir=args.particle_dir,
            tng_root=args.tng_root,
            spec=spec,
            snapshot=args.snapshot,
            hubble_param=args.hubble_param,
            normal_radius_kpc=args.normal_radius_kpc,
            bar_radius_kpc=args.bar_radius_kpc,
            check_image_size=args.check_image_size,
        )
        attrs = {
            **row_data,
            "coordinate_frame": "disk_bar_aligned",
            "radial_spacing": "logarithmic_with_zero_inner_edge",
            "phi_spacing": "linear",
            "z_spacing": "linear",
            "density_unit": "Msun/kpc^3",
            "mass_unit": "Msun",
            "length_unit": "kpc",
        }
        write_cylindrical_density_hdf5(output_dir / rel_path, grid, attrs=attrs)
        rows.append(row_data)

    catalog = pd.DataFrame(rows)
    catalog.to_csv(output_dir / "density_grid_catalog.csv", index=False)
    diagnostics = {
        "n_galaxies": int(len(catalog)),
        "median_mass_fraction_in_grid": float(catalog["mass_fraction_in_grid"].median()),
        "min_mass_fraction_in_grid": float(catalog["mass_fraction_in_grid"].min()),
        "max_mass_fraction_in_grid": float(catalog["mass_fraction_in_grid"].max()),
        "density_shape": [args.n_r, args.n_phi, args.n_z],
    }
    (output_dir / "density_grid_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {len(catalog)} cylindrical density grids to {output_dir}")


if __name__ == "__main__":
    main()
