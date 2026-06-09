from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from dgdp.config import load_config
from dgdp.coordinates import rotate_points, rotation_matrix_x, rotation_matrix_z
from dgdp.manifest import validate_split_by_galaxy
from dgdp.orientation import align_particles_to_disk_bar_frame, rotate_to_faceon
from dgdp.tng50 import iter_subhalo_stars_from_chunks, load_particle_set_hdf5
from dgdp.types import Geometry, ParticleSet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--particle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tng-root", type=Path, default=None)
    parser.add_argument("--snapshot", type=int, default=99)
    parser.add_argument("--hubble-param", type=float, default=0.6774)
    parser.add_argument("--inclinations-deg", default="20,40,60")
    parser.add_argument("--bar-angles-deg", default="0,45,90")
    parser.add_argument("--disk-pa-deg", type=float, default=0.0)
    parser.add_argument("--normal-radius-kpc", type=float, default=30.0)
    parser.add_argument("--bar-radius-kpc", type=float, default=5.0)
    return parser.parse_args()


def parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one numeric value")
    return values


def expand_projection_grid(
    manifest: pd.DataFrame,
    *,
    inclinations_deg: tuple[float, ...],
    bar_angles_deg: tuple[float, ...],
    disk_pa_deg: float,
) -> pd.DataFrame:
    if {"projection_id", "inclination_deg", "disk_pa_deg", "bar_angle_deg"}.issubset(manifest.columns):
        return manifest.copy()

    rows = []
    for row in manifest.itertuples(index=False):
        base = row._asdict()
        projection_id = 0
        for inclination in inclinations_deg:
            for bar_angle in bar_angles_deg:
                rows.append(
                    {
                        **base,
                        "projection_id": projection_id,
                        "inclination_deg": float(inclination),
                        "disk_pa_deg": float(disk_pa_deg),
                        "bar_angle_deg": float(bar_angle),
                    }
                )
                projection_id += 1
    return pd.DataFrame(rows)


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
    return iter_subhalo_stars_from_chunks(
        snap_dir=tng_root / f"snapdir_{snapshot:03d}",
        offsets_path=tng_root / "postprocessing" / "offsets" / f"offsets_{snapshot:03d}.hdf5",
        subhalo_id=int(getattr(row, "subhalo_id")),
        star_particle_count=int(getattr(row, "star_particles")),
        subhalo_center_ckpc_h=_center_from_row(row),
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


def project_particle_chunk_to_image(
    particles: ParticleSet,
    geometry: Geometry,
    *,
    image_size: int,
    pixel_scale_kpc: float,
) -> np.ndarray:
    rotated = rotate_points(particles.positions_kpc, rotation_matrix_z(geometry.bar_angle_deg))
    rotated = rotate_points(rotated, rotation_matrix_z(geometry.disk_pa_deg))
    rotated = rotate_points(rotated, rotation_matrix_x(geometry.inclination_deg))
    half_size_kpc = 0.5 * image_size * pixel_scale_kpc
    edges = np.linspace(-half_size_kpc, half_size_kpc, image_size + 1)
    image, _, _ = np.histogram2d(
        rotated[:, 1],
        rotated[:, 0],
        bins=(edges, edges),
        weights=particles.masses_msun,
    )
    return image.astype(float)


def build_images_for_galaxy(
    *,
    galaxy_rows: pd.DataFrame,
    particle_dir: Path,
    tng_root: Path | None,
    snapshot: int,
    hubble_param: float,
    normal_radius_kpc: float,
    bar_radius_kpc: float,
    image_size: int,
    pixel_scale_kpc: float,
    psf_sigma_pixels: float,
    noise_sigma_fraction: float,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, float | int | str]]]:
    first = next(galaxy_rows.itertuples(index=False))
    subhalo_id = int(first.subhalo_id)
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
    rows = list(galaxy_rows.itertuples(index=False))
    geometries = [
        Geometry(
            inclination_deg=float(row.inclination_deg),
            disk_pa_deg=float(row.disk_pa_deg),
            bar_angle_deg=float(row.bar_angle_deg),
        )
        for row in rows
    ]
    images = [np.zeros((image_size, image_size), dtype=float) for _ in rows]
    input_mass = 0.0
    source = "tng_chunks_all_formed_stars" if tng_root is not None else "compact_particle_file"
    raw_chunks = _source_chunks(
        row=first,
        tng_root=tng_root,
        snapshot=snapshot,
        hubble_param=hubble_param,
        fallback_particles=orientation_particles,
    )
    for chunk in _aligned_chunks(
        raw_chunks,
        disk_normal=alignment.disk_normal,
        bar_angle_deg=alignment.bar_angle_deg,
    ):
        input_mass += float(np.sum(chunk.masses_msun))
        for image, geometry in zip(images, geometries, strict=True):
            image += project_particle_chunk_to_image(
                chunk,
                geometry,
                image_size=image_size,
                pixel_scale_kpc=pixel_scale_kpc,
            )

    output_rows = []
    catalog_rows = []
    for image, row, geometry in zip(images, rows, geometries, strict=True):
        noiseless = image.astype(float)
        if psf_sigma_pixels > 0.0:
            noiseless = gaussian_filter(noiseless, sigma=psf_sigma_pixels, mode="constant")
        if noise_sigma_fraction > 0.0:
            rng = np.random.default_rng(seed + subhalo_id * 100 + int(row.projection_id))
            sigma = noise_sigma_fraction * max(float(noiseless.max()), 1.0)
            image_out = np.clip(noiseless + rng.normal(0.0, sigma, size=noiseless.shape), 0.0, None)
        else:
            image_out = noiseless
        output_rows.append(
            {
                "galaxy_id": subhalo_id,
                "projection_id": int(row.projection_id),
                "split": str(row.split),
                "image": image_out.astype(np.float32),
                "metadata": np.asarray(
                    [geometry.inclination_deg, geometry.disk_pa_deg, geometry.bar_angle_deg],
                    dtype=np.float32,
                ),
            }
        )
        catalog_rows.append(
            {
                "subhalo_id": subhalo_id,
                "projection_id": int(row.projection_id),
                "split": str(row.split),
                "source_particles": source,
                "source_input_mass_msun": input_mass,
                "image_mass_msun": float(np.sum(image_out)),
                "inclination_deg": geometry.inclination_deg,
                "disk_pa_deg": geometry.disk_pa_deg,
                "bar_angle_deg": geometry.bar_angle_deg,
                "orientation_method": alignment.orientation_method,
                "orientation_bar_angle_deg": float(alignment.bar_angle_deg),
            }
        )
    return output_rows, catalog_rows


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    manifest = expand_projection_grid(
        pd.read_csv(args.manifest),
        inclinations_deg=parse_float_list(args.inclinations_deg),
        bar_angles_deg=parse_float_list(args.bar_angles_deg),
        disk_pa_deg=args.disk_pa_deg,
    )
    if not validate_split_by_galaxy(manifest.rename(columns={"subhalo_id": "galaxy_id"})):
        raise RuntimeError("manifest leaks subhalo IDs across splits")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    catalog_rows: list[dict[str, float | int | str]] = []
    for _, galaxy_rows in manifest.groupby("subhalo_id", sort=False):
        new_rows, new_catalog_rows = build_images_for_galaxy(
            galaxy_rows=galaxy_rows.reset_index(drop=True),
            particle_dir=args.particle_dir,
            tng_root=args.tng_root,
            snapshot=args.snapshot,
            hubble_param=args.hubble_param,
            normal_radius_kpc=args.normal_radius_kpc,
            bar_radius_kpc=args.bar_radius_kpc,
            image_size=cfg.image_size,
            pixel_scale_kpc=cfg.pixel_scale_kpc,
            psf_sigma_pixels=cfg.psf_sigma_pixels,
            noise_sigma_fraction=cfg.noise_sigma_fraction,
            seed=cfg.seed,
        )
        rows.extend(new_rows)
        catalog_rows.extend(new_catalog_rows)

    np.savez_compressed(
        output_dir / "residual_table.npz",
        images=np.stack([row["image"] for row in rows]),
        baseline=np.zeros((len(rows), 0), dtype=np.float32),
        truth=np.zeros((len(rows), 0), dtype=np.float32),
        delta=np.zeros((len(rows), 0), dtype=np.float32),
        metadata=np.stack([row["metadata"] for row in rows]),
        split=np.array([row["split"] for row in rows]),
        galaxy_id=np.array([row["galaxy_id"] for row in rows], dtype=int),
        projection_id=np.array([row["projection_id"] for row in rows], dtype=int),
        summary_names=np.array([], dtype=str),
    )
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    catalog = pd.DataFrame(catalog_rows)
    catalog.to_csv(output_dir / "all_particle_image_catalog.csv", index=False)
    diagnostics = {
        "n_rows": int(len(catalog)),
        "n_galaxies": int(catalog["subhalo_id"].nunique()),
        "source_particles": str(catalog["source_particles"].iloc[0]) if len(catalog) else "",
        "median_image_mass_msun": float(catalog["image_mass_msun"].median()),
        "min_image_mass_msun": float(catalog["image_mass_msun"].min()),
        "max_image_mass_msun": float(catalog["image_mass_msun"].max()),
        "image_shape": [cfg.image_size, cfg.image_size],
        "pixel_scale_kpc": cfg.pixel_scale_kpc,
        "psf_sigma_pixels": cfg.psf_sigma_pixels,
        "noise_sigma_fraction": cfg.noise_sigma_fraction,
    }
    (output_dir / "all_particle_image_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )
    print(f"wrote all-particle TNG50 image table to {output_dir / 'residual_table.npz'}")


if __name__ == "__main__":
    main()
