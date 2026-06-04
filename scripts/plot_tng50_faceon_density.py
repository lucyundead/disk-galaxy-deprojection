from __future__ import annotations

import argparse
import csv
import html
import math
import struct
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from dgdp.tng50 import load_particle_set_hdf5
from dgdp.types import ParticleSet


_INFERNO_STOPS = np.array(
    [
        [0, 0, 4],
        [40, 11, 84],
        [101, 21, 110],
        [159, 42, 99],
        [212, 72, 66],
        [245, 125, 21],
        [250, 193, 39],
        [252, 255, 164],
    ],
    dtype=float,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--particle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--radius-kpc", type=float, default=30.0)
    parser.add_argument("--normal-radius-kpc", type=float, default=30.0)
    parser.add_argument("--montage-columns", type=int, default=8)
    return parser.parse_args()


def _write_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError("PNG writer expects an RGB image")

    def chunk(tag: bytes, data: bytes) -> bytes:
        payload = tag + data
        return (
            struct.pack(">I", len(data))
            + payload
            + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + row.astype(np.uint8).tobytes() for row in rgb)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=6))
        + chunk(b"IEND", b"")
    )


def _try_write_matplotlib_density(
    path: Path,
    log_density: np.ndarray,
    *,
    extent_kpc: tuple[float, float, float, float],
    title: str,
    vmin: float,
    vmax: float,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 4.6), dpi=150)
    image = ax.imshow(
        log_density,
        origin="lower",
        extent=extent_kpc,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x face-on [kpc]")
    ax.set_ylabel("y face-on [kpc]")
    ax.set_aspect("equal")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(r"log$_{10}$ $\Sigma_\star$ [M$_\odot$ kpc$^{-2}$]")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def _try_write_matplotlib_montage(
    path: Path,
    log_images: list[np.ndarray],
    titles: list[str],
    *,
    columns: int,
    vmin: float,
    vmax: float,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    rows = math.ceil(len(log_images) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 2.0, rows * 2.0), dpi=140)
    axes_array = np.atleast_1d(axes).reshape(rows, columns)
    last_image = None
    for index, ax in enumerate(axes_array.ravel()):
        ax.set_xticks([])
        ax.set_yticks([])
        if index >= len(log_images):
            ax.axis("off")
            continue
        last_image = ax.imshow(
            log_images[index],
            origin="lower",
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(titles[index], fontsize=7)
    if last_image is not None:
        fig.colorbar(last_image, ax=axes_array.ravel().tolist(), fraction=0.018, pad=0.01)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def _inferno(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    scaled = values * (len(_INFERNO_STOPS) - 1)
    low = np.floor(scaled).astype(int)
    high = np.clip(low + 1, 0, len(_INFERNO_STOPS) - 1)
    frac = (scaled - low)[..., None]
    return ((1.0 - frac) * _INFERNO_STOPS[low] + frac * _INFERNO_STOPS[high]).astype(np.uint8)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("cannot normalize zero vector")
    return np.asarray(vector, dtype=float) / norm


def _faceon_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_axis = _unit(normal)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(z_axis, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    x_axis = _unit(np.cross(reference, z_axis))
    y_axis = _unit(np.cross(z_axis, x_axis))
    return x_axis, y_axis, z_axis


def estimate_disk_normal(particles: ParticleSet, *, radius_kpc: float) -> tuple[np.ndarray, str]:
    positions = np.asarray(particles.positions_kpc, dtype=float)
    masses = np.asarray(particles.masses_msun, dtype=float)
    radii = np.linalg.norm(positions, axis=1)
    mask = np.isfinite(radii) & (radii <= radius_kpc) & np.isfinite(masses) & (masses > 0.0)
    if int(mask.sum()) < 10:
        mask = np.isfinite(radii) & np.isfinite(masses) & (masses > 0.0)
    pos = positions[mask]
    weight = masses[mask]
    if particles.velocities_kms is not None and len(pos) >= 10:
        velocities = np.asarray(particles.velocities_kms, dtype=float)[mask]
        mean_vel = np.average(velocities, axis=0, weights=weight)
        angular_momentum = np.sum(weight[:, None] * np.cross(pos, velocities - mean_vel), axis=0)
        if np.linalg.norm(angular_momentum) > 0.0:
            return _unit(angular_momentum), "angular_momentum"

    centered = pos - np.average(pos, axis=0, weights=weight)
    covariance = (centered * weight[:, None]).T @ centered / float(weight.sum())
    _, eigenvectors = np.linalg.eigh(covariance)
    return _unit(eigenvectors[:, 0]), "minor_axis"


def surface_density_image(
    particles: ParticleSet,
    *,
    image_size: int,
    radius_kpc: float,
    normal_radius_kpc: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str]]:
    normal, method = estimate_disk_normal(particles, radius_kpc=normal_radius_kpc)
    x_axis, y_axis, normal = _faceon_basis(normal)
    positions = np.asarray(particles.positions_kpc, dtype=float)
    xy = np.column_stack((positions @ x_axis, positions @ y_axis))
    edges = np.linspace(-radius_kpc, radius_kpc, image_size + 1)
    mass_grid, _, _ = np.histogram2d(
        xy[:, 1],
        xy[:, 0],
        bins=(edges, edges),
        weights=particles.masses_msun,
    )
    pixel_area = (2.0 * radius_kpc / image_size) ** 2
    density = mass_grid / pixel_area
    positive = density[density > 0.0]
    log_density = np.zeros_like(density, dtype=float)
    if positive.size:
        log_density[density > 0.0] = np.log10(positive)
        vmin = float(np.percentile(log_density[density > 0.0], 1.0))
        vmax = float(np.percentile(log_density[density > 0.0], 99.5))
        if vmax <= vmin:
            vmax = vmin + 1.0
        normalized = np.clip((log_density - vmin) / (vmax - vmin), 0.0, 1.0)
    else:
        vmin = 0.0
        vmax = 1.0
        normalized = np.zeros_like(density, dtype=float)
    rgb = _inferno(normalized)
    rgb[density <= 0.0] = np.array([0, 0, 4], dtype=np.uint8)
    metadata = {
        "orientation_method": method,
        "normal_x": float(normal[0]),
        "normal_y": float(normal[1]),
        "normal_z": float(normal[2]),
        "log10_sigma_vmin": vmin,
        "log10_sigma_vmax": vmax,
        "total_mass_in_frame_msun": float(mass_grid.sum()),
    }
    return rgb, log_density, metadata


def _make_montage(images: list[np.ndarray], *, columns: int) -> np.ndarray:
    if not images:
        raise ValueError("no images to place in montage")
    tile_h, tile_w, _ = images[0].shape
    rows = math.ceil(len(images) / columns)
    montage = np.zeros((rows * tile_h, columns * tile_w, 3), dtype=np.uint8)
    montage[:, :] = np.array([0, 0, 4], dtype=np.uint8)
    for index, image in enumerate(images):
        row = index // columns
        col = index % columns
        montage[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = image
    return montage


def _write_gallery(path: Path, rows: list[dict[str, str]]) -> None:
    cards = []
    for row in rows:
        caption = (
            f"Subhalo {row['subhalo_id']} | {row['split']} | "
            f"A2={row['bar_a2_catalog']} | Rbar={row['bar_length_catalog']} kpc"
        )
        cards.append(
            "<figure>"
            f"<img src=\"{html.escape(row['filename'])}\" alt=\"{html.escape(caption)}\">"
            f"<figcaption>{html.escape(caption)}</figcaption>"
            "</figure>"
        )
    path.write_text(
        """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>TNG50 barred sample face-on stellar surface density</title>
<style>
body { font-family: sans-serif; margin: 24px; background: #111; color: #eee; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 16px; }
figure { margin: 0; }
img { width: 100%; image-rendering: pixelated; border: 1px solid #333; }
figcaption { font-size: 12px; color: #ccc; margin-top: 4px; }
</style>
</head>
<body>
<h1>TNG50 barred sample face-on stellar surface density</h1>
<p>Estimated face-on orientation from stellar angular momentum where available.</p>
<div class="grid">
"""
        + "\n".join(cards)
        + """
</div>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = []
    log_images = []
    titles = []
    metadata_rows = []
    gallery_rows = []
    montage_vmin = math.inf
    montage_vmax = -math.inf

    for row in manifest.itertuples(index=False):
        subhalo_id = int(row.subhalo_id)
        particles = load_particle_set_hdf5(
            args.particle_dir / f"subhalo_{subhalo_id}.hdf5",
            length_unit_kpc=1.0,
            mass_unit_msun=1.0,
        )
        rgb, log_density, metadata = surface_density_image(
            particles,
            image_size=args.image_size,
            radius_kpc=args.radius_kpc,
            normal_radius_kpc=args.normal_radius_kpc,
        )
        filename = f"subhalo_{subhalo_id}_faceon_density.png"
        title = (
            f"Subhalo {subhalo_id} | {row.split} | "
            f"A2={float(getattr(row, 'bar_a2_catalog', np.nan)):.3f}"
        )
        wrote_matplotlib = _try_write_matplotlib_density(
            args.output_dir / filename,
            log_density,
            extent_kpc=(-args.radius_kpc, args.radius_kpc, -args.radius_kpc, args.radius_kpc),
            title=title,
            vmin=float(metadata["log10_sigma_vmin"]),
            vmax=float(metadata["log10_sigma_vmax"]),
        )
        if not wrote_matplotlib:
            _write_png(args.output_dir / filename, rgb)
        images.append(rgb)
        log_images.append(log_density)
        titles.append(f"{subhalo_id} {row.split}")
        montage_vmin = min(montage_vmin, float(metadata["log10_sigma_vmin"]))
        montage_vmax = max(montage_vmax, float(metadata["log10_sigma_vmax"]))
        metadata_rows.append(
            {
                "subhalo_id": subhalo_id,
                "split": str(row.split),
                "filename": filename,
                "bar_a2_catalog": float(getattr(row, "bar_a2_catalog", np.nan)),
                "bar_length_catalog": float(getattr(row, "bar_length_catalog", np.nan)),
                **metadata,
            }
        )
        gallery_rows.append(
            {
                "subhalo_id": str(subhalo_id),
                "split": str(row.split),
                "filename": filename,
                "bar_a2_catalog": f"{float(getattr(row, 'bar_a2_catalog', np.nan)):.3f}",
                "bar_length_catalog": f"{float(getattr(row, 'bar_length_catalog', np.nan)):.2f}",
            }
        )

    if not _try_write_matplotlib_montage(
        args.output_dir / "montage.png",
        log_images,
        titles,
        columns=args.montage_columns,
        vmin=montage_vmin,
        vmax=montage_vmax,
    ):
        montage = _make_montage(images, columns=args.montage_columns)
        _write_png(args.output_dir / "montage.png", montage)
    with (args.output_dir / "faceon_density_metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)
    _write_gallery(args.output_dir / "gallery.html", gallery_rows)
    print(f"wrote {len(images)} face-on density plots to {args.output_dir}")


if __name__ == "__main__":
    main()
