from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np

from dgdp.types import ParticleSet


def load_particle_set_hdf5(
    path: Path,
    *,
    length_unit_kpc: float,
    mass_unit_msun: float,
) -> ParticleSet:
    with h5py.File(path, "r") as handle:
        coords = np.asarray(handle["PartType4/Coordinates"], dtype=float) * length_unit_kpc
        masses = np.asarray(handle["PartType4/Masses"], dtype=float) * mass_unit_msun
        velocities = (
            np.asarray(handle["PartType4/Velocities"], dtype=float)
            if "PartType4/Velocities" in handle
            else None
        )
    return ParticleSet(positions_kpc=coords, masses_msun=masses, velocities_kms=velocities)


@lru_cache(maxsize=8)
def _snapshot_file_names(snap_dir: str, snapshot: int) -> tuple[str, ...]:
    def _chunk_number(path: Path) -> int:
        return int(path.name.removesuffix(".hdf5").rsplit(".", maxsplit=1)[1])

    paths = sorted(Path(snap_dir).glob(f"snap_{snapshot:03d}.*.hdf5"), key=_chunk_number)
    return tuple(str(path) for path in paths)


def _snapshot_files(snap_dir: Path, snapshot: int) -> list[Path]:
    return [Path(path) for path in _snapshot_file_names(str(snap_dir), snapshot)]


@lru_cache(maxsize=8)
def _star_counts_by_file_names(files: tuple[str, ...]) -> tuple[int, ...]:
    counts = []
    for path in files:
        with h5py.File(path, "r") as handle:
            counts.append(int(handle["Header"].attrs["NumPart_ThisFile"][4]))
    return tuple(counts)


def _star_counts_by_file(files: list[Path]) -> list[int]:
    return list(_star_counts_by_file_names(tuple(str(path) for path in files)))


def _read_optional_dataset(
    group: h5py.Group,
    name: str,
    selection: slice | np.ndarray,
) -> np.ndarray | None:
    if name not in group:
        return None
    return np.asarray(group[name][selection])


def load_subhalo_stars_from_chunks(
    *,
    snap_dir: Path,
    offsets_path: Path,
    subhalo_id: int,
    star_particle_count: int,
    subhalo_center_ckpc_h: np.ndarray,
    snapshot: int,
    hubble_param: float,
    max_particles: int,
    sampling_mode: str = "stride",
) -> ParticleSet:
    with h5py.File(offsets_path, "r") as handle:
        start = int(handle["Subhalo/SnapByType"][subhalo_id, 4])
    full_length = int(star_particle_count)
    length = min(full_length, int(max_particles))
    requested_indices = None
    requested_ranges = None
    if sampling_mode == "first" or length >= full_length:
        requested_ranges = [(start, start + length)]
    elif sampling_mode == "stride":
        requested_indices = start + np.linspace(0, full_length - 1, length, dtype=np.int64)
    elif sampling_mode == "block_stride":
        n_blocks = min(32, length)
        block_size = max(1, length // n_blocks)
        relative_starts = np.linspace(0, full_length - block_size, n_blocks, dtype=np.int64)
        requested_ranges = [
            (start + int(relative_start), start + int(relative_start) + block_size)
            for relative_start in relative_starts
        ]
    else:
        raise ValueError(f"unknown sampling_mode: {sampling_mode}")

    files = _snapshot_files(snap_dir, snapshot)
    counts = _star_counts_by_file(files)
    coords_parts = []
    mass_parts = []
    vel_parts = []
    formation_parts = []
    file_start = 0

    for path, count in zip(files, counts):
        if length <= 0:
            break
        file_end = file_start + count
        selections: list[slice | np.ndarray] = []
        if requested_indices is not None:
            in_file = (requested_indices >= file_start) & (requested_indices < file_end)
            if np.any(in_file):
                selections.append(requested_indices[in_file] - file_start)
        if requested_ranges is not None:
            for global_start, global_end in requested_ranges:
                overlap_start = max(global_start, file_start)
                overlap_end = min(global_end, file_end)
                if overlap_start < overlap_end:
                    selections.append(slice(overlap_start - file_start, overlap_end - file_start))
        if selections:
            with h5py.File(path, "r") as handle:
                stars = handle["PartType4"]
                for selection in selections:
                    coords_parts.append(np.asarray(stars["Coordinates"][selection], dtype=float))
                    mass_parts.append(np.asarray(stars["Masses"][selection], dtype=float))
                    vel = _read_optional_dataset(stars, "Velocities", selection)
                    form = _read_optional_dataset(stars, "GFM_StellarFormationTime", selection)
                    if vel is not None:
                        vel_parts.append(vel.astype(float))
                    if form is not None:
                        formation_parts.append(form.astype(float))
        file_start = file_end

    if not coords_parts:
        return ParticleSet(positions_kpc=np.empty((0, 3)), masses_msun=np.empty((0,)))

    coords = np.vstack(coords_parts)
    masses = np.concatenate(mass_parts)
    velocities = np.vstack(vel_parts) if vel_parts else None
    if formation_parts:
        formed = np.concatenate(formation_parts) > 0.0
        coords = coords[formed]
        masses = masses[formed]
        if velocities is not None:
            velocities = velocities[formed]

    positions_kpc = (coords - np.asarray(subhalo_center_ckpc_h, dtype=float)) / hubble_param
    masses_msun = masses * 1.0e10 / hubble_param
    return ParticleSet(positions_kpc=positions_kpc, masses_msun=masses_msun, velocities_kms=velocities)


def iter_subhalo_stars_from_chunks(
    *,
    snap_dir: Path,
    offsets_path: Path,
    subhalo_id: int,
    star_particle_count: int,
    subhalo_center_ckpc_h: np.ndarray,
    snapshot: int,
    hubble_param: float,
) -> Iterator[ParticleSet]:
    with h5py.File(offsets_path, "r") as handle:
        start = int(handle["Subhalo/SnapByType"][subhalo_id, 4])
    end = start + int(star_particle_count)

    file_start = 0
    files = _snapshot_files(snap_dir, snapshot)
    counts = _star_counts_by_file(files)
    center = np.asarray(subhalo_center_ckpc_h, dtype=float)
    for path, count in zip(files, counts):
        file_end = file_start + count
        overlap_start = max(start, file_start)
        overlap_end = min(end, file_end)
        if overlap_start >= overlap_end:
            file_start = file_end
            continue

        selection = slice(overlap_start - file_start, overlap_end - file_start)
        with h5py.File(path, "r") as handle:
            stars = handle["PartType4"]
            coords = np.asarray(stars["Coordinates"][selection], dtype=float)
            masses = np.asarray(stars["Masses"][selection], dtype=float)
            velocities = _read_optional_dataset(stars, "Velocities", selection)
            formation = _read_optional_dataset(stars, "GFM_StellarFormationTime", selection)
            if formation is not None:
                formed = formation > 0.0
                coords = coords[formed]
                masses = masses[formed]
                if velocities is not None:
                    velocities = velocities[formed]
            if len(masses):
                yield ParticleSet(
                    positions_kpc=(coords - center) / hubble_param,
                    masses_msun=masses * 1.0e10 / hubble_param,
                    velocities_kms=velocities.astype(float) if velocities is not None else None,
                )
        file_start = file_end


def write_particle_set_hdf5(path: Path, particles: ParticleSet, *, subhalo_id: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["subhalo_id"] = int(subhalo_id)
        stars = handle.create_group("PartType4")
        stars["Coordinates"] = particles.positions_kpc
        stars["Masses"] = particles.masses_msun
        if particles.velocities_kms is not None:
            stars["Velocities"] = particles.velocities_kms
