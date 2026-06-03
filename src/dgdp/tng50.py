from __future__ import annotations

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


def _snapshot_files(snap_dir: Path, snapshot: int) -> list[Path]:
    def _chunk_number(path: Path) -> int:
        return int(path.name.removesuffix(".hdf5").rsplit(".", maxsplit=1)[1])

    return sorted(snap_dir.glob(f"snap_{snapshot:03d}.*.hdf5"), key=_chunk_number)


def _star_counts_by_file(files: list[Path]) -> list[int]:
    counts = []
    for path in files:
        with h5py.File(path, "r") as handle:
            counts.append(int(handle["Header"].attrs["NumPart_ThisFile"][4]))
    return counts


def _read_optional_dataset(group: h5py.Group, name: str, selection: slice) -> np.ndarray | None:
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
) -> ParticleSet:
    with h5py.File(offsets_path, "r") as handle:
        start = int(handle["Subhalo/SnapByType"][subhalo_id, 4])
    length = int(star_particle_count)
    length = min(length, int(max_particles))

    files = _snapshot_files(snap_dir, snapshot)
    counts = _star_counts_by_file(files)
    remaining_start = start
    remaining_length = length
    coords_parts = []
    mass_parts = []
    vel_parts = []
    formation_parts = []

    for path, count in zip(files, counts):
        if remaining_start >= count:
            remaining_start -= count
            continue
        if remaining_length <= 0:
            break
        local_start = remaining_start
        take = min(count - local_start, remaining_length)
        selection = slice(local_start, local_start + take)
        with h5py.File(path, "r") as handle:
            stars = handle["PartType4"]
            coords_parts.append(np.asarray(stars["Coordinates"][selection], dtype=float))
            mass_parts.append(np.asarray(stars["Masses"][selection], dtype=float))
            vel = _read_optional_dataset(stars, "Velocities", selection)
            form = _read_optional_dataset(stars, "GFM_StellarFormationTime", selection)
            if vel is not None:
                vel_parts.append(vel.astype(float))
            if form is not None:
                formation_parts.append(form.astype(float))
        remaining_length -= take
        remaining_start = 0

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


def write_particle_set_hdf5(path: Path, particles: ParticleSet, *, subhalo_id: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["subhalo_id"] = int(subhalo_id)
        stars = handle.create_group("PartType4")
        stars["Coordinates"] = particles.positions_kpc
        stars["Masses"] = particles.masses_msun
        if particles.velocities_kms is not None:
            stars["Velocities"] = particles.velocities_kms
