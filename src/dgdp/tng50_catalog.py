from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TNG50GroupCatalog:
    hubble_param: float
    subhalo_len_type: np.ndarray
    subhalo_mass_type: np.ndarray
    subhalo_pos: np.ndarray
    subhalo_vel: np.ndarray
    subhalo_halfmass_rad_type: np.ndarray
    subhalo_grnr: np.ndarray
    central_subhalo_ids: np.ndarray


def _group_files(group_dir: Path, snapshot: int) -> list[Path]:
    def _chunk_number(path: Path) -> int:
        return int(path.name.removesuffix(".hdf5").rsplit(".", maxsplit=1)[1])

    return sorted(group_dir.glob(f"fof_subhalo_tab_{snapshot:03d}.*.hdf5"), key=_chunk_number)


def read_group_catalog(group_dir: Path, *, snapshot: int = 99) -> TNG50GroupCatalog:
    files = _group_files(group_dir, snapshot)
    if not files:
        raise FileNotFoundError(f"no group catalog files found in {group_dir}")

    len_parts = []
    mass_parts = []
    pos_parts = []
    vel_parts = []
    halfmass_parts = []
    grnr_parts = []
    central_ids = []
    subhalo_offset = 0
    hubble = 0.6774

    for path in files:
        with h5py.File(path, "r") as handle:
            hubble = float(handle["Header"].attrs.get("HubbleParam", hubble))
            if "SubhaloLenType" not in handle["Subhalo"]:
                continue

            subhalo_len = np.asarray(handle["Subhalo/SubhaloLenType"], dtype=np.int64)
            if "GroupFirstSub" in handle["Group"]:
                group_first_sub = np.asarray(handle["Group/GroupFirstSub"], dtype=np.int64)
                valid = group_first_sub[group_first_sub >= 0]
                if valid.size and int(valid.max()) < subhalo_len.shape[0]:
                    valid = valid + subhalo_offset
                central_ids.append(valid)
            len_parts.append(subhalo_len)
            mass_parts.append(np.asarray(handle["Subhalo/SubhaloMassType"], dtype=np.float64))
            pos_parts.append(np.asarray(handle["Subhalo/SubhaloPos"], dtype=np.float64))
            vel_parts.append(np.asarray(handle["Subhalo/SubhaloVel"], dtype=np.float64))
            halfmass_parts.append(
                np.asarray(handle["Subhalo/SubhaloHalfmassRadType"], dtype=np.float64)
            )
            grnr_parts.append(np.asarray(handle["Subhalo/SubhaloGrNr"], dtype=np.int64))
            subhalo_offset += subhalo_len.shape[0]

    return TNG50GroupCatalog(
        hubble_param=hubble,
        subhalo_len_type=np.vstack(len_parts),
        subhalo_mass_type=np.vstack(mass_parts),
        subhalo_pos=np.vstack(pos_parts),
        subhalo_vel=np.vstack(vel_parts),
        subhalo_halfmass_rad_type=np.vstack(halfmass_parts),
        subhalo_grnr=np.concatenate(grnr_parts),
        central_subhalo_ids=np.concatenate(central_ids) if central_ids else np.array([], dtype=int),
    )


def _split_for_rank(rank: int, n_rows: int) -> str:
    train_end = int(round(0.60 * n_rows))
    val_end = int(round(0.80 * n_rows))
    if rank < train_end:
        return "train"
    if rank < val_end:
        return "val"
    return "test"


def build_candidate_manifest(
    catalog: TNG50GroupCatalog,
    *,
    min_star_particles: int,
    min_stellar_mass_msun: float,
    max_candidates: int,
    snapshot: int = 99,
) -> pd.DataFrame:
    subhalo_id = np.arange(catalog.subhalo_len_type.shape[0], dtype=int)
    star_particles = catalog.subhalo_len_type[:, 4].astype(int)
    stellar_mass_msun = catalog.subhalo_mass_type[:, 4] * 1.0e10 / catalog.hubble_param
    central = np.isin(subhalo_id, catalog.central_subhalo_ids)
    keep = (
        central
        & (star_particles >= min_star_particles)
        & (stellar_mass_msun >= min_stellar_mass_msun)
    )
    rows = pd.DataFrame(
        {
            "subhalo_id": subhalo_id[keep],
            "snapshot": snapshot,
            "star_particles": star_particles[keep],
            "stellar_mass_msun": stellar_mass_msun[keep],
            "subhalo_grnr": catalog.subhalo_grnr[keep],
            "subhalo_pos_x_ckpc_h": catalog.subhalo_pos[keep, 0],
            "subhalo_pos_y_ckpc_h": catalog.subhalo_pos[keep, 1],
            "subhalo_pos_z_ckpc_h": catalog.subhalo_pos[keep, 2],
        }
    )
    rows = rows.sort_values("stellar_mass_msun", ascending=False).head(max_candidates)
    rows = rows.reset_index(drop=True)
    rows["split"] = [_split_for_rank(i, len(rows)) for i in range(len(rows))]
    return rows
