from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def inspect_hdf5_datasets(path: Path) -> dict[str, str]:
    """Return dataset names and shapes from a bar-catalog HDF5 file."""
    info: dict[str, str] = {}
    with h5py.File(path, "r") as handle:

        def _walk(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            if isinstance(obj, h5py.Dataset):
                info[name] = str(obj.shape)

        handle.visititems(_walk)
    return info


def inspect_bar_catalog(path: Path) -> dict[str, str]:
    return inspect_hdf5_datasets(path)


_COLUMN_CANDIDATES = {
    "subhalo_id": ("SubfindID", "SubhaloID", "SubhaloId", "subhalo_id", "subfind_id"),
    "barred_catalog": ("Barred", "barred"),
    "bar_type": ("Bartype", "BarType", "bar_type"),
    "bar_a2_catalog": ("BarStrength", "A2max", "A2_bar", "bar_strength", "bar_a2", "A2"),
    "bar_length_catalog": ("BarSize", "Rbar", "bar_length", "bar_length_kpc", "BarLength"),
    "bar_stellar_mass_catalog": ("StellarMass",),
}


def _catalog_root(handle: h5py.File, snapshot: int | None) -> h5py.File | h5py.Group:
    if snapshot is not None and f"Snapshot_{snapshot}" in handle:
        return handle[f"Snapshot_{snapshot}"]
    return handle


def _dataset_by_basename(
    handle: h5py.File | h5py.Group,
    candidates: tuple[str, ...],
) -> np.ndarray | None:
    found: np.ndarray | None = None
    candidate_set = set(candidates)

    def _walk(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        nonlocal found
        if found is not None or not isinstance(obj, h5py.Dataset):
            return
        if name.rsplit("/", maxsplit=1)[-1] in candidate_set:
            found = np.asarray(obj)

    handle.visititems(_walk)
    return found


def _primary_values(values: np.ndarray) -> np.ndarray:
    if values.ndim == 2:
        return values[0]
    return values


def read_bar_strengths(path: Path, *, snapshot: int | None = None) -> pd.DataFrame:
    """Read and normalize key bar catalog columns when present."""
    with h5py.File(path, "r") as handle:
        root = _catalog_root(handle, snapshot)
        records: dict[str, np.ndarray] = {}
        for output_name, candidates in _COLUMN_CANDIDATES.items():
            values = _dataset_by_basename(root, candidates)
            if values is not None:
                records[output_name] = _primary_values(values)
                if output_name == "bar_a2_catalog" and values.ndim == 2 and values.shape[0] > 1:
                    records["bar_a2_secondary_catalog"] = values[1]
                if output_name == "bar_length_catalog" and values.ndim == 2 and values.shape[0] > 1:
                    records["bar_length_secondary_catalog"] = values[1]
        if not records:
            raise KeyError(f"No recognized bar-catalog columns in {path}")

    df = pd.DataFrame(records)
    aliases = {"bar_a2_catalog": "A2_bar", "bar_length_catalog": "bar_length"}
    for src, dst in aliases.items():
        if src in df.columns:
            df[dst] = df[src]
    return df


def join_bar_catalog(
    manifest: pd.DataFrame,
    bar_path: Path,
    *,
    subhalo_col: str = "subhalo_id",
    bar_id_col: str = "SubhaloID",
    snapshot: int | None = None,
) -> pd.DataFrame:
    """Left-join bar catalog columns onto a candidate manifest."""
    bar_df = read_bar_strengths(bar_path, snapshot=snapshot)
    if bar_id_col in bar_df.columns:
        bar_df = bar_df.rename(columns={bar_id_col: subhalo_col})
    elif "subhalo_id" in bar_df.columns and subhalo_col != "subhalo_id":
        bar_df = bar_df.rename(columns={"subhalo_id": subhalo_col})
    return manifest.merge(bar_df, on=subhalo_col, how="left")
