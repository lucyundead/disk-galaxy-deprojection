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
    "bar_type": ("Bartype", "BarType", "bar_type"),
    "bar_a2_catalog": ("A2max", "A2_bar", "bar_strength", "bar_a2", "A2"),
    "bar_length_catalog": ("Rbar", "bar_length", "bar_length_kpc", "BarLength"),
}


def _dataset_by_basename(handle: h5py.File, candidates: tuple[str, ...]) -> np.ndarray | None:
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


def read_bar_strengths(path: Path) -> pd.DataFrame:
    """Read and normalize key bar catalog columns when present."""
    with h5py.File(path, "r") as handle:
        records: dict[str, np.ndarray] = {}
        for output_name, candidates in _COLUMN_CANDIDATES.items():
            values = _dataset_by_basename(handle, candidates)
            if values is not None:
                records[output_name] = values
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
) -> pd.DataFrame:
    """Left-join bar catalog columns onto a candidate manifest."""
    bar_df = read_bar_strengths(bar_path)
    if bar_id_col in bar_df.columns:
        bar_df = bar_df.rename(columns={bar_id_col: subhalo_col})
    elif "subhalo_id" in bar_df.columns and subhalo_col != "subhalo_id":
        bar_df = bar_df.rename(columns={"subhalo_id": subhalo_col})
    return manifest.merge(bar_df, on=subhalo_col, how="left")
