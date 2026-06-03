import h5py
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from dgdp.bar_catalog import inspect_bar_catalog, join_bar_catalog, read_bar_strengths


def _write_fake_bar_catalog(path: Path):
    with h5py.File(path, "w") as handle:
        handle["SubhaloID"] = np.array([100, 200, 300], dtype=int)
        handle["A2_bar"] = np.array([0.15, 0.35, 0.55], dtype=float)
        handle["bar_length"] = np.array([2.0, 5.0, 8.0], dtype=float)


def test_inspect_bar_catalog_returns_dataset_shapes(tmp_path):
    path = tmp_path / "bar_catalog.hdf5"
    _write_fake_bar_catalog(path)
    info = inspect_bar_catalog(path)

    assert "SubhaloID" in info
    assert "A2_bar" in info
    assert "(3,)" in info["SubhaloID"] or info["SubhaloID"] == "(3,)"


def test_read_bar_strengths_reads_known_columns(tmp_path):
    path = tmp_path / "bar_catalog.hdf5"
    _write_fake_bar_catalog(path)
    df = read_bar_strengths(path)

    assert len(df) == 3
    assert "A2_bar" in df.columns
    assert df["A2_bar"].iloc[1] == pytest.approx(0.35)


def test_read_bar_strengths_raises_on_empty_file(tmp_path):
    path = tmp_path / "empty.hdf5"
    with h5py.File(path, "w"):
        pass

    with pytest.raises(KeyError):
        read_bar_strengths(path)


def test_join_bar_catalog_merges_on_subhalo_id(tmp_path):
    bar_path = tmp_path / "bar_catalog.hdf5"
    _write_fake_bar_catalog(bar_path)

    manifest = pd.DataFrame({
        "subhalo_id": [100, 200, 400],
        "stellar_mass_msun": [1e10, 2e10, 3e10],
        "split": ["train", "test", "train"],
    })

    merged = join_bar_catalog(manifest, bar_path)

    assert len(merged) == 3
    assert "A2_bar" in merged.columns
    # subhalo 400 is not in bar catalog, should be NaN
    assert pd.isna(merged.loc[merged["subhalo_id"] == 400, "A2_bar"].iloc[0])
    # subhalo 100 is in bar catalog
    assert merged.loc[merged["subhalo_id"] == 100, "A2_bar"].iloc[0] == pytest.approx(0.15)