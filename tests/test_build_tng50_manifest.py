import subprocess
import sys

import h5py
import numpy as np
import pandas as pd


def test_build_tng50_manifest_writes_csv(tmp_path):
    tng_root = tmp_path / "TNG50-1"
    group_dir = tng_root / "groups_099"
    group_dir.mkdir(parents=True)

    def write_group_chunk(path):
        with h5py.File(path, "w") as handle:
            header = handle.create_group("Header")
            group = handle.create_group("Group")
            subhalo = handle.create_group("Subhalo")
            header.attrs["HubbleParam"] = 0.6774
            group["GroupFirstSub"] = np.array([0])
            lens = np.zeros((3, 6), dtype=np.int64)
            lens[:, 4] = np.array([5000, 100, 7000])
            masses = np.zeros((3, 6), dtype=np.float32)
            masses[:, 4] = np.array([1.0, 0.01, 2.0])
            subhalo["SubhaloLenType"] = lens
            subhalo["SubhaloMassType"] = masses
            subhalo["SubhaloPos"] = np.zeros((3, 3), dtype=np.float32)
            subhalo["SubhaloVel"] = np.zeros((3, 3), dtype=np.float32)
            subhalo["SubhaloHalfmassRadType"] = np.ones((3, 6), dtype=np.float32)
            subhalo["SubhaloGrNr"] = np.arange(3)

    write_group_chunk(group_dir / "fof_subhalo_tab_099.0.hdf5")
    output = tmp_path / "manifest.csv"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_tng50_manifest.py",
            "--tng-root",
            str(tng_root),
            "--output",
            str(output),
            "--max-candidates",
            "2",
            "--min-star-particles",
            "1000",
            "--min-stellar-mass-msun",
            "1e9",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "wrote TNG50 manifest" in result.stdout
    assert output.exists()


def test_build_tng50_manifest_filters_barred_before_max_candidates(tmp_path):
    tng_root = tmp_path / "TNG50-1"
    group_dir = tng_root / "groups_099"
    group_dir.mkdir(parents=True)
    with h5py.File(group_dir / "fof_subhalo_tab_099.0.hdf5", "w") as handle:
        header = handle.create_group("Header")
        group = handle.create_group("Group")
        subhalo = handle.create_group("Subhalo")
        header.attrs["HubbleParam"] = 1.0
        group["GroupFirstSub"] = np.array([0, 1, 2, 3])
        lens = np.zeros((4, 6), dtype=np.int64)
        lens[:, 4] = np.array([9000, 8000, 7000, 6000])
        masses = np.zeros((4, 6), dtype=np.float32)
        masses[:, 4] = np.array([4.0, 3.0, 2.0, 1.0])
        subhalo["SubhaloLenType"] = lens
        subhalo["SubhaloMassType"] = masses
        subhalo["SubhaloPos"] = np.zeros((4, 3), dtype=np.float32)
        subhalo["SubhaloVel"] = np.zeros((4, 3), dtype=np.float32)
        subhalo["SubhaloHalfmassRadType"] = np.ones((4, 6), dtype=np.float32)
        subhalo["SubhaloGrNr"] = np.arange(4)

    bar_catalog = tmp_path / "morphs_kinematic_bars.hdf5"
    with h5py.File(bar_catalog, "w") as handle:
        snap = handle.create_group("Snapshot_99")
        snap["SubhaloID"] = np.array([0, 1, 2, 3])
        snap["Barred"] = np.array([False, True, True, True])
        snap["BarStrength"] = np.array([[0.5, 0.1, 0.3, 0.4], [0.4, 0.1, 0.2, 0.3]])
        snap["BarSize"] = np.array([[3.0, 3.0, 0.5, 2.0], [2.0, 2.0, 0.4, 1.5]])

    output = tmp_path / "barred_manifest.csv"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_tng50_manifest.py",
            "--tng-root",
            str(tng_root),
            "--output",
            str(output),
            "--bar-catalog",
            str(bar_catalog),
            "--max-candidates",
            "1",
            "--min-star-particles",
            "1000",
            "--min-stellar-mass-msun",
            "1e9",
            "--min-bar-strength",
            "0.2",
            "--min-bar-size-kpc",
            "1.0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = pd.read_csv(output)
    assert "with 1 rows" in result.stdout
    assert manifest["subhalo_id"].tolist() == [3]
