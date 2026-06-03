import subprocess
import sys

import h5py
import numpy as np


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
