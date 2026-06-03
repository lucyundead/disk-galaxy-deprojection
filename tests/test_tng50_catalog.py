from pathlib import Path

import h5py
import numpy as np

from dgdp.tng50_catalog import build_candidate_manifest, read_group_catalog


def _write_group_chunk(path: Path, first_subs: list[int], subhalo_offset: int) -> None:
    with h5py.File(path, "w") as handle:
        header = handle.create_group("Header")
        group = handle.create_group("Group")
        subhalo = handle.create_group("Subhalo")
        header.attrs["HubbleParam"] = 0.6774
        group["GroupFirstSub"] = np.asarray(first_subs, dtype=np.int64)
        n_sub = 3
        lens = np.zeros((n_sub, 6), dtype=np.int64)
        lens[:, 4] = np.asarray([5000, 100, 7000], dtype=np.int64)
        masses = np.zeros((n_sub, 6), dtype=np.float32)
        masses[:, 4] = np.asarray([1.0, 0.01, 2.0], dtype=np.float32)
        subhalo["SubhaloLenType"] = lens
        subhalo["SubhaloMassType"] = masses
        subhalo["SubhaloPos"] = np.arange(n_sub * 3, dtype=np.float32).reshape(n_sub, 3)
        subhalo["SubhaloVel"] = np.zeros((n_sub, 3), dtype=np.float32)
        subhalo["SubhaloHalfmassRadType"] = np.ones((n_sub, 6), dtype=np.float32)
        subhalo["SubhaloGrNr"] = np.arange(subhalo_offset, subhalo_offset + n_sub)


def test_read_group_catalog_concatenates_subhalos(tmp_path):
    group_dir = tmp_path / "groups_099"
    group_dir.mkdir()
    _write_group_chunk(group_dir / "fof_subhalo_tab_099.0.hdf5", [0], 0)
    _write_group_chunk(group_dir / "fof_subhalo_tab_099.1.hdf5", [0], 3)

    catalog = read_group_catalog(group_dir, snapshot=99)

    assert catalog.hubble_param == 0.6774
    assert catalog.subhalo_len_type.shape == (6, 6)
    assert catalog.subhalo_mass_type.shape == (6, 6)
    assert catalog.central_subhalo_ids.tolist() == [0, 3]


def test_read_group_catalog_keeps_global_central_ids(tmp_path):
    group_dir = tmp_path / "groups_099"
    group_dir.mkdir()
    _write_group_chunk(group_dir / "fof_subhalo_tab_099.0.hdf5", [0], 0)
    _write_group_chunk(group_dir / "fof_subhalo_tab_099.1.hdf5", [4], 3)

    catalog = read_group_catalog(group_dir, snapshot=99)

    assert catalog.central_subhalo_ids.tolist() == [0, 4]


def test_build_candidate_manifest_filters_centrals_and_particle_count(tmp_path):
    group_dir = tmp_path / "groups_099"
    group_dir.mkdir()
    _write_group_chunk(group_dir / "fof_subhalo_tab_099.0.hdf5", [0, 2], 0)

    catalog = read_group_catalog(group_dir, snapshot=99)
    manifest = build_candidate_manifest(
        catalog,
        min_star_particles=1000,
        min_stellar_mass_msun=1.0e9,
        max_candidates=2,
    )

    assert manifest["subhalo_id"].tolist() == [2, 0]
    assert manifest["split"].tolist() == ["train", "val"]
    assert manifest["star_particles"].tolist() == [7000, 5000]
