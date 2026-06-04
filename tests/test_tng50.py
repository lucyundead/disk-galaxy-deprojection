import h5py
import numpy as np

from dgdp.tng50 import load_particle_set_hdf5, load_subhalo_stars_from_chunks, write_particle_set_hdf5
from dgdp.types import ParticleSet


def test_load_particle_set_hdf5_reads_required_fields(tmp_path):
    path = tmp_path / "particles.hdf5"
    with h5py.File(path, "w") as handle:
        handle["PartType4/Coordinates"] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        handle["PartType4/Masses"] = np.array([10.0, 20.0])

    particles = load_particle_set_hdf5(path, length_unit_kpc=1.0, mass_unit_msun=1.0)

    assert particles.positions_kpc.shape == (2, 3)
    assert particles.masses_msun.tolist() == [10.0, 20.0]


def test_load_subhalo_stars_from_chunks_reads_across_files(tmp_path):
    snap_dir = tmp_path / "snapdir_099"
    snap_dir.mkdir()
    offsets = tmp_path / "offsets_099.hdf5"

    with h5py.File(offsets, "w") as handle:
        subhalo = handle.create_group("Subhalo")
        data = np.zeros((2, 6), dtype=np.int64)
        data[1, 4] = 1
        subhalo["SnapByType"] = data

    coords0 = np.array([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]])
    coords1 = np.array([[12.0, 0.0, 0.0], [13.0, 0.0, 0.0]])
    for chunk, coords in enumerate([coords0, coords1]):
        with h5py.File(snap_dir / f"snap_099.{chunk}.hdf5", "w") as handle:
            header = handle.create_group("Header")
            stars = handle.create_group("PartType4")
            header.attrs["NumPart_ThisFile"] = np.array([0, 0, 0, 0, len(coords), 0])
            header.attrs["HubbleParam"] = 0.5
            stars["Coordinates"] = coords
            stars["Masses"] = np.ones(len(coords))
            stars["Velocities"] = np.zeros((len(coords), 3))
            stars["GFM_StellarFormationTime"] = np.ones(len(coords))
            stars["ParticleIDs"] = np.arange(chunk * 10, chunk * 10 + len(coords))

    particles = load_subhalo_stars_from_chunks(
        snap_dir=snap_dir,
        offsets_path=offsets,
        subhalo_id=1,
        star_particle_count=3,
        subhalo_center_ckpc_h=np.array([10.0, 0.0, 0.0]),
        snapshot=99,
        hubble_param=0.5,
        max_particles=10,
    )

    assert particles.positions_kpc.shape == (3, 3)
    assert particles.masses_msun.tolist() == [2.0e10, 2.0e10, 2.0e10]
    assert np.allclose(particles.positions_kpc[:, 0], [2.0, 4.0, 6.0])


def test_load_subhalo_stars_from_chunks_sorts_snapshot_chunks_numerically(tmp_path):
    snap_dir = tmp_path / "snapdir_099"
    snap_dir.mkdir()
    offsets = tmp_path / "offsets_099.hdf5"

    with h5py.File(offsets, "w") as handle:
        subhalo = handle.create_group("Subhalo")
        data = np.zeros((1, 6), dtype=np.int64)
        data[0, 4] = 1
        subhalo["SnapByType"] = data

    for chunk, x_coord in [(0, 0.0), (10, 10.0), (2, 2.0)]:
        with h5py.File(snap_dir / f"snap_099.{chunk}.hdf5", "w") as handle:
            header = handle.create_group("Header")
            stars = handle.create_group("PartType4")
            header.attrs["NumPart_ThisFile"] = np.array([0, 0, 0, 0, 1, 0])
            stars["Coordinates"] = np.array([[x_coord, 0.0, 0.0]])
            stars["Masses"] = np.ones(1)
            stars["GFM_StellarFormationTime"] = np.ones(1)

    particles = load_subhalo_stars_from_chunks(
        snap_dir=snap_dir,
        offsets_path=offsets,
        subhalo_id=0,
        star_particle_count=2,
        subhalo_center_ckpc_h=np.zeros(3),
        snapshot=99,
        hubble_param=1.0,
        max_particles=10,
    )

    assert particles.positions_kpc[:, 0].tolist() == [2.0, 10.0]


def test_load_subhalo_stars_from_chunks_stride_samples_full_subhalo(tmp_path):
    snap_dir = tmp_path / "snapdir_099"
    snap_dir.mkdir()
    offsets = tmp_path / "offsets_099.hdf5"

    with h5py.File(offsets, "w") as handle:
        subhalo = handle.create_group("Subhalo")
        data = np.zeros((1, 6), dtype=np.int64)
        subhalo["SnapByType"] = data

    chunk_coords = [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        np.array([[4.0, 0.0, 0.0]]),
    ]
    for chunk, coords in enumerate(chunk_coords):
        with h5py.File(snap_dir / f"snap_099.{chunk}.hdf5", "w") as handle:
            header = handle.create_group("Header")
            stars = handle.create_group("PartType4")
            header.attrs["NumPart_ThisFile"] = np.array([0, 0, 0, 0, len(coords), 0])
            stars["Coordinates"] = coords
            stars["Masses"] = np.ones(len(coords))
            stars["GFM_StellarFormationTime"] = np.ones(len(coords))

    particles = load_subhalo_stars_from_chunks(
        snap_dir=snap_dir,
        offsets_path=offsets,
        subhalo_id=0,
        star_particle_count=5,
        subhalo_center_ckpc_h=np.zeros(3),
        snapshot=99,
        hubble_param=1.0,
        max_particles=3,
        sampling_mode="stride",
    )

    assert particles.positions_kpc[:, 0].tolist() == [0.0, 2.0, 4.0]


def test_load_subhalo_stars_from_chunks_block_stride_samples_full_subhalo(tmp_path):
    snap_dir = tmp_path / "snapdir_099"
    snap_dir.mkdir()
    offsets = tmp_path / "offsets_099.hdf5"

    with h5py.File(offsets, "w") as handle:
        subhalo = handle.create_group("Subhalo")
        subhalo["SnapByType"] = np.zeros((1, 6), dtype=np.int64)

    coords = np.column_stack((np.arange(10, dtype=float), np.zeros((10, 2))))
    for chunk, chunk_coords in enumerate([coords[:5], coords[5:]]):
        with h5py.File(snap_dir / f"snap_099.{chunk}.hdf5", "w") as handle:
            header = handle.create_group("Header")
            stars = handle.create_group("PartType4")
            header.attrs["NumPart_ThisFile"] = np.array([0, 0, 0, 0, len(chunk_coords), 0])
            stars["Coordinates"] = chunk_coords
            stars["Masses"] = np.ones(len(chunk_coords))
            stars["GFM_StellarFormationTime"] = np.ones(len(chunk_coords))

    particles = load_subhalo_stars_from_chunks(
        snap_dir=snap_dir,
        offsets_path=offsets,
        subhalo_id=0,
        star_particle_count=10,
        subhalo_center_ckpc_h=np.zeros(3),
        snapshot=99,
        hubble_param=1.0,
        max_particles=4,
        sampling_mode="block_stride",
    )

    assert particles.positions_kpc[:, 0].tolist() == [0.0, 3.0, 6.0, 9.0]


def test_write_particle_set_hdf5_round_trips_velocities(tmp_path):
    path = tmp_path / "compact.hdf5"
    particles = ParticleSet(
        positions_kpc=np.ones((2, 3)),
        masses_msun=np.array([1.0, 2.0]),
        velocities_kms=np.zeros((2, 3)),
    )

    write_particle_set_hdf5(path, particles, subhalo_id=42)
    loaded = load_particle_set_hdf5(path, length_unit_kpc=1.0, mass_unit_msun=1.0)

    assert loaded.positions_kpc.shape == (2, 3)
    assert loaded.velocities_kms.shape == (2, 3)
