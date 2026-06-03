import h5py
import numpy as np

from dgdp.tng50 import load_particle_set_hdf5


def test_load_particle_set_hdf5_reads_required_fields(tmp_path):
    path = tmp_path / "particles.hdf5"
    with h5py.File(path, "w") as handle:
        handle["PartType4/Coordinates"] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        handle["PartType4/Masses"] = np.array([10.0, 20.0])

    particles = load_particle_set_hdf5(path, length_unit_kpc=1.0, mass_unit_msun=1.0)

    assert particles.positions_kpc.shape == (2, 3)
    assert particles.masses_msun.tolist() == [10.0, 20.0]
