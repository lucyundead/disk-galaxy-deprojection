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
    return ParticleSet(positions_kpc=coords, masses_msun=masses)
