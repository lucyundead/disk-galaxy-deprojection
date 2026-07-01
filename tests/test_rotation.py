import builtins

import numpy as np
import pytest

from dgdp.rotation import potential, v_circ


def test_v_circ_point_mass_scaling():
    # a compact central ring -> v_c declines outward
    r = np.linspace(0.5, 20, 32)
    phi = np.linspace(-np.pi, np.pi, 24, endpoint=False)
    z = np.linspace(-1, 1, 8)
    mass = np.zeros((r.size, phi.size, z.size))
    mass[0] += 1e10                                          # all mass at the inner radius
    vc = v_circ(mass, r, phi, z, np.array([2.0, 8.0]))
    assert vc[0] > vc[1] > 0


def test_potential_without_agama_raises_clearly(monkeypatch):
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "agama":
            raise ImportError("no agama")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(RuntimeError, match="AGAMA"):
        potential(np.ones((4, 4, 4)), np.linspace(1, 4, 4),
                  np.linspace(-np.pi, np.pi, 4, endpoint=False), np.linspace(-1, 1, 4),
                  total_mass=1e10, query_R=np.array([1.0]), query_z=np.array([0.0]))
