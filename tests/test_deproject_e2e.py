from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

from dgdp import deproject

FITS = Path("NGC4321_m_c_r_f.fits")
BUNDLE = Path(str(files("dgdp.models").joinpath("dgdp_fixed_dict.npz")))
pytestmark = pytest.mark.skipif(not (FITS.exists() and BUNDLE.exists()),
                                reason="need NGC4321 fits + bundled model (Task 8)")


def test_ngc4321_end_to_end():
    r = deproject(str(FITS), distance_mpc=15.2, inclination_deg=30.0, pa_onsky_deg=153.0,
                  ml=1.0, stellar_mass=6.0e10)
    assert r.density_3d.ndim == 3
    assert np.all(r.density_3d >= 0)
    assert np.isfinite(r.density_3d).all()
    np.testing.assert_allclose(r.total_mass, 6.0e10, rtol=1e-3)
    radii = np.linspace(1, 15, 40)
    vc = r.v_circ(radii)
    assert 150.0 < vc.max() < 190.0                          # NGC4321 params -> peak ~155-180
    rms = r.rms_z(radii)
    assert rms[radii < 3].mean() > 0.6                       # thick/flaring, not the 0.3 baseline
