"""Regression gate: the bundled model reproduces the validated NGC 4371 physics (thick disk,
undepressed rotation curve). Skips unless the collaborator data + bundle are present."""
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

from dgdp import deproject

FITS = Path("NGC4371/final/NGC4371_S4G_cut_EL_Fil.fits")
BUNDLE = Path(str(files("dgdp.models").joinpath("dgdp_fixed_dict.npz")))
pytestmark = pytest.mark.skipif(not (FITS.exists() and BUNDLE.exists()),
                                reason="need NGC4371 data + bundled model")


def test_ngc4371_thick_and_conserving():
    r = deproject(str(FITS), distance_mpc=16.194, inclination_deg=58.0, pa_pix_deg=1.8,
                  center=(254.6, 152.8), pix_arcsec=0.75, mask="NGC4371/final/NGC4371_mask.fits",
                  ml=1.0, stellar_mass=3.53e10)
    rms15 = r.rms_z(np.array([1.5]))[0]
    vc2 = r.v_circ(np.array([2.0]))[0]
    np.testing.assert_allclose(r.total_mass, 3.53e10, rtol=1e-3)
    assert 0.6 < rms15 < 2.5, f"NGC4371 RMS|z|(1.5)={rms15:.2f} not thick/flaring"      # ~0.9, MGE regime
    assert 130.0 < vc2 < 210.0, f"NGC4371 v_c(2.0)={vc2:.1f} depressed/wrong"            # ~177, not depressed
