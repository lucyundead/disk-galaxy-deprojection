import os
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

FITS = Path("NGC4321_m_c_r_f.fits")
BUNDLE = Path(str(files("dgdp.models").joinpath("dgdp_fixed_dict.npz")))
pytestmark = pytest.mark.skipif(not (FITS.exists() and BUNDLE.exists()),
                                reason="need NGC4321 fits + bundled model (Task 8)")


def test_cli_writes_outputs(tmp_path):
    out = tmp_path / "o"
    env = {**os.environ, "PYTHONPATH": "src"}
    rc = subprocess.run(
        [sys.executable, "-m", "dgdp.cli", str(FITS), "--distance-mpc", "15.2",
         "--inclination-deg", "30", "--pa-onsky-deg", "153", "--ml", "1.0",
         "--stellar-mass", "6e10", "--no-figures", "-o", str(out)],
        env=env, capture_output=True).returncode
    assert rc == 0
    assert (out / "density.npz").exists()
    assert (out / "rotation_curve.csv").exists()
    d = np.load(out / "density.npz")
    assert d["density_3d"].ndim == 3
