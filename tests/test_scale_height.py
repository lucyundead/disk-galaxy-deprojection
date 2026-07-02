import numpy as np

from dgdp.vertical_mixture import sech2_height_fit


def test_sech2_height_fit_recovers_known_heights():
    z = np.linspace(-4.84, 4.84, 32)                     # the bundle's z grid
    hs_true = np.array([0.3, 0.6, 1.6, 3.2])             # rho ∝ sech^2(z/h_z) convention
    prof = 1.0 / np.cosh(z[None, :] / hs_true[:, None]) ** 2
    np.testing.assert_allclose(sech2_height_fit(prof, z), hs_true, rtol=0.02)

    # thin+thick mixture (Comeron-like 0.4/2.4): the sech^2 FIT reads near the dominant
    # thin component, while the RMS moment is tail-weighted -- the reason h_z is the
    # obs-comparable number
    mix = 0.75 / np.cosh(z / 0.4) ** 2 / 0.8 + 0.25 / np.cosh(z / 2.4) ** 2 / 4.8
    hmix = sech2_height_fit(mix[None], z)[0]
    h_from_rms = np.sqrt((mix * z**2).sum() / mix.sum()) / (np.pi / np.sqrt(12.0) / 2.0)
    assert 0.4 < hmix < h_from_rms

    assert sech2_height_fit(np.zeros((1, len(z))), z)[0] == 0.0   # no mass -> 0
