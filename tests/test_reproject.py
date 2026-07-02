import numpy as np

from dgdp.density3d import cylindrical_bin_volumes, make_cylindrical_grid_spec
from dgdp.reproject import anchors_from_sigma, project_to_sky, refine_sigma

SPEC = make_cylindrical_grid_spec(z_max_kpc=5.0, n_z=32)
R = 0.5 * (SPEC.r_edges_kpc[:-1] + SPEC.r_edges_kpc[1:])
Z = 0.5 * (SPEC.z_edges_kpc[:-1] + SPEC.z_edges_kpc[1:])
PHI = 0.5 * (SPEC.phi_edges_rad[:-1] + SPEC.phi_edges_rad[1:])
VOL = cylindrical_bin_volumes(SPEC).astype(float)
DZ = float(np.diff(SPEC.z_edges_kpc)[0])
AREA = VOL[:, 0, 0] / DZ
EDGES = np.linspace(-33.6, 33.6, 97)                       # 96 x 0.7 kpc test sky grid

Q = 1.0 / np.cosh(Z / 1.0) ** 2                            # fixed vertical pdf (h=0.5)
Q /= Q.sum() * DZ


def sigma2d(bar_amp):
    return np.exp(-R / 4.0)[:, None] * (1.0 + bar_amp * np.cos(2 * PHI))[None, :]


def rho_from_anchor(anchor):
    """Analytic 'model': even-m anchors x the fixed vertical pdf Q (clip >= 0)."""
    s = anchor[0][0].real[:, None] + 0j
    for m in (2, 4):
        s = s + 2.0 * (anchor[m][0][:, None] * np.exp(1j * m * PHI)[None, :])
    return np.clip(s.real, 0.0, None)[:, :, None] * Q[None, None, :]


def test_projector_faceon_matches_column_mass_and_conserves():
    rho = sigma2d(0.0)[:, :, None] * Q[None, None, :]      # [mass/kpc^3], Sigma = e^{-R/4}
    img = project_to_sky(rho, R, PHI, Z, 0.0, EDGES, n_los=201)
    c = 0.5 * (EDGES[:-1] + EDGES[1:])
    xs, ys = np.meshgrid(c, c, indexing="xy")
    rad = np.hypot(xs, ys)
    sel = (rad > 2) & (rad < 8)
    pix = (EDGES[1] - EDGES[0]) ** 2
    np.testing.assert_allclose(img[sel], np.exp(-rad[sel] / 4.0) * pix, rtol=0.05)
    total = float((rho * VOL).sum())
    assert abs(img.sum() / total - 1.0) < 0.02
    img55 = project_to_sky(rho, R, PHI, Z, 55.0, EDGES, n_los=201)
    assert abs(img55.sum() / total - 1.0) < 0.02            # conserves when inclined too


def test_refine_sigma_recovers_bar_from_image():
    # truth has an m=2 bar; the initial sigma is axisymmetric with the wrong scalelength.
    # The loop must pull both the bar and the radial profile out of the observed image.
    sig_true = sigma2d(0.4) * AREA[:, None]
    obs = project_to_sky(rho_from_anchor(anchors_from_sigma(sig_true, AREA)),
                         R, PHI, Z, 55.0, EDGES, n_los=161)
    sig0 = np.exp(-R / 3.0)[:, None] * np.ones_like(PHI)[None, :] * AREA[:, None]
    sig0 *= sig_true.sum() / sig0.sum()
    sig, hist, ratio = refine_sigma(sig0, obs, rho_from_anchor, AREA, R, PHI, Z, 55.0,
                                    EDGES, iters=3, n_los=161)
    # hist[0] = uncorrected residual; the loop must improve it a lot on this in-class mock
    assert min(hist) < 0.5 * hist[0], f"residual not reduced: {hist}"

    def m2_amp(s):                                          # mean |2 Sigma_2|/Sigma_0 in the bar
        co = np.fft.rfft(s / AREA[:, None], axis=1) / s.shape[1]
        sel = (R > 2) & (R < 6)
        return float((2 * np.abs(co[sel, 2]) / co[sel, 0].real).mean())

    assert m2_amp(sig0) < 0.01
    assert abs(m2_amp(sig) - 0.4) < 0.12, f"bar amplitude not recovered: {m2_amp(sig):.3f}"
    sel = (R > 1) & (R < 10)
    s0_true = (sig_true / AREA[:, None]).mean(axis=1)[sel]
    err0 = np.linalg.norm((sig0 / AREA[:, None]).mean(axis=1)[sel] - s0_true)
    err1 = np.linalg.norm((sig / AREA[:, None]).mean(axis=1)[sel] - s0_true)
    assert err1 < 0.35 * err0, f"radial profile not improved: {err1:.3g} vs {err0:.3g}"
