import numpy as np

from dgdp.harmonics import EVEN_M, harmonics, reconstruct_density


def test_harmonics_sigma_is_zintegral():
    nR, nphi, nz = 6, 8, 12
    z = np.linspace(-5, 5, nz)
    dz = z[1] - z[0]
    field = np.abs(np.random.default_rng(0).normal(1, 0.1, (1, nR, nphi, nz)))
    a, sigma = harmonics(field, dz)
    # m=0 Sigma == the axisymmetric (phi-averaged) z-integral
    np.testing.assert_allclose(sigma[0], field.mean(axis=2).sum(axis=2) * dz, rtol=1e-5)
    assert set(a) == set(EVEN_M)


def test_reconstruct_density_conserves_column():
    r_grid = np.linspace(0.5, 20, 16)
    z_grid = np.linspace(-5, 5, 32)
    phi = np.linspace(-np.pi, np.pi, 48, endpoint=False)
    heights = np.geomspace(0.2, 3.5, 7)
    rk = {0: np.geomspace(0.12, 15, 12), 2: np.geomspace(0.12, 15, 8), 4: np.geomspace(0.12, 15, 6)}
    K = len(heights)
    vec = []
    for m in EVEN_M:
        w = np.zeros((1, len(rk[m]), K))
        w[..., 1] = 1.0                                       # all weight on the 2nd height
        vec.append(w.reshape(1, -1))
        if m != 0:
            vec.append(np.zeros((1, len(rk[m]) * K)))         # zero imaginary part
    vec = np.concatenate(vec, axis=1)
    anchor = {0: np.ones((1, len(r_grid)), complex), 2: np.zeros((1, len(r_grid)), complex),
              4: np.zeros((1, len(r_grid)), complex)}
    rho = reconstruct_density(vec, anchor, rk, {m: K for m in EVEN_M}, heights,
                              r_grid, z_grid, phi)
    assert rho.shape == (1, len(r_grid), len(phi), len(z_grid))
    assert np.all(rho >= 0)
    col = rho[0].sum(axis=(1, 2)) * (z_grid[1] - z_grid[0])    # int over phi,z ~ nphi (flat anchor)
    assert np.allclose(col / col.mean(), 1.0, atol=1e-6)
