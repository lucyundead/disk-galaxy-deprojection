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


def test_reconstruct_density_conserving_clip_under_ringing():
    # hostile anchors at the phi-delta limit (|2 Sigma_m|/Sigma_0 = 2) with a thin m=0 profile
    # and tall m=2/4 profiles: truncated-Fourier ringing at high |z| must not rectify into mass,
    # i.e. the phi-mean of the clipped density must still equal a_0(R,z) = Sigma_0 * q_0(z).
    r_grid = np.linspace(0.5, 20, 16)
    z_grid = np.linspace(-5, 5, 32)
    phi = np.linspace(-np.pi, np.pi, 48, endpoint=False)
    heights = np.geomspace(0.2, 3.5, 7)
    rk = {0: np.geomspace(0.12, 15, 12), 2: np.geomspace(0.12, 15, 8), 4: np.geomspace(0.12, 15, 6)}
    K = len(heights)
    vec = []
    for m in EVEN_M:
        w = np.zeros((1, len(rk[m]), K))
        w[..., 0 if m == 0 else K - 1] = 1.0                  # m=0 thin, m>0 tallest height
        vec.append(w.reshape(1, -1))
        if m != 0:
            vec.append(np.zeros((1, len(rk[m]) * K)))
    vec = np.concatenate(vec, axis=1)
    ones = np.ones((1, len(r_grid)), complex)
    rho = reconstruct_density(vec, {0: ones, 2: ones, 4: ones}, rk, {m: K for m in EVEN_M},
                              heights, r_grid, z_grid, phi)
    assert np.all(rho >= 0)
    dz = z_grid[1] - z_grid[0]
    thin = 1.0 / np.cosh(np.abs(z_grid) / (2 * heights[0])) ** 2 / (4 * heights[0])
    a0 = thin / (thin.sum() * dz)                             # Sigma_0 = 1 -> a_0(z) = q_0(z)
    np.testing.assert_allclose(rho.mean(axis=2)[0], np.broadcast_to(a0, (len(r_grid), len(z_grid))),
                               rtol=1e-9, atol=1e-12)


def test_reconstruct_density_sigma_hi_column_exact_and_mean_free():
    # anchors + sigma_hi derived from ONE surface density with m=2 AND m=3 content must give
    # back that surface density column-by-column (this catches phi-phase-convention mismatches
    # between the rfft anchors and the cos/sin evaluation: a half-cell rotation of the m=2
    # part breaks the identity wherever its gradient is strong); and under clipping the
    # phi-mean must stay pinned to a_0
    from dgdp.reproject import anchors_from_sigma, high_m_sigma

    r_grid = np.linspace(0.5, 20, 10)
    z_grid = np.linspace(-5, 5, 32)
    phi = np.linspace(-np.pi, np.pi, 48, endpoint=False) + np.pi / 48   # cell centres
    heights = np.geomspace(0.2, 3.5, 7)
    rk = {0: np.geomspace(0.12, 15, 12), 2: np.geomspace(0.12, 15, 8), 4: np.geomspace(0.12, 15, 6)}
    K = len(heights)
    vec = []
    for m in EVEN_M:
        w = np.zeros((1, len(rk[m]), K))
        w[..., 1] = 1.0            # same kernel for every m -> columns positive => no clipping
        vec.append(w.reshape(1, -1))
        if m != 0:
            vec.append(np.zeros((1, len(rk[m]) * K)))
    vec = np.concatenate(vec, axis=1)
    dz = z_grid[1] - z_grid[0]
    area = np.ones(len(r_grid))

    s2d = (1.0 + 0.35 * np.cos(2 * phi + 0.4) + 0.25 * np.cos(3 * phi))[None, :] \
        * np.ones((len(r_grid), 1))                           # mild -> no clipping anywhere
    anchor = anchors_from_sigma(s2d, area)
    hi = high_m_sigma(s2d, area, phi)
    rho = reconstruct_density(vec, anchor, rk, {m: K for m in EVEN_M}, heights,
                              r_grid, z_grid, phi, sigma_hi=hi[None])
    col = rho[0].sum(axis=2) * dz                             # (nR, nphi) column density
    np.testing.assert_allclose(col, s2d, rtol=1e-9, atol=1e-12)

    flat = {0: np.ones((1, len(r_grid)), complex), 2: np.zeros((1, len(r_grid)), complex),
            4: np.zeros((1, len(r_grid)), complex)}
    hi_strong = 1.5 * np.cos(7 * phi)[None, :] * np.ones((len(r_grid), 1))  # forces clipping
    rho2 = reconstruct_density(vec, flat, rk, {m: K for m in EVEN_M}, heights,
                               r_grid, z_grid, phi, sigma_hi=hi_strong[None])
    assert np.all(rho2 >= 0)
    b = 1.0 / np.cosh(np.abs(z_grid) / (2 * heights[1])) ** 2 / (4 * heights[1])
    q0 = b / (b.sum() * dz)
    np.testing.assert_allclose(rho2.mean(axis=2)[0],
                               np.broadcast_to(q0, (len(r_grid), len(z_grid))),
                               rtol=1e-9, atol=1e-12)         # phi-mean still exactly a_0
