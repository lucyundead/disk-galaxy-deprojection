"""Reprojection-consistency loop: project the reconstruction back to the sky and correct
the in-plane Sigma anchors until the deprojection actually reproduces the observed image.

The geometric baseline stretches the image assuming zero thickness, but the learned q_m
then makes the disk thick -- mutually inconsistent (a thick component smears along the
minor axis by ~RMS|z| sin i in projection, which the thin stretch mis-assigns in-plane).
With the predicted vertical profiles FIXED, iterate

    rho  = reconstruct(anchors(sigma))
    proj = LOS-project rho at inclination i onto the mock sky grid
    sigma *= backmap(obs / proj)      (midplane mapping; ratio clipped; obs>0 pixels only)

The fixed point is a Sigma whose *thick* projection matches the image within the m<=4
azimuthal band the anchors carry (anchors_from_sigma filters higher m automatically).
The final ratio map is a per-galaxy self-consistency diagnostic. PSF is ignored: the S4G
PSF (~0.16 kpc) is below the 0.35 kpc mock pixel. Pure numpy -- scipy is not a core dep.
"""
from __future__ import annotations

import numpy as np

EVEN_M = (0, 2, 4)


def _frac_index(grid, x):
    i = np.clip(np.searchsorted(grid, x) - 1, 0, len(grid) - 2)
    f = (x - grid[i]) / (grid[i + 1] - grid[i])
    return i, np.clip(f, 0.0, 1.0)


def _sample_cyl(rho_ext, r_grid, phi_ext, z_grid, rad, ang, zz):
    """Trilinear sample of the phi-extended density; 0 outside the (R,z) domain."""
    ir, fr = _frac_index(r_grid, rad)
    ip, fp = _frac_index(phi_ext, ang)
    iz, fz = _frac_index(z_grid, zz)
    out = np.zeros_like(rad)
    for dr in (0, 1):
        wr = np.where(dr, fr, 1.0 - fr)
        for dp in (0, 1):
            wp = np.where(dp, fp, 1.0 - fp)
            for dzi in (0, 1):
                wz = np.where(dzi, fz, 1.0 - fz)
                out += rho_ext[ir + dr, ip + dp, iz + dzi] * wr * wp * wz
    inside = (rad >= r_grid[0]) & (rad <= r_grid[-1]) & (zz >= z_grid[0]) & (zz <= z_grid[-1])
    return np.where(inside, out, 0.0)


def project_to_sky(rho, r_grid, phi_centers, z_grid, incl_deg, edges, n_los=281):
    """LOS-integrate density rho (nR,nphi,nz) [mass/kpc^3] at inclination about the major
    (x) axis -> mass per pixel on the mock sky grid (axis 0 = y_minor, axis 1 = x_major,
    square bins `edges`). Sky->galaxy: y_g = y_s cos i + l sin i, z = -y_s sin i + l cos i.
    """
    r_grid = np.asarray(r_grid, float)
    phi_ext = np.concatenate([phi_centers, [phi_centers[0] + 2 * np.pi]])
    rho_ext = np.concatenate([rho, rho[:, :1, :]], axis=1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    xs, ys = np.meshgrid(centers, centers, indexing="xy")        # [iy, ix]
    ci, si = np.cos(np.radians(incl_deg)), np.sin(np.radians(incl_deg))
    span = float(r_grid[-1] + z_grid[-1])
    los = np.linspace(-span, span, int(n_los))
    dl = los[1] - los[0]
    img = np.zeros_like(xs)
    for chunk in np.array_split(los, max(1, len(los) // 32)):
        y_g = ys[:, :, None] * ci + chunk[None, None, :] * si
        z_g = -ys[:, :, None] * si + chunk[None, None, :] * ci
        rad = np.hypot(xs[:, :, None], y_g)
        ang = phi_centers[0] + np.mod(np.arctan2(y_g, xs[:, :, None]) - phi_centers[0],
                                      2 * np.pi)
        img += _sample_cyl(rho_ext, r_grid, phi_ext, z_grid, rad, ang, z_g).sum(axis=2)
    pix_area = float(edges[1] - edges[0]) ** 2
    return img * dl * pix_area


def anchors_from_sigma(sigma_mass, base_area):
    """(R,phi) cell masses -> even-m anchor harmonics Sigma_m(R) [mass/kpc^2], shape (1,nR)
    complex per m. Identical to harmonics(baseline_density) -- the z-profile integrates out.
    """
    s2d = sigma_mass / base_area[:, None]
    coeff = np.fft.rfft(s2d, axis=1) / s2d.shape[1]
    return {m: coeff[None, :, m] for m in EVEN_M}


def refine_sigma(sigma_mass, obs_img, reconstruct_fn, base_area, r_grid, phi_centers,
                 z_grid, incl_deg, edges, *, iters=2, ratio_clip=3.0, n_los=281):
    """Iteratively correct sigma_mass (R,phi) so the reconstruction reprojects to obs_img.

    reconstruct_fn(anchor_dict) -> rho (nR,nphi,nz); anchors rebuilt from sigma each pass.
    Early-stop with revert: each candidate state is MEASURED (obs-weighted mean |log ratio|);
    a correction that does not improve is discarded and the loop stops, so the returned sigma
    is never worse-reprojecting than the input (residuals dominated by m>4 structure the
    m<=4 anchors cannot carry, e.g. strong spiral arms, would otherwise cause overshoot).
    Returns (best sigma, history of measured residuals, ratio map at the best state).
    """
    total = sigma_mass.sum()
    obs = np.clip(np.asarray(obs_img, float), 0.0, None)
    obs_n = obs / max(obs.sum(), 1e-300)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ci = np.cos(np.radians(incl_deg))
    x_cell = r_grid[:, None] * np.cos(phi_centers)[None, :]
    y_cell = r_grid[:, None] * np.sin(phi_centers)[None, :] * ci  # midplane projection

    def measure(sig):
        rho = reconstruct_fn(anchors_from_sigma(sig, base_area))
        proj = project_to_sky(rho, r_grid, phi_centers, z_grid, incl_deg, edges, n_los=n_los)
        proj_n = proj / max(proj.sum(), 1e-300)
        valid = (obs_n > 0) & (proj_n > 0)
        ratio = np.ones_like(obs)
        ratio[valid] = np.clip(obs_n[valid] / proj_n[valid], 1.0 / ratio_clip, ratio_clip)
        res = float(np.abs(np.log(ratio[valid])) @ obs_n[valid] / obs_n[valid].sum())
        return res, ratio

    best_sig = sigma_mass.copy()
    best_res, best_ratio = measure(best_sig)
    history = [best_res]
    for _ in range(int(iters)):
        # bilinear backmap of the ratio at each cell's midplane sky position (outside -> 1)
        iy, fy = _frac_index(centers, y_cell)
        ix, fx = _frac_index(centers, x_cell)
        corr = (best_ratio[iy, ix] * (1 - fy) * (1 - fx) + best_ratio[iy, ix + 1] * (1 - fy) * fx
                + best_ratio[iy + 1, ix] * fy * (1 - fx) + best_ratio[iy + 1, ix + 1] * fy * fx)
        inside = ((y_cell >= centers[0]) & (y_cell <= centers[-1])
                  & (x_cell >= centers[0]) & (x_cell <= centers[-1]))
        sig = best_sig * np.where(inside, corr, 1.0)
        sig *= total / max(sig.sum(), 1e-300)
        res, ratio = measure(sig)
        history.append(res)
        if res >= best_res:
            break
        best_sig, best_res, best_ratio = sig, res, ratio
    return best_sig, history, best_ratio
