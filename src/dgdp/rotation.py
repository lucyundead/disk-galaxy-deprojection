"""Rotation curve (pure numpy) and optional AGAMA potential."""
from __future__ import annotations

import numpy as np


def v_circ(mass_grid, r_grid, phi_centers, z_grid, radii, *, eps=0.15) -> np.ndarray:
    """Midplane v_c(R) [km/s] by direct softened summation over cells treated as point masses."""
    grav = 4.300917270e-6  # kpc (km/s)^2 / Msun
    rr, pp, zz = np.meshgrid(r_grid, phi_centers, z_grid, indexing="ij")
    cx, cy, cz = (rr * np.cos(pp)).ravel(), (rr * np.sin(pp)).ravel(), zz.ravel()
    m = mass_grid.ravel()
    keep = m > 0
    cx, cy, cz, m = cx[keep], cy[keep], cz[keep], m[keep]
    out = np.empty(len(radii))
    for i, radius in enumerate(radii):
        dx = cx - radius
        inv = (dx * dx + cy * cy + cz * cz + eps * eps) ** -1.5
        accel_x = grav * np.sum(m * inv * dx)
        out[i] = np.sqrt(max(-accel_x * radius, 0.0))
    return out


def potential(density_grid, r_grid, phi_centers, z_grid, *, total_mass, query_R, query_z):
    """AGAMA CylSpline potential at (query_R, query_z, phi=0). Requires a source-built AGAMA.

    density_grid: (nR, nphi, nz) mass density; the field is Fourier-fit over phi (m=0,2,4).
    """
    try:
        import agama  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "AGAMA is required for the potential field but is not importable. Build it from "
            "source (https://github.com/GalacticDynamics-Oxford/Agama) and add it to PYTHONPATH."
        ) from exc
    from dgdp.agama_density import fourier_rz_to_agama_density
    from dgdp.fourier_rz import fit_fourier_rz_from_grid

    r_max, z_max = float(r_grid[-1]), float(np.abs(z_grid).max())
    model = fit_fourier_rz_from_grid(density_grid, r_grid, z_grid, n_r=25, n_z_half=12,
                                     r_max=r_max, z_max=z_max)
    dens = fourier_rz_to_agama_density(model, total_mass=float(total_mass), r_max=r_max, z_max=z_max)
    import agama
    pot = agama.Potential(type="CylSpline", density=dens)
    qR, qz = np.atleast_1d(query_R), np.atleast_1d(query_z)
    return pot.potential(np.column_stack([qR, np.zeros_like(qR), qz]))
