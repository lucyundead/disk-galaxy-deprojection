"""Isolated (vacuum-boundary) gravitational potential and forces on a uniform grid.

A dependency-free Poisson solver via the Hockney-Eastwood zero-padded FFT
convolution: the potential of an isolated mass distribution is
``Phi = -G * (mass (*) 1/r)`` and the acceleration is
``a = -G * (mass (*) r_hat/r^2)``, evaluated as cyclic convolutions on a grid
zero-padded to twice its size in each dimension (so the cyclic convolution equals
the linear one and the boundary is open, Phi -> 0 at infinity).

Used to validate that a smooth density representation (the even-m Fourier x (R,z)
maps) reproduces the truth's potential/forces - the "potential-ready" claim - when
AGAMA's CylSpline is unavailable (no compiler in the sandbox). The forces use an
exact antisymmetric kernel (zero self-term), so they are free of softening choices;
only the potential carries a softened cell self-term.
"""

from __future__ import annotations

import numpy as np

# G in galactic units: kpc * (km/s)^2 / Msun.
GRAV_KPC_KMS2_MSUN = 4.300917270e-6


def _signed_axis(n: int, spacing: float) -> np.ndarray:
    """Signed offset coordinates for a periodic axis of length ``n`` (n even)."""
    i = np.arange(n)
    return np.where(i >= n // 2, i - n, i).astype(float) * spacing


def isolated_potential_and_forces(
    density: np.ndarray,
    spacing: float,
    *,
    grav: float = GRAV_KPC_KMS2_MSUN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Potential and acceleration of a density cube with open boundaries.

    Parameters
    ----------
    density : (nx, ny, nz) array in Msun / kpc^3, on a uniform Cartesian grid.
    spacing : cell size in kpc (cubic cells).

    Returns
    -------
    (phi, ax, ay, az): phi in (km/s)^2, accelerations in (km/s)^2 / kpc, same shape
    as ``density``. ``a = -grad(phi)`` points toward mass (attractive).
    """
    nx, ny, nz = density.shape
    mass = density * spacing**3
    pad = (2 * nx, 2 * ny, 2 * nz)
    x = _signed_axis(pad[0], spacing)[:, None, None]
    y = _signed_axis(pad[1], spacing)[None, :, None]
    z = _signed_axis(pad[2], spacing)[None, None, :]
    r = np.sqrt(x * x + y * y + z * z)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_r = np.where(r > 0, 1.0 / r, 0.0)
        inv_r3 = np.where(r > 0, 1.0 / r**3, 0.0)
    inv_r[0, 0, 0] = 1.0 / (0.5 * spacing)  # softened self-term (affects phi only)

    mass_padded = np.zeros(pad)
    mass_padded[:nx, :ny, :nz] = mass
    mass_ft = np.fft.rfftn(mass_padded)

    axes = (0, 1, 2)

    def convolve(kernel: np.ndarray) -> np.ndarray:
        return np.fft.irfftn(mass_ft * np.fft.rfftn(kernel, axes=axes), s=pad, axes=axes)[:nx, :ny, :nz]

    phi = -grav * convolve(inv_r)
    ax = -grav * convolve(x * inv_r3)
    ay = -grav * convolve(y * inv_r3)
    az = -grav * convolve(z * inv_r3)
    return phi, ax, ay, az


def circular_velocity_profile(
    ax: np.ndarray,
    ay: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    r_bin_edges: np.ndarray,
    *,
    z_slab: float,
) -> np.ndarray:
    """Azimuthally-averaged circular velocity v_c(R) in the midplane.

    ``v_c^2 = -R * <a_R>_phi`` with the inward radial acceleration averaged over a
    thin midplane slab ``|z| < z_slab``. Returns v_c [km/s] per radial bin
    (NaN where no cells fall in the bin).
    """
    gx, gy, gz = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
    radius = np.hypot(gx, gy)
    a_radial = np.divide(gx * ax + gy * ay, radius, out=np.zeros_like(ax), where=radius > 0)
    in_slab = np.abs(gz) < z_slab
    v_c = np.full(len(r_bin_edges) - 1, np.nan)
    for i in range(len(r_bin_edges) - 1):
        sel = in_slab & (radius >= r_bin_edges[i]) & (radius < r_bin_edges[i + 1])
        if sel.any():
            r_mid = 0.5 * (r_bin_edges[i] + r_bin_edges[i + 1])
            v_c[i] = np.sqrt(max(-r_mid * float(a_radial[sel].mean()), 0.0))
    return v_c
