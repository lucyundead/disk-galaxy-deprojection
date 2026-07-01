"""Wrap a predicted 3D stellar density as a native AGAMA ``Density`` / ``Potential``.

The even-m Fourier x (R,z) deprojection (:mod:`dgdp.fourier_rz`) and its
mass-conserving image-anchored variant produce a smooth cylindrical stellar density
that is already in the AGAMA ``CylSpline`` / ``DensityAzimuthalHarmonic`` form. These
helpers wrap such a *predicted* density -- any Cartesian density callable, or a
``fourier_rz`` model -- as a genuine ``agama.Density`` object, and as an
``agama.Potential``, with **no retraining**. The predicted total mass is preserved
*exactly* (the harmonic fit is linear in the input density, so a single global
rescale is exact to machine precision), so a mass-conserving deprojection stays
mass-conserving end to end: image -> predicted density -> potential -> dynamics.

AGAMA is an optional dependency (the user's prebuilt copy, imported via the Agama
build directory on ``PYTHONPATH``); it is imported lazily so the rest of ``dgdp``
never depends on it.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

DensityCallable = Callable[[np.ndarray], np.ndarray]


def _import_agama():
    try:
        import agama
    except ImportError as exc:  # pragma: no cover - exercised only without AGAMA
        raise RuntimeError(
            "AGAMA is not importable. Add the prebuilt Agama build dir to PYTHONPATH, e.g. "
            "PYTHONPATH=/home/lucyundead/Agama/build/lib.linux-x86_64-cpython-313, and call "
            "agama.setUnits(mass=1, length=1, velocity=1) before use."
        ) from exc
    return agama


def to_agama_density(
    density: DensityCallable,
    *,
    total_mass: float | None = None,
    mmax: int = 6,
    symmetry: str = "triaxial",
    gridsize_r: int = 25,
    gridsize_z: int = 25,
    r_min: float = 0.1,
    r_max: float = 15.0,
    z_min: float = 0.05,
    z_max: float = 5.0,
) -> object:
    """Wrap a Cartesian density callable as an ``agama.DensityAzimuthalHarmonic``.

    Parameters
    ----------
    density:
        Callable mapping Cartesian positions ``(N, 3)`` in kpc to densities ``(N,)``
        in Msun/kpc^3 (non-negative). AGAMA samples it on its own (R, z, phi) lattice.
    total_mass:
        If given, the returned density is rescaled so ``totalMass()`` equals
        ``total_mass`` exactly (the azimuthal-harmonic fit is linear in the input, so
        one rescale is exact). Pass the predicted density's own total to keep a
        mass-conserving deprojection intact through the wrap. The grid domain
        ``(r_max, z_max)`` must cover the density's support, or the conserved mass is
        only the mass within the domain.

    Notes
    -----
    ``symmetry='triaxial'`` keeps even azimuthal harmonics m=0,2,..,mmax with the bar
    on the x axis -- the same even-m bar symmetry the Fourier x (R,z) target assumes.
    """
    agama = _import_agama()

    def _build(scale: float) -> object:
        return agama.Density(
            type="DensityAzimuthalHarmonic",
            density=lambda x: scale * np.asarray(density(np.atleast_2d(np.asarray(x, dtype=float)))),
            gridsizeR=gridsize_r, gridsizez=gridsize_z, mmax=mmax,
            Rmin=r_min, Rmax=r_max, zmin=z_min, zmax=z_max, symmetry=symmetry,
        )

    dens = _build(1.0)
    if total_mass is not None:
        fitted = dens.totalMass()
        if fitted > 0:
            dens = _build(float(total_mass) / float(fitted))
    return dens


def fourier_rz_to_agama_density(
    model: dict,
    *,
    nonneg: bool = True,
    total_mass: float | None = None,
    **kwargs,
) -> object:
    """Wrap a :mod:`dgdp.fourier_rz` model as an ``agama.Density``.

    Convenience over :func:`to_agama_density` that reconstructs the model with
    :func:`dgdp.fourier_rz.reconstruct_fourier_rz`. Extra keyword arguments
    (``mmax``, ``total_mass`` domain, grid sizes, ...) are forwarded.
    """
    from dgdp.fourier_rz import reconstruct_fourier_rz

    def _density(points: np.ndarray) -> np.ndarray:
        return reconstruct_fourier_rz(np.atleast_2d(np.asarray(points, dtype=float)), model, nonneg=nonneg)

    return to_agama_density(_density, total_mass=total_mass, **kwargs)


def to_agama_potential(
    density: object,
    *,
    mmax: int = 6,
    symmetry: str = "triaxial",
    gridsize_r: int = 25,
    gridsize_z: int = 25,
    r_min: float = 0.1,
    r_max: float = 25.0,
    z_min: float = 0.05,
    z_max: float = 10.0,
) -> object:
    """Solve the ``CylSpline`` potential of an ``agama.Density`` (or density callable).

    The CylSpline grid is wider than the density grid by default so the potential is
    well behaved out to large radius. ``density`` may be an ``agama.Density`` object
    (e.g. from :func:`to_agama_density`) or a Cartesian density callable.
    """
    agama = _import_agama()
    return agama.Potential(
        type="CylSpline", density=density, symmetry=symmetry, mmax=mmax,
        gridSizeR=gridsize_r, gridSizeZ=gridsize_z, Rmin=r_min, Rmax=r_max, zmin=z_min, zmax=z_max,
    )


def circular_velocity(potential: object, radii: np.ndarray, *, n_phi: int = 16) -> np.ndarray:
    """Azimuthally-averaged circular velocity v_c(R) [km/s] from an ``agama.Potential``.

    For a barred (non-axisymmetric) potential the in-plane radial force depends on phi;
    this averages the inward radial acceleration over ``n_phi`` azimuths at z=0 and
    returns ``sqrt(R * <-F_R>)`` (the m=0 circular speed).
    """
    radii = np.atleast_1d(np.asarray(radii, dtype=float))
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    out = np.empty_like(radii)
    for i, r in enumerate(radii):
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), np.zeros_like(phi)])
        force = np.asarray(potential.force(pts))
        f_radial = force[:, 0] * np.cos(phi) + force[:, 1] * np.sin(phi)  # F . R_hat (inward < 0)
        out[i] = np.sqrt(max(-float(np.mean(f_radial)) * r, 0.0))
    return out
