"""Even-m azimuthal Fourier x smooth-(R,z) density representation.

The adopted grid-free 3D backbone (see
``docs/reports/2026-06-13-nbody-shen2010-deprojection.md``): represent a
cylindrical stellar density as

    rho(R, phi, z) = sum_{m in 0,2,4,6,8,10} a_m(R,z) cos(m phi) + b_m(R,z) sin(m phi)

with each complex amplitude map ``a_m(R,z)`` a *free* 2D field on a set of
control knots (log-spaced R, dense-near-plane z), reconstructed by smooth
interpolation. This is one non-stratified component (no image / disk-bulge
decomposition): a flat disk and a rounder bulge coexist natively, the X lives in
the z-structure of a_0,a_2,a_4, and the form is the AGAMA ``CylSpline`` form
(potential-ready). It beats the cylindrical grid in 3D rel-L2 on Shen2010 + TNG
554189 / 392276 (full particles).

A *uniform* (R,z) knot grid for every harmonic costs ~2000 coefficients, but the
high harmonics carry almost no power (m=6,8,10 ~0.1% each), so a power-weighted
per-harmonic (R,z) allocation (:func:`derive_power_allocation`) compresses the
target to a few hundred coefficients at nearly unchanged 3D fidelity.

The maps can be fit from a density callable (:func:`fit_fourier_rz`, used with an
SPH-KDE truth) or from an existing cylindrical (R, phi, z) grid
(:func:`fit_fourier_rz_from_grid`, used for the cross-galaxy deprojection
target). Both return the same model dict consumed by
:func:`reconstruct_fourier_rz`.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

EVEN_M: tuple[int, ...] = (0, 2, 4, 6, 8, 10)

Allocation = dict[int, tuple[int, int]]
"""Per-harmonic knot counts ``{m: (n_R, n_z_half)}``; total z knots = 2*n_z_half+1.
Harmonics absent from the mapping are dropped from the representation."""


def r_knots(n_r: int, r_max: float, r_min: float = 0.12) -> np.ndarray:
    """Log-spaced radial control knots."""
    return np.geomspace(r_min, r_max, int(n_r))


def z_knots(n_z_half: int, z_max: float, z_min: float = 0.12) -> np.ndarray:
    """Vertical control knots: dense near the plane, symmetric about z=0.

    Returns ``2*n_z_half + 1`` knots: the mirrored geomspace plus an explicit 0.
    """
    zp = np.geomspace(z_min, z_max, int(n_z_half))
    return np.concatenate([-zp[::-1], [0.0], zp])


def _resample_complex(
    cmap: np.ndarray,
    r_src: np.ndarray,
    z_src: np.ndarray,
    r_dst: np.ndarray,
    z_dst: np.ndarray,
) -> np.ndarray:
    """Bilinearly resample a complex (R,z) map from one knot grid to another."""
    query = np.array(np.meshgrid(r_dst, z_dst, indexing="ij")).reshape(2, -1).T
    shape = (len(r_dst), len(z_dst))
    re = RegularGridInterpolator((r_src, z_src), cmap.real, bounds_error=False, fill_value=None)(query)
    im = RegularGridInterpolator((r_src, z_src), cmap.imag, bounds_error=False, fill_value=None)(query)
    return (re + 1j * im).reshape(shape)


def _harmonic_mult(m: int) -> float:
    """Power/coefficient multiplicity: m=0 is real (1), m>0 carries +-m (2)."""
    return 1.0 if m == 0 else 2.0


def n_coefficients(model: dict) -> int:
    """Real-coefficient count of a model (m=0 real; m>0 real+imag)."""
    return int(sum(_harmonic_mult(m) * v["map"].size for m, v in model["maps"].items()))


def _build_model(
    full: dict[int, np.ndarray],
    r_full: np.ndarray,
    z_full: np.ndarray,
    *,
    r_max: float,
    z_max: float,
    alloc: Allocation | None,
    r_min: float,
    z_min: float,
) -> dict:
    """Assemble a model from full-resolution per-harmonic maps and an allocation.

    ``alloc=None`` keeps the full grid for every harmonic (uniform allocation).
    """
    if alloc is None:
        alloc = {m: (len(r_full), (len(z_full) - 1) // 2) for m in full}
    maps: dict[int, dict] = {}
    for m, (n_r_m, n_zh_m) in alloc.items():
        rk = r_knots(n_r_m, r_max, r_min)
        zk = z_knots(n_zh_m, z_max, z_min)
        maps[m] = {"R": rk, "z": zk, "map": _resample_complex(full[m], r_full, z_full, rk, zk)}
    model = {"maps": maps, "full": full, "R": r_full, "z": z_full, "r_max": r_max, "z_max": z_max}
    model["n_coeff"] = n_coefficients(model)
    return model


def fit_fourier_rz(
    rho,
    *,
    n_phi: int = 64,
    n_r: int = 14,
    n_z_half: int = 6,
    r_max: float = 15.0,
    z_max: float = 4.0,
    r_min: float = 0.12,
    z_min: float = 0.12,
    alloc: Allocation | None = None,
) -> dict:
    """Fit the even-m Fourier x (R,z) model to a density callable ``rho(points)``.

    Samples ``rho`` on a (R, phi, z) control lattice, takes the azimuthal FFT,
    keeps even m<=10, and (optionally) resamples each harmonic onto its allocated
    (R,z) knots. With ``alloc=None`` this is the uniform full-grid model.
    """
    r = r_knots(n_r, r_max, r_min)
    z = z_knots(n_z_half, z_max, z_min)
    phi = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    rr, pp, zz = np.meshgrid(r, phi, z, indexing="ij")
    pts = np.column_stack([(rr * np.cos(pp)).ravel(), (rr * np.sin(pp)).ravel(), zz.ravel()])
    dens = np.asarray(rho(pts)).reshape(len(r), n_phi, len(z))
    coeff = np.fft.rfft(dens, axis=1) / n_phi
    full = {m: coeff[:, m, :] for m in EVEN_M}
    return _build_model(full, r, z, r_max=r_max, z_max=z_max, alloc=alloc, r_min=r_min, z_min=z_min)


def fit_fourier_rz_from_grid(
    field_rphiz: np.ndarray,
    r_centers: np.ndarray,
    z_centers: np.ndarray,
    *,
    n_r: int = 14,
    n_z_half: int = 6,
    r_max: float = 15.0,
    z_max: float = 4.0,
    r_min: float = 0.12,
    z_min: float = 0.12,
    alloc: Allocation | None = None,
) -> dict:
    """Fit the model to an existing cylindrical (R, phi, z) density/residual grid.

    Azimuthally FFTs the grid (phi = axis 1), keeps even m<=10, resamples each
    harmonic from the grid's native (r_centers, z_centers) onto the canonical
    control knots, then applies ``alloc``. Used to build the cross-galaxy
    deprojection target from the milestone density-residual grids.
    """
    coeff = np.fft.rfft(field_rphiz, axis=1) / field_rphiz.shape[1]
    r = r_knots(n_r, r_max, r_min)
    z = z_knots(n_z_half, z_max, z_min)
    full = {m: _resample_complex(coeff[:, m, :], r_centers, z_centers, r, z) for m in EVEN_M}
    return _build_model(full, r, z, r_max=r_max, z_max=z_max, alloc=alloc, r_min=r_min, z_min=z_min)


def reconstruct_fourier_rz(points: np.ndarray, model: dict, *, nonneg: bool = True) -> np.ndarray:
    """Evaluate the reconstructed field at Cartesian ``points``.

    ``nonneg=True`` (default) clips to >=0 for a density; pass ``nonneg=False`` to
    reconstruct a signed field such as a baseline-subtracted residual.
    """
    radius = np.hypot(points[:, 0], points[:, 1])
    phi = np.arctan2(points[:, 1], points[:, 0])
    out = np.zeros(points.shape[0])
    for m, v in model["maps"].items():
        rk, zk, cmap = v["R"], v["z"], v["map"]
        query = np.column_stack([np.clip(radius, rk[0], rk[-1]), np.clip(points[:, 2], zk[0], zk[-1])])
        re = RegularGridInterpolator((rk, zk), cmap.real, bounds_error=False, fill_value=0.0)(query)
        if m == 0:
            out += re
        else:
            im = RegularGridInterpolator((rk, zk), cmap.imag, bounds_error=False, fill_value=0.0)(query)
            out += 2.0 * (re * np.cos(m * phi) - im * np.sin(m * phi))
    return np.clip(out, 0.0, None) if nonneg else out


def coefficient_vector(model: dict) -> np.ndarray:
    """Flatten a model's maps to a real coefficient vector (m=0 real; m>0 re,im).

    Harmonics are concatenated in :data:`EVEN_M` order; within a harmonic the
    (R,z) map is flattened C-order. Invertible with :func:`model_from_vector`
    given the same allocation. This is the predictand for the MDN / flow.
    """
    parts = []
    for m in EVEN_M:
        if m not in model["maps"]:
            continue
        cmap = model["maps"][m]["map"]
        parts.append(cmap.real.ravel())
        if m != 0:
            parts.append(cmap.imag.ravel())
    return np.concatenate(parts)


def model_from_vector(vector: np.ndarray, template: dict) -> dict:
    """Inverse of :func:`coefficient_vector`; rebuild a model from a flat vector.

    ``template`` supplies the knots/allocation (its map *values* are ignored).
    """
    maps: dict[int, dict] = {}
    i = 0
    for m in EVEN_M:
        if m not in template["maps"]:
            continue
        rk, zk = template["maps"][m]["R"], template["maps"][m]["z"]
        size = len(rk) * len(zk)
        re = vector[i : i + size].reshape(len(rk), len(zk))
        i += size
        if m == 0:
            cmap = re.astype(complex)
        else:
            im = vector[i : i + size].reshape(len(rk), len(zk))
            i += size
            cmap = re + 1j * im
        maps[m] = {"R": rk, "z": zk, "map": cmap}
    model = {"maps": maps, "R": template["R"], "z": template["z"],
             "r_max": template["r_max"], "z_max": template["z_max"]}
    model["n_coeff"] = n_coefficients(model)
    return model


def _trapezoid_weights(x: np.ndarray) -> np.ndarray:
    """Non-negative trapezoidal quadrature weights for samples at ``x``."""
    x = np.asarray(x, dtype=float)
    w = np.empty_like(x)
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    return np.abs(w)


def harmonic_power(model: dict) -> dict[int, float]:
    """Cylindrical-volume-weighted power per harmonic (proxy for L2 contribution).

    ``P_m = mult_m * sum_{R,z} |a_m(R,z)|^2 * R * dR * dz`` with the +-m
    multiplicity. Operates on whichever maps the model stores (full or compact).
    """
    powers: dict[int, float] = {}
    for m, v in model["maps"].items():
        rk, zk, cmap = v["R"], v["z"], v["map"]
        weight = (_trapezoid_weights(rk) * rk)[:, None] * _trapezoid_weights(zk)[None, :]
        powers[m] = float(_harmonic_mult(m) * np.sum((np.abs(cmap) ** 2) * weight))
    return powers


def _coarsening_tables(
    model: dict,
    cand_n_r: tuple[int, ...],
    cand_n_z_half: tuple[int, ...],
) -> tuple[dict[int, float], dict[int, dict[tuple[int, int], float]]]:
    """Per-harmonic full power and per-candidate discarded power for one model."""
    full = model["full"]
    r_full, z_full = model["R"], model["z"]
    weight = (_trapezoid_weights(r_full) * r_full)[:, None] * _trapezoid_weights(z_full)[None, :]
    r_max, z_max = model["r_max"], model["z_max"]
    r_min, z_min = float(r_full[0]), float(z_full[len(z_full) // 2 + 1])

    powers = {m: float(_harmonic_mult(m) * np.sum((np.abs(full[m]) ** 2) * weight)) for m in full}
    discarded: dict[int, dict[tuple[int, int], float]] = {}
    for m, cmap in full.items():
        mult = _harmonic_mult(m)
        table: dict[tuple[int, int], float] = {}
        for n_r_m in cand_n_r:
            for n_zh_m in cand_n_z_half:
                rk = r_knots(n_r_m, r_max, r_min)
                zk = z_knots(n_zh_m, z_max, z_min)
                coarse = _resample_complex(cmap, r_full, z_full, rk, zk)
                back = _resample_complex(coarse, rk, zk, r_full, z_full)
                table[(n_r_m, n_zh_m)] = float(mult * np.sum((np.abs(cmap - back) ** 2) * weight))
        discarded[m] = table
    return powers, discarded


def _waterfill(
    powers: dict[int, float],
    discarded: dict[int, dict[tuple[int, int], float]],
    *,
    capture: float,
) -> tuple[Allocation, dict]:
    """Reverse water-filling over (aggregated) power/discarded tables.

    A single global threshold ``T`` is raised until the cumulative discarded power
    (coarsening error for kept harmonics + full power for any harmonic whose total
    power falls below ``T``) reaches ``(1 - capture)`` of the total; each harmonic
    then takes the fewest-coefficient grid whose coarsening error is within ``T``.
    Low-power harmonics collapse to tiny grids or drop out entirely.
    """
    total = sum(powers.values())

    def coeffs(m: int, grid: tuple[int, int]) -> int:
        return int(_harmonic_mult(m) * grid[0] * (2 * grid[1] + 1))

    def allocate(threshold: float) -> tuple[Allocation, float]:
        alloc: Allocation = {}
        used = 0.0
        for m in powers:
            if powers[m] <= threshold:
                used += powers[m]  # drop the harmonic entirely
                continue
            feasible = [(coeffs(m, g), g, d) for g, d in discarded[m].items() if d <= threshold]
            _, grid, d = min(feasible) if feasible else (0, min(discarded[m], key=discarded[m].get), 0.0)
            used += discarded[m][grid] if feasible else min(discarded[m].values())
            alloc[m] = grid
        return alloc, used

    budget = (1.0 - capture) * total
    thresholds = sorted({0.0, *(d for tab in discarded.values() for d in tab.values()), *powers.values()})
    best_alloc, best_used = allocate(0.0)
    for threshold in thresholds:
        alloc, used = allocate(threshold)
        if used <= budget:
            best_alloc, best_used = alloc, used
        else:
            break
    info = {
        "powers": powers,
        "power_fraction": {m: powers[m] / total if total > 0 else 0.0 for m in powers},
        "total_power": total,
        "captured_fraction": 1.0 - best_used / total if total > 0 else 1.0,
        "capture_target": capture,
    }
    return best_alloc, info


def derive_power_allocation(
    model: dict,
    *,
    capture: float = 0.99,
    cand_n_r: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 10, 12, 14),
    cand_n_z_half: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
) -> tuple[Allocation, dict]:
    """Power-weighted per-harmonic (R,z) allocation for a single model.

    See :func:`_waterfill`. Returns ``(allocation, info)``.
    """
    powers, discarded = _coarsening_tables(model, cand_n_r, cand_n_z_half)
    return _waterfill(powers, discarded, capture=capture)


def derive_power_allocation_multi(
    models: list[dict],
    *,
    capture: float = 0.99,
    cand_n_r: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 10, 12, 14),
    cand_n_z_half: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
) -> tuple[Allocation, dict]:
    """Single shared allocation for a population: water-fill on summed power/discarded
    tables across ``models`` (all sharing the same control knots). More stable than a
    per-galaxy union when high harmonics are partly shot-noise. Returns ``(allocation, info)``.
    """
    agg_powers: dict[int, float] = {}
    agg_discarded: dict[int, dict[tuple[int, int], float]] = {}
    for model in models:
        powers, discarded = _coarsening_tables(model, cand_n_r, cand_n_z_half)
        for m in powers:
            agg_powers[m] = agg_powers.get(m, 0.0) + powers[m]
            tab = agg_discarded.setdefault(m, {})
            for grid, d in discarded[m].items():
                tab[grid] = tab.get(grid, 0.0) + d
    return _waterfill(agg_powers, agg_discarded, capture=capture)
