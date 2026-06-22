"""Tilt-invariant boxy/peanut/X strength metric, with validation before any census.

A boxy/peanut/X bulge is a CONTOUR-SHAPE feature: the edge-on (bar side-on)
surface density has boxy / pinched iso-contours. The standard quantification is
the m=4 ("boxiness") Fourier term of the iso-density shape. We make it
tilt-invariant by measuring in the shape's OWN inertia principal frame, so a
mere tilt (which my earlier off-plane metric mistook for a peanut) is absorbed
by the frame and scores ~0.

This is a measurement choice only; it does NOT de-tilt the deprojection target
(that stays in the whole-galaxy disk frame). It just answers "is there a peanut"
invariant to orientation.

`--validate` runs four ground-truth cases (run this FIRST, sanity-check it):
  A. Shen2010 fine truth          -> strong peanut  -> expect HIGH
  B. thinnest real TNG disk       -> non-peanut     -> expect ~0
  C. synthetic tilted ellipsoid   -> non-peanut     -> expect ~0  (tilt control)
  D. Shen2010 tilted by 15 deg    -> tilted peanut  -> expect HIGH (tilt control)
`--census` (separate, after the metric is trusted) runs the 185 TNG galaxies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator

from dgdp.density3d import build_cylindrical_density_grid, make_cylindrical_grid_spec
from dgdp.orientation import align_particles_to_disk_bar_frame
from dgdp.types import ParticleSet


def edgeon_surface_density(density, r_edges, phi_edges, z_edges, *, half_xy=8.0, y_int=4.0, vox=0.1):
    """Sigma(x,z): integrate the cyl density over the LOS |y|<y_int (bar side-on)."""
    x = np.arange(-half_xy + 0.5 * vox, half_xy, vox)
    z = np.arange(-z_edges[-1] + 0.5 * vox, z_edges[-1], vox)
    y = np.arange(-y_int + 0.5 * vox, y_int, vox)
    nR, nphi, nz = len(r_edges) - 1, len(phi_edges) - 1, len(z_edges) - 1
    Xg, Yg = np.meshgrid(x, y, indexing="ij")
    Rg = np.hypot(Xg, Yg)
    Pg = np.arctan2(Yg, Xg)
    ir = np.clip(np.searchsorted(r_edges, Rg) - 1, 0, nR - 1)
    ip = np.clip(np.searchsorted(phi_edges, Pg) - 1, 0, nphi - 1)
    valid = Rg < r_edges[-1]
    sigma = np.zeros((len(x), len(z)))
    for k, zv in enumerate(z):
        iz = int(np.searchsorted(z_edges, zv) - 1)
        if 0 <= iz < nz:
            cell = np.where(valid, density[ir, ip, iz], 0.0)
            sigma[:, k] = cell.sum(axis=1) * vox
    return x, z, sigma


def peanut_strength(x, z, sigma, *, r_in=0.6, r_out=4.0, z_max=3.0, n_theta=120, n_rho=200):
    """Isophote b4 boxiness, tilt-invariant. For a set of iso-density contours,
    trace the contour radius vs angle in the shape's circularised principal frame
    and take the m=4 term. An ellipse (any axis ratio, any tilt -> incl. a thin
    disk) gives a circle in circularised coords -> b4=0; a boxy/peanut gives the
    contour bulging at the diagonals -> b4>0. Radial profile is normalised out
    because each contour is a fixed density level. >0 boxy/peanut, <0 disky."""
    X, Z = np.meshgrid(x, z, indexing="ij")
    w = np.clip(sigma, 0.0, None)
    reg = (np.abs(X) <= r_out) & (np.abs(Z) <= z_max)
    W = w * reg
    tot = float(W.sum())
    if tot <= 0:
        return {"strength": 0.0, "n_levels": 0, "tilt_deg": 0.0, "q": 1.0}
    xc = float((W * X).sum() / tot)
    zc = float((W * Z).sum() / tot)
    Xc, Zc = X - xc, Z - zc
    Ixx = float((W * Xc * Xc).sum() / tot)
    Izz = float((W * Zc * Zc).sum() / tot)
    Ixz = float((W * Xc * Zc).sum() / tot)
    theta = 0.5 * np.arctan2(2 * Ixz, Ixx - Izz)          # principal axis = the tilt
    tr, det = Ixx + Izz, Ixx * Izz - Ixz * Ixz
    disc = np.sqrt(max(tr * tr / 4 - det, 0.0))
    q = np.sqrt(max(tr / 2 - disc, 1e-12)) / np.sqrt(max(tr / 2 + disc, 1e-12))
    interp = RegularGridInterpolator((x, z), sigma, bounds_error=False, fill_value=0.0)

    th = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    rho = np.linspace(0.05, r_out, n_rho)
    TH, RHO = np.meshgrid(th, rho, indexing="ij")          # (n_theta, n_rho)
    Xp, Zp = RHO * np.cos(TH), q * RHO * np.sin(TH)        # circularised -> principal ellipse
    c, s = np.cos(theta), np.sin(theta)
    xr = xc + c * Xp - s * Zp                              # principal -> real (absorbs tilt)
    zr = zc + s * Xp + c * Zp
    s_polar = interp(np.column_stack((xr.ravel(), zr.ravel()))).reshape(TH.shape)

    prof = s_polar.mean(axis=0)
    levels = prof.max() * np.array([0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.65, 0.8])
    b4s = []
    for level in levels:
        rc = np.zeros(n_theta)
        for i in range(n_theta):
            above = np.flatnonzero(s_polar[i] >= level)
            rc[i] = rho[above[-1]] if above.size else 0.0
        if np.mean(rc > 0) < 0.95:
            continue
        rm = rc.mean()
        if rm < r_in or rm > r_out:
            continue
        b4s.append(float(np.mean(rc * np.cos(4 * th)) / rm))   # boxy => rc large at 45deg => b4<0
    if not b4s:
        return {"strength": 0.0, "strength_boxiest": 0.0, "n_levels": 0, "tilt_deg": float(np.degrees(theta)), "q": float(q)}
    return {
        "strength": float(-np.median(b4s)),               # median over levels (disk+bulge mixed)
        "strength_boxiest": float(-np.quantile(b4s, 0.1)),  # boxiest isophote = the bulge/peanut
        "n_levels": len(b4s),
        "tilt_deg": float(np.degrees(theta)),
        "q": float(q),
    }


def _rotation_xz(deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def _grid_from_particles(pos, *, total_mass=4.5e10, z_max=4.0, n_z=64):
    spec = make_cylindrical_grid_spec(z_max_kpc=z_max, n_z=n_z)
    masses = np.full(pos.shape[0], total_mass / pos.shape[0])
    g = build_cylindrical_density_grid(ParticleSet(positions_kpc=pos, masses_msun=masses), spec)
    vol = (
        0.5 * (spec.r_edges_kpc[1:] ** 2 - spec.r_edges_kpc[:-1] ** 2)[:, None, None]
        * np.diff(spec.phi_edges_rad)[None, :, None]
        * np.diff(spec.z_edges_kpc)[None, None, :]
    )
    return (g.mass_msun / vol), spec.r_edges_kpc, spec.phi_edges_rad, spec.z_edges_kpc


def _shen_aligned_positions(cache_dir):
    pos = np.load(cache_dir / "positions_raw.npy").astype(np.float64)
    vel_path = cache_dir / "velocities_raw.npy"
    vel = np.load(vel_path).astype(np.float64) if vel_path.exists() else None
    pos = pos - pos.mean(axis=0)
    masses = np.full(pos.shape[0], 1.0)
    al = align_particles_to_disk_bar_frame(
        ParticleSet(positions_kpc=pos, masses_msun=masses, velocities_kms=vel),
        normal_radius_kpc=10.0, bar_radius_kpc=5.0,
    )
    return al.particles.positions_kpc


def validate(args):
    cases = []

    # A. Shen2010 fine truth (real strong peanut)
    g = np.load(args.fine_truth)
    cases.append(("A: Shen2010 truth", g["truth_density"], g["r_edges_kpc"], g["phi_edges_rad"], g["z_edges_kpc"]))

    # B. thinnest real TNG disk (non-peanut)
    t = np.load(args.table)
    gid = np.asarray(t["galaxy_id"])
    re, pe, ze = t["r_edges_kpc"], t["phi_edges_rad"], t["z_edges_kpc"]
    rc = 0.5 * (re[:-1] + re[1:])
    zc = 0.5 * (ze[:-1] + ze[1:])
    bar = rc < 5.0
    truth = t["truth_density"]
    offfrac = {}
    for gg in np.unique(gid):
        row = int(np.flatnonzero(gid == gg)[0])
        d = truth[row][bar]
        num = d[:, :, np.abs(zc) > 0.8].sum()
        den = d.sum()
        offfrac[int(gg)] = num / den if den > 0 else 0.0
    thin = min(offfrac, key=offfrac.get)
    row = int(np.flatnonzero(gid == thin)[0])
    print(f"thinnest TNG disk = subhalo {thin} (off-plane frac {offfrac[thin]:.3f})")
    cases.append((f"B: thin TNG {thin}", truth[row], re, pe, ze))

    # C. synthetic tilted smooth ellipsoid (non-peanut)  -- Gaussian triaxial "bar"
    rng = np.random.default_rng(7)
    n = 600000
    ell = np.column_stack((rng.normal(0, 2.5, n), rng.normal(0, 1.0, n), rng.normal(0, 0.5, n)))
    ell = ell @ _rotation_xz(15.0).T
    dC, rC, pC, zC = _grid_from_particles(ell)
    cases.append(("C: tilted ellipsoid 15deg", dC, rC, pC, zC))

    # D. Shen2010 tilted 15 deg (tilted real peanut)
    shen = _shen_aligned_positions(args.cache_dir) @ _rotation_xz(15.0).T
    dD, rD, pD, zD = _grid_from_particles(shen)
    cases.append(("D: Shen2010 tilted 15deg", dD, rD, pD, zD))

    # E. synthetic thin exponential disk (axisymmetric, NO peanut) -- false-positive control
    rng2 = np.random.default_rng(11)
    rr = -3.0 * np.log(1.0 - rng2.random(700000))
    rr = rr[rr < 15.0]
    ph = rng2.uniform(0, 2 * np.pi, rr.size)
    zz = 0.3 * np.arctanh(np.clip(2 * rng2.random(rr.size) - 1, -0.999, 0.999))
    disk = np.column_stack((rr * np.cos(ph), rr * np.sin(ph), zz))
    dE, rE, pE, zE = _grid_from_particles(disk)
    cases.append(("E: thin exp disk", dE, rE, pE, zE))

    fig, axes = plt.subplots(1, len(cases), figsize=(4.2 * len(cases), 4.4), constrained_layout=True)
    print("\n=== validation (strength>0 = boxy/peanut, ~0 = disky/ellipse) ===")
    results = {}
    for ax, (name, dens, r_e, p_e, z_e) in zip(axes, cases, strict=True):
        x, z, sigma = edgeon_surface_density(dens.astype(np.float64), r_e, p_e, z_e)
        m = peanut_strength(x, z, sigma)
        results[name] = m
        print(f"{name:28s} strength={m['strength']:+.4f}  b4_min={m.get('b4_min', 0.0):+.4f}  "
              f"nlev={m['n_levels']}  tilt={m['tilt_deg']:+.1f}  q={m['q']:.2f}")
        vmax = float(sigma.max())
        ax.imshow(sigma.T, origin="lower", extent=[x[0], x[-1], z[0], z[-1]], cmap="magma",
                  norm=LogNorm(vmin=vmax * 5e-3, vmax=vmax), aspect="auto")
        ax.contour(x, z, sigma.T, levels=vmax * np.array([0.04, 0.1, 0.25, 0.5]),
                   colors="cyan", linewidths=0.7)
        ax.set_title(f"{name}\nstrength={m['strength']:+.3f} (tilt {m['tilt_deg']:+.0f}deg)", fontsize=10)
        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("z [kpc]")
        ax.set_xlim(-6, 6)
        ax.set_ylim(-3, 3)
    fig.suptitle("Peanut-strength metric validation (m=4 boxiness in principal frame)", fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {args.output}")
    sA, sB, sC, sD, sE = (results[k]["strength"] for k in list(results))
    # ground-truth controls: peanuts (A,D) boxy>0; ellipsoid (C) neutral ~0;
    # thin disk (E) disky<0 (NOT a false positive); all well separated.
    ok = (sA > 0.015) and (sD > 0.015) and (abs(sC) < 0.015) and (sE < 0.005)
    print(f"VALIDATION {'PASS' if ok else 'CHECK'} (ground truth): "
          f"peanut A={sA:+.3f} D={sD:+.3f} (boxy>0) | ellipsoid C={sC:+.3f} (~0) | "
          f"thin disk E={sE:+.3f} (disky<0)  [real B={sB:+.3f}]")


def _strength_of(dens, r_e, p_e, z_e, **region):
    x, z, sig = edgeon_surface_density(np.asarray(dens, dtype=np.float64), r_e, p_e, z_e)
    return x, z, sig, peanut_strength(x, z, sig, **region)


def _region_for(bar_length_kpc):
    """Bulge-focused, bar-scaled measurement region (isolates bulge from disk)."""
    L = float(bar_length_kpc)
    return {
        "r_in": max(0.12 * L, 0.3),
        "r_out": min(max(0.8 * L, 2.0), 6.0),
        "z_max": min(max(0.6 * L, 1.5), 4.5),
    }


def census(args):
    import pandas as pd

    t = np.load(args.table)
    gid = np.asarray(t["galaxy_id"])
    re, pe, ze = t["r_edges_kpc"], t["phi_edges_rad"], t["z_edges_kpc"]
    truth = t["truth_density"]
    zmax, nz = float(ze[-1]), len(ze) - 1
    man = pd.read_csv(args.manifest).drop_duplicates("subhalo_id").set_index("subhalo_id")
    barlen = man["bar_length"].to_dict()

    # anchors (bar-scaled to their own structure), report median & boxiest-isophote
    shen = _shen_aligned_positions(args.cache_dir)
    aS = _strength_of(*_grid_from_particles(shen, z_max=zmax, n_z=nz), **_region_for(4.4))[3]
    rng = np.random.default_rng(7)
    ell = np.column_stack((rng.normal(0, 2.5, 600000), rng.normal(0, 1.0, 600000), rng.normal(0, 0.5, 600000)))
    aE = _strength_of(*_grid_from_particles(ell, z_max=zmax, n_z=nz), **_region_for(6.0))[3]
    rng2 = np.random.default_rng(11)
    rr = -3.0 * np.log(1.0 - rng2.random(700000))
    rr = rr[rr < 15.0]
    ph = rng2.uniform(0, 2 * np.pi, rr.size)
    zz = 0.3 * np.arctanh(np.clip(2 * rng2.random(rr.size) - 1, -0.999, 0.999))
    disk = np.column_stack((rr * np.cos(ph), rr * np.sin(ph), zz))
    aD = _strength_of(*_grid_from_particles(disk, z_max=zmax, n_z=nz), **_region_for(3.0))[3]
    print("anchors (median | boxiest-isophote):")
    for lab, m in (("thin disk", aD), ("ellipsoid", aE), ("Shen2010 peanut", aS)):
        print(f"  {lab:16s} {m['strength']:+.3f} | {m['strength_boxiest']:+.3f}")

    uniq = np.unique(gid)
    med, box = {}, {}
    for g in uniq:
        row = int(np.flatnonzero(gid == g)[0])
        m = _strength_of(truth[row], re, pe, ze, **_region_for(barlen.get(int(g), 3.0)))[3]
        med[int(g)], box[int(g)] = m["strength"], m["strength_boxiest"]
    ids = np.array(list(med.keys()))
    vals = np.array(list(med.values()))    # median b4 in the bar-scaled bulge region (controls-validated)
    bx = np.array([box[i] for i in ids])
    shen_v = aS["strength"]
    boxy_thr = 0.5 * (aE["strength"] + shen_v)   # midway ellipsoid<->Shen
    print(f"\ncensus of {len(vals)} TNG galaxies (disk frame, no de-tilt, bar-scaled bulge region):")
    print("  using median b4 in the bulge region (anchors: "
          f"disk {aD['strength']:+.3f}, ellipsoid {aE['strength']:+.3f}, Shen {shen_v:+.3f}):")
    print(f"  median={np.median(vals):+.3f}  p16={np.percentile(vals, 16):+.3f}  "
          f"p84={np.percentile(vals, 84):+.3f}  max={vals.max():+.3f}")
    print(f"  fraction disky (<0)                              = {np.mean(vals < 0):.2f}")
    print(f"  fraction boxy/peanut (> {boxy_thr:+.3f}, midway ell->Shen) = {np.mean(vals > boxy_thr):.2f}")
    print(f"  fraction >= Shen2010 ({shen_v:+.3f})                      = {np.mean(vals >= shen_v):.2f}")

    order = np.argsort(vals)
    pd.DataFrame({"subhalo_id": ids[order], "strength_median": vals[order],
                  "strength_boxiest": bx[order]}).to_csv(
        args.output.parent / "tng_peanut_census.csv", index=False)

    picks = [("most disky", ids[order][0]), ("median", ids[order][len(order) // 2]),
             (f"~Shen ({shen_v:+.2f})", ids[np.argmin(np.abs(vals - shen_v))]),
             ("most boxy", ids[order][-1])]
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 4)
    axh = fig.add_subplot(gs[0, :])
    axh.hist(vals, bins=np.linspace(min(aD["strength"], vals.min()) - 0.01, vals.max() + 0.01, 35),
             color="#4c72b0", edgecolor="k", alpha=0.85)
    for val, lab, col in ((aD["strength"], "thin disk", "green"), (aE["strength"], "ellipsoid", "gray"),
                          (shen_v, "Shen2010", "#b3403c")):
        axh.axvline(val, color=col, lw=2, ls="--", label=f"{lab} ({val:+.3f})")
    axh.axvline(0, color="k", lw=0.8)
    axh.set_xlabel("bulge boxiness (median b4; <0 disky, ~0 ellipse, >0 boxy/peanut)")
    axh.set_ylabel("galaxies")
    axh.set_title(f"TNG50 bulge boxy/peanut census ({len(vals)} galaxies): "
                  f"{np.mean(vals > boxy_thr) * 100:.0f}% boxy/peanut, {np.mean(vals >= shen_v) * 100:.0f}% >= Shen2010")
    axh.legend(fontsize=9)
    for col, (lab, g) in enumerate(picks):
        row = int(np.flatnonzero(gid == g)[0])
        x, z, sig, m = _strength_of(truth[row], re, pe, ze, **_region_for(barlen.get(int(g), 3.0)))
        a = fig.add_subplot(gs[1, col])
        vmax = float(sig.max())
        a.imshow(sig.T, origin="lower", extent=[x[0], x[-1], z[0], z[-1]], cmap="magma",
                 norm=LogNorm(vmin=vmax * 5e-3, vmax=vmax), aspect="auto")
        a.contour(x, z, sig.T, levels=vmax * np.array([0.04, 0.1, 0.25, 0.5]), colors="cyan", linewidths=0.7)
        a.set_title(f"{lab}\n{g}: {m['strength']:+.3f} (L={barlen.get(int(g), 3.0):.1f})", fontsize=10)
        a.set_xlabel("x [kpc]")
        a.set_ylabel("z [kpc]")
        a.set_xlim(-6, 6)
        a.set_ylim(-3, 3)
    fig.suptitle("Is TNG50 peanut-poor? Bulge boxy/peanut census (bar-scaled, disk frame)", fontsize=14)
    out = args.output.parent / "tng_peanut_census.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} and tng_peanut_census.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("validate", "census"), default="validate")
    ap.add_argument("--fine-truth", type=Path, default=Path("outputs/nbody_shen2010/nbody_shen2010_fine_truth_grid.npz"))
    ap.add_argument("--table", type=Path, default=Path("/mnt/e/dgdp-milestone2d/density_residual_table.npz"))
    ap.add_argument("--manifest", type=Path, default=Path("outputs/tng50_milestone2c_clean3d/manifest.csv"))
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/nbody_shen2010/cache"))
    ap.add_argument("--output", type=Path, default=Path("outputs/nbody_shen2010/figures/peanut_metric_validation.png"))
    args = ap.parse_args()
    if args.mode == "validate":
        validate(args)
    else:
        census(args)


if __name__ == "__main__":
    main()
