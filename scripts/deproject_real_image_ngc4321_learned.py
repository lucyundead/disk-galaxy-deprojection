"""Learned vertical step on the real NGC 4321 image (OOD) vs the geometric baseline.

Trains the conserving method's q_m(z;R) vertical-profile head on the milestone-2d TNG
table, then applies it to the real S4G image of NGC 4321 (resampled into the TNG mock
format) and compares the LEARNED vertical structure to the baseline sech^2(z/h). The
in-plane image anchor Sigma_m(R) is IDENTICAL in both reconstructions; only the vertical
profile q_m(z;R) differs, so this isolates what the learned step changes.

OOD handling: the head is trained on TNG barred-galaxy mocks; NGC 4321 is a real grand-
design spiral whose stellar mass is ~1 dex below the TNG sample. Of the 586 features only
two are absolute (log image mass, log baseline mass); they are set to the TNG train median
(in-distribution), while the other 584 -- mass-normalized image shape, central flux
fractions, geometry -- come from NGC 4321's actual light. M/L only scales v_c amplitude.

Run:
    PYTHONPATH=/home/lucyundead/Agama/build/lib.linux-x86_64-cpython-313 \\
        .venv/bin/python scripts/deproject_real_image_ngc4321_learned.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.colors import LogNorm

try:
    import agama
except ImportError:
    sys.path.insert(0, "/home/lucyundead/Agama/build/lib.linux-x86_64-cpython-313")
    import agama

from astropy import log as astropy_log

from deproject_fourier_rz_compare import fit_pca, train_mdn
from deproject_fourier_rz_conserving import EVEN_M, harmonics, r_resample
from dgdp import vertical_mixture as vm
from dgdp.agama_density import fourier_rz_to_agama_density
from dgdp.density3d import cylindrical_bin_volumes, make_cylindrical_grid_spec
from dgdp.fourier_rz import fit_fourier_rz_from_grid, r_knots
from train_density_residual_pca import make_density_residual_features, standardize_with_train

agama.setUnits(mass=1, length=1, velocity=1)
astropy_log.setLevel("ERROR")
ARCSEC_PER_RAD = 206264.806


def onsky_pa(wcs, cx, cy, pa_pix):
    c0 = wcs.pixel_to_world(cx, cy)
    c1 = wcs.pixel_to_world(cx + 50.0 * np.cos(pa_pix), cy + 50.0 * np.sin(pa_pix))
    return float(c0.position_angle(c1).deg)


def pixel_pa_from_onsky(wcs, cx, cy, pa_sky_deg):
    grid = np.radians(np.linspace(0.0, 180.0, 721, endpoint=False))
    skies = np.array([onsky_pa(wcs, cx, cy, g) % 180.0 for g in grid])
    diff = np.abs((skies - (pa_sky_deg % 180.0) + 90.0) % 180.0 - 90.0)
    return float(grid[int(np.argmin(diff))])


def reconstruct_smooth(vec_rows, anchor, rk_by_m, k_by_m, heights, r_grid, z_grid, phi_centers, vol):
    """Predicted mixture weights -> normalised q_m(z;R) -> anchor Sigma_m(R) -> cell mass.

    q is z-symmetric, >=0 (m=0) and int q dz=1 by construction (vertical_mixture.reconstruct);
    linear R-interpolation preserves the unit integral, so anchoring by Sigma_m(R) conserves the
    column mass exactly -- no post-hoc anchor renorm / z-symmetrise / taper.
    """
    rows = vec_rows.shape[0]
    rho = np.zeros((rows, len(r_grid), len(phi_centers), len(z_grid)))
    i = 0
    for m in EVEN_M:
        rk, kk = rk_by_m[m], k_by_m[m]
        size = len(rk) * kk
        wre = vec_rows[:, i:i + size].reshape(rows, len(rk), kk)
        i += size
        if m == 0:
            w = wre.astype(np.complex128)
        else:
            wim = vec_rows[:, i:i + size].reshape(rows, len(rk), kk)
            i += size
            w = wre + 1j * wim
        q_rk = vm.reconstruct(w, z_grid, heights, signed=(m != 0))
        q_grid = r_resample(q_rk, rk, r_grid)
        a_m = anchor[m][:, :, None] * q_grid
        if m == 0:
            rho += a_m.real[:, :, None, :]
        else:
            cos_m, sin_m = np.cos(m * phi_centers), np.sin(m * phi_centers)
            rho += 2.0 * (a_m.real[:, :, None, :] * cos_m[None, None, :, None]
                          - a_m.imag[:, :, None, :] * sin_m[None, None, :, None])
    return np.clip(rho, 0.0, None) * vol


def rms_z_profile(mass_grid, z_centers):
    """RMS |z| as a function of R (summed over phi)."""
    m_rz = mass_grid.sum(axis=1)  # (R, z)
    tot = np.maximum(m_rz.sum(axis=1), 1e-30)
    return np.sqrt((m_rz * z_centers[None, :] ** 2).sum(axis=1) / tot)


def direct_vc(mass_grid, r_grid, phi_centers, z_grid, radii, *, eps=0.15):
    """Independent midplane v_c(R): direct softened sum over grid cells as point masses.

    No AGAMA, no Fourier rep -- isolates the pure thin-vs-thick vertical effect at z=0.
    """
    grav = 4.300917270e-6  # kpc (km/s)^2 / Msun
    rr, pp, zz = np.meshgrid(r_grid, phi_centers, z_grid, indexing="ij")
    cx, cy, cz, m = (rr * np.cos(pp)).ravel(), (rr * np.sin(pp)).ravel(), zz.ravel(), mass_grid.ravel()
    keep = m > 0
    cx, cy, cz, m = cx[keep], cy[keep], cz[keep], m[keep]
    out = np.empty(len(radii))
    for i, radius in enumerate(radii):
        dx = cx - radius  # (cell - eval)_x at eval point (radius, 0, 0)
        inv = (dx * dx + cy * cy + cz * cz + eps * eps) ** -1.5
        accel_x = grav * np.sum(m * inv * dx)  # inward (<0) for interior mass
        out[i] = np.sqrt(max(-accel_x * radius, 0.0))
    return out


def deproject_ngc4321(args, spec):
    """Geometric deprojection of the S4G image -> baseline density grid + TNG-format image."""
    with fits.open(args.fits) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        header = hdul[0].header
    if args.pix_arcsec is not None:   # explicit geometry, e.g. an S4G cutout with no WCS
        pix_arcsec, wcs = float(args.pix_arcsec), None
    else:
        wcs = WCS(header)
        cd = np.array([[header["CD1_1"], header["CD1_2"]], [header["CD2_1"], header["CD2_2"]]])
        pix_arcsec = float(np.sqrt(np.abs(np.linalg.det(cd))) * 3600.0)
    pix_kpc = pix_arcsec / ARCSEC_PER_RAD * (args.distance_mpc * 1e3)

    if args.mask is not None:         # blank flagged contaminants before measuring the galaxy
        with fits.open(args.mask) as mh:
            data = np.where(np.asarray(mh[0].data, dtype=float) > 0, np.nan, data)
    border = np.concatenate([data[0], data[-1], data[:, 0], data[:, -1]])
    bkg, bkg_std = float(np.nanmedian(border)), float(np.nanstd(border))
    light = np.clip(np.nan_to_num(data, nan=bkg) - bkg, 0.0, None)
    ny, nx = light.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    if args.center_x is not None:
        cx, cy = float(args.center_x), float(args.center_y)
    else:
        iy0, ix0 = np.unravel_index(np.argmax(light), light.shape)
        win = (np.hypot(xx - ix0, yy - iy0) < 80) & (light > 5 * bkg_std)
        cx = float((light[win] * xx[win]).sum() / light[win].sum())
        cy = float((light[win] * yy[win]).sum() / light[win].sum())

    incl = args.inclination_deg
    if args.pa_pix_deg is not None:
        pa_pix = np.radians(args.pa_pix_deg)
    else:
        pa_pix = pixel_pa_from_onsky(wcs, cx, cy, args.pa_onsky_deg)
    sel = light > 2 * bkg_std
    x_sky = (xx[sel] - cx) * pix_kpc
    y_sky = (yy[sel] - cy) * pix_kpc
    mass = light[sel].astype(float)
    cpa, spa = np.cos(pa_pix), np.sin(pa_pix)
    x_major = x_sky * cpa + y_sky * spa
    y_minor = -x_sky * spa + y_sky * cpa

    # deprojected disk-plane Sigma(R,phi) x sech^2(z/h) -> baseline density grid
    y_disk = y_minor / max(np.cos(np.radians(incl)), 1e-3)
    radius = np.hypot(x_major, y_disk)
    phi = np.arctan2(y_disk, x_major)
    r_edges, phi_edges, z_edges = spec.r_edges_kpc, spec.phi_edges_rad, spec.z_edges_kpc
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    sigma_mass, _, _ = np.histogram2d(radius, phi, bins=(r_edges, phi_edges), weights=mass)
    wz = 1.0 / np.cosh(z_centers / args.scale_height_kpc) ** 2
    wz /= wz.sum()
    mass3d = sigma_mass[:, :, None] * wz[None, None, :]
    mass3d *= args.stellar_mass / mass3d.sum()
    baseline_density = mass3d / cylindrical_bin_volumes(spec)

    # OBSERVED (not deprojected) image in the TNG mock format: major axis -> axis 1
    fov = 0.5 * 192 * 0.35
    edges = np.linspace(-fov, fov, 193)
    img_tng, _, _ = np.histogram2d(y_minor, x_major, bins=(edges, edges), weights=mass)  # axis0=minor
    return {
        "baseline_density": baseline_density, "image_tng": img_tng.astype(np.float32),
        "light": light, "cx": cx, "cy": cy, "pix_kpc": pix_kpc, "incl": incl,
        "pa_pix": pa_pix, "bkg": bkg, "M_star": float(mass3d.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", type=Path, default=Path("NGC4321_m_c_r_f.fits"))
    ap.add_argument("--table", type=Path, default=Path("/mnt/e/dgdp-milestone2d/density_residual_table.npz"))
    ap.add_argument("--allocation", type=Path, default=Path("configs/fourier_rz_mixture.json"))
    ap.add_argument("--distance-mpc", type=float, default=15.2)
    ap.add_argument("--inclination-deg", type=float, default=30.0)
    ap.add_argument("--pa-onsky-deg", type=float, default=153.0)
    ap.add_argument("--bar-angle-deg", type=float, default=0.0)
    ap.add_argument("--stellar-mass", type=float, default=6.0e10)
    ap.add_argument("--scale-height-kpc", type=float, default=0.3)
    ap.add_argument("--n-comp", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260624)
    ap.add_argument("--r-max-rep", type=float, default=20.0)
    ap.add_argument("--n-r-out", type=int, default=32, help="R bins for the NGC4321 output grid (no retrain)")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/real_images"))
    ap.add_argument("--galaxy-name", default="NGC4321")
    ap.add_argument("--pix-arcsec", type=float, default=None, help="pixel scale arcsec; set to bypass WCS (S4G cutout)")
    ap.add_argument("--pa-pix-deg", type=float, default=None, help="disk major-axis PA in pixel frame (math, from +x)")
    ap.add_argument("--center-x", type=float, default=None)
    ap.add_argument("--center-y", type=float, default=None)
    ap.add_argument("--mask", type=Path, default=None, help="mask FITS; pixels>0 blanked before measuring")
    args = ap.parse_args()
    slug = args.galaxy_name.lower()

    spec = make_cylindrical_grid_spec(z_max_kpc=5.0, n_z=32)
    r_grid = 0.5 * (spec.r_edges_kpc[:-1] + spec.r_edges_kpc[1:])
    z_grid = 0.5 * (spec.z_edges_kpc[:-1] + spec.z_edges_kpc[1:])
    phi_centers = 0.5 * (spec.phi_edges_rad[:-1] + spec.phi_edges_rad[1:])
    vol = cylindrical_bin_volumes(spec).astype(np.float64)
    dz = float(np.diff(spec.z_edges_kpc)[0])

    payload = json.loads(args.allocation.read_text(encoding="utf-8"))
    alloc = {int(m): (int(a), int(b)) for m, (a, b) in payload["allocation"].items()}  # (n_R_knots, K)
    kp = payload["knot_params"]
    heights = np.asarray(payload["heights"], dtype=float)  # fixed sech^2 dictionary [kpc]
    rk_by_m = {m: r_knots(nr, kp["r_max"], kp["r_min"]) for m, (nr, _k) in alloc.items()}
    k_by_m = {m: len(heights) for m in alloc}

    # ---- (A) train the conserving q_m vertical-profile head on the milestone-2d table ----
    table = np.load(args.table)
    split = table["split"].astype(str)
    train, val = split == "train", split == "val"
    truth = table["truth_density"].astype(np.float32)
    a_true, _ = harmonics(truth, dz)
    feat = make_density_residual_features(
        table["images"].astype(np.float32), table["metadata"].astype(np.float32),
        baseline_grid_mass_msun=table["baseline_grid_mass_msun"].astype(np.float32),
        image_feature_size=24, central_pixel_scale_kpc=0.35,
    )
    x_all, feat_mean, feat_scale = standardize_with_train(feat, train)

    target_parts = []
    for m in EVEN_M:
        rk = rk_by_m[m]
        a_rk = r_resample(a_true[m], r_grid, rk)                          # (rows, nRk, nz), R-only
        w = vm.weights_target(a_rk, z_grid, heights, signed=(m != 0))
        target_parts.append(w.real.reshape(w.shape[0], -1))
        if m != 0:
            target_parts.append(w.imag.reshape(w.shape[0], -1))
    target = np.concatenate(target_parts, axis=1).astype(np.float32)
    coeff, vec, mean = fit_pca(target, train, args.n_comp)
    y_all, y_mean, y_std = standardize_with_train(coeff, train)
    print(f"training q_m mixture head: target {target.shape[1]} -> PCA{args.n_comp}; "
          f"rows train {train.sum()} val {val.sum()}")
    mdn = train_mdn(x_all, y_all, train, val, seed=args.seed, epochs=args.epochs)

    # TNG-train medians for the two absolute mass features (keep NGC4321 in-distribution there)
    img_mass_train = np.median(table["images"].astype(np.float64).sum(axis=(1, 2))[train])
    base_mass_train = np.median(table["baseline_grid_mass_msun"].astype(np.float64)[train])

    # ---- finer OUTPUT grid for NGC4321: the radial anchor Sigma_m(R) is image-measured and the
    # predicted q_m (on fixed coarse knots) is interpolated onto it -> NO retraining; the q_m head
    # above stays trained on the 32-R milestone-2d table. ----
    spec_out = make_cylindrical_grid_spec(z_max_kpc=5.0, n_z=32, n_r=args.n_r_out)
    r_grid = 0.5 * (spec_out.r_edges_kpc[:-1] + spec_out.r_edges_kpc[1:])
    vol = cylindrical_bin_volumes(spec_out).astype(np.float64)

    # ---- (B) deproject NGC4321; predict q_m with its own image anchor ----
    ngc = deproject_ngc4321(args, spec_out)
    baseline_mass = ngc["baseline_density"] * vol
    _, sigma_img = harmonics(ngc["baseline_density"][None], dz)
    anchor = {m: sigma_img[m] for m in EVEN_M}
    print(f"NGC4321 M* {ngc['M_star']:.2e} Msun vs TNG train median image mass {img_mass_train:.2e} "
          f"(absolute mass features are ~{np.log10(img_mass_train / ngc['M_star']):.1f} dex apart)")

    def predict_learned(image_total, base_mass_feat):
        img = ngc["image_tng"] * (image_total / max(ngc["image_tng"].sum(), 1e-30))
        feat_row = make_density_residual_features(
            img[None], np.array([[args.inclination_deg, 0.0, args.bar_angle_deg]], dtype=np.float32),
            baseline_grid_mass_msun=np.array([base_mass_feat], dtype=np.float32),
            image_feature_size=24, central_pixel_scale_kpc=0.35,
        )
        x = ((feat_row - feat_mean) / feat_scale).astype(np.float32)
        with torch.no_grad():
            sc = mdn.sample(torch.tensor(x), 128).numpy().mean(axis=1)
        pv = (sc * y_std + y_mean) @ vec + mean
        # int q dz=1 is exact on the fine z grid and preserved by linear R-interpolation, so the image
        # anchor Sigma_m(R,phi) is matched by construction -- the old post-hoc per-column renorm is gone.
        lm = reconstruct_smooth(pv, anchor, rk_by_m, k_by_m, heights, r_grid, z_grid, phi_centers, vol)[0]
        return lm * (args.stellar_mass / lm.sum())

    # ---- (C) primary: mass features at TNG median (in-distribution) -> isolates morphology ----
    learned_mass = predict_learned(img_mass_train, base_mass_train)
    # sensitivity: NGC4321's real (M/L-scaled) mass features (extrapolates below TNG range)
    learned_mass_real = predict_learned(ngc["M_star"], float(baseline_mass.sum()))

    rms_base = rms_z_profile(baseline_mass, z_grid)
    rms_learn = rms_z_profile(learned_mass, z_grid)
    tot_base = baseline_mass.sum(axis=(0, 1))
    tot_base /= tot_base.sum()
    tot_learn = learned_mass.sum(axis=(0, 1))
    tot_learn /= tot_learn.sum()
    rms_learn_real = rms_z_profile(learned_mass_real, z_grid)
    inner = r_grid < 12.0
    w_in = baseline_mass.sum(axis=(1, 2))[inner]

    def eff(rms):
        return float(np.sum(rms[inner] * w_in) / w_in.sum())

    rmsz_base_eff, rmsz_learn_eff, rmsz_real_eff = eff(rms_base), eff(rms_learn), eff(rms_learn_real)
    print(f"mass-weighted RMS|z| (R<12 kpc): baseline {rmsz_base_eff:.3f} kpc (sech2 h={args.scale_height_kpc}) "
          f"-> learned[TNG-median mass] {rmsz_learn_eff:.3f} kpc ({rmsz_learn_eff / rmsz_base_eff:.2f}x); "
          f"learned[real low mass] {rmsz_real_eff:.3f} kpc ({rmsz_real_eff / rmsz_base_eff:.2f}x)")

    # rotation curves by direct softened summation over the (finer-R) grid cells -- uses the FULL
    # radial resolution of the image-measured Sigma_m(R), no representation cap; this is exactly what
    # the finer R bins buy. uniform-h1 = NGC4321's own Sigma(R,phi) x a UNIFORM sech^2(h=1.0) control.
    wz_unif = 1.0 / np.cosh(z_grid / 1.0) ** 2
    wz_unif /= wz_unif.sum()
    uniform_mass = baseline_mass.sum(axis=2)[:, :, None] * wz_unif[None, None, :]
    radii = np.linspace(0.3, 18.0, 80)
    vc = {name: direct_vc(mg, r_grid, phi_centers, z_grid, radii)
          for name, mg in (("baseline", baseline_mass), ("learned", learned_mass),
                           ("uniform_h1", uniform_mass))}
    print(f"direct-sum v_c [km/s] on {len(r_grid)} R bins  baseline / learned (ratio) / uniform-h1 (ratio):")
    for rq in (1.0, 2.0, 3.0, 5.0, 10.0, 15.0):
        vb = float(np.interp(rq, radii, vc["baseline"]))
        vl = float(np.interp(rq, radii, vc["learned"]))
        vu = float(np.interp(rq, radii, vc["uniform_h1"]))
        print(f"  R={rq:5.1f} kpc: {vb:6.1f} / {vl:6.1f} ({vl / vb:.2f}) / {vu:6.1f} ({vu / vb:.2f})")

    # is Sigma(R) actually preserved between baseline and learned?  (else the comparison is confounded)
    sig_base = baseline_mass.sum(axis=(1, 2))
    sig_learn = learned_mass.sum(axis=(1, 2))
    rel = np.abs(sig_learn - sig_base) / np.maximum(sig_base, sig_base.max() * 1e-3)
    print(f"Sigma(R) preservation (baseline vs learned): max rel diff R<15 kpc {rel[r_grid < 15].max():.3f}; "
          f"mass frac R<3 kpc base {sig_base[r_grid < 3].sum() / sig_base.sum():.3f} "
          f"learn {sig_learn[r_grid < 3].sum() / sig_learn.sum():.3f}")
    print("learned RMS|z|(R) inner: "
          + "  ".join(f"R={rq}:{np.interp(rq, r_grid, rms_learn):.2f}" for rq in (0.3, 0.5, 0.8, 1.0, 1.5)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_dir / f"{slug}_arrays.npz",
             baseline_mass=baseline_mass, learned_mass=learned_mass, uniform_mass=uniform_mass,
             r_grid=r_grid, z_grid=z_grid, phi_centers=phi_centers,
             incl_deg=float(args.inclination_deg), pa_pix=float(ngc["pa_pix"]),
             cx=float(ngc["cx"]), cy=float(ngc["cy"]), pix_kpc=float(ngc["pix_kpc"]),
             bkg=float(ngc["bkg"]), scale_height=float(args.scale_height_kpc),
             bar_angle_deg=float(args.bar_angle_deg), stellar_mass=float(args.stellar_mass))
    metrics_path = args.output_dir / f"{slug}_learned_vs_baseline_metrics.json"
    metrics_path.write_text(json.dumps({
        "galaxy": args.galaxy_name, "rmsz_baseline_kpc": rmsz_base_eff, "rmsz_learned_kpc": rmsz_learn_eff,
        "rmsz_learned_realmass_kpc": rmsz_real_eff,
        "rmsz_ratio": rmsz_learn_eff / rmsz_base_eff, "scale_height_baseline_kpc": args.scale_height_kpc,
        "vc_baseline_10kpc": float(np.interp(10, radii, vc["baseline"])),
        "vc_learned_10kpc": float(np.interp(10, radii, vc["learned"])),
        "vc_radii_kpc": radii.tolist(), "n_r_out": int(args.n_r_out),
        "vc_baseline_kms": vc["baseline"].tolist(), "vc_learned_kms": vc["learned"].tolist(),
        "vc_uniform_kms": vc["uniform_h1"].tolist(),
        "note": "learned q_m head trained on TNG milestone-2d; OOD applied to real S4G NGC4321; "
                "absolute mass features set to TNG train median; M* normalized to literature.",
    }, indent=2, sort_keys=True), encoding="utf-8")

    # ---- figure: edge-on baseline | edge-on learned | RMS-z(R) | vertical profile | v_c ----
    fig, ax = plt.subplots(1, 5, figsize=(23, 4.5), constrained_layout=True)
    xs = np.linspace(-args.r_max_rep, args.r_max_rep, 200)
    zs = np.linspace(-2.5, 2.5, 120)
    xe, ze = np.meshgrid(xs, zs, indexing="ij")
    for axis, mass_grid, ttl in ((ax[0], baseline_mass, f"baseline edge-on\nsech^2 h={args.scale_height_kpc} kpc"),
                                 (ax[1], learned_mass, "learned edge-on\n(TNG q_m head, OOD)")):
        model = fit_fourier_rz_from_grid(mass_grid / vol, r_grid, z_grid, n_r=25, n_z_half=12,
                                         r_max=args.r_max_rep, z_max=5.0)
        azh = fourier_rz_to_agama_density(model, total_mass=float(mass_grid.sum()),
                                          r_max=args.r_max_rep, z_max=5.0)
        dmap = azh.density(np.column_stack([xe.ravel(), np.zeros(xe.size), ze.ravel()])).reshape(xe.shape)
        dmax = float(dmap.max())
        axis.imshow(dmap.T, origin="lower", extent=[-args.r_max_rep, args.r_max_rep, -2.5, 2.5], cmap="magma",
                    aspect="auto", norm=LogNorm(vmin=dmax * 3e-3, vmax=dmax))
        axis.set_title(ttl)
        axis.set_xlabel("x [kpc]")
        axis.set_ylabel("z [kpc]")

    ax[2].plot(r_grid, rms_base, "-", color="#4c72b0", lw=2, label=f"baseline ({rmsz_base_eff:.2f} kpc)")
    ax[2].plot(r_grid, rms_learn, "-", color="#c44e52", lw=2, label=f"learned ({rmsz_learn_eff:.2f} kpc)")
    ax[2].set_xlim(0, 18)
    ax[2].set_xlabel("R [kpc]")
    ax[2].set_ylabel("RMS |z| [kpc]")
    ax[2].set_title("vertical thickness vs R")
    ax[2].legend(fontsize=8)

    ax[3].plot(z_grid, tot_base, "-", color="#4c72b0", lw=2, label="baseline (sech^2)")
    ax[3].plot(z_grid, tot_learn, "-", color="#c44e52", lw=2, label="learned")
    ax[3].set_xlabel("z [kpc]")
    ax[3].set_ylabel("normalized mass(z)")
    ax[3].set_title("global vertical profile")
    ax[3].set_yscale("log")
    ax[3].set_ylim(1e-3, None)
    ax[3].legend(fontsize=8)

    ax[4].plot(radii, vc["baseline"], "-", color="#4c72b0", lw=2, label="baseline (thin)")
    ax[4].plot(radii, vc["learned"], "--", color="#c44e52", lw=2, label="learned (flaring)")
    ax[4].plot(radii, vc["uniform_h1"], ":", color="#2ca02c", lw=2, label="uniform h=1 (control)")
    ax[4].set_xlabel("R [kpc]")
    ax[4].set_ylabel("v_c [km/s]")
    ax[4].set_ylim(0, None)
    ax[4].set_title("rotation curve\n(same Sigma(R), different vertical)")
    ax[4].legend(fontsize=7)

    fig.suptitle(f"{args.galaxy_name}: learned TNG q_m vertical step (OOD) vs geometric sech^2 baseline "
                 "(same in-plane image anchor)", fontsize=12)
    fig_path = args.output_dir / f"{slug}_learned_vs_baseline.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {metrics_path} and {fig_path}")


if __name__ == "__main__":
    main()
