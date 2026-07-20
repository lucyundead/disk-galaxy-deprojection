"""End-to-end deprojection: image -> geometric baseline -> features -> learned q_m -> density."""
from __future__ import annotations

import csv
import os
import warnings
from dataclasses import dataclass
from importlib.resources import files

import numpy as np

from dgdp import rotation
from dgdp.density3d import cylindrical_bin_volumes, make_cylindrical_grid_spec
from dgdp.features import make_features
from dgdp.harmonics import reconstruct_density
from dgdp.image import geometric_baseline, load_image
from dgdp.model import DeprojectionModel
from dgdp.reproject import anchors_from_sigma, high_m_sigma, refine_sigma

_BUNDLED = files("dgdp.models").joinpath("dgdp_fixed_dict.npz")


def _resolve_mass(*, stellar_mass, luminosity, ml, image_light, zeropoint, band_solar_mag,
                  distance_mpc):
    """Resolve the absolute stellar mass (Msun) and whether we fell back to a relative scale."""
    if stellar_mass is not None:
        return float(stellar_mass), False                    # M/L already folded in; ml ignored
    if luminosity is not None:
        return float(ml) * float(luminosity), False
    if zeropoint is not None and band_solar_mag is not None and image_light is not None:
        flux = float(np.nansum(image_light))
        app_mag = -2.5 * np.log10(max(flux, 1e-30)) + float(zeropoint)
        dist_mod = 5.0 * np.log10(distance_mpc * 1e6) - 5.0
        abs_mag = app_mag - dist_mod
        lum = 10.0 ** (-0.4 * (abs_mag - float(band_solar_mag)))
        return float(ml) * lum, False
    return 1.0, True                                         # relative-only scale


@dataclass
class DeprojectionResult:
    density_3d: np.ndarray                                   # (nR,nphi,nz) Msun/kpc^3 (or relative)
    grid: dict                                              # r, phi, z bin centres
    total_mass: float
    relative: bool
    _vol: np.ndarray
    reproj: dict | None = None                              # reprojection-loop diagnostics:
    #   history: obs-weighted mean |log(obs/model)| per measured state (history[0] = before any
    #   correction; the result uses the best = min); ratio: sky-plane obs/model map at the best
    #   state (axis 0 = minor axis) on edges_kpc bins -- a per-galaxy self-consistency check.
    _samples: dict | None = None                            # posterior-sampling context
    _ctx: dict | None = None                                # regrid context (native sig + model)
    ood: dict | None = None                                 # training-distribution diagnostic
    #   (dgdp.ood): d2 + empirical percentile among TNG training rows + nearest training
    #   analogs (subhalo ids). None for custom bundles the shipped reference doesn't match.

    def regrid(self, *, n_r=None, n_phi=None, n_z=None, r_min=None, r_max=None,
               z_max=None) -> "DeprojectionResult":
        """Re-evaluate the deprojection on a finer cylindrical grid (e.g. for hydro ICs).

        Not an interpolation of the coarse cube: the vertical mixture and the m<=4 azimuthal
        terms are ANALYTIC and are evaluated exactly on the new grid; only the in-plane
        surface density Sigma(R,phi) is bilinearly interpolated (in ln R and periodic phi)
        from the native deposit grid, so no in-plane information beyond the native
        resolution is invented. Mass conservation (per-ring anchoring + total) is re-imposed
        on the new grid. Unspecified parameters keep the native values; regridding a
        regridded result still interpolates from the ORIGINAL native grid (no compounding).
        """
        c = self._ctx
        if c is None:
            raise ValueError("no regrid context -- this result was not built by deproject()")
        from dgdp.harmonics import reconstruct_density as _rec
        from dgdp.reproject import anchors_from_sigma, high_m_sigma
        g = c["grid"]
        spec = make_cylindrical_grid_spec(
            r_min_kpc=r_min if r_min is not None else g["r_min"],
            r_max_kpc=r_max if r_max is not None else g["r_max"],
            n_r=n_r if n_r is not None else g["n_r"],
            n_phi=n_phi if n_phi is not None else g["n_phi"],
            z_max_kpc=z_max if z_max is not None else g["z_max"],
            n_z=n_z if n_z is not None else g["n_z"])
        r_f = 0.5 * (spec.r_edges_kpc[:-1] + spec.r_edges_kpc[1:])
        z_f = 0.5 * (spec.z_edges_kpc[:-1] + spec.z_edges_kpc[1:])
        phi_f = 0.5 * (spec.phi_edges_rad[:-1] + spec.phi_edges_rad[1:])
        vol_f = cylindrical_bin_volumes(spec).astype(float)
        area_f = vol_f[:, 0, 0] / float(np.diff(spec.z_edges_kpc)[0])

        # bilinear (ln R, periodic phi) interpolation of the native Sigma2D field
        r_n, phi_n = c["r_grid"], c["phi_centers"]
        s2d = c["sig"] / c["base_area"][:, None]
        s2d_ext = np.concatenate([s2d, s2d[:, :1]], axis=1)
        phi_ext = np.concatenate([phi_n, [phi_n[0] + 2 * np.pi]])
        lr_n, lr_f = np.log(r_n), np.log(np.clip(r_f, r_n[0], r_n[-1]))
        ir = np.clip(np.searchsorted(lr_n, lr_f) - 1, 0, len(lr_n) - 2)
        fr = np.clip((lr_f - lr_n[ir]) / (lr_n[ir + 1] - lr_n[ir]), 0.0, 1.0)
        pw = phi_ext[0] + np.mod(phi_f - phi_ext[0], 2 * np.pi)
        ip = np.clip(np.searchsorted(phi_ext, pw) - 1, 0, len(phi_ext) - 2)
        fp = np.clip((pw - phi_ext[ip]) / (phi_ext[ip + 1] - phi_ext[ip]), 0.0, 1.0)
        s2d_f = (s2d_ext[np.ix_(ir, ip)] * (1 - fr)[:, None] * (1 - fp)[None, :]
                 + s2d_ext[np.ix_(ir, ip + 1)] * (1 - fr)[:, None] * fp[None, :]
                 + s2d_ext[np.ix_(ir + 1, ip)] * fr[:, None] * (1 - fp)[None, :]
                 + s2d_ext[np.ix_(ir + 1, ip + 1)] * fr[:, None] * fp[None, :])
        s2d_f[r_f > c["r_edges_max"]] = 0.0                  # never extrapolate beyond the data
        sig_f = s2d_f * area_f[:, None]

        rho = _rec(c["vec"], anchors_from_sigma(sig_f, area_f), c["rk_by_m"], c["k_by_m"],
                   c["heights"], r_f, z_f, phi_f,
                   sigma_hi=high_m_sigma(sig_f, area_f, phi_f)[None])[0]
        mass = rho * vol_f
        if not self.relative:
            mass *= self.total_mass / max(mass.sum(), 1e-30)
        return DeprojectionResult(mass / vol_f, {"r": r_f, "phi": phi_f, "z": z_f},
                                  float(mass.sum()), self.relative, vol_f, None, None, c)

    def bisymmetrize_for_dynamics(self, *, inner_radius_kpc: float,
                                  outer_radius_kpc: float) -> "DeprojectionResult":
        """Return a smooth, bisymmetric density derived for dynamical modelling.

        The full 3D mass grid is truncated to ``m=0,2,4`` at every ``(R,z)``.
        This removes high-order image texture and enforces exact 180-degree
        symmetry.  A raised-cosine radial taper retains those modes inside
        ``inner_radius_kpc`` and removes ``m=2,4`` at and beyond
        ``outer_radius_kpc``, leaving an axisymmetric outer density.

        Positivity is enforced by damping the retained non-axisymmetric modes
        only where necessary.  Total mass and the azimuthally averaged ``(R,z)``
        mass distribution are conserved.  The native result is not modified.
        The returned object deliberately has no reprojection or posterior
        context because this dynamics product is not an exact fit to the input
        image; retain the native result as the projection-consistent
        intermediate.
        """
        inner = float(inner_radius_kpc)
        outer = float(outer_radius_kpc)
        if not (np.isfinite(inner) and np.isfinite(outer) and 0.0 <= inner < outer):
            raise ValueError("inner_radius_kpc must be finite, non-negative, and below outer_radius_kpc")
        if self.density_3d.shape[1] % 2:
            raise ValueError("bisymmetrize_for_dynamics requires an even number of azimuth bins")

        mass = self.density_3d * self._vol
        coeff = np.fft.rfft(mass, axis=1)
        retained = np.zeros_like(coeff)
        for mode in (0, 2, 4):
            if mode < coeff.shape[1]:
                retained[:, mode, :] = coeff[:, mode, :]
        filtered = np.fft.irfft(retained, n=mass.shape[1], axis=1)

        axisymmetric = filtered.mean(axis=1, keepdims=True)
        nonaxisymmetric = filtered - axisymmetric
        minimum_nonaxisymmetric = nonaxisymmetric.min(axis=1)
        positivity_scale = np.ones_like(minimum_nonaxisymmetric)
        needs_damping = minimum_nonaxisymmetric < 0.0
        positivity_scale[needs_damping] = np.minimum(
            1.0,
            axisymmetric[:, 0, :][needs_damping] / -minimum_nonaxisymmetric[needs_damping],
        ) * (1.0 - 32.0 * np.finfo(float).eps)
        filtered = axisymmetric + positivity_scale[:, None, :] * nonaxisymmetric

        radius = self.grid["r"]
        taper = np.ones_like(radius)
        transition = (radius > inner) & (radius < outer)
        taper[radius >= outer] = 0.0
        phase = (radius[transition] - inner) / (outer - inner)
        taper[transition] = 0.5 * (1.0 + np.cos(np.pi * phase))
        processed_mass = axisymmetric + taper[:, None, None] * (filtered - axisymmetric)

        source_rz = mass.sum(axis=1)
        processed_rz = processed_mass.sum(axis=1)
        processed_mass *= np.divide(
            source_rz,
            processed_rz,
            out=np.ones_like(source_rz),
            where=processed_rz > 0.0,
        )[:, None, :]
        return DeprojectionResult(
            processed_mass / self._vol,
            dict(self.grid),
            float(processed_mass.sum()),
            self.relative,
            self._vol,
            None,
            None,
            None,
            self.ood,
        )

    def _sample_masses(self):
        """(S,nR,nphi,nz) mass grids reconstructed from the posterior weight draws (cached).

        Samples share the (loop-corrected) Sigma anchors of the mean prediction: the bands
        quantify the vertical-profile posterior at fixed in-plane surface density.
        """
        s = self._samples
        if s is None:
            raise ValueError("no posterior samples -- call deproject(..., n_samples=N)")
        if "mass" not in s:
            from dgdp.harmonics import reconstruct_density as _rec
            rho = _rec(s["weights"], s["anchor"], s["rk_by_m"], s["k_by_m"], s["heights"],
                       self.grid["r"], self.grid["z"], self.grid["phi"],
                       sigma_hi=s.get("sigma_hi"))
            mass = rho * self._vol[None]
            if not self.relative:
                tot = np.maximum(mass.sum(axis=(1, 2, 3), keepdims=True), 1e-30)
                mass *= self.total_mass / tot
            s["mass"] = mass
        return s["mass"]

    def rms_z_samples(self, radii_kpc):
        """Posterior draws of RMS|z|(R), shape (n_samples, len(radii))."""
        mass = self._sample_masses()
        m_rz = mass.sum(axis=2)
        tot = np.maximum(m_rz.sum(axis=2), 1e-30)
        rms = np.sqrt((m_rz * self.grid["z"][None, None, :] ** 2).sum(axis=2) / tot)
        radii = np.asarray(radii_kpc, float)
        return np.stack([np.interp(radii, self.grid["r"], row) for row in rms])

    def v_circ_samples(self, radii_kpc):
        """Posterior draws of v_c(R) [km/s], shape (n_samples, len(radii))."""
        mass = self._sample_masses()
        radii = np.asarray(radii_kpc, float)
        return np.stack([rotation.v_circ(mg, self.grid["r"], self.grid["phi"],
                                         self.grid["z"], radii) for mg in mass])

    def v_circ(self, radii_kpc):
        return rotation.v_circ(self.density_3d * self._vol, self.grid["r"], self.grid["phi"],
                               self.grid["z"], np.asarray(radii_kpc, float))

    def rms_z(self, radii_kpc):
        m_rz = (self.density_3d * self._vol).sum(axis=1)     # (R,z)
        tot = np.maximum(m_rz.sum(axis=1), 1e-30)
        rms = np.sqrt((m_rz * self.grid["z"][None, :] ** 2).sum(axis=1) / tot)
        return np.interp(np.asarray(radii_kpc, float), self.grid["r"], rms)

    def scale_height(self, radii_kpc):
        """Sech^2 scale height h_z(R) [kpc]: rho(z) ∝ sech^2(z / h_z) fit to each radius'
        phi-summed vertical profile (exponential scale height = h_z/2). Same convention as
        the geometric baseline and Comeron+2018-style edge-on decompositions -- directly
        comparable with observations, unlike the tail-weighted RMS|z| moment."""
        from dgdp import vertical_mixture as vm
        m_rz = (self.density_3d * self._vol).sum(axis=1)
        h = vm.sech2_height_fit(m_rz, self.grid["z"])
        return np.interp(np.asarray(radii_kpc, float), self.grid["r"], h)

    def scale_height_samples(self, radii_kpc):
        """Posterior draws of h_z(R), shape (n_samples, len(radii))."""
        from dgdp import vertical_mixture as vm
        mass = self._sample_masses()
        m_rz = mass.sum(axis=2)                              # (S,R,z)
        h = vm.sech2_height_fit(m_rz.reshape(-1, m_rz.shape[-1]),
                                self.grid["z"]).reshape(m_rz.shape[:2])
        radii = np.asarray(radii_kpc, float)
        return np.stack([np.interp(radii, self.grid["r"], row) for row in h])

    @property
    def face_on(self):
        return (self.density_3d * self._vol).sum(axis=2)     # (R,phi) column mass

    @property
    def edge_on(self):
        return (self.density_3d * self._vol).sum(axis=1)     # (R,z)

    def potential(self, R, z):
        return rotation.potential(self.density_3d, self.grid["r"], self.grid["phi"],
                                  self.grid["z"], total_mass=self.total_mass,
                                  query_R=np.atleast_1d(R), query_z=np.atleast_1d(z))

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        np.savez(os.path.join(out_dir, "density.npz"), density_3d=self.density_3d,
                 r=self.grid["r"], phi=self.grid["phi"], z=self.grid["z"],
                 total_mass=self.total_mass)
        radii = np.linspace(self.grid["r"][0], min(self.grid["r"][-1], 18.0), 80)
        vc = self.v_circ(radii)
        with open(os.path.join(out_dir, "rotation_curve.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["R_kpc", "v_c_kms"])
            w.writerows(zip(radii, vc))


def deproject(image, *, distance_mpc, inclination_deg, pa_pix_deg=None, pa_onsky_deg=None,
              center=None, pix_arcsec=None, mask=None, ml=1.0, stellar_mass=None,
              luminosity=None, zeropoint=None, band_solar_mag=None, bar_angle_deg=0.0,
              scale_height_kpc=0.3, model=None, reproject_iters=2, n_samples=0,
              seed=None) -> DeprojectionResult:
    """Deproject a galaxy image into a 3D stellar-mass cube + rotation curve.

    See the package README / spec for the M/L scaling paths. Returns a DeprojectionResult.

    reproject_iters: max correction passes of the reprojection-consistency loop
    (dgdp.reproject) adjusting the in-plane Sigma anchors so the THICK reconstruction
    actually reprojects to the observed image (the plain geometric stretch assumes zero
    thickness). Early-stops with revert, so the result never reprojects worse than the
    single-shot anchors. 0 = legacy single-shot. Diagnostics land in result.reproj.

    n_samples: posterior draws of the vertical-profile weights (needs a bundle with the
    mixture scale heads). The result then supports rms_z_samples / v_circ_samples for
    uncertainty bands; the density_3d itself stays the mixture-mean prediction.
    """
    m = model if isinstance(model, DeprojectionModel) else DeprojectionModel.load(str(model or _BUNDLED))
    g = m.grid
    spec = make_cylindrical_grid_spec(r_min_kpc=g["r_min"], r_max_kpc=g["r_max"], n_r=g["n_r"],
                                      n_phi=g["n_phi"], z_max_kpc=g["z_max"], n_z=g["n_z"])
    r_grid = 0.5 * (spec.r_edges_kpc[:-1] + spec.r_edges_kpc[1:])
    z_grid = 0.5 * (spec.z_edges_kpc[:-1] + spec.z_edges_kpc[1:])
    phi = 0.5 * (spec.phi_edges_rad[:-1] + spec.phi_edges_rad[1:])
    dz = float(np.diff(spec.z_edges_kpc)[0])
    vol = cylindrical_bin_volumes(spec).astype(float)

    gi = load_image(image, distance_mpc=distance_mpc, inclination_deg=inclination_deg,
                    pix_arcsec=pix_arcsec, pa_pix_deg=pa_pix_deg, pa_onsky_deg=pa_onsky_deg,
                    center=center, mask=mask)
    total_mass, relative = _resolve_mass(stellar_mass=stellar_mass, luminosity=luminosity, ml=ml,
                                         image_light=gi.light, zeropoint=zeropoint,
                                         band_solar_mag=band_solar_mag, distance_mpc=distance_mpc)
    base = geometric_baseline(gi, spec, scale_height_kpc=scale_height_kpc,
                              stellar_mass=max(total_mass, 1.0))

    # Scale the TNG-format image to the training median total (sets the 2 absolute mass features
    # in-distribution; every other feature is scale-invariant), then predict the mixture weights.
    img = base["image_tng"]
    img = img * (m.img_mass_median / max(img.sum(), 1e-30))
    feat = make_features(img[None], np.array([[inclination_deg, 0.0, bar_angle_deg]], np.float32),
                         baseline_grid_mass_msun=np.array([m.base_mass_median], np.float32),
                         image_feature_size=m.image_feature_size,
                         central_pixel_scale_kpc=m.central_pixel_scale_kpc)
    vec = m.predict_weights(feat)

    from dgdp.ood import ood_check
    ood = ood_check(feat, m.feat_mean, m.feat_scale)
    if ood is not None and ood["percentile"] > 99.0:
        warnings.warn(
            f"input galaxy sits at the {ood['percentile']:.1f}th percentile of the TNG "
            "training feature distribution -- the prediction is an extrapolation and the "
            "uncertainty bands are not calibrated there", stacklevel=2)

    base_area = vol[:, 0, 0] / dz

    def _reconstruct(sig):
        return reconstruct_density(vec, anchors_from_sigma(sig, base_area), m.rk_by_m,
                                   m.k_by_m, m.heights, r_grid, z_grid, phi,
                                   sigma_hi=high_m_sigma(sig, base_area, phi)[None])[0]

    sig, reproj = base["sigma_mass"], None
    if reproject_iters:
        sig, hist, ratio = refine_sigma(sig, base["image_tng"], _reconstruct,
                                        r_grid, phi, z_grid, inclination_deg,
                                        base["image_edges_kpc"], iters=reproject_iters)
        reproj = {"history": hist, "ratio": ratio, "edges_kpc": base["image_edges_kpc"]}

    rho = _reconstruct(sig)
    mass = rho * vol
    if not relative:
        mass *= total_mass / max(mass.sum(), 1e-30)
    density = mass / vol

    samples = None
    if n_samples:
        ws = m.sample_weights(feat, int(n_samples), np.random.default_rng(seed))[0]
        samples = {"weights": ws, "anchor": anchors_from_sigma(sig, base_area),
                   "sigma_hi": high_m_sigma(sig, base_area, phi)[None], "rk_by_m": m.rk_by_m,
                   "k_by_m": m.k_by_m, "heights": m.heights}
    ctx = {"sig": sig, "base_area": base_area, "vec": vec, "rk_by_m": m.rk_by_m,
           "k_by_m": m.k_by_m, "heights": m.heights, "grid": dict(g),
           "r_grid": r_grid, "phi_centers": phi, "r_edges_max": float(spec.r_edges_kpc[-1])}
    return DeprojectionResult(density, {"r": r_grid, "phi": phi, "z": z_grid},
                              float(mass.sum()), relative, vol, reproj, samples, ctx, ood)
