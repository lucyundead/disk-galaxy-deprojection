"""End-to-end deprojection: image -> geometric baseline -> features -> learned q_m -> density."""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from importlib.resources import files

import numpy as np

from dgdp import rotation
from dgdp.density3d import cylindrical_bin_volumes, make_cylindrical_grid_spec
from dgdp.features import make_features
from dgdp.harmonics import reconstruct_density
from dgdp.image import geometric_baseline, load_image
from dgdp.model import DeprojectionModel
from dgdp.reproject import anchors_from_sigma, refine_sigma

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

    def v_circ(self, radii_kpc):
        return rotation.v_circ(self.density_3d * self._vol, self.grid["r"], self.grid["phi"],
                               self.grid["z"], np.asarray(radii_kpc, float))

    def rms_z(self, radii_kpc):
        m_rz = (self.density_3d * self._vol).sum(axis=1)     # (R,z)
        tot = np.maximum(m_rz.sum(axis=1), 1e-30)
        rms = np.sqrt((m_rz * self.grid["z"][None, :] ** 2).sum(axis=1) / tot)
        return np.interp(np.asarray(radii_kpc, float), self.grid["r"], rms)

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
              scale_height_kpc=0.3, model=None, reproject_iters=2) -> DeprojectionResult:
    """Deproject a galaxy image into a 3D stellar-mass cube + rotation curve.

    See the package README / spec for the M/L scaling paths. Returns a DeprojectionResult.

    reproject_iters: max correction passes of the reprojection-consistency loop
    (dgdp.reproject) adjusting the in-plane Sigma anchors so the THICK reconstruction
    actually reprojects to the observed image (the plain geometric stretch assumes zero
    thickness). Early-stops with revert, so the result never reprojects worse than the
    single-shot anchors. 0 = legacy single-shot. Diagnostics land in result.reproj.
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

    def _reconstruct(anchor):
        return reconstruct_density(vec, anchor, m.rk_by_m, m.k_by_m, m.heights,
                                   r_grid, z_grid, phi)[0]

    base_area = vol[:, 0, 0] / dz
    sig, reproj = base["sigma_mass"], None
    if reproject_iters:
        sig, hist, ratio = refine_sigma(sig, base["image_tng"], _reconstruct, base_area,
                                        r_grid, phi, z_grid, inclination_deg,
                                        base["image_edges_kpc"], iters=reproject_iters)
        reproj = {"history": hist, "ratio": ratio, "edges_kpc": base["image_edges_kpc"]}

    rho = _reconstruct(anchors_from_sigma(sig, base_area))
    mass = rho * vol
    if not relative:
        mass *= total_mass / max(mass.sum(), 1e-30)
    density = mass / vol
    return DeprojectionResult(density, {"r": r_grid, "phi": phi, "z": z_grid},
                              float(mass.sum()), relative, vol, reproj)
