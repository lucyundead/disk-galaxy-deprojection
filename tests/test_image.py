import numpy as np
from astropy.io import fits

from dgdp.density3d import cylindrical_bin_volumes, make_cylindrical_grid_spec
from dgdp.image import geometric_baseline, load_image


def test_geometric_baseline_mass_and_shape(tmp_path):
    ny = nx = 200
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - 100, yy - 100)
    data = 100.0 * np.exp(-r / 20.0)                       # face-on exp disk
    p = tmp_path / "disk.fits"
    fits.PrimaryHDU(data.astype(np.float32)).writeto(p)
    spec = make_cylindrical_grid_spec(z_max_kpc=5.0, n_z=32)
    gi = load_image(str(p), pix_arcsec=1.0, pa_pix_deg=0.0, center=(100, 100),
                    inclination_deg=0.0, distance_mpc=15.0)
    base = geometric_baseline(gi, spec, scale_height_kpc=0.3, stellar_mass=5e10)
    vol = cylindrical_bin_volumes(spec)
    assert np.isclose((base["baseline_density"] * vol).sum(), 5e10, rtol=1e-6)
    assert base["image_tng"].shape == (192, 192)
    assert base["baseline_density"].shape == (spec.r_edges_kpc.size - 1, 48, 32)


def test_load_image_fills_masked_pixels(tmp_path):
    # masked foreground stars must be filled from their surroundings, not read as zero
    # flux (a zero there becomes a fake dip in the deprojected density)
    ny = nx = 128
    yy, xx = np.mgrid[0:ny, 0:nx]
    data = 100.0 * np.exp(-np.hypot(xx - 64, yy - 64) / 20.0)
    maskarr = np.zeros((ny, nx), np.float32)
    maskarr[70:78, 50:58] = 1.0                              # a masked 'star' on the disk
    fits.PrimaryHDU(data.astype(np.float32)).writeto(tmp_path / "d.fits")
    fits.PrimaryHDU(maskarr).writeto(tmp_path / "m.fits")
    gi = load_image(str(tmp_path / "d.fits"), pix_arcsec=1.0, pa_pix_deg=0.0,
                    center=(64, 64), inclination_deg=0.0, distance_mpc=15.0,
                    mask=str(tmp_path / "m.fits"))
    ratio = gi.light[70:78, 50:58] / data[70:78, 50:58]
    assert np.abs(ratio - 1.0).max() < 0.2                   # ~local values, not zeros


def test_geometric_baseline_subpixel_deposit_spreads_central_pixel(tmp_path):
    # a pixel is wider than the inner log-R rings: depositing pixel CENTRES made the central
    # anchors phi-deltas (|2 Sigma_2|/Sigma_0 = 2) and Sigma(R) a comb of single-ring teeth
    ny = nx = 64
    data = np.zeros((ny, nx), np.float32)
    data[33, 33] = 100.0                                   # one pixel ~0.10 kpc off-centre
    p = tmp_path / "pix.fits"
    fits.PrimaryHDU(data).writeto(p)
    spec = make_cylindrical_grid_spec(z_max_kpc=5.0, n_z=32)
    gi = load_image(str(p), pix_arcsec=1.0, pa_pix_deg=0.0, center=(32, 32),
                    inclination_deg=0.0, distance_mpc=15.0)
    base = geometric_baseline(gi, spec, scale_height_kpc=0.3, stellar_mass=1e9)
    sig = (base["baseline_density"] * cylindrical_bin_volumes(spec)).sum(axis=2)   # (R,phi)
    assert (sig.sum(axis=1) > 0).sum() >= 2                # spread over rings, not one tooth
    coeff = np.fft.rfft(sig, axis=1) / sig.shape[1]
    ratio = 2 * np.abs(coeff[:, 2].sum()) / coeff[:, 0].real.sum()
    assert ratio < 1.95                                    # below the phi-delta limit of 2
