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
