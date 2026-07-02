import numpy as np
import pytest

from dgdp.deproject import deproject
from dgdp.model import DeprojectionModel, save_bundle

HIDDEN, K, NPCA, TARGET = 4, 2, 2, 30            # target: m0 2x3 + (m2,m4) 2x3x2 each = 30
D_FEAT = 4 * 4 + 3 + 2 + 5                       # image_feature_size=4 features


def _bundle(path, *, with_scales, pca_vec, pca_mean):
    rng = np.random.default_rng(0)
    k = K if with_scales else 1                  # old-format bundles have a K=1 mean head
    # W1=W2=0: the trunk output is b2 regardless of input -> head math is hand-checkable
    mlp = dict(W1=np.zeros((HIDDEN, D_FEAT)), b1=np.ones(HIDDEN),
               W2=np.zeros((HIDDEN, HIDDEN)), b2=np.arange(1.0, HIDDEN + 1),
               Wm=0.1 * rng.normal(size=(k * NPCA, HIDDEN)), bm=rng.normal(size=k * NPCA))
    cfg = dict(heights=[0.3, 0.9, 2.0], alloc={0: [2, 3], 2: [2, 3], 4: [2, 3]},
               r_min=0.12, r_max=15.0,
               grid=dict(r_min=0.05, r_max=30.0, n_r=16, n_phi=16, z_max=5.0, n_z=16),
               image_feature_size=4, central_pixel_scale_kpc=0.35,
               img_mass_median=1e9, base_mass_median=1e9)
    if with_scales:
        mlp.update(Wl=np.zeros((K, HIDDEN)), bl=np.array([0.0, 2.0]),
                   Ws=np.zeros((K * NPCA, HIDDEN)), bs=np.full(K * NPCA, -6.0))
        cfg["n_components"] = K
    save_bundle(path, cfg=cfg,
                pca=dict(vec=pca_vec, mean=pca_mean,
                         y_mean=np.zeros(NPCA), y_std=np.ones(NPCA)),
                mlp=mlp, feat=dict(mean=np.zeros(D_FEAT), scale=np.ones(D_FEAT)))
    return DeprojectionModel.load(str(path)), mlp


def test_old_format_loads_and_refuses_sampling(tmp_path):
    m, mlp = _bundle(tmp_path / "old.npz", with_scales=False,
                     pca_vec=np.zeros((NPCA, TARGET)), pca_mean=np.full(TARGET, 0.5))
    assert m.n_components == 1 and m.Wl is None
    feat = np.zeros((1, D_FEAT))
    np.testing.assert_allclose(m.predict_weights(feat), np.full((1, TARGET), 0.5))
    with pytest.raises(ValueError, match="scale head"):
        m.sample_weights(feat, 4)


def test_mixture_mean_and_sampling_math(tmp_path):
    rng = np.random.default_rng(1)
    pca_vec = rng.normal(size=(NPCA, TARGET))
    m, mlp = _bundle(tmp_path / "new.npz", with_scales=True,
                     pca_vec=pca_vec, pca_mean=np.zeros(TARGET))
    feat = np.zeros((1, D_FEAT))
    h = np.arange(1.0, HIDDEN + 1)                          # trunk output by construction
    mu = (h @ mlp["Wm"].T + mlp["bm"]).reshape(K, NPCA)     # component means (scores)
    pi = np.exp([0.0, 2.0]) / np.exp([0.0, 2.0]).sum()
    np.testing.assert_allclose(m.predict_weights(feat)[0], (pi @ mu) @ pca_vec, rtol=1e-10)

    draws = m.sample_weights(feat, 400, rng=2)
    assert draws.shape == (1, 400, TARGET)
    comp_w = mu @ pca_vec                                   # the two component weight vectors
    d = np.linalg.norm(draws[0][:, None, :] - comp_w[None], axis=2)
    assert d.min(axis=1).max() < 0.05                       # sigma=e^-6: draws sit on a component
    frac1 = float((d.argmin(axis=1) == 1).mean())
    assert abs(frac1 - pi[1]) < 0.08                        # mixture proportions respected


def test_deproject_posterior_bands_plumbing(tmp_path):
    m, _ = _bundle(tmp_path / "det.npz", with_scales=True,
                   pca_vec=np.zeros((NPCA, TARGET)), pca_mean=np.full(TARGET, 0.5))
    yy, xx = np.mgrid[0:64, 0:64]
    img = 100.0 * np.exp(-np.hypot(xx - 32, yy - 32) / 8.0)
    r = deproject(img, distance_mpc=15.0, inclination_deg=30.0, pa_pix_deg=0.0,
                  center=(32, 32), pix_arcsec=1.0, stellar_mass=1e10, model=m,
                  reproject_iters=0, n_samples=6, seed=3)
    rz = r.rms_z_samples([0.5, 1.0, 2.0])
    vc = r.v_circ_samples([1.0, 2.0])
    assert rz.shape == (6, 3) and vc.shape == (6, 2)
    assert (rz > 0).all() and (vc > 0).all()
    assert rz.std(axis=0).max() < 1e-3                      # pca_vec=0 -> deterministic draws
    np.testing.assert_allclose(r.total_mass, 1e10, rtol=1e-3)